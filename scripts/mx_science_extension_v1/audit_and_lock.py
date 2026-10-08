"""Audit patient boundaries and lock the MX-05 add-on design before new results.

This command never trains, accesses image voxels, or performs outer evaluation.
Its lock is a design lock, not GPU authorization: execution source hashes and a
completed parent MX-07 freeze are required by a later execution preregistration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.audit_inputs import PHYSICAL, read_csv
from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2_1.supervise_v21 import immutable_json

CONFIG = Path("configs/experiments/method_x_science_extension_v1_20260927.json")
PARENT = Path("results/final_results/method_x_fasttrack_v2_20260927_mx05_gate_import")
OUTPUT = Path("results/final_results/method_x_science_extension_v1_20260927")
AUDIT_NAME = "day1_patient_provenance_audit.json"
LOCK_NAME = "design_lock.json"
REQUIRED = ("patient_hash", "sample_hash", "substudy", "ab_label",
            "pet_day_t0", "t1_day_t0", "abs_pet_t1_gap_days", "t1_file_hash")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _ids(rows: list[dict[str, str]]) -> set[str]:
    return {str(row.get("patient_hash") or "") for row in rows}


def partition_audit(index_rows: list[dict[str, str]],
                    tests: list[list[dict[str, str]]],
                    trains: list[list[dict[str, str]]],
                    held: list[dict[str, str]],
                    *, seeds: tuple[int, ...] = (13, 29, 47),
                    val_fraction: float = 0.15) -> dict:
    """Aggregate-only split audit; no subject-level rows leave this function."""
    blockers: set[str] = set()
    if len(tests) != 5 or len(trains) != 5 or not 0 < val_fraction < 1:
        raise ValueError("EXTENSION_SPLIT_SHAPE_INVALID")
    outer = [row for fold in tests for row in fold]
    all_rows = outer + held
    if any(not all(str(row.get(key) or "").strip() for key in REQUIRED)
           for row in outer):
        blockers.add("SPLIT_REQUIRED_MRI_PET_PROVENANCE_FIELD_MISSING")
    if any(not str(row.get("patient_hash") or "").strip()
           or not str(row.get("sample_hash") or "").strip() for row in held):
        blockers.add("HOLDOUT_MEMBERSHIP_ID_MISSING")
    patient_ids = [str(row.get("patient_hash") or "") for row in all_rows]
    sample_ids = [str(row.get("sample_hash") or "") for row in all_rows]
    if (not all(patient_ids) or not all(sample_ids)
        or len(set(patient_ids)) != len(patient_ids)
        or len(set(sample_ids)) != len(sample_ids)):
        blockers.add("OUTER_OR_HOLDOUT_PATIENT_DUPLICATION")
    if _ids(outer) & _ids(held):
        blockers.add("OUTER_HOLDOUT_PATIENT_OVERLAP")
    real = {str(row.get("sample_hash")): row for row in index_rows
            if str(row.get("is_synthetic", "")).lower() in {"0", "false"}}
    if len(real) != sum(str(row.get("is_synthetic", "")).lower() in {"0", "false"}
                        for row in index_rows):
        blockers.add("REAL_INDEX_SAMPLE_DUPLICATION")
    outer_by_sample = {str(row.get("sample_hash")): row for row in outer}
    all_split_samples = set(sample_ids)
    if set(real) - all_split_samples:
        blockers.add("REAL_INDEX_NOT_IN_SPLITS")
    if any(str(real[sample].get("patient_hash")) != str(row.get("patient_hash"))
           or int(float(real[sample].get("label", -1))) != int(float(row.get("ab_label", -2)))
           for row in outer for sample in [str(row.get("sample_hash"))] if sample in real):
        blockers.add("INDEX_PATIENT_OR_LABEL_MISMATCH")
    gaps_bad = 0
    mri_hash_missing = 0
    cohort = Counter()
    for row in outer:
        sample = str(row.get("sample_hash"))
        if sample not in real:
            continue
        if not str(row.get("t1_file_hash") or "").strip():
            mri_hash_missing += 1
        try:
            pet, mri, gap = (float(row[key]) for key in
                             ("pet_day_t0", "t1_day_t0", "abs_pet_t1_gap_days"))
            if (not all(map(math.isfinite, (pet, mri, gap)))
                or gap < 0 or abs(abs(pet - mri) - gap) > 1e-6):
                gaps_bad += 1
            cohort[(str(row["substudy"]).upper(), int(float(row["ab_label"])))] += 1
        except (KeyError, TypeError, ValueError):
            gaps_bad += 1
    if gaps_bad:
        blockers.add("PET_MRI_GAP_INCONSISTENT")
    if mri_hash_missing:
        blockers.add("MRI_FILE_HASH_MISSING")
    folds = []
    for fold in range(5):
        outer_samples = {str(row.get("sample_hash")) for row in tests[fold]}
        train_samples = [str(row.get("sample_hash")) for row in trains[fold]]
        if (len(train_samples) != len(set(train_samples))
            or set(train_samples) != set(outer_by_sample) - outer_samples
            or any(row != outer_by_sample.get(str(row.get("sample_hash")))
                   for row in trains[fold])):
            blockers.add(f"TRAIN_SPLIT_MEMBERSHIP_INVALID:{fold}")
        retained_train = [row for row in trains[fold]
                          if str(row.get("sample_hash")) in real]
        inner = []
        for seed in seeds:
            shuffled = retained_train.copy()
            random.Random(seed + fold).shuffle(shuffled)
            n_val = min(max(int(len(shuffled) * val_fraction), 1), max(len(shuffled) - 1, 0))
            val, fit = shuffled[:n_val], shuffled[n_val:]
            if (not val or not fit or _ids(val) & _ids(fit)
                or _ids(val) & _ids(tests[fold])
                or _ids(fit) & _ids(tests[fold])
                or _ids(val) & _ids(held) or _ids(fit) & _ids(held)):
                blockers.add(f"INNER_PATIENT_LEAKAGE_OR_EMPTY:{fold}:{seed}")
            inner.append({"seed": seed, "fit_n": len(fit), "inner_validation_n": len(val),
                          "fit_positive_n": sum(int(float(r["ab_label"])) for r in fit),
                          "inner_positive_n": sum(int(float(r["ab_label"])) for r in val),
                          "membership_sha256": _digest({
                              "fit": sorted(_ids(fit)), "validation": sorted(_ids(val))})})
        folds.append({"fold": fold, "outer_n": len(tests[fold]),
                      "outer_qc_n": sum(str(r.get("sample_hash")) in real for r in tests[fold]),
                      "train_qc_n": len(retained_train), "inner": inner})
    perfect_confounding = (set(cohort) == {("A4", 1), ("LEARN", 0)})
    return {
        "status": "BLOCKED_LEAKAGE_OR_PROVENANCE" if blockers else "READY_INTERNAL_ONLY",
        "blockers": sorted(blockers),
        "outer_n": len(outer), "holdout_membership_n": len(held),
        "real_index_n": len(real), "outer_qc_n": sum(s in real for s in outer_by_sample),
        "mri_file_hash_missing_n": mri_hash_missing,
        "pet_mri_gap_inconsistent_n": gaps_bad,
        "cohort_by_label_qc": {f"{site}|{label}": count
                               for (site, label), count in sorted(cohort.items())},
        "cohort_label_perfectly_confounding": perfect_confounding,
        "folds": folds,
        "pet_image_file_hash_available_in_split": False,
        "site_scanner_metadata_available_in_split": False,
        "holdout_labels_used_for_this_audit": False,
        "subject_level_exported": False,
    }


def build_audit(root: Path) -> dict:
    config_path = root / CONFIG
    config = read(config_path)
    if (config.get("protocol_id") != "METHOD_X_MX05_SCIENCE_EXTENSION_V1_20260927"
        or config.get("original_seed") != 13
        or config.get("replication_seeds") != [29, 47]
        or config.get("outer_folds") != list(range(5))
        or config.get("training", {}).get("epochs_per_fold_max") != 20
        or config.get("claim_gates", {}).get("external_validation_without_external_data") is not False):
        raise RuntimeError("EXTENSION_CONFIG_CONTRACT_INVALID")
    parent = root / PARENT
    prereg = read(parent / "reports/fasttrack_preregistration.json")
    prior = read(parent / "reports/data_semantics_audit.json")
    if (prereg.get("protocol_id") != config["parent_protocol_id"]
        or prereg.get("status") != "LOCKED_BEFORE_GPU"
        or prior.get("status") != "READY"
        or prior.get("cache_sha256") != prereg.get("cache_sha256")
        or prior.get("physical_order") != list(PHYSICAL)
        or prior.get("id_label_mismatches") != 0
        or prior.get("index_tabular_mismatches") != 0):
        raise RuntimeError("EXTENSION_PARENT_DATA_CONTRACT_INVALID")
    split_dir = root / "splits"
    expected_split_sha = prior["split_sha256"]
    if (sha(root / "docs/fasttrack_private/cache_full_index.csv") != prior["index_sha256"]
        or any(sha(split_dir / f"fold_{i}_{phase}.csv") !=
               expected_split_sha[f"fold_{i}_{phase}"]
               for i in range(5) for phase in ("train", "test"))
        or sha(split_dir / "external_holdout.csv") != expected_split_sha["external_holdout"]):
        raise RuntimeError("EXTENSION_SPLIT_OR_INDEX_HASH_DRIFT")
    tests = [read_csv(split_dir / f"fold_{i}_test.csv") for i in range(5)]
    trains = [read_csv(split_dir / f"fold_{i}_train.csv") for i in range(5)]
    held = read_csv(split_dir / "external_holdout.csv")
    index_rows = read_csv(root / "docs/fasttrack_private/cache_full_index.csv")
    partition = partition_audit(
        index_rows, tests, trains, held,
        seeds=(config["original_seed"], *config["replication_seeds"]),
        val_fraction=float(config["inner_validation_fraction"]),
    )
    partition.update({
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "design_config_sha256": sha(config_path),
        "parent_preregistration_sha256": sha(parent / "reports/fasttrack_preregistration.json"),
        "parent_data_audit_sha256": sha(parent / "reports/data_semantics_audit.json"),
        "split_sha256": prior["split_sha256"],
        "index_sha256": prior["index_sha256"],
        "cache_sha256_reused_from_parent_audit": prior["cache_sha256"],
        "pet_image_provenance": "UNVERIFIED_IN_AVAILABLE_SPLIT",
        "mri_file_provenance": "T1_HASH_TOKEN_PRESENT_BUT_SOURCE_FILE_NOT_REHASHED",
        "scanner_overlap": "UNKNOWN",
        "pseudo_holdout_prior_access": "UNKNOWN",
        "claim_cap": "INTERNAL_EXPLORATORY_NO_INDEPENDENT_DISEASE_REGION_OR_GENERALIZATION_CLAIM",
    })
    if partition["cohort_label_perfectly_confounding"] != prior["cohort_label_perfectly_confounding"]:
        raise RuntimeError("EXTENSION_COHORT_AUDIT_DISAGREEMENT")
    return partition


def parent_terminal(root: Path) -> bool:
    state_path = root / PARENT / "reports/fasttrack_supervisor_state.json"
    if not state_path.is_file():
        return False
    steps = read(state_path).get("steps") or {}
    if not all((steps.get(name) or {}).get("status") == "COMPLETED" for name in
               ("mx7_freeze_scoped_report", "mx7_v22_thesis_addendum")):
        return False
    reports = root / PARENT / "reports"
    frozen_path = reports / "fasttrack_thesis_traceability.json"
    manifest_path = reports / "fasttrack_v22_thesis_manifest.json"
    if not frozen_path.is_file() or not manifest_path.is_file():
        return False
    return (read(frozen_path).get("immutable") is True
            and read(manifest_path).get("parent_freeze_sha256") == sha(frozen_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "lock", "verify"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if root != ROOT.resolve():
        raise RuntimeError("EXTENSION_PROJECT_ROOT_INVALID")
    report = build_audit(root)
    out = root / OUTPUT / "reports"
    if args.mode == "lock":
        if report["status"] != "READY_INTERNAL_ONLY":
            raise RuntimeError("EXTENSION_LEAKAGE_GATE_BLOCKED")
        immutable_json(out / AUDIT_NAME, report)
        lock = {
            "schema_version": 1, "status": "DESIGN_LOCKED_IMPLEMENTATION_PENDING",
            "protocol_id": report["protocol_id"],
            "design_config_sha256": report["design_config_sha256"],
            "day1_audit_sha256": sha(out / AUDIT_NAME),
            "audit_source_sha256": sha(Path(__file__)),
            "parent_preregistration_sha256": report["parent_preregistration_sha256"],
            "gpu_authorized": False,
            "strong_claim_allowed": False,
        }
        immutable_json(out / LOCK_NAME, lock)
    if args.mode == "verify":
        if not (out / AUDIT_NAME).is_file() or read(out / AUDIT_NAME) != report:
            raise RuntimeError("EXTENSION_DAY1_AUDIT_DRIFT")
        lock = read(out / LOCK_NAME)
        if (lock.get("design_config_sha256") != report["design_config_sha256"]
            or lock.get("day1_audit_sha256") != sha(out / AUDIT_NAME)
            or lock.get("audit_source_sha256") != sha(Path(__file__))
            or lock.get("gpu_authorized") is not False):
            raise RuntimeError("EXTENSION_DESIGN_LOCK_DRIFT")
    print(json.dumps({"status": report["status"], "mode": args.mode,
                      "parent_mx7_terminal": parent_terminal(root),
                      "gpu_authorized": False,
                      "blockers": report["blockers"]}, ensure_ascii=False))
    return 0 if report["status"] == "READY_INTERNAL_ONLY" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
