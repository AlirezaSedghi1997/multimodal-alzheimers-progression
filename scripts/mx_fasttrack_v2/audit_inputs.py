"""Read-only, aggregate-only MX fast-track data and feature-order audit.

Run against the FULL HDF5 cache on the training machine.  --index-only is for
reviewing the shipped CSV snapshot and never authorizes a GPU experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

PHYSICAL = ("age", "education_years", "apoe4_carrier", "mmse_base", "sex")
CANONICAL = ("age", "sex", "education_years", "apoe4_carrier", "mmse_base")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def value(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    try:
        return float(raw) if raw not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def expected(row: dict[str, str]) -> tuple[float, ...]:
    sx = str(row.get("sex", "")).strip()
    sex_binary = 1.0 if sx in {"1", "M", "Male", "male"} else 0.0
    return (value(row, "age"), value(row, "education_years"),
            value(row, "apoe4_carrier"), value(row, "mmse_base"), sex_binary)


def decode(x: object) -> str:
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def audit(index: Path, splits: Path, cache: Path | None = None) -> dict:
    blockers: list[str] = []
    index_rows = read_csv(index)
    real = [r for r in index_rows if str(r["is_synthetic"]).lower() in {"0", "false"}]
    if len(real) != 1669 or len({r["sample_hash"] for r in real}) != len(real):
        blockers.append("REAL_CACHE_INDEX_COUNT_OR_HASH_UNEXPECTED")
    real_by_sample = {r["sample_hash"]: r for r in real}
    tests = [read_csv(splits / f"fold_{i}_test.csv") for i in range(5)]
    held = read_csv(splits / "external_holdout.csv")
    development = [r for part in tests for r in part]
    all_split = development + held
    development_hashes = {r["sample_hash"] for r in development}
    development_by_hash = {r["sample_hash"]: r for r in development}
    for fold in range(5):
        train_rows = read_csv(splits / f"fold_{fold}_train.csv")
        train_hashes = [row["sample_hash"] for row in train_rows]
        expected_train = development_hashes - {row["sample_hash"] for row in tests[fold]}
        if (len(train_hashes) != len(set(train_hashes)) or set(train_hashes) != expected_train
            or any(row != development_by_hash.get(row["sample_hash"]) for row in train_rows)):
            blockers.append(f"TRAIN_FOLD_MEMBERSHIP_INVALID:{fold}")
    hashes = [r["sample_hash"] for r in all_split]
    patients = [r["patient_hash"] for r in all_split]
    if len(all_split) != 1699 or len(set(hashes)) != len(hashes) or len(set(patients)) != len(patients):
        blockers.append("SPLIT_UNIQUE_PATIENT_CONTRACT_FAILED")
    if set(r["patient_hash"] for r in development) & set(r["patient_hash"] for r in held):
        blockers.append("PSEUDO_HOLDOUT_PATIENT_OVERLAP")
    mismatches = 0
    hash_label_mismatches = 0
    for row in all_split:
        indexed = real_by_sample.get(row["sample_hash"])
        if indexed is None:
            continue  # QC-excluded split subjects; report denominator separately.
        if decode(indexed["patient_hash"]) != row["patient_hash"] or int(indexed["label"]) != int(row["ab_label"]):
            hash_label_mismatches += 1
        observed = tuple(float(indexed[f"tab{i}"]) for i in range(5))
        if any(not math.isclose(a, b, abs_tol=1e-3, rel_tol=1e-6)
               for a, b in zip(observed, expected(row), strict=True)):
            mismatches += 1
    if len([r for r in all_split if r["sample_hash"] in real_by_sample]) != len(real):
        blockers.append("REAL_CACHE_VERSUS_SPLIT_COVERAGE_MISMATCH")
    development_qc_n = sum(r["sample_hash"] in real_by_sample for r in development)
    pseudo_qc_n = sum(r["sample_hash"] in real_by_sample for r in held)
    if (development_qc_n, pseudo_qc_n) != (1336, 333):
        blockers.append("EXPECTED_DEVELOPMENT_OR_HOLDOUT_QC_COUNT_CHANGED")
    if mismatches or hash_label_mismatches:
        blockers.append("FEATURE_OR_ID_LABEL_MAPPING_MISMATCH")
    cohort = Counter((r["substudy"], int(r["ab_label"])) for r in all_split)
    if cohort != Counter({("A4", 1): 1163, ("LEARN", 0): 536}):
        blockers.append("COHORT_LABEL_COUNTS_CHANGED_REVIEW_DESIGN")
    retained = [r for r in all_split if r["sample_hash"] in real_by_sample]
    gaps = [float(r["abs_pet_t1_gap_days"]) for r in retained]
    days = Counter("PET_AFTER_MRI" if value(r, "pet_day_t0") > value(r, "t1_day_t0")
                   else "PET_BEFORE_MRI" if value(r, "pet_day_t0") < value(r, "t1_day_t0")
                   else "SAME_DAY" for r in retained)
    temporal = {}
    for threshold in (90, 180, 365):
        subset = [r for r in retained if float(r["abs_pet_t1_gap_days"]) <= threshold]
        temporal[str(threshold)] = {
            "n": len(subset), "positive": sum(int(r["ab_label"]) for r in subset),
            "negative": sum(1 - int(r["ab_label"]) for r in subset),
        }
    h5_checked = 0
    cache_sha256 = None
    if cache is None:
        blockers.append("FULL_HDF5_NOT_VERIFIED")
    else:
        try:
            import h5py  # available in the project's trainer environment
            with h5py.File(cache, "r") as h5:
                if bool(h5.attrs.get("contains_pet_input", True)):
                    blockers.append("CACHE_PET_INPUT_FLAG_INVALID")
                if int(h5.attrs.get("resolution", -1)) != 128:
                    blockers.append("CACHE_RESOLUTION_INVALID")
                for row in real:
                    i = int(row["row"])
                    if i >= len(h5["label"]) or decode(h5["sample_hash"][i]) != row["sample_hash"]:
                        blockers.append("CACHE_SAMPLE_INDEX_MISMATCH")
                        break
                    if decode(h5["status"][i]) != "OK":
                        blockers.append("CACHE_REAL_STATUS_NOT_OK")
                        break
                    if int(h5["label"][i]) != int(row["label"]):
                        blockers.append("CACHE_LABEL_MISMATCH")
                        break
                    if any(not math.isclose(float(h5["tabular"][i, j]), float(row[f"tab{j}"]),
                                            abs_tol=1e-3, rel_tol=1e-6) for j in range(5)):
                        blockers.append("CACHE_FULL_TABULAR_MISMATCH")
                        break
                    h5_checked += 1
            if h5_checked == len(real) and not blockers:
                cache_sha256 = digest(cache)
        except (OSError, ImportError, KeyError, ValueError) as exc:
            blockers.append(f"FULL_HDF5_AUDIT_FAILED:{type(exc).__name__}")
    return {
        "schema_version": 2, "scope": "FASTTRACK_V2_READ_ONLY_AUDIT",
        "status": "READY" if not blockers else "BLOCKED",
        "blockers": sorted(set(blockers)), "cache_sha256": cache_sha256,
        "index_sha256": digest(index),
        "split_sha256": {**{f"fold_{i}_test": digest(splits / f"fold_{i}_test.csv") for i in range(5)},
                         **{f"fold_{i}_train": digest(splits / f"fold_{i}_train.csv") for i in range(5)},
                         "external_holdout": digest(splits / "external_holdout.csv")},
        "physical_order": list(PHYSICAL), "semantic_order": list(CANONICAL),
        "feature_drop_label_correction": {name: PHYSICAL[CANONICAL.index(name)] for name in CANONICAL},
        "index_real_n": len(real), "verified_full_hdf5_real_n": h5_checked,
        "development_split_n": len(development), "pseudo_holdout_split_n": len(held),
        "development_qc_n": development_qc_n, "pseudo_holdout_qc_n": pseudo_qc_n,
        "real_retained_n": len(retained), "index_tabular_mismatches": mismatches,
        "id_label_mismatches": hash_label_mismatches,
        "cohort_by_label": {f"{k[0]}|{k[1]}": n for k, n in sorted(cohort.items())},
        "cohort_label_perfectly_confounding": (set(cohort) == {("A4", 1), ("LEARN", 0)}),
        "pet_mri_gap_median_days": statistics.median(gaps) if gaps else None,
        "pet_vs_mri_direction": dict(days), "time_window_sensitivity_counts": temporal,
        "site_scanner_overlap": "UNKNOWN_UNLESS_SEPARATELY_VERIFIED",
        "holdout_prior_access": "UNKNOWN_UNLESS_AUDITED_ON_ORIGINAL_MACHINE",
        "subject_level_exported": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.index, args.splits, args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps({"status": report["status"], "blockers": report["blockers"],
                      "output": str(args.out)}, ensure_ascii=False))
    return 0 if report["status"] == "READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
