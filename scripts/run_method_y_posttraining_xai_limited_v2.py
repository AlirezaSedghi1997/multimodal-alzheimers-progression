"""Resume XAI with affine-aware brain masking; migrate valid v1 inference.

The v1 maps are evidence, not a valid anatomical explanation. Their 46
committed inference products are re-used by an exact post-map mask operation;
uncomputed subjects are inferred from the same immutable checkpoints. The
source v1 root is never modified. No outer performance evaluation occurs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import msvcrt
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts import run_method_y_posttraining_xai_limited_v1 as old  # noqa: E402
from src.method_y_revival import pet_attribution_v4 as pa  # noqa: E402
from src.method_y_revival.atomic_io import atomic_write_bytes, atomic_write_json  # noqa: E402
from src.method_y_revival.contracts import sha256_mapping  # noqa: E402
from src.method_y_revival.posttraining_correction_v4_2 import CorrectionV42Error  # noqa: E402
from src.method_y_revival.posttraining_xai_limited_v1 import cluster_bootstrap  # noqa: E402
from src.method_y_revival.posttraining_xai_limited_v2 import (  # noqa: E402
    brain_outside_perturbation, corrected_agreement, load_locked_brain_mask,
    mask_maps_and_qc,
)

V1_ROOT = PROJECT / "results" / "method_y_revival_v4_posttraining_xai_limited_v1"
OUTPUT = PROJECT / "results" / "method_y_revival_v4_posttraining_xai_limited_v2"
TEMPLATE = PROJECT / "data" / "templates" / "method_y_mni152nlin2009casym_v1"
MASK_PATH = TEMPLATE / "tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz"
REFERENCE_PATH = TEMPLATE / "reference_128_1p5mm.nii.gz"


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise CorrectionV42Error(code)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v1_plan() -> dict:
    path = V1_ROOT / "private" / "inference_plan.json"
    _require(path.is_file(), "XAI_V1_PLAN_MISSING")
    original = json.loads(path.read_text(encoding="utf-8"))
    _require(original == old.build_plan(), "XAI_V1_PLAN_OR_SOURCE_DRIFT")
    _require(original.get("plan_sha256") == sha256_mapping(
        {key: value for key, value in original.items() if key != "plan_sha256"}
    ), "XAI_V1_PLAN_HASH_INVALID")
    return original


def build_plan() -> tuple[dict, np.ndarray]:
    prior = _v1_plan()
    mask, mask_contract = load_locked_brain_mask(MASK_PATH, REFERENCE_PATH)
    helper = PROJECT / "src" / "method_y_revival" / "posttraining_xai_limited_v2.py"
    plan = {
        "schema_version": "2.0", "status": "LOCKED_MASKED_INFERENCE_ONLY",
        "science_gate_status": "BLOCKED", "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False, "external_claim_allowed": False,
        "v1_plan_sha256": prior["plan_sha256"],
        "source_v4_run_contract_sha256": prior["source_v4_run_contract_sha256"],
        "source_runner_sha256": _sha(Path(old.__file__)),
        "source_attribution_module_sha256": prior["attribution_module_sha256"],
        "runner_sha256": _sha(Path(__file__)), "helper_sha256": _sha(helper),
        "mask_contract": mask_contract, "license_sha256": _sha(TEMPLATE / "LICENSE"),
        "selection_rule": prior["selection_rule"],
        "subjects_per_cell": prior["subjects_per_cell"],
        "cells": prior["cells"], "device": "cpu",
        "migrate_committed_v1_maps_without_model_reinference": True,
        "outer_test_metric_recomputation": False,
    }
    plan["plan_sha256"] = sha256_mapping(plan)
    return plan, mask


def _load_or_lock_plan(output: Path) -> tuple[dict, np.ndarray]:
    plan, mask = build_plan()
    target = output / "private" / "masked_inference_plan.json"
    if target.exists():
        _require(json.loads(target.read_text(encoding="utf-8")) == plan,
                 "XAI_V2_PLAN_OR_SOURCE_MISMATCH")
    else:
        atomic_write_json(target, plan)
        atomic_write_json(output / "private" / "v1_invalidation.json", {
            "status": "INVALIDATED_FOR_ANATOMICAL_LOCALIZATION",
            "source_plan_sha256": plan["v1_plan_sha256"],
            "reason": ["UNMASKED_ATTRIBUTION_OUTSIDE_TRUE_BRAIN",
                       "GRADCAM_ZERO_MAPS_REQUIRE_NOT_ESTIMABLE",
                       "PARTIAL_RUN_STOPPED_BY_USER"],
            "source_root_preserved": True,
        })
    return plan, mask


def _paths(output: Path, row: dict, subject: str) -> tuple[Path, Path]:
    return old._stored_map_paths(output, row, subject)


def _read_committed(output: Path, plan: dict, row: dict, subject: str) -> dict | None:
    map_path, meta_path = _paths(output, row, subject)
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _require(meta.get("status") == "COMMITTED_MASKED"
             and meta.get("plan_sha256") == plan["plan_sha256"]
             and meta.get("subject") == subject
             and meta.get("checkpoint_sha256") == row["checkpoint_sha256"]
             and meta.get("brain_mask_sha256")
             == plan["mask_contract"]["resampled_mask_sha256"]
             and meta.get("derivation") in ("MASKED_V1_MAP", "FROZEN_INFERENCE_MASKED")
             and map_path.is_file() and _sha(map_path) == meta.get("maps_sha256"),
             "XAI_V2_COMMITTED_MAP_MISMATCH")
    if meta["derivation"] == "MASKED_V1_MAP":
        source_map, source_receipt = old._stored_map_paths(V1_ROOT, row, subject)
        _require(source_map.is_file() and source_receipt.is_file(),
                 "XAI_V2_MIGRATED_SOURCE_MISSING")
        original = json.loads(source_receipt.read_text(encoding="utf-8"))
        _require(original.get("status") == "COMMITTED"
                 and original.get("plan_sha256") == plan["v1_plan_sha256"]
                 and original.get("subject") == subject
                 and original.get("checkpoint_sha256") == row["checkpoint_sha256"]
                 and original.get("maps_sha256") == meta.get("source_v1_maps_sha256")
                 and _sha(source_map) == original["maps_sha256"],
                 "XAI_V2_MIGRATED_SOURCE_HASH_MISMATCH")
    return meta


def _perturbation_path(output: Path, row: dict, subject: str) -> Path:
    _, meta_path = _paths(output, row, subject)
    return meta_path.with_name(meta_path.stem + ".perturbation.json")


def _read_perturbation(output: Path, plan: dict, row: dict,
                       subject: str) -> dict | None:
    target = _perturbation_path(output, row, subject)
    if not target.exists():
        return None
    receipt = json.loads(target.read_text(encoding="utf-8"))
    _require(receipt.get("receipt_sha256") == sha256_mapping(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    ) and receipt.get("plan_sha256") == plan["plan_sha256"]
             and receipt.get("subject") == subject
             and receipt.get("checkpoint_sha256") == row["checkpoint_sha256"]
             and receipt.get("brain_mask_sha256")
             == plan["mask_contract"]["resampled_mask_sha256"]
             and receipt.get("result", {}).get("status")
             == "COMPUTED_OOD_DIAGNOSTIC_ONLY", "XAI_V2_PERTURBATION_RECEIPT_MISMATCH")
    return receipt


def _commit_perturbation(output: Path, plan: dict, row: dict, subject: str,
                         result: dict) -> dict:
    receipt = {
        "status": "COMMITTED", "privacy_tier": "PRIVATE_SUBJECT_DIAGNOSTIC",
        "plan_sha256": plan["plan_sha256"], "subject": subject,
        "branch": row["branch"], "arm": row["arm"], "fold": row["outer"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "brain_mask_sha256": plan["mask_contract"]["resampled_mask_sha256"],
        "result": result,
    }
    receipt["receipt_sha256"] = sha256_mapping(receipt)
    atomic_write_json(_perturbation_path(output, row, subject), receipt)
    return receipt


def _commit(output: Path, plan: dict, row: dict, subject: str,
            maps: dict, qc: dict, agreement: dict,
            *, derivation: str, source_v1_sha256: str | None = None) -> dict:
    map_path, meta_path = _paths(output, row, subject)
    payload = io.BytesIO()
    np.savez_compressed(payload, **maps)
    atomic_write_bytes(map_path, payload.getvalue())
    meta = {
        "status": "COMMITTED_MASKED", "privacy_tier": "PRIVATE_SUBJECT_MAP",
        "plan_sha256": plan["plan_sha256"], "subject": subject,
        "branch": row["branch"], "arm": row["arm"], "fold": row["outer"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "brain_mask_sha256": plan["mask_contract"]["resampled_mask_sha256"],
        "maps_sha256": _sha(map_path), "derivation": derivation,
        "source_v1_maps_sha256": source_v1_sha256,
        "map_qc": qc, "agreement": agreement,
    }
    atomic_write_json(meta_path, meta)
    return meta


def _migrate_if_available(output: Path, plan: dict, prior: dict, row: dict,
                          subject: str, mask: np.ndarray) -> bool:
    old_meta = old._read_committed(V1_ROOT, prior, row, subject)
    if old_meta is None:
        return False
    old_map_path, _ = old._stored_map_paths(V1_ROOT, row, subject)
    with np.load(old_map_path, allow_pickle=False) as payload:
        maps = {name: payload[name] for name in payload.files}
    masked, qc = mask_maps_and_qc(maps, mask)
    agreement = corrected_agreement(masked)
    _commit(output, plan, row, subject, masked, qc, agreement,
            derivation="MASKED_V1_MAP", source_v1_sha256=old_meta["maps_sha256"])
    return True


def _aggregate(output: Path, plan: dict) -> dict:
    records = []
    mask, mask_contract = load_locked_brain_mask(MASK_PATH, REFERENCE_PATH)
    _require(mask_contract == plan["mask_contract"], "XAI_V2_MASK_CONTRACT_DRIFT")
    for row in plan["cells"]:
        for subject in row["subjects"]:
            receipt = _read_committed(output, plan, row, subject)
            _require(receipt is not None, "XAI_V2_INCOMPLETE")
            map_path, _ = _paths(output, row, subject)
            with np.load(map_path, allow_pickle=False) as stored:
                _require(set(stored.files) == {"layercam_primary", "layercam_sensitivity",
                                               "gradcam_primary", "scorecam_primary"},
                         "XAI_V2_MAP_SCHEMA_INVALID")
                for name in stored.files:
                    arr = stored[name]
                    _require(arr.shape == mask.shape and np.isfinite(arr).all()
                             and np.all(arr[~mask] == 0),
                             "XAI_V2_MAP_NOT_ANATOMICALLY_MASKED")
            perturbation = _read_perturbation(output, plan, row, subject)
            _require(perturbation is not None, "XAI_V2_PERTURBATION_INCOMPLETE")
            receipt["perturbation"] = perturbation["result"]
            records.append(receipt)
    _require(len(records) == 100, "XAI_V2_SAMPLE_COUNT_INVALID")
    groups = {}
    for branch in ("amyloid", "tau"):
        for arm in ("A0", "A1", "A2", "A3", "A4"):
            subset = [r for r in records if r["branch"] == branch and r["arm"] == arm]
            _require(len(subset) == 10 and {r["fold"] for r in subset} == set(range(5)),
                     "XAI_V2_BRANCH_ARM_FOLD_COVERAGE_INVALID")
            agreement = {}
            for kind in ("risk_target", "layer_vs_score", "layer_vs_grad"):
                values = [{"fold": r["fold"], "rho": r["agreement"][kind]["rho"]}
                          for r in subset]
                agreement[kind] = cluster_bootstrap(
                    values, field="rho", seed=20260714 + len(groups),
                )
            qc = {}
            for kind in ("layercam_primary", "layercam_sensitivity",
                         "gradcam_primary", "scorecam_primary"):
                statuses = [r["map_qc"][kind] for r in subset]
                mass = [r["outside_brain_absolute_mass_fraction_before_mask"]
                        for r in statuses if r["outside_brain_absolute_mass_fraction_before_mask"]
                        is not None]
                qc[kind] = {
                    "zero_inside_brain_count": sum(
                        r["status"] == "NOT_ESTIMABLE_ZERO_MAP" for r in statuses),
                    "median_outside_brain_mass_before_mask":
                        float(np.median(mass)) if mass else None,
                    "all_exact_zero_outside_after_mask": all(
                        r["outside_brain_exact_zero_after_mask"] for r in statuses),
                }
            perturb = {}
            for mode in ("logit_sum", "mortality_last", "mortality_at_locked_horizon"):
                rows = [r["perturbation"]["risk_modes"][mode] for r in subset]
                perturb[mode] = {
                    "outside_change_ge_brain_change_count": sum(
                        r["outside_change_ge_brain_change"] for r in rows),
                    "median_absolute_change_remove_outside": float(np.median([
                        r["absolute_change_remove_outside"] for r in rows])),
                    "median_absolute_change_remove_brain": float(np.median([
                        r["absolute_change_remove_brain"] for r in rows])),
                    "interpretation": "OOD_SHORTCUT_DIAGNOSTIC_NOT_LOCALIZATION_PROOF",
                }
            groups[f"{branch}/{arm}"] = {"n_subject_maps": len(subset),
                                           "agreement": agreement, "map_qc": qc,
                                           "brain_outside_perturbation": perturb}
    total_zero_gradcam = sum(
        r["map_qc"]["gradcam_primary"]["status"] == "NOT_ESTIMABLE_ZERO_MAP"
        for r in records
    )
    report = {
        "schema_version": "2.0", "artifact_status": "COMPLETED",
        "status": "COMPLETED_LIMITED_MASKED_XAI",
        "xai_anatomical_validity": "FAIL" if total_zero_gradcam else "NOT_ESTABLISHED",
        "science_gate_status": "FAIL", "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "positive_claim_allowed": False, "external_claim_allowed": False,
        "plan_sha256": plan["plan_sha256"], "cells_completed": 50,
        "subject_maps_completed": len(records),
        "migrated_v1_subject_maps": sum(r["derivation"] == "MASKED_V1_MAP" for r in records),
        "new_frozen_inference_subject_maps": sum(
            r["derivation"] == "FROZEN_INFERENCE_MASKED" for r in records),
        "brain_mask_fraction": plan["mask_contract"]["brain_fraction"],
        "outer_test_metric_recomputation": False,
        "frozen_model_brain_vs_outside_perturbations": 100,
        "gradcam_zero_inside_brain_total": total_zero_gradcam,
        "limitations": ["TWO_OUTCOME_BLIND_SUBJECTS_PER_CELL_ONLY",
                        "ZERO_GRADCAM_MAPS_ARE_NOT_ESTIMABLE",
                        "BRAIN_MASKING_DOES_NOT_VALIDATE_PET_QUANTIFICATION",
                        "BRAIN_VS_OUTSIDE_ZEROING_IS_OUT_OF_DISTRIBUTION_DIAGNOSTIC_ONLY",
                        "CLUSTER_BOOTSTRAP_CONDITIONAL_ON_FITTED_MODELS",
                        "SCANNER_AND_EXTERNAL_GATES_REMAIN_BLOCKED"],
        "branch_arm_aggregate": groups,
    }
    report["report_sha256"] = sha256_mapping(report)
    target = output / "public_bundle" / "masked_xai_aggregate.json"
    if target.exists():
        _require(json.loads(target.read_text(encoding="utf-8")) == report,
                 "XAI_V2_PUBLIC_REPORT_MISMATCH")
    else:
        atomic_write_json(target, report)
    _verify_public_bundle(output, report["report_sha256"])
    return report


def _verify_public_bundle(output: Path, report_sha256: str) -> None:
    root = output / "public_bundle"
    files = [p for p in root.rglob("*") if p.is_file()]
    _require(len(files) == 1 and files[0].name == "masked_xai_aggregate.json",
             "XAI_V2_PUBLIC_BUNDLE_UNEXPECTED_FILE")
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    _require(payload.get("report_sha256") == report_sha256,
             "XAI_V2_PUBLIC_REPORT_HASH_INVALID")
    forbidden = {"subject", "subject_key", "scan_key", "source_subject_id",
                 "raw_path", "event", "time_days", "prediction", "predictions",
                 "embedding", "embeddings", "source_v1_maps_sha256", "maps_sha256"}

    def visit(value):
        if isinstance(value, dict):
            _require(not (forbidden & set(value)), "XAI_V2_PUBLIC_ROW_LEVEL_LEAK")
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, str):
            _require("H:\\" not in value and "A4 Learn" not in value,
                     "XAI_V2_PUBLIC_PATH_LEAK")
    visit(payload)


def _partial_status(output: Path, plan: dict) -> dict:
    count = 0
    migrated = 0
    perturbations = 0
    per_branch: dict[str, int] = defaultdict(int)
    for row in plan["cells"]:
        for subject in row["subjects"]:
            meta = _read_committed(output, plan, row, subject)
            if meta is not None:
                count += 1
                migrated += meta["derivation"] == "MASKED_V1_MAP"
                per_branch[row["branch"]] += 1
            perturbations += _read_perturbation(output, plan, row, subject) is not None
    return {"status": "IN_PROGRESS" if count < 100 else "MAPS_COMMITTED",
            "committed_subject_maps": count, "expected_subject_maps": 100,
            "migrated_v1_subject_maps": migrated,
            "committed_perturbation_receipts": perturbations,
            "branch_counts": dict(sorted(per_branch.items()))}


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise CorrectionV42Error("XAI_V2_ACTIVE_WRITER") from exc
        try:
            plan, mask = _load_or_lock_plan(output)
            prior = _v1_plan()
            xai, factor, rc, cells = old._locked_inputs()
            by_key = {(c["branch"], c["arm"], c["outer"]): c for c in cells}
            torch.set_num_threads(4)
            for row in plan["cells"]:
                cell = by_key[(row["branch"], row["arm"], row["outer"])]
                pending_maps = []
                pending_perturbations = []
                for subject in row["subjects"]:
                    if _read_committed(output, plan, row, subject) is None:
                        if _migrate_if_available(output, plan, prior, row, subject, mask):
                            print(json.dumps({"status": "MASKED_V1_MAP_MIGRATED",
                                              "branch": row["branch"], "arm": row["arm"],
                                              "fold": row["outer"]}), flush=True)
                        else:
                            pending_maps.append(subject)
                    if _read_perturbation(output, plan, row, subject) is None:
                        pending_perturbations.append(subject)
                if not pending_maps and not pending_perturbations:
                    continue
                prov = pa.verify_final_provenance(
                    cell["final_checkpoint"], v4_root=old.V4_ROOT,
                    branch=row["branch"], arm=row["arm"], outer=row["outer"],
                )
                _require(prov["checkpoint_sha256"] == row["checkpoint_sha256"],
                         "XAI_V2_CHECKPOINT_DRIFT")
                payload = pa.read_only_load_checkpoint(
                    cell["final_checkpoint"], expected_sha256=row["checkpoint_sha256"],
                )
                features = factor["branches"][row["branch"]]["model_features"]
                net = old._build_real_arch(row["arm"], old._locked_model_cfg(),
                                           tabular_dim=len(features),
                                           bins=old._infer_bins_from_payload(payload))
                net.load_state_dict(payload["model_state"], strict=True)
                net.eval()
                needed = [subject for subject in row["subjects"]
                          if subject in pending_maps or subject in pending_perturbations]
                batch = old._load_cell_batch(dict(cell, firewall_subjects=needed),
                                             run_contract=rc, model_cfg=old._locked_model_cfg(),
                                             model_features=features)
                family = old.FAMILIES[row["arm"]]
                layer = xai["targets"][family][0]
                shallow = xai["targets"]["shallow_fusion"][family]
                for subject, image, tabular in batch:
                    if subject in pending_maps:
                        raw_maps, _ = old._compute_maps(net, image, tabular,
                                                        layer=layer, shallow=shallow)
                        masked, qc = mask_maps_and_qc(raw_maps, mask)
                        agreement = corrected_agreement(masked)
                        _commit(output, plan, row, subject, masked, qc, agreement,
                                derivation="FROZEN_INFERENCE_MASKED")
                        print(json.dumps({"status": "MASKED_MAP_COMMITTED",
                                          "branch": row["branch"], "arm": row["arm"],
                                          "fold": row["outer"]}), flush=True)
                    if subject in pending_perturbations:
                        edges = factor["shared"]["hazard_bin_edges_days"]
                        horizons = factor["shared"]["requested_horizons_days"]
                        _require(1095.75 in horizons,
                                 "XAI_V2_THREE_YEAR_HORIZON_NOT_LOCKED")
                        result = brain_outside_perturbation(
                            net, image, tabular, mask, bin_edges=edges,
                            horizon_days=1095.75,
                        )
                        _commit_perturbation(output, plan, row, subject, result)
                del net, payload, batch
                gc.collect()
            return _aggregate(output, plan)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plan", "run", "status", "verify"), required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    _require(output.is_relative_to((PROJECT / "results").resolve()),
             "XAI_V2_OUTPUT_NOT_PROJECT_RESULTS")
    if args.mode == "run":
        report = run(output)
        print(json.dumps({"status": report["status"],
                          "subject_maps": report["subject_maps_completed"],
                          "migrated": report["migrated_v1_subject_maps"],
                          "science_gate_status": report["science_gate_status"]}), flush=True)
    else:
        plan, _ = _load_or_lock_plan(output)
        if args.mode == "plan":
            print(json.dumps({"status": "PLAN_LOCKED", "cells": len(plan["cells"]),
                              "expected_maps": len(plan["cells"]) * plan["subjects_per_cell"],
                              "plan_sha256": plan["plan_sha256"]}), flush=True)
        elif args.mode == "status":
            print(json.dumps(_partial_status(output, plan), sort_keys=True), flush=True)
        else:
            report = _aggregate(output, plan)
            print(json.dumps({"status": "PASS_TECHNICAL_MASKED_MAP_LINEAGE_ONLY",
                              "report_sha256": report["report_sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": f"{type(exc).__name__}:{exc}"}),
              flush=True)
        raise SystemExit(3)
