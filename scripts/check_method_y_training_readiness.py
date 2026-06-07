#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.pet_spatial_qc import atomic_write_json, atomic_write_text, load_yaml


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _count_min_class(feas: dict, tier: str = "primary_strict") -> int:
    item = feas.get(tier, {}) if isinstance(feas, dict) else {}
    return int(min(item.get("pos_like", 0), item.get("neg_like", 0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracer", default="fbp")
    ap.add_argument("--policy", default="configs/model/method_y_spatial_preprocess_policy.yaml")
    ap.add_argument("--modeling_design", default="docs/method_y_phase2_modeling_design.md")
    args = ap.parse_args()

    policy = load_yaml(args.policy)
    manifest_summary = read_json(PROJECT_ROOT / "results" / "tables" / "method_y_pet_manifest_summary.json")
    gap = read_json(PROJECT_ROOT / "results" / "tables" / "method_y_gap_filter_report.json")
    preprocess = read_json(PROJECT_ROOT / "results" / "tables" / "method_y_pet_preprocess_summary.json")
    cache = read_json(PROJECT_ROOT / "results" / "tables" / "method_y_pet_cache_summary.json")
    blockers: list[str] = []
    warnings: list[str] = []

    if not manifest_summary:
        blockers.append("MISSING_PRIVATE_MANIFEST_SUMMARY")
    if manifest_summary and manifest_summary.get("regex_fallback_used"):
        blockers.append("REGEX_FALLBACK_MANIFEST_NOT_TRAINING_READY")
    if manifest_summary and not manifest_summary.get("training_ready_manifest"):
        blockers.append("PRIVATE_MANIFEST_NOT_TRAINING_READY")
    if not gap:
        blockers.append("MISSING_GAP_FILTER_REPORT")
    else:
        if gap.get("missing_gap_rows", 0) > 0:
            warnings.append("MISSING_GAP_ROWS_PRESENT")
        feas = gap.get("feasibility_by_tier", {})
        min_class = _count_min_class(feas, "primary_strict")
        min_warn = int(policy.get("training_readiness", {}).get("min_per_class_warn", 150))
        min_block = int(policy.get("training_readiness", {}).get("min_per_class_block", 100))
        if min_class < min_block:
            blockers.append("PRIMARY_TIER_MIN_CLASS_BELOW_BLOCK_THRESHOLD")
        elif min_class < min_warn:
            warnings.append("PRIMARY_TIER_MIN_CLASS_BELOW_WARN_THRESHOLD")
        for audit_name in ["cohort_label_audit", "tracer_label_audit", "gap_label_audit"]:
            audit = gap.get(audit_name, {})
            if not audit:
                blockers.append(f"MISSING_{audit_name.upper()}")
            elif audit.get("status") == "SINGLE_ROW_OR_COLUMN":
                warnings.append(f"{audit_name.upper()}_SINGLE_ROW_OR_COLUMN_REVIEW")
            elif audit.get("chi2", 0) and audit.get("chi2", 0) > 50:
                warnings.append(f"{audit_name.upper()}_POSSIBLE_STRONG_COUPLING")
    if not preprocess:
        blockers.append("MISSING_PREPROCESS_SUMMARY")
    else:
        if preprocess.get("mode") != "execute":
            blockers.append("PREPROCESS_NOT_EXECUTED")
        if preprocess.get("status_counts", {}).get("PASS", 0) <= 0:
            blockers.append("NO_QC_PASS_PET_OUTPUTS")
        if preprocess.get("spatial_mode") == "identity_for_tests":
            blockers.append("IDENTITY_SPATIAL_MODE_NOT_TRAINING_READY")
        if preprocess.get("spatial_mode") == "ants_rigid_pet_to_t1":
            warnings.append("SUBJECT_T1_ONLY_SPATIAL_OUTPUT_REQUIRES_MNI_STEP_FOR_REFERENCE_REGION_POLICY")
        if preprocess.get("manifest_source") == "filename_regex_fallback":
            blockers.append("FILENAME_REGEX_FALLBACK_USED_PROVIDE_PRIVATE_MANIFEST")
    if not cache:
        warnings.append("PET_CACHE_NOT_BUILT_YET")
    if not (PROJECT_ROOT / args.modeling_design).exists():
        blockers.append("MISSING_METHOD_Y_MODELING_DESIGN")

    status = "READY_FOR_EXPLORATORY_METHOD_Y_TRAINING" if not blockers else "NOT_TRAINING_READY"
    report = {
        "stage": "method_y_training_readiness", "version": "v3.8-literature-aligned", "tracer": args.tracer,
        "status": status, "blockers": blockers, "warnings": warnings,
        "method_x_claim_unchanged": True,
        "pet_training_if_ready_is_method_y_only": True,
        "clinical_alzheimer_diagnosis_allowed": False,
        "required_before_training": [
            "PET-only baseline plan frozen", "MRI-only baseline on Method Y split plan frozen", "calibration/ECE/Brier/CI plan frozen",
            "patient-level split and one-PET-per-patient verified", "cohort/tracer/gap confounding audits reviewed",
        ],
    }
    atomic_write_json(PROJECT_ROOT / "results" / "tables" / "method_y_training_readiness.json", report)
    md = ["# Method Y Training Readiness\n\n", f"- Status: `{status}`\n", f"- Blockers: `{blockers}`\n", f"- Warnings: `{warnings}`\n", "- Method X claim unchanged: true\n", "- PET training if ready is Method Y only: true\n"]
    atomic_write_text(PROJECT_ROOT / "results" / "tables" / "method_y_training_readiness.md", "".join(md))
    print(status)
    return 0 if not blockers else 1

if __name__ == "__main__":
    raise SystemExit(main())
