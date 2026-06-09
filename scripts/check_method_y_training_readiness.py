#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))

from src.utils.pet_preprocess_progress import summarize_pet_preprocess_progress
from src.utils.pet_spatial_qc import atomic_write_json, atomic_write_text, load_yaml


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _count_min_class(feas: dict, tier: str = "primary_strict") -> int:
    item = feas.get(tier, {}) if isinstance(feas, dict) else {}
    return int(min(item.get("pos_like", 0), item.get("neg_like", 0)))


def nested_bool(obj: dict | None, *keys: str) -> bool | None:
    cur: object = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return bool(cur) if cur is not None else None


def _load_imbalance_policy() -> dict:
    path = PROJECT_ROOT / "configs/policy/method_y_class_imbalance_policy.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracer", default="fbp")
    ap.add_argument("--policy", default="configs/model/method_y_spatial_preprocess_policy.yaml")
    ap.add_argument("--modeling_design", default="docs/method_y_phase2_modeling_design.md")
    args = ap.parse_args()

    load_yaml(args.policy)
    imbalance = _load_imbalance_policy()
    approved_tier = imbalance.get("approved_primary_training_tier", "primary_feasible_90")
    exploratory_block = int(imbalance.get("thresholds", {}).get("exploratory_min_per_class_block", 30))
    exploratory_warn = int(imbalance.get("thresholds", {}).get("exploratory_min_per_class_warn", 50))
    min_neg_auc = int(imbalance.get("minimum_counts", {}).get("min_neg_for_auc", 20))
    forbidden_tiers = set(imbalance.get("forbidden_primary_training_tiers", ["gold_standard_30"]))

    manifest_summary = read_json(PROJECT_ROOT / "results/tables/method_y_pet_manifest_summary.json")
    gap = read_json(PROJECT_ROOT / "results/tables/method_y_gap_filter_report.json")
    preprocess = read_json(PROJECT_ROOT / "results/tables/method_y_pet_preprocess_summary.json")
    cache = read_json(PROJECT_ROOT / "results/tables/method_y_pet_cache_summary.json")

    design_blockers: list[str] = []
    execution_blockers: list[str] = []
    warnings: list[str] = []

    if not manifest_summary:
        design_blockers.append("MISSING_PRIVATE_MANIFEST_SUMMARY")
    else:
        if manifest_summary.get("regex_fallback_used"):
            design_blockers.append("REGEX_FALLBACK_MANIFEST_NOT_TRAINING_READY")
        if not manifest_summary.get("training_ready_manifest"):
            design_blockers.append("PRIVATE_MANIFEST_NOT_TRAINING_READY")
        if manifest_summary.get("manifest_source") == "filename_regex_fallback":
            design_blockers.append("FILENAME_REGEX_FALLBACK_USED_PROVIDE_PRIVATE_MANIFEST")

    linked = int(
        manifest_summary.get("link_summary", {}).get("linked_work_items")
        or manifest_summary.get("linked_work_items")
        or 266
    )
    pet_progress = summarize_pet_preprocess_progress(linked_work_items=linked, project_root=PROJECT_ROOT)

    if not gap:
        design_blockers.append("MISSING_GAP_FILTER_REPORT")
    else:
        if gap.get("missing_gap_rows", 0) > 0:
            warnings.append("MISSING_GAP_ROWS_PRESENT")
        feas = gap.get("feasibility_by_tier", {})
        min_class_approved = _count_min_class(feas, approved_tier)
        if approved_tier not in feas:
            frozen = imbalance.get(f"class_counts_{approved_tier}", {})
            min_class_approved = int(frozen.get("min_class", 0))
        min_class_strict = _count_min_class(feas, "primary_strict")
        min_class_gold = _count_min_class(feas, "gold_standard_30")
        if min_class_approved < exploratory_block:
            design_blockers.append("PRIMARY_FEASIBLE_90_MIN_CLASS_BELOW_EXPLORATORY_BLOCK")
        elif min_class_approved < exploratory_warn:
            warnings.append("PRIMARY_FEASIBLE_90_MIN_CLASS_BELOW_WARN_THRESHOLD")
        if min_class_strict < exploratory_block:
            warnings.append("PRIMARY_STRICT_TIER_TOO_IMBALANCED_FOR_SOLE_PRIMARY_TRAINING")
        if min_class_gold < min_neg_auc or "gold_standard_30" in forbidden_tiers:
            warnings.append("GOLD_STANDARD_30_FORBIDDEN_FOR_PRIMARY_TRAINING_MIN_NEG_VIOLATION")
        for audit_name in ["cohort_label_audit", "tracer_label_audit", "gap_label_audit"]:
            audit = gap.get(audit_name, {})
            if not audit:
                design_blockers.append(f"MISSING_{audit_name.upper()}")
            elif audit.get("status") == "SINGLE_ROW_OR_COLUMN":
                warnings.append(f"{audit_name.upper()}_SINGLE_ROW_OR_COLUMN_REVIEW")
            elif audit.get("chi2", 0) and audit.get("chi2", 0) > 50:
                warnings.append(f"{audit_name.upper()}_POSSIBLE_STRONG_COUPLING")

    if not preprocess or preprocess.get("mode") != "execute":
        execution_blockers.append("PREPROCESS_NOT_EXECUTED")
    elif pet_progress.get("resume_done", 0) <= 0:
        execution_blockers.append("NO_QC_PASS_PET_OUTPUTS")
    elif not pet_progress.get("preprocessing_complete"):
        execution_blockers.append("PET_PREPROCESS_INCOMPLETE")
    if preprocess.get("spatial_mode") == "identity_for_tests":
        execution_blockers.append("IDENTITY_SPATIAL_MODE_NOT_TRAINING_READY")
    if preprocess.get("spatial_mode") == "ants_rigid_pet_to_t1":
        warnings.append("SUBJECT_T1_ONLY_SPATIAL_OUTPUT_REQUIRES_MNI_STEP_FOR_REFERENCE_REGION_POLICY")

    if not cache:
        execution_blockers.append("PET_CACHE_NOT_BUILT_YET")
    if not (PROJECT_ROOT / args.modeling_design).exists():
        design_blockers.append("MISSING_METHOD_Y_MODELING_DESIGN")

    qc_audit = read_json(PROJECT_ROOT / "results/tables/preprocess_quality_audit.json")
    visual_manifest = read_json(PROJECT_ROOT / "results/qc/visual_review_manifest.json")
    if not qc_audit:
        execution_blockers.append("PREPROCESS_QUALITY_AUDIT_NOT_RUN")
    elif qc_audit.get("gate_verdict", {}).get("training_ready_after_audit") is False:
        execution_blockers.append("PREPROCESS_QUALITY_AUDIT_BLOCKERS")
        warnings.extend(qc_audit.get("gate_verdict", {}).get("blockers") or [])
    if not visual_manifest:
        execution_blockers.append("VISUAL_QC_MANIFEST_MISSING")
    elif visual_manifest.get("pending_signoff_count", 0) > 0:
        execution_blockers.append("VISUAL_QC_SIGNOFF_INCOMPLETE")

    if design_blockers:
        status = "NOT_TRAINING_READY"
    elif execution_blockers:
        status = "DESIGN_READY_EXECUTION_GATED"
    else:
        status = "READY_FOR_EXPLORATORY_METHOD_Y_TRAINING"

    report = {
        "stage": "method_y_training_readiness",
        "version": "v3.9-readiness-split",
        "tracer": args.tracer,
        "status": status,
        "blockers": design_blockers,
        "execution_blockers": execution_blockers,
        "warnings": warnings,
        "pet_preprocess_progress": pet_progress,
        "method_x_claim_unchanged": True,
        "pet_training_if_ready_is_method_y_only": True,
        "clinical_alzheimer_diagnosis_allowed": False,
        "approved_training_tier": approved_tier,
        "forbidden_primary_training_tiers": sorted(forbidden_tiers),
        "preprocess_quality_audit": {
            "present": bool(qc_audit),
            "training_ready_after_audit": nested_bool(qc_audit, "gate_verdict", "training_ready_after_audit"),
            "blockers": (qc_audit or {}).get("gate_verdict", {}).get("blockers", []),
        },
        "visual_qc_manifest": {
            "present": bool(visual_manifest),
            "pending_signoff_count": (visual_manifest or {}).get("pending_signoff_count"),
            "all_signed_off": (visual_manifest or {}).get("all_signed_off"),
        },
        "class_imbalance_policy": "configs/policy/method_y_class_imbalance_policy.json",
        "interpretation": (
            "DESIGN_READY_EXECUTION_GATED means manifest/gap/tier policy pass but PET preprocess or cache incomplete. "
            "This is not a contradiction with thesis report when both design and execution sections are read."
        ),
    }
    atomic_write_json(PROJECT_ROOT / "results/tables/method_y_training_readiness.json", report)
    md = [
        "# Method Y Training Readiness\n\n",
        f"- Status: `{status}`\n",
        f"- Design blockers: `{design_blockers}`\n",
        f"- Execution blockers: `{execution_blockers}`\n",
        f"- Warnings: `{warnings}`\n",
        f"- PET progress: `{pet_progress.get('resume_done')}/{linked}`\n",
        f"- Manifest regex_fallback_used: `{manifest_summary.get('regex_fallback_used')}`\n",
        f"- Manifest project_inventory_used: `{manifest_summary.get('project_inventory_used')}`\n",
        f"- Approved tier: `{approved_tier}`\n",
        f"- Forbidden tiers for primary training: `{sorted(forbidden_tiers)}`\n",
        f"- Interpretation: {report['interpretation']}\n",
    ]
    atomic_write_text(PROJECT_ROOT / "results/tables/method_y_training_readiness.md", "".join(md))
    print(status)
    return 0 if status == "READY_FOR_EXPLORATORY_METHOD_Y_TRAINING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
