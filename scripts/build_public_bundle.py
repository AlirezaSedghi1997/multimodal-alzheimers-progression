from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.public_sanitize import (
    refresh_mri_smoke_summary_if_superseded,
    sanitize_public_text,
    scan_for_local_paths,
    scan_json_for_subject_vectors,
)
from src.utils.state_manager import StateManager


PUBLIC_FILES = [
    "README.md",
    "governance/method_x_operating_contract.md",
    "governance/method_x_cog_operating_contract.md",
    "governance/method_y_pet_exploratory_policy.md",
    "governance/method_y_multimodal_operating_contract.md",
    "governance/method_y_class_imbalance_policy.md",
    "governance/cognitive_feature_policy.md",
    "governance/data_schema_policy.md",
    "governance/calibration_policy.md",
    "governance/feasibility_assessment.md",
    "governance/compute_fallback_policy.md",
    "governance/external_drive_policy.md",
    "governance/sf_internal_decoupling_policy.md",
    "governance/clinical_endpoint_progression_policy.md",
    "docs/final_execution_program.md",
    "docs/final_writeup_method_x_cog_method_y_pet_mm.md",
    "docs/clinical_alzheimer_target_feasibility_and_governance.md",
    "docs/method_x_cog_validation_plan.md",
    "docs/method_y_pet_assisted_exploratory_plan.md",
    "docs/method_y_pet_mri_multimodal_plan.md",
    "docs/method_y_status_summary.md",
    "docs/clinical_endpoint_readiness.md",
    "docs/bio_cog_blood_column_mapping_audit.md",
    "docs/research_methods_limitations_ethics_statistics.md",
    "docs/spm_incomplete_safe_workflow.md",
    "configs/model/method_y_multimodal_plan.yaml",
    "results/tables/dataset_inventory.md",
    "results/tables/master_catalog_summary.md",
    "results/tables/split_summary.md",
    "results/tables/cross_cohort_bias_report.md",
    "results/tables/power_analysis.md",
    "results/tables/implementation_gap_matrix.md",
    "results/tables/full_mri_preprocess_toolchain_report.md",
    "results/tables/spm_preprocess_toolchain_report.md",
    "results/tables/site_scanner_drift_sensitivity_report.md",
    "results/tables/method_y_pet_inventory.md",
    "results/tables/method_y_multimodal_design.md",
    "results/tables/clinical_ad_target_feasibility.md",
    "results/tables/clinical_ad_target_feasibility.json",
    "results/tables/mri_volume_cache_smoke_summary.md",
    "results/tables/mri_volume_cache_smoke_summary.json",
    "results/tables/method_y_gap_filter_report.md",
    "results/tables/method_y_pet_manifest_summary.json",
    "results/tables/method_y_pet_manifest_summary.md",
    "results/tables/method_y_pet_preprocess_summary.json",
    "results/tables/method_y_training_readiness.json",
    "results/tables/method_y_training_readiness.md",
    "results/tables/phase1_incremental_value_table.md",
    "results/tables/phase2_mvp/phase2_mvp_status.md",
    "results/tables/cohort_probe/cohort_probe_summary.md",
    "results/tables/headline_temporal_gap_evaluation.md",
    "results/tables/decision_curve_analysis.md",
    "results/tables/matched_propensity_subset.md",
    "results/tables/final_metrics_table.md",
    "results/tables/gate_matrix.md",
    "results/tables/bio_cog_method_z_readiness.md",
    "results/tables/extended_modality_inventory.md",
    "results/tables/extended_modality_inventory.json",
    "results/tables/augmentation_policy.md",
    "results/tables/augmentation_policy.json",
    "results/tables/synthetic_mri_pet_supplement.md",
    "results/tables/synthetic_mri_pet_supplement.json",
    "results/tables/method_y_incremental_value_table.md",
    "results/tables/method_y_incremental_value_table.json",
    "results/tables/proposal_alignment_report.md",
    "results/tables/proposal_alignment_report.json",
    "results/tables/tau_feasibility_report.md",
    "results/final_results/thesis/thesis_grade_report.md",
    "results/final_results/method_y/phase_y_summary.md",
    "results/final_results/method_y/mri_pet_cognitive_training/method_y_mri_pet_cognitive_metrics.md",
    "results/final_results/subgroup_performance_audit.md",
    "results/final_results/three_d_ablation/three_d_ablation_summary.md",
    "results/final_results/xai/xai_summary.md",
    "results/final_results/final_analysis/final_analysis_report.md",
    "results/final_results/three_d/three_d_metrics.md",
    "results/final_results/centiloid_regression/centiloid_regression_metrics.md",
    "results/final_results/method_x_cog_sf/sf_stress_tests_summary.md",
    "results/final_results/method_x_cog/method_x_cog_summary.md",
    "results/final_results/bio_cog_method_z/bio_cog_method_z_summary.md",
    "results/final_results/clinical_endpoint/clinical_endpoint_baseline_summary.md",
    "results/final_results/tau/tau_regression_metrics.md",
    "results/governance/domain_ceiling_citation_list.json",
    "results/governance/tripod_probast_checklists.md",
    "results/governance/tripod_manuscript_scaffold.md",
    "results/governance/cohort_label_coupling_audit.md",
]

FORBIDDEN_SUFFIXES = {".csv", ".h5", ".hdf5", ".nii", ".gz", ".pt", ".ckpt"}
FORBIDDEN_TERMS = ("PREVBID", "patient_id", "sample_id")


def copy_if_exists(src: Path, root: Path, out: Path) -> list[str]:
    copied = []
    if not src.exists():
        return copied
    dst = out / src.relative_to(root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml"}:
        dst.write_text(sanitize_public_text(src.read_text(encoding="utf-8")), encoding="utf-8")
    else:
        shutil.copy2(src, dst)
    copied.append(dst.relative_to(out).as_posix())
    return copied


def validate_public_bundle(out: Path) -> list[str]:
    violations = []
    for p in out.rglob("*"):
        if not p.is_file():
            continue
        lower = p.name.lower()
        if any(lower.endswith(s) for s in FORBIDDEN_SUFFIXES):
            violations.append(f"forbidden_suffix:{p.relative_to(out)}")
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"non_text:{p.relative_to(out)}")
            continue
        for term in FORBIDDEN_TERMS:
            if term in text:
                violations.append(f"forbidden_term:{term}:{p.relative_to(out)}")
        if scan_for_local_paths(text):
            violations.append(f"raw_local_path:{p.relative_to(out)}")
        if p.suffix.lower() == ".json":
            try:
                if scan_json_for_subject_vectors(json.loads(text)):
                    violations.append(f"subject_level_vectors:{p.relative_to(out)}")
            except json.JSONDecodeError:
                violations.append(f"invalid_json:{p.relative_to(out)}")
    return violations


def main() -> int:
    refresh_mri_smoke_summary_if_superseded()
    root = Path(".").resolve()
    out = Path("public_bundle")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    copied: list[str] = []
    for rel in PUBLIC_FILES:
        copied.extend(copy_if_exists(root / rel, root, out))
    split_summary = root / "splits/split_summary.md"
    if split_summary.exists():
        dst = out / "results/tables/split_summary.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(sanitize_public_text(split_summary.read_text(encoding="utf-8")), encoding="utf-8")
        copied.append(dst.relative_to(out).as_posix())

    violations = validate_public_bundle(out)
    manifest = {
        "public_bundle": str(out),
        "copied_files": sorted(set(copied)),
        "contains_raw_data": False,
        "contains_hdf5": False,
        "contains_images": False,
        "contains_raw_identifiers": False,
        "contains_raw_paths": any(v.startswith("raw_local_path:") for v in violations),
        "contains_subject_level_vectors": any(v.startswith("subject_level_vectors:") for v in violations),
        "violations": violations,
    }
    atomic_write_json(out / "PUBLIC_BUNDLE_MANIFEST.json", manifest)
    readme = [
        "# Public Reproducibility Bundle\n\n",
        "Aggregate governance and results reports only.\n\n",
        "- Raw data included: `False`\n",
        "- HDF5/cache files included: `False`\n",
        "- MRI/PET images included: `False`\n",
        "- Raw identifiers included: `False`\n",
        "- Local filesystem paths redacted: `True`\n",
        "- Subject-level prediction vectors included: `False`\n",
        "- External holdout type: `locked_pseudo_external_random_patient_holdout`\n",
    ]
    atomic_write_text(out / "README_PUBLIC.md", "".join(readme))
    StateManager().append_event("public_bundle", "COMPLETED" if not violations else "FAILED", "Public bundle generated.")
    print(json.dumps({"files": len(copied), "violations": violations}, indent=2))
    return 0 if not violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
