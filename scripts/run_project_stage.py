from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import resolve_data_root, resolve_required_file
from scripts.spm_completion_guard import spm_completion_ready, spm_completion_status


STAGE_ORDER = [
    "preflight",
    "external-drive-check",
    "compute-plan",
    "phase0-governance",
    "branch-catalogs",
    "cohort-label-coupling-audit",
    "leakage-review-report",
    "gpu-preflight",
    "phase1-baselines",
    "phase1-incremental-value",
    "extract-3d-embeddings-oof",
    "cohort-probe",
    "centiloid-regression",
    "phase2-mvp-audit",
    "three-d-ablation",
    "method-x-cog-sf-stress-tests",
    "bio-cog-method-z-readiness",
    "inventory",
    "extended-modality-inventory",
    "catalog",
    "splits",
    "power-analysis",
    "site-drift",
    "subgroup-audit",
    "preprocess",
    "preprocess-full-mri",
    "mri-volume-cache",
    "mri-features",
    "baseline",
    "mri-feature-baseline",
    "train-3d",
    "cross-cohort-bias",
    "optuna",
    "fusion",
    "external",
    "calibration",
    "tau",
    "clinical-ad-feasibility",
    "final-analysis",
    "gap-matrix",
    "public-bundle",
]

METHOD_X_COG_STAGE_ORDER = [
    "cognitive-features",
    "cognitive-ablation",
]

METHOD_Y_STAGE_ORDER = [
    "method-y-gap-filter",
    "method-y-pet-manifest",
    "method-y-pet-dry",
    "method-y-readiness",
    "method-y-pet-cache",
    "method-y-pet-train",
    "method-y-summary",
]

STAGE_COMMANDS = {
    "preflight": ["scripts/preflight_check.py"],
    "external-drive-check": ["scripts/external_drive_check.py"],
    "compute-plan": ["scripts/compute_plan.py"],
    "phase0-governance": ["scripts/write_phase0_governance_artifacts.py"],
    "branch-catalogs": ["scripts/build_branch_catalogs.py"],
    "cohort-label-coupling-audit": ["scripts/run_cohort_label_coupling_audit.py"],
    "leakage-review-report": ["scripts/write_leakage_review_report.py"],
    "gpu-preflight": ["scripts/gpu_preflight_check.py"],
    "phase1-baselines": ["scripts/run_branch_baselines.py"],
    "phase1-incremental-value": ["scripts/build_incremental_value_table.py"],
    "extract-3d-embeddings-oof": ["scripts/extract_3d_mri_embeddings_oof.py"],
    "cohort-probe": ["scripts/run_cohort_probe.py"],
    "centiloid-regression": ["scripts/run_centiloid_regression.py"],
    "phase2-mvp-audit": ["scripts/run_phase2_mvp_audit.py"],
    "three-d-ablation": ["scripts/run_3d_ablation.py", "--cache", "data/cached/t1_volume_cache_96.h5", "--folds", "5", "--epochs", "4", "--batch_size", "1"],
    "method-x-cog-sf-stress-tests": ["scripts/run_method_x_cog_sf_stress_tests.py"],
    "bio-cog-method-z-readiness": ["scripts/audit_bio_cog_method_z_readiness.py"],
    "thesis-grade-report": ["scripts/build_thesis_grade_report.py"],
    "inventory": ["scripts/run_inventory.py"],
    "extended-modality-inventory": ["scripts/build_extended_modality_inventory.py"],
    "catalog": ["scripts/build_master_catalog.py"],
    "splits": ["scripts/make_splits.py"],
    "power-analysis": ["scripts/power_analysis.py"],
    "site-drift": ["scripts/site_scanner_drift_sensitivity.py"],
    "subgroup-audit": ["scripts/subgroup_performance_audit.py"],
    "preprocess": ["scripts/preprocess_mri.py"],
    "preprocess-full-mri": ["scripts/preprocess_mri_full.py"],
    "mri-volume-cache": ["scripts/build_mri_volume_cache.py"],
    "mri-volume-cache-smoke": ["scripts/mri_volume_cache_smoke.py"],
    "baseline": ["scripts/train_baseline.py"],
    "cross-cohort-bias": ["scripts/cross_cohort_bias_test.py"],
    "mri-features": ["scripts/extract_mri_features.py"],
    "mri-feature-baseline": ["scripts/train_mri_feature_baseline.py"],
    "train-3d": ["scripts/train_3d_mri.py"],
    "optuna": ["scripts/run_optuna.py"],
    "fusion": ["scripts/run_fusion.py"],
    "external": ["scripts/run_external_eval.py"],
    "calibration": ["scripts/run_calibration.py"],
    "final-analysis": ["scripts/final_analysis.py"],
    "gap-matrix": ["scripts/build_implementation_gap_matrix.py"],
    "public-bundle": ["scripts/build_public_bundle.py"],
    "tau": ["scripts/run_tau_branch.py"],
    "clinical-ad-feasibility": ["scripts/clinical_ad_target_feasibility.py"],
    "cognitive-features": ["scripts/build_cognitive_features.py"],
    "cognitive-ablation": ["scripts/run_cognitive_ablation.py"],
    "method-y-gap-filter": ["scripts/method_y_gap_filter.py"],
    "method-y-pet-manifest": ["scripts/build_pet_private_manifest.py"],
    "method-y-pet-dry": ["scripts/preprocess_pet_spatial.py", "--dry_run", "--limit", "10"],
    "method-y-readiness": ["scripts/check_method_y_training_readiness.py"],
    "method-y-pet-cache": ["scripts/build_pet_volume_cache.py", "--tracer", "fbp"],
    "method-y-pet-train": ["scripts/train_pet_method_y.py", "--tracer", "fbp"],
    "method-y-summary": ["scripts/build_phase_y_summary.py"],
}

STAGE_ARTIFACTS = {
    "preflight": Path("results/tables/preflight_report.json"),
    "external-drive-check": Path("results/tables/external_drive_check.json"),
    "compute-plan": Path("results/tables/compute_plan.json"),
    "phase0-governance": Path("results/governance/phase0_governance_manifest.json"),
    "branch-catalogs": Path("data/processed/catalogs/branch_catalog_manifest.json"),
    "cohort-label-coupling-audit": Path("results/governance/cohort_label_coupling_audit.json"),
    "leakage-review-report": Path("results/governance/leakage_review_report.json"),
    "gpu-preflight": Path("results/governance/gpu_preflight_check.json"),
    "phase1-baselines": Path("results/final_results/phase1_baselines/phase1_baselines_summary.json"),
    "phase1-incremental-value": Path("results/tables/phase1_incremental_value_table.json"),
    "extract-3d-embeddings-oof": Path("results/tables/cohort_probe/method_x_3d_embedding_extraction_summary.json"),
    "cohort-probe": Path("results/tables/cohort_probe/cohort_probe_summary.json"),
    "centiloid-regression": Path("results/final_results/centiloid_regression/centiloid_regression_metrics.json"),
    "phase2-mvp-audit": Path("results/tables/phase2_mvp/phase2_mvp_status.json"),
    "three-d-ablation": Path("results/final_results/three_d_ablation/three_d_ablation_summary.json"),
    "method-x-cog-sf-stress-tests": Path("results/final_results/method_x_cog_sf/sf_stress_tests_summary.json"),
    "bio-cog-method-z-readiness": Path("results/tables/bio_cog_method_z_readiness.json"),
    "thesis-grade-report": Path("results/final_results/thesis/thesis_grade_report.md"),
    "inventory": Path("results/tables/dataset_inventory.json"),
    "extended-modality-inventory": Path("results/tables/extended_modality_inventory.json"),
    "catalog": Path("data/processed/master_catalog.csv"),
    "splits": Path("splits/split_summary.json"),
    "power-analysis": Path("results/tables/power_analysis.json"),
    "site-drift": Path("results/tables/site_scanner_drift_sensitivity_report.json"),
    "subgroup-audit": Path("results/final_results/subgroup_performance_audit.json"),
    "preprocess": Path("data/cached/t1_header_cache.h5"),
    "preprocess-full-mri": Path("results/tables/full_mri_preprocess_toolchain_report.json"),
    "mri-volume-cache": Path("data/cached/t1_volume_cache_64.h5"),
    "mri-features": Path("data/processed/mri_features.csv"),
    "baseline": Path("results/final_results/baseline/baseline_metrics.json"),
    "mri-feature-baseline": Path("results/final_results/mri_feature_baseline/mri_feature_baseline_metrics.json"),
    "train-3d": Path("results/final_results/three_d/three_d_metrics.json"),
    "cross-cohort-bias": Path("results/tables/cross_cohort_bias_report.json"),
    "optuna": Path("results/final_results/optuna/optuna_summary.json"),
    "fusion": Path("results/final_results/fusion/fusion_summary.json"),
    "external": Path("results/final_results/external/external_metrics.json"),
    "calibration": Path("results/final_results/calibration/calibration_report.json"),
    "final-analysis": Path("results/final_results/final_analysis/final_analysis_summary.json"),
    "gap-matrix": Path("results/tables/implementation_gap_matrix.json"),
    "public-bundle": Path("public_bundle/PUBLIC_BUNDLE_MANIFEST.json"),
    "tau": Path("results/final_results/tau/tau_regression_metrics.json"),
    "clinical-ad-feasibility": Path("results/tables/clinical_ad_target_feasibility.json"),
    "cognitive-features": Path("data/processed/cognitive_features.csv"),
    "cognitive-ablation": Path("results/final_results/method_x_cog/method_x_cog_summary.json"),
    "method-y-gap-filter": Path("results/tables/method_y_gap_filter_report.json"),
    "method-y-pet-manifest": Path("results/tables/method_y_pet_manifest_summary.json"),
    "method-y-pet-dry": Path("results/tables/method_y_pet_preprocess_summary.json"),
    "method-y-readiness": Path("results/tables/method_y_training_readiness.json"),
    "method-y-pet-cache": Path("results/tables/method_y_pet_cache_summary.json"),
    "method-y-pet-train": Path("results/final_results/method_y/pet_training/method_y_pet_metrics.json"),
    "method-y-summary": Path("results/final_results/method_y/phase_y_summary.md"),
}

DATA_ROOT_STAGES = {
    "preflight",
    "external-drive-check",
    "inventory",
    "catalog",
    "preprocess",
    "preprocess-full-mri",
    "mri-volume-cache",
    "mri-features",
    "tau",
    "clinical-ad-feasibility",
    "cognitive-features",
    "method-y-pet-manifest",
    "method-y-pet-dry",
}
REQUIRED_VOLUME_CACHE_KEYS = {"image", "label", "sample_hash", "patient_hash", "tabular", "status"}
POST_SPM_GATED_STAGES = {
    "mri-volume-cache",
    "extract-3d-embeddings-oof",
    "three-d-ablation",
    "train-3d",
    "optuna",
    "fusion",
    "external",
    "calibration",
    "final-analysis",
}
PRE_MODELING_GATED_STAGES = {
    "baseline",
    "mri-feature-baseline",
    "phase1-baselines",
    "train-3d",
    "cross-cohort-bias",
    "optuna",
    "fusion",
    "external",
    "calibration",
    "final-analysis",
    "cognitive-ablation",
}


def tau_target_available(data_root: str | None) -> bool:
    if not data_root:
        return False
    root = resolve_data_root(data_root)
    return resolve_required_file(root, "tau").exists()


def newer_than(path: Path, dependency: Path) -> bool:
    return path.exists() and dependency.exists() and dependency.stat().st_mtime > path.stat().st_mtime


def valid_volume_cache(path: Path) -> bool:
    try:
        import h5py

        with h5py.File(path, "r") as h5:
            return REQUIRED_VOLUME_CACHE_KEYS.issubset(set(h5.keys()))
    except Exception:
        return False


def artifact_is_current(stage: str, data_root: str | None = None) -> bool:
    artifact = STAGE_ARTIFACTS[stage]
    if not artifact.exists():
        return False
    if stage == "mri-volume-cache":
        return valid_volume_cache(artifact)
    if stage == "power-analysis":
        try:
            obj = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return obj.get("status") == "DESCRIPTIVE_POWER_ANALYSIS"
    if stage == "preprocess-full-mri":
        try:
            obj = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return obj.get("status") == "COMPLETED"
    if stage == "final-analysis" and newer_than(artifact, STAGE_ARTIFACTS["train-3d"]):
        return False
    if stage == "final-analysis" and newer_than(artifact, STAGE_ARTIFACTS["tau"]):
        return False
    if stage == "public-bundle" and newer_than(artifact, STAGE_ARTIFACTS["final-analysis"]):
        return False
    if stage == "clinical-ad-feasibility":
        try:
            obj = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            obj.get("status") == "FEASIBILITY_ONLY_NOT_TRAINED"
            and obj.get("training_started") is False
            and obj.get("clinical_alzheimer_output_allowed_now") is False
            and obj.get("subject_level_outputs_exported") is False
        )
    if stage == "method-y-readiness":
        try:
            obj = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return obj.get("status") == "READY_FOR_EXPLORATORY_METHOD_Y_TRAINING"
    if stage == "method-y-pet-cache":
        try:
            obj = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(obj.get("n", 0) > 0 and obj.get("method_x_cache") is False)
    if stage == "method-y-pet-train":
        try:
            obj = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return obj.get("status") in {"COMPLETED", "CPU_DEBUG_ONLY"}
    if stage == "tau":
        try:
            obj = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if obj.get("schema_version") != 2:
            return False
        if obj.get("mode") == "SKIP_WITH_JUSTIFICATION" and tau_target_available(data_root):
            return False
        return True
    return True


def stage_order(include_method_x_cog: bool = False, include_method_y: bool = False) -> list[str]:
    order = list(STAGE_ORDER)
    if include_method_x_cog:
        order.extend(METHOD_X_COG_STAGE_ORDER)
    if include_method_y:
        order.extend(METHOD_Y_STAGE_ORDER)
    return order


def next_missing_stage(data_root: str | None = None, *, include_method_x_cog: bool = False, include_method_y: bool = False) -> str | None:
    for stage in stage_order(include_method_x_cog, include_method_y):
        if not artifact_is_current(stage, data_root):
            return stage
    return None


def run_stage(stage: str, data_root: str | None) -> int:
    if stage in PRE_MODELING_GATED_STAGES:
        guard = [
            sys.executable,
            "scripts/phase0_readiness_guard.py",
            "--require_label_freeze",
            "--require_feature_matrix_check",
        ]
        guard_code = subprocess.call(guard)
        if guard_code != 0:
            return guard_code
    if stage in POST_SPM_GATED_STAGES:
        spm_status = spm_completion_status()
        if not spm_completion_ready(spm_status):
            print(json.dumps({"stage": stage, "blocked": "SPM_NOT_COMPLETE", "spm_completion": spm_status}, indent=2))
            return 4
    cmd = [sys.executable, *STAGE_COMMANDS[stage]]
    if stage in DATA_ROOT_STAGES and data_root:
        cmd += ["--data_root", data_root]
    print(f"[stage] {stage}")
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGE_COMMANDS))
    ap.add_argument("--next", action="store_true")
    ap.add_argument("--run_all", action="store_true")
    ap.add_argument(
        "--include_method_x_cog",
        action="store_true",
        help="Include optional Method X-Cog cognitive feature and ablation stages after the primary Method X pipeline.",
    )
    ap.add_argument(
        "--include_method_y",
        action="store_true",
        help="Include optional gated Method Y PET dry-run/readiness stages after the primary Method X pipeline.",
    )
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    args = ap.parse_args()
    if args.run_all:
        while True:
            stage = next_missing_stage(args.data_root, include_method_x_cog=args.include_method_x_cog, include_method_y=args.include_method_y)
            if stage is None:
                print("all_stages_complete")
                return 0
            code = run_stage(stage, args.data_root)
            if code != 0:
                return code
    if args.next:
        stage = next_missing_stage(args.data_root, include_method_x_cog=args.include_method_x_cog, include_method_y=args.include_method_y)
        if stage is None:
            print("all_stages_complete")
            return 0
        return run_stage(stage, args.data_root)
    return run_stage(args.stage or "preflight", args.data_root)


if __name__ == "__main__":
    raise SystemExit(main())

