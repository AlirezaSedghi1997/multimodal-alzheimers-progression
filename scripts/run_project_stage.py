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
    "inventory",
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

STAGE_COMMANDS = {
    "preflight": ["scripts/preflight_check.py"],
    "external-drive-check": ["scripts/external_drive_check.py"],
    "compute-plan": ["scripts/compute_plan.py"],
    "inventory": ["scripts/run_inventory.py"],
    "catalog": ["scripts/build_master_catalog.py"],
    "splits": ["scripts/make_splits.py"],
    "power-analysis": ["scripts/power_analysis.py"],
    "site-drift": ["scripts/site_scanner_drift_sensitivity.py"],
    "subgroup-audit": ["scripts/subgroup_performance_audit.py"],
    "preprocess": ["scripts/preprocess_mri.py"],
    "preprocess-full-mri": ["scripts/preprocess_mri_full.py"],
    "mri-volume-cache": ["scripts/build_mri_volume_cache.py"],
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
}

STAGE_ARTIFACTS = {
    "preflight": Path("results/tables/preflight_report.json"),
    "external-drive-check": Path("results/tables/external_drive_check.json"),
    "compute-plan": Path("results/tables/compute_plan.json"),
    "inventory": Path("results/tables/dataset_inventory.json"),
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
}
REQUIRED_VOLUME_CACHE_KEYS = {"image", "label", "sample_hash", "patient_hash", "tabular", "status"}
POST_SPM_GATED_STAGES = {
    "mri-volume-cache",
    "train-3d",
    "optuna",
    "fusion",
    "external",
    "calibration",
    "final-analysis",
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


def stage_order(include_method_x_cog: bool = False) -> list[str]:
    if include_method_x_cog:
        return [*STAGE_ORDER, *METHOD_X_COG_STAGE_ORDER]
    return STAGE_ORDER


def next_missing_stage(data_root: str | None = None, *, include_method_x_cog: bool = False) -> str | None:
    for stage in stage_order(include_method_x_cog):
        if not artifact_is_current(stage, data_root):
            return stage
    return None


def run_stage(stage: str, data_root: str | None) -> int:
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
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    args = ap.parse_args()
    if args.run_all:
        while True:
            stage = next_missing_stage(args.data_root, include_method_x_cog=args.include_method_x_cog)
            if stage is None:
                print("all_stages_complete")
                return 0
            code = run_stage(stage, args.data_root)
            if code != 0:
                return code
    if args.next:
        stage = next_missing_stage(args.data_root, include_method_x_cog=args.include_method_x_cog)
        if stage is None:
            print("all_stages_complete")
            return 0
        return run_stage(stage, args.data_root)
    return run_stage(args.stage or "preflight", args.data_root)


if __name__ == "__main__":
    raise SystemExit(main())
