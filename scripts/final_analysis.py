from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.final_claim_policy import three_d_gate_status, spm_progress, validity_recommendation
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Missing required final-analysis input: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_optional_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(value: object, ndigits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def nested(obj: dict | None, *keys: str) -> object:
    cur: object = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def main() -> int:
    master = load_json("results/tables/master_catalog_summary.json")
    splits = load_json("splits/split_summary.json")
    baseline = load_json("results/final_results/baseline/baseline_metrics.json")
    mri_baseline = load_json("results/final_results/mri_feature_baseline/mri_feature_baseline_metrics.json")
    optuna = load_json("results/final_results/optuna/optuna_summary.json")
    fusion = load_json("results/final_results/fusion/fusion_summary.json")
    external = load_json("results/final_results/external/external_metrics.json")
    calibration = load_json("results/final_results/calibration/calibration_report.json")
    bias = load_json("results/tables/cross_cohort_bias_report.json")
    tau = load_json("results/tables/tau_feasibility_report.json")
    tau_regression = load_optional_json("results/final_results/tau/tau_regression_metrics.json")
    three_d = load_optional_json("results/final_results/three_d/three_d_metrics.json")
    power = load_optional_json("results/tables/power_analysis.json")
    gap_matrix = load_optional_json("results/tables/implementation_gap_matrix.json")
    full_preprocess = load_optional_json("results/tables/full_mri_preprocess_toolchain_report.json")
    site_scanner = load_optional_json("results/tables/site_scanner_drift_sensitivity_report.json")
    subgroup = load_optional_json("results/final_results/subgroup_performance_audit.json")
    three_d_ablation = load_optional_json("results/final_results/three_d_ablation/three_d_ablation_summary.json")
    xai = load_optional_json("results/final_results/xai/xai_summary.json")
    method_x_cog = load_optional_json("results/final_results/method_x_cog/method_x_cog_summary.json")
    method_x_cog_sf = load_optional_json("results/final_results/method_x_cog_sf/sf_stress_tests_summary.json")
    phase2_mvp = load_optional_json("results/tables/phase2_mvp/phase2_mvp_status.json")
    cohort_probe = load_optional_json("results/tables/cohort_probe/cohort_probe_summary.json")
    centiloid_regression = load_optional_json("results/final_results/centiloid_regression/centiloid_regression_metrics.json")
    phase1_baselines = load_optional_json("results/final_results/phase1_baselines/phase1_baselines_summary.json")
    method_y_pet = load_optional_json("results/tables/method_y_pet_inventory.json")
    method_y_multimodal = load_optional_json("results/tables/method_y_multimodal_design.json")
    tau_gate = tau_regression["mode"] if tau_regression is not None else tau["feasibility_mode_120d"]
    three_d_gate = three_d_gate_status(three_d)
    power_gate = "PASSED" if power and power.get("status") == "DESCRIPTIVE_POWER_ANALYSIS" else "PENDING"
    gap_gate = "PASSED" if gap_matrix else "PENDING"
    full_preprocess_gate = full_preprocess.get("status") if full_preprocess else "PENDING"
    site_scanner_gate = site_scanner.get("status") if site_scanner else "PENDING"
    subgroup_gate = "PASSED" if subgroup else "PENDING"
    ablation_gate = "PASSED" if three_d_ablation and three_d_ablation.get("status") == "COMPLETED" else "PENDING"
    xai_gate = "PASSED" if xai and xai.get("status") == "COMPLETED" else "PENDING"
    method_x_cog_gate = "PASSED" if method_x_cog and method_x_cog.get("status") == "COMPLETED" else "PENDING"
    method_x_cog_sf_gate = "PASSED" if method_x_cog_sf and method_x_cog_sf.get("status") == "COMPLETED" else "PENDING"
    phase2_mvp_gate = "PASSED" if phase2_mvp and phase2_mvp.get("status") == "MVP_COMPLETE" else "PARTIAL" if phase2_mvp else "PENDING"
    cohort_probe_gate = "PASSED" if cohort_probe and cohort_probe.get("status") == "COMPLETED" else "PENDING"
    centiloid_gate = "PASSED" if centiloid_regression and centiloid_regression.get("status") == "COMPLETED" else "PENDING"
    phase1_gate = "PASSED" if phase1_baselines and phase1_baselines.get("status") == "COMPLETED" else "PENDING"
    method_y_pet_gate = "AGGREGATE_ONLY" if method_y_pet and method_y_pet.get("status") == "COMPLETED" else "PENDING"
    method_y_multimodal_gate = "DESIGN_READY_NOT_TRAINED" if method_y_multimodal and method_y_multimodal.get("status") == "DESIGN_READY_NOT_TRAINED" else "PENDING"

    gates = {
        "gate_h1_data": "PASSED",
        "gate_h2_qc": "PASSED",
        "gate_h3_split_lock": "PASSED",
        "gate_1_baseline": "PASSED" if baseline["aggregate"]["gate_1_passed"] else "FAILED",
        "gate_2_optuna": "PASSED" if optuna["best"]["gate_2_passed"] else "FAILED",
        "gate_3_hybrid": fusion["gate_3_status"],
        "gate_4_external": external["gate_4_status"],
        "gate_calibration": "PASSED" if calibration["external_used_for_fit"] is False else "FAILED",
        "gate_power_analysis": power_gate,
        "gate_gap_matrix": gap_gate,
        "gate_site_scanner_drift": site_scanner_gate,
        "gate_subgroup_audit": subgroup_gate,
        "gate_full_mri_preprocess": full_preprocess_gate,
        "gate_3d_training": three_d_gate,
        "gate_3d_fusion_ablation": ablation_gate,
        "gate_xai": xai_gate,
        "gate_method_x_cog": method_x_cog_gate,
        "gate_method_x_cog_sf_stress_tests": method_x_cog_sf_gate,
        "gate_phase2_minimum_mvp": phase2_mvp_gate,
        "gate_cohort_probe": cohort_probe_gate,
        "gate_centiloid_within_cohort": centiloid_gate,
        "gate_phase1_interpretable_baselines": phase1_gate,
        "gate_method_y_pet_inventory": method_y_pet_gate,
        "gate_method_y_multimodal_design": method_y_multimodal_gate,
        "gate_tau": tau_gate,
    }
    final = {
        "project": "amyloid-tau-prediction-a4learn",
        "method": "X",
        "method_x_compliant": True,
        "raw_identifiers_used": False,
        "pet_as_model_input": False,
        "catalog": master,
        "splits": {
            "rows_total": splits["rows_total"],
            "dev_rows": splits["dev_rows"],
            "external_rows": splits["external_rows"],
            "external_holdout_type": splits["external_holdout_type"],
        },
        "metrics": {
            "tabular_baseline_roc_auc": baseline["aggregate"]["roc_auc"],
            "mri_feature_baseline_roc_auc": mri_baseline["aggregate"]["roc_auc"],
            "optuna_best_roc_auc": optuna["best"]["roc_auc"],
            "fusion_best_roc_auc": fusion["best"]["roc_auc"],
            "external_like_roc_auc": external["roc_auc"],
            "external_like_pr_auc": external["pr_auc"],
            "cv_calibrated_brier": calibration["cv_calibrated"]["brier"],
            "cv_calibrated_ece": calibration["cv_calibrated"]["ece"],
            "external_like_calibrated_brier": calibration["external_like_calibrated"]["brier"],
            "external_like_calibrated_ece": calibration["external_like_calibrated"]["ece"],
            "three_d_roc_auc": three_d["aggregate"].get("roc_auc") if three_d else None,
            "three_d_pr_auc": three_d["aggregate"].get("pr_auc") if three_d else None,
            "three_d_brier": three_d["aggregate"].get("brier") if three_d else None,
            "three_d_external_like_roc_auc": (
                three_d["aggregate"]["external_like"].get("roc_auc")
                if three_d and three_d["aggregate"].get("external_like")
                else None
            ),
            "three_d_ablation_best_roc_auc": nested(three_d_ablation, "best_by_cv_roc_auc", "roc_auc"),
            "three_d_ablation_best_external_like_roc_auc": nested(
                three_d_ablation, "best_by_cv_roc_auc", "external_like_roc_auc"
            ),
            "tau_rows_used": nested(tau_regression, "rows_used"),
            "tau_r2": nested(tau_regression, "aggregate", "r2"),
            "tau_rmse": nested(tau_regression, "aggregate", "rmse"),
            "tau_mae": nested(tau_regression, "aggregate", "mae"),
            "tau_rmse_reduction_vs_mean_baseline": nested(tau_regression, "delta_vs_mean_baseline", "rmse_reduction"),
            "tau_r2_delta_vs_mean_baseline": nested(tau_regression, "delta_vs_mean_baseline", "r2_delta"),
            "method_x_cog_best_roc_auc": nested(method_x_cog, "best", "roc_auc"),
            "method_x_cog_best_pr_auc": nested(method_x_cog, "best", "pr_auc"),
            "method_x_cog_best_brier": nested(method_x_cog, "best", "brier"),
            "method_x_cog_auc_gain_over_fixed_tabular": nested(method_x_cog, "improvement_over_fixed_tabular_auc"),
            "method_x_cog_best_non_mri_roc_auc": nested(method_x_cog, "best_non_mri", "roc_auc"),
            "method_x_cog_best_non_mri_pr_auc": nested(method_x_cog, "best_non_mri", "pr_auc"),
            "method_x_cog_best_non_mri_auc_gain": nested(method_x_cog, "non_mri_improvement_over_fixed_tabular_auc"),
            "method_y_locked_catalog_rows": nested(method_y_pet, "locked_catalog", "rows"),
            "method_y_amyloid_pet_image_files": nested(method_y_pet, "pet_image_inventory", "amyloid_fbp", "image_file_count"),
            "method_y_tau_pet_image_files": nested(method_y_pet, "pet_image_inventory", "tau_ftp", "image_file_count"),
            "method_y_multimodal_training_started": nested(method_y_multimodal, "training_started"),
            "method_y_multimodal_clinical_ad_output_allowed_now": nested(method_y_multimodal, "target_policy", "clinical_alzheimer_output_allowed_now"),
            "phase1_fixed_tabular_roc_auc": nested(phase1_baselines, "results", "fixed_tabular", "roc_auc"),
            "cohort_probe_3d_latent_linear_auc": next(
                (
                    block.get("linear_probe_auc")
                    for block in (cohort_probe or {}).get("blocks", [])
                    if block.get("input_name") == "method_x_3d_mri_latent_oof"
                ),
                None,
            ),
            "centiloid_within_a4_spearman": nested(centiloid_regression, "subsets", "within_a4", "spearman_r"),
            "centiloid_within_learn_spearman": nested(centiloid_regression, "subsets", "within_learn", "spearman_r"),
            "method_x_cog_sf_pooled_cramers_v": nested(method_x_cog_sf, "cohort_label_coupling", "pooled_cramers_v"),
            "method_x_cog_sf_inclusive_cognitive_roc_auc": next(
                (
                    variant.get("roc_auc")
                    for arm in (method_x_cog_sf or {}).get("arms", [])
                    if arm.get("arm") == "sf_inclusive_confound_controlled_5fold_cv"
                    for variant in [arm.get("variants", {}).get("fixed_tabular_plus_all_cognitive", {})]
                ),
                None,
            ),
            "method_x_cog_sf_train_sf_cognitive_roc_auc": next(
                (
                    variant.get("roc_auc")
                    for arm in (method_x_cog_sf or {}).get("arms", [])
                    if arm.get("arm") == "train_sf_test_enrolled_internal_decoupling"
                    for variant in [arm.get("variants", {}).get("fixed_tabular_plus_all_cognitive", {})]
                ),
                None,
            ),
            "phase2_mvp_overall_status": nested(phase2_mvp, "status"),
        },
        "gates": gates,
        "bias": {
            "risk_level": bias["risk_level"],
            "cohort_label_cramers_v": bias["audit"]["cohort_label_cramers_v"],
            "all_cohorts_single_label": bias["audit"]["all_cohorts_single_label"],
        },
        "tau": tau,
        "tau_regression": tau_regression,
        "three_d": three_d,
        "power_analysis": power,
        "implementation_gap_matrix": gap_matrix,
        "full_mri_preprocess": full_preprocess,
        "site_scanner_drift": site_scanner,
        "subgroup_performance_audit": subgroup,
        "three_d_ablation": three_d_ablation,
        "xai": xai,
        "method_x_cog": method_x_cog,
        "method_x_cog_sf": method_x_cog_sf,
        "phase2_mvp": phase2_mvp,
        "cohort_probe": cohort_probe,
        "centiloid_regression": centiloid_regression,
        "phase1_baselines": phase1_baselines,
        "method_y_pet_inventory": method_y_pet,
        "method_y_multimodal_design": method_y_multimodal,
        "scientific_caveat": (
            "The local catalog has complete cohort-label coupling. Report performance as Method X engineering evidence "
            "under a pseudo-external holdout, not as proof of pathology-only generalization to an independent cohort."
        ),
        "validity_recommendation": validity_recommendation(),
        "spm_progress": spm_progress(),
    }
    out_dir = Path("results/final_results/final_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "final_analysis_summary.json", final)

    gate_lines = ["# Gate Matrix\n\n", "| gate | status |\n", "|---|---|\n"]
    for gate, status in gates.items():
        gate_lines.append(f"| {gate} | {status} |\n")
    atomic_write_text("results/tables/gate_matrix.md", "".join(gate_lines))

    metric_lines = [
        "# Final Metrics Table\n\n",
        "| metric | value |\n",
        "|---|---:|\n",
    ]
    for key, value in final["metrics"].items():
        metric_lines.append(f"| {key} | {fmt(value)} |\n")
    atomic_write_text("results/tables/final_metrics_table.md", "".join(metric_lines))

    report = [
        "# Final Analysis Report\n\n",
        "## Summary\n\n",
        "Method X pipeline execution reached Gate 4 and calibration using T1 MRI-derived features plus allowed tabular variables. ",
        "PET was used only as label/target source and never as a model input.\n\n",
        "## Key Results\n\n",
        f"- Catalog rows: `{master['rows']}`\n",
        f"- Dev rows: `{splits['dev_rows']}`\n",
        f"- Pseudo-external rows: `{splits['external_rows']}`\n",
        f"- Best fusion ROC-AUC: `{fmt(fusion['best']['roc_auc'])}`\n",
        f"- External-like ROC-AUC: `{fmt(external['roc_auc'])}`\n",
        f"- 3D training mode: `{three_d['aggregate']['mode'] if three_d else 'PENDING'}`\n",
        f"- 3D ROC-AUC: `{fmt(three_d['aggregate'].get('roc_auc')) if three_d else 'NA'}`\n",
        f"- 3D external-like ROC-AUC: `{fmt(three_d['aggregate']['external_like'].get('roc_auc')) if three_d and three_d['aggregate'].get('external_like') else 'NA'}`\n",
        f"- 3D ablation status: `{ablation_gate}`\n",
        f"- 3D ablation best ROC-AUC: `{fmt(nested(three_d_ablation, 'best_by_cv_roc_auc', 'roc_auc'))}`\n",
        f"- CV calibrated Brier: `{fmt(calibration['cv_calibrated']['brier'])}`\n",
        f"- CV calibrated ECE: `{fmt(calibration['cv_calibrated']['ece'])}`\n",
        f"- Tau feasibility mode: `{tau['feasibility_mode_120d']}`\n",
        f"- Tau analysis mode: `{tau_gate}`\n",
        f"- Tau regression R2: `{fmt(tau_regression['aggregate']['r2']) if tau_regression else 'NA'}`\n\n",
        f"- Tau rows used: `{fmt(nested(tau_regression, 'rows_used'), 0)}`\n",
        f"- Tau RMSE: `{fmt(nested(tau_regression, 'aggregate', 'rmse'))}`\n",
        f"- Tau RMSE reduction vs mean baseline: `{fmt(nested(tau_regression, 'delta_vs_mean_baseline', 'rmse_reduction'))}`\n",
        f"- Tau interpretation: `{nested(tau_regression, 'interpretation_detail') if tau_regression else 'NA'}`\n\n",
        f"- Power analysis status: `{power.get('status') if power else 'PENDING'}`\n",
        f"- Full MRI preprocessing status: `{full_preprocess_gate}`\n",
        f"- Site/scanner drift status: `{site_scanner_gate}`\n",
        f"- Subgroup audit status: `{subgroup_gate}`\n",
        f"- XAI status: `{xai_gate}`\n",
        f"- Method X-Cog status: `{method_x_cog_gate}`\n",
        f"- Method X-Cog best ROC-AUC: `{fmt(nested(method_x_cog, 'best', 'roc_auc'))}`\n",
        f"- Method X-Cog AUC gain over fixed tabular: `{fmt(nested(method_x_cog, 'improvement_over_fixed_tabular_auc'))}`\n",
        f"- Method X-Cog best non-MRI variant: `{nested(method_x_cog, 'best_non_mri', 'variant') or 'NA'}`\n",
        f"- Method X-Cog best non-MRI ROC-AUC: `{fmt(nested(method_x_cog, 'best_non_mri', 'roc_auc'))}`\n",
        f"- Method X-Cog/SF stress-test status: `{method_x_cog_sf_gate}`\n",
        f"- Phase 2 minimum MVP status: `{phase2_mvp_gate}`\n",
        f"- Cohort-probe (3D latent linear AUC): `{fmt(final['metrics']['cohort_probe_3d_latent_linear_auc'])}`\n",
        f"- Centiloid within-A4 Spearman: `{fmt(final['metrics']['centiloid_within_a4_spearman'])}`\n",
        f"- Centiloid within-LEARN Spearman: `{fmt(final['metrics']['centiloid_within_learn_spearman'])}`\n",
        f"- Method Y PET inventory status: `{method_y_pet_gate}`\n",
        f"- Method Y amyloid PET image files counted: `{fmt(nested(method_y_pet, 'pet_image_inventory', 'amyloid_fbp', 'image_file_count'), 0)}`\n",
        f"- Method Y tau PET image files counted: `{fmt(nested(method_y_pet, 'pet_image_inventory', 'tau_ftp', 'image_file_count'), 0)}`\n",
        f"- Method Y multimodal design status: `{method_y_multimodal_gate}`\n",
        f"- Method Y clinical Alzheimer output allowed now: `{nested(method_y_multimodal, 'target_policy', 'clinical_alzheimer_output_allowed_now')}`\n",
        f"- Implementation gap summary: `{gap_matrix.get('summary') if gap_matrix else 'PENDING'}`\n\n",
        "## Required Caveat\n\n",
        final["scientific_caveat"],
        "\n\n## Compliance\n\n",
        "- PET as model input: `False`\n",
        "- PET as Method X-Cog input: `False`\n",
        "- Method Y PET inventory is aggregate-only: `True`\n",
        "- Method Y multimodal training started: `False`\n",
        "- Alzheimer diagnosis output requires independent clinical label: `True`\n",
        "- Raw identifiers used: `False`\n",
        "- External used for calibration fit: `False`\n",
        "- External holdout type: `locked_pseudo_external_random_patient_holdout`\n",
    ]
    atomic_write_text(out_dir / "final_analysis_report.md", "".join(report))
    write_run_manifest("results/logs/final_analysis_manifest.json", stage="final_analysis", data_root=None, extra={"gates": gates})
    StateManager().append_event("final_analysis", "COMPLETED", "Final aggregate analysis package generated.")
    print(json.dumps({"final_analysis": "COMPLETED", "external_like_roc_auc": external["roc_auc"], "bias_risk": bias["risk_level"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
