#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "JSON_READ_FAILED", "path": str(path), "error": str(exc)}


def fmt(value: object, ndigits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def nested(obj: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def probe_auc(cohort_probe: dict[str, Any], input_name: str, probe: str) -> float | None:
    for block in cohort_probe.get("blocks", []):
        if block.get("input_name") != input_name:
            continue
        if probe == "linear":
            return block.get("linear_probe_auc")
        return block.get("nonlinear_probe_auc")
    return None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    out_dir = PROJECT_ROOT / "results" / "final_results" / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)

    validity = safe_json(PROJECT_ROOT / "results/tables/validity_audit/validity_audit_summary.json")
    temporal = safe_json(PROJECT_ROOT / "results/tables/temporal_window_sensitivity.json")
    external = safe_json(PROJECT_ROOT / "results/final_results/external/external_final_metrics.json")
    cog = safe_json(PROJECT_ROOT / "results/final_results/method_x_cog/method_x_cog_summary.json")
    cog_sf = safe_json(PROJECT_ROOT / "results/final_results/method_x_cog_sf/sf_stress_tests_summary.json")
    phase2 = safe_json(PROJECT_ROOT / "results/tables/phase2_mvp/phase2_mvp_status.json")
    phase1 = safe_json(PROJECT_ROOT / "results/final_results/phase1_baselines/phase1_baselines_summary.json")
    cohort_probe = safe_json(PROJECT_ROOT / "results/tables/cohort_probe/cohort_probe_summary.json")
    centiloid = safe_json(PROJECT_ROOT / "results/final_results/centiloid_regression/centiloid_regression_metrics.json")
    three_d = safe_json(PROJECT_ROOT / "results/final_results/three_d/three_d_metrics.json")
    ablation = safe_json(PROJECT_ROOT / "results/final_results/three_d_ablation/three_d_ablation_summary.json")
    method_y = safe_json(PROJECT_ROOT / "results/tables/method_y_training_readiness.json")
    bio_z = safe_json(PROJECT_ROOT / "results/tables/bio_cog_method_z_readiness.json")
    coupling = safe_json(PROJECT_ROOT / "results/governance/cohort_label_coupling_audit.json")

    lines = [
        "# Thesis-grade Method X Report — execution program v3.9\n\n",
        "## Required caveats\n\n",
        "1. The output is **Aβ-positive vs Aβ-negative amyloid biomarker status**, not Alzheimer disease diagnosis.\n",
        "2. PET is label/target only in Method X and Method X-Cog; PET image/SUVR/centiloid are not model inputs.\n",
        "3. Current external-like evaluation is a locked pseudo-external holdout, not a truly independent cohort.\n",
        "4. A4/LEARN cohort-label coupling must be discussed wherever performance is reported.\n",
        "5. SF is an internal decoupling/stress-test instrument only — never external validation.\n",
        "6. High ROI/WMH nonlinear cohort-probe AUC reflects cohort artifact and must be reported as a limitation.\n\n",
        "## Phase 2 Minimum MVP\n\n",
        f"- overall status: `{phase2.get('status', 'MISSING')}`\n",
        f"- method_x_strict_tabular: `{nested(phase2, 'models', 'method_x_strict_tabular', 'completion', 'mvp_complete')}`\n",
        f"- method_x_strict_3d_cnn: `{nested(phase2, 'models', 'method_x_strict_3d_cnn', 'completion', 'mvp_complete')}`\n",
        f"- mri_roi_wmh_reads_branch: `{nested(phase2, 'models', 'mri_roi_wmh_reads_branch', 'completion', 'mvp_complete')}`\n",
        f"- method_x_cog_sf_stress_tests: `{nested(phase2, 'method_x_cog_sf_stress_tests', 'status')}`\n\n",
        "## Validity audit summary\n\n",
        f"- status: `{nested(validity, 'verdict', 'final_claim_recommendation') or validity.get('status')}`\n\n",
        "## Cohort-probe (headline validity metric)\n\n",
        f"- 3D MRI latent linear AUC: `{fmt(probe_auc(cohort_probe, 'method_x_3d_mri_latent_oof', 'linear'))}`\n",
        f"- 3D MRI latent nonlinear AUC: `{fmt(probe_auc(cohort_probe, 'method_x_3d_mri_latent_oof', 'nonlinear'))}`\n",
        f"- tabular linear AUC: `{fmt(probe_auc(cohort_probe, 'method_x_strict_tabular', 'linear'))}`\n",
        f"- ROI/WMH/reads nonlinear AUC: `{fmt(probe_auc(cohort_probe, 'mri_roi_wmh_reads', 'nonlinear'))}`\n",
        f"- reading: 3D latent does not encode cohort; ROI branch shows strong cohort artifact.\n\n",
        "## Continuous centiloid (target only)\n\n",
        f"- pooled Spearman r: `{fmt(nested(centiloid, 'subsets', 'pooled', 'spearman_r'))}`\n",
        f"- within-A4 Spearman r: `{fmt(nested(centiloid, 'subsets', 'within_a4', 'spearman_r'))}`\n",
        f"- within-LEARN Spearman r: `{fmt(nested(centiloid, 'subsets', 'within_learn', 'spearman_r'))}`\n",
        f"- centiloid_as_predictor: `{centiloid.get('centiloid_as_predictor')}`\n\n",
        "## Phase 1 interpretable baselines\n\n",
        f"- status: `{phase1.get('status')}`\n",
        f"- fixed tabular ROC-AUC: `{fmt(nested(phase1, 'results', 'fixed_tabular', 'roc_auc'))}`\n",
        f"- tabular+cognitive ROC-AUC: `{fmt(nested(phase1, 'results', 'fixed_tabular_plus_cognitive', 'roc_auc'))}`\n",
        f"- ROI/WMH/reads ROC-AUC: `{fmt(nested(phase1, 'results', 'roi_wmh_reads', 'roc_auc'))}`\n\n",
        "## Method X-Strict 3D CNN\n\n",
        f"- ROC-AUC (5-fold CV): `{fmt(nested(three_d, 'aggregate', 'roc_auc'))}`\n",
        f"- external-like ROC-AUC: `{fmt(nested(three_d, 'aggregate', 'external_like', 'roc_auc'))}`\n",
        f"- final_claim_allowed: `{nested(three_d, 'aggregate', 'final_claim_allowed')}`\n",
        f"- ablation status: `{ablation.get('status')}`\n",
        f"- ablation best variant: `{nested(ablation, 'best_by_cv_roc_auc', 'variant')}`\n",
        f"- ablation best ROC-AUC: `{fmt(nested(ablation, 'best_by_cv_roc_auc', 'roc_auc'))}`\n",
        f"- ablation dev-mode variants pending refresh: `{nested(ablation, 'training_contract', 'variants_with_dev_mode_artifact')}`\n\n",
        "## Method X-Cog (exploratory; does not replace Method X-Strict)\n\n",
        f"- status: `{cog.get('status')}`\n",
        f"- best ROC-AUC: `{fmt(nested(cog, 'best', 'roc_auc'))}`\n",
        f"- best non-MRI ROC-AUC: `{fmt(nested(cog, 'best_non_mri', 'roc_auc'))}`\n",
        f"- AUC gain over fixed tabular: `{fmt(cog.get('improvement_over_fixed_tabular_auc'))}`\n\n",
        "## Method X-Cog/SF internal stress tests\n\n",
        f"- status: `{cog_sf.get('status')}`\n",
        f"- pooled Cramér's V: `{fmt(nested(cog_sf, 'cohort_label_coupling', 'pooled_cramers_v'))}`\n",
        f"- SF-inclusive cognitive ROC-AUC: `{fmt(nested(cog_sf, 'scientific_interpretation', 'pooled_sf_inclusive_caution', 'cognitive_roc_auc'))}`\n",
        f"- train-SF→enrolled cognitive ROC-AUC: `{fmt(nested(cog_sf, 'scientific_interpretation', 'internal_decoupling_transfer', 'train_sf_cognitive_roc_auc'))}`\n",
        f"- disclaimer: SF is internal decoupling only, not external validation.\n\n",
        "## Cohort-label coupling audit\n\n",
        f"- before SF Cramér's V: `{fmt(nested(coupling, 'before_sf', 'cohort_label_cramers_v'))}`\n",
        f"- after SF Cramér's V: `{fmt(nested(coupling, 'after_sf', 'cohort_label_cramers_v'))}`\n\n",
        "## Temporal window sensitivity\n\n",
        f"- status: `{temporal.get('version', temporal.get('status'))}`\n",
        f"- main_catalog_overwritten: `{temporal.get('main_catalog_overwritten')}`\n\n",
        "## External-like evaluation\n\n",
        f"- external_holdout_type: `{external.get('external_holdout_type')}`\n",
        f"- is_truly_independent_external: `{external.get('is_truly_independent_external')}`\n",
        f"- DeLong status: `{nested(external, 'delong', 'status')}`\n\n",
        "## Method Y PET (exploratory branch)\n\n",
        f"- training readiness: `{method_y.get('status')}`\n",
        f"- method_x_claim_unchanged: `{method_y.get('method_x_claim_unchanged')}`\n",
        f"- blockers: `{method_y.get('blockers')}`\n\n",
        "## Bio-Cog / Method Z (Phase 3)\n\n",
        f"- readiness: `{bio_z.get('status', 'PENDING_AUDIT')}`\n",
        f"- rows: `{bio_z.get('rows')}`\n",
        f"- blockers: `{bio_z.get('blockers')}`\n\n",
        "## Required wording\n\n",
        "The current A4/LEARN enrolled catalog exhibits strong cohort-label coupling (Cramér's V≈1 within enrolled-only analyses). ",
        "Therefore, model performance cannot be interpreted as pathology-only generalization. ",
        "All external-like results are reported as locked pseudo-external holdout results, not as validation on a truly independent cohort. ",
        "Method X-Strict remains the protected thesis anchor; Method X-Cog/SF and blood/PET branches are extensions with separate claim boundaries.\n",
    ]

    atomic_write_text(out_dir / "thesis_grade_report.md", "".join(lines))
    print(f"Wrote {out_dir / 'thesis_grade_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
