#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.pet_preprocess_progress import summarize_pet_preprocess_progress
from src.utils.report_sources import (
    best_centiloid_spearman,
    ci_overlap,
    fmt,
    fmt_ci,
    load_json,
    nested,
    pick_nested,
    validate_required_keys,
)


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


def method_y_section(
    method_y: dict[str, Any],
    manifest: dict[str, Any],
    incremental: dict[str, Any],
    synthetic: dict[str, Any],
    augmentation: dict[str, Any],
) -> list[str]:
    linked = int(
        nested(manifest, "link_summary", "linked_work_items")
        or manifest.get("linked_work_items")
        or 266
    )
    progress = summarize_pet_preprocess_progress(linked_work_items=linked, project_root=PROJECT_ROOT)
    design_blockers = list(method_y.get("blockers") or [])
    execution_blockers = list(method_y.get("execution_blockers") or [])
    warnings = list(method_y.get("warnings") or [])
    status = method_y.get("status", "MISSING")
    pet_ref = incremental.get("pet_only_reference") or {}
    rows = incremental.get("rows") or []
    lines = [
        "- branch status: `PET_ASSISTED_EXPLORATORY_ONLY`\n",
        f"- design/training readiness status: `{status}`\n",
        f"- approved tier: `{method_y.get('approved_training_tier', 'primary_feasible_90')}`\n",
        f"- design blockers: `{design_blockers}`\n",
        f"- execution blockers (preprocess/cache): `{execution_blockers}`\n",
        f"- warnings: `{warnings}`\n",
        f"- PET preprocess progress: `{progress.get('resume_done', 0)}/{linked}` PASS\n",
        f"- regex_fallback_used (manifest): `{manifest.get('regex_fallback_used')}`\n",
        f"- project_inventory_used: `{manifest.get('project_inventory_used')}`\n",
        f"- matched effective N for Method Y deltas: `{incremental.get('matched_effective_n', 'NA')}`\n",
        f"- PET-only reference ROC-AUC: `{fmt(pet_ref.get('roc_auc'))}`\n",
        "- PET-label tautology caveat: PET-containing models predict a PET-derived amyloid label using PET inputs, so they cannot support the PET-free thesis claim.\n",
        "- Method X input status: PET and synthetic PET remain forbidden; Method Y rows are reported separately.\n",
        "- clinical Alzheimer diagnosis allowed: `False`; report amyloid biomarker status and preclinical risk only.\n",
        "- threshold metrics use fold-internal validation-selected thresholds only.\n",
        "- current Method Y rows:\n",
    ]
    for row in rows:
        threshold = row.get("threshold_metrics") or {}
        lines.append(
            f"  - `{row.get('comparison_id')}`: AUC `{fmt(row.get('roc_auc'))}`, "
            f"delta vs PET `{fmt(row.get('delta_auc_vs_pet_only'))}`, "
            f"accuracy `{fmt(threshold.get('accuracy'))}`, sensitivity `{fmt(threshold.get('sensitivity'))}`, "
            f"specificity `{fmt(threshold.get('specificity'))}`, F1 `{fmt(threshold.get('f1'))}`\n"
        )
    lines.extend(
        [
            f"- synthetic MRI-to-PET status: `{synthetic.get('status', 'MISSING')}`; fit scope `{synthetic.get('fit_generator_scope', 'NA')}`; counted as real subjects `{synthetic.get('synthetic_counted_as_real_subjects', 'NA')}`\n",
            f"- train-only augmentation exposures: `{augmentation.get('augmented_training_exposures', 'NA')}` from real train N `{augmentation.get('real_train_n', 'NA')}`; counted as real subjects `{augmentation.get('synthetic_or_augmented_counted_as_real_subjects', 'NA')}`\n",
        ]
    )
    return lines


def main() -> int:
    out_dir = PROJECT_ROOT / "results" / "final_results" / "thesis"
    out_dir.mkdir(parents=True, exist_ok=True)

    validity = load_json(PROJECT_ROOT / "results/tables/validity_audit/validity_audit_summary.json")
    temporal = load_json(PROJECT_ROOT / "results/tables/headline_temporal_gap_evaluation.json")
    external = load_json(PROJECT_ROOT / "results/final_results/external/external_final_metrics.json")
    cog = load_json(PROJECT_ROOT / "results/final_results/method_x_cog/method_x_cog_summary.json")
    cog_sf = load_json(PROJECT_ROOT / "results/final_results/method_x_cog_sf/sf_stress_tests_summary.json")
    phase2 = load_json(PROJECT_ROOT / "results/tables/phase2_mvp/phase2_mvp_status.json")
    phase1 = load_json(PROJECT_ROOT / "results/final_results/phase1_baselines/phase1_baselines_summary.json")
    incr = load_json(PROJECT_ROOT / "results/tables/phase1_incremental_value_table.json")
    cohort_probe = load_json(PROJECT_ROOT / "results/tables/cohort_probe/cohort_probe_summary.json")
    centiloid = load_json(PROJECT_ROOT / "results/final_results/centiloid_regression/centiloid_regression_metrics.json")
    three_d = load_json(PROJECT_ROOT / "results/final_results/three_d/three_d_metrics.json")
    ablation = load_json(PROJECT_ROOT / "results/final_results/three_d_ablation/three_d_ablation_summary.json")
    method_y = load_json(PROJECT_ROOT / "results/tables/method_y_training_readiness.json")
    manifest = load_json(PROJECT_ROOT / "results/tables/method_y_pet_manifest_summary.json")
    method_y_incremental = load_json(PROJECT_ROOT / "results/tables/method_y_incremental_value_table.json")
    synthetic_policy = load_json(PROJECT_ROOT / "results/tables/synthetic_mri_pet_supplement.json")
    augmentation_policy = load_json(PROJECT_ROOT / "results/tables/augmentation_policy.json")
    bio_z = load_json(PROJECT_ROOT / "results/final_results/bio_cog_method_z/bio_cog_method_z_summary.json")
    bio_ablation = load_json(PROJECT_ROOT / "results/final_results/bio_cog_method_z/bio_cog_blood_ablation.json")
    coupling = load_json(PROJECT_ROOT / "results/governance/cohort_label_coupling_audit.json")
    matched = load_json(PROJECT_ROOT / "results/tables/matched_propensity_subset.json")
    dca = load_json(PROJECT_ROOT / "results/tables/decision_curve_analysis.json")
    clinical = load_json(PROJECT_ROOT / "results/final_results/clinical_endpoint/clinical_endpoint_baseline_summary.json")

    source_warnings: list[str] = []
    source_warnings.extend(
        validate_required_keys(
            centiloid,
            [("subsets", "pooled_enrolled"), ("subsets", "within_a4"), ("subsets", "within_learn")],
            context="centiloid_regression_metrics",
        )
    )
    source_warnings.extend(
        validate_required_keys(
            coupling,
            [("before_sf_inclusion", "cohort_label_cramers_v"), ("after_sf_inclusion", "cohort_label_cramers_v")],
            context="cohort_label_coupling_audit",
        )
    )

    roi_nonlinear = probe_auc(cohort_probe, "mri_roi_wmh_reads", "nonlinear")
    tabular_auc = nested(phase1, "results", "fixed_tabular", "roc_auc")
    roi_combo_auc = nested(phase1, "results", "fixed_tabular_plus_roi_wmh_reads", "roc_auc")
    three_d_auc = nested(three_d, "aggregate", "roc_auc")

    matched_overlap = ci_overlap(
        nested(matched, "full_cohort", "roc_auc_ci"),
        nested(matched, "matched_subset", "roc_auc_ci"),
    )

    lines = [
        "# Thesis-grade Method X Report - Early Prediction of Amyloid Biomarker Status and Preclinical Risk\n\n",
        "## Required caveats\n\n",
        "1. Output is **amyloid biomarker status and preclinical risk**, not Alzheimer disease diagnosis.\n",
        "2. PET is label/target only; never a Method X predictor.\n",
        "3. Evaluation is locked **pseudo-external**, not independent external validation.\n",
        "4. A4/LEARN cohort-label coupling (Cramér's V≈1.0 enrolled-only) limits pooled binary interpretation.\n",
        "5. SF is internal decoupling only — never external validation.\n",
        "6. ROI/WMH nonlinear cohort-probe ≈0.99 → high pooled AUC on ROI+tabular is **descriptive**, not biological proof.\n",
        "7. WMH/reads alone ≈0.50 AUC; signal in MRI branch is driven by ROI volumes, not WMH/reads.\n",
        "8. 3D CNN AUC (~0.72) below tabular/ROI descriptive AUC reflects architecture/cohort coupling; `final_claim_allowed=false`.\n\n",
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
        f"- ROI/WMH/reads nonlinear AUC: `{fmt(roi_nonlinear)}`\n",
        "- reading: 3D latent ≈0.50 (cohort not encoded); ROI branch ≈0.99 (cohort artifact).\n\n",
        "## Incremental value vs validity (critical)\n\n",
        f"- highest **descriptive** pooled ROC-AUC step: `{incr.get('best_descriptive_roc_auc_step', 'fixed_tabular_plus_roi_wmh_reads')}` "
        f"({fmt(roi_combo_auc)})\n",
        f"- validity-safe headline anchor: `{incr.get('validity_safe_headline_step', 'fixed_tabular')}` "
        f"(tabular {fmt(tabular_auc)}; 3D {fmt(three_d_auc)} with cohort-probe ≈0.50)\n",
        f"- ROI nonlinear cohort-probe: `{fmt(roi_nonlinear)}` — do **not** rank ROI+tabular as best biological model.\n\n",
        "## Continuous centiloid (target only; tabular predictors)\n\n",
        f"- pooled enrolled best Spearman r: `{fmt(best_centiloid_spearman(nested(centiloid, 'subsets', 'pooled_enrolled')))}`\n",
        f"- within-A4 best Spearman r: `{fmt(best_centiloid_spearman(nested(centiloid, 'subsets', 'within_a4')))}`\n",
        f"- within-LEARN best Spearman r: `{fmt(best_centiloid_spearman(nested(centiloid, 'subsets', 'within_learn')))}`\n",
        f"- centiloid_as_predictor: `{centiloid.get('centiloid_as_predictor')}`\n",
        "- interpretation: pooled descriptive; within-LEARN signal weak (~0.08).\n\n",
        "## Phase 1 interpretable baselines\n\n",
        f"- status: `{phase1.get('status')}`\n",
        f"- fixed tabular ROC-AUC: `{fmt(tabular_auc)}`\n",
        f"- tabular+cognitive ROC-AUC: `{fmt(nested(phase1, 'results', 'fixed_tabular_plus_cognitive', 'roc_auc'))}`\n",
        f"- ROI volumes only ROC-AUC: `{fmt(nested(phase1, 'results', 'roi_icv_norm_only', 'roc_auc'))}`\n",
        f"- WMH only ROC-AUC: `{fmt(nested(phase1, 'results', 'wmh_only', 'roc_auc'))}` (single official ICV-normalized WMH column)\n",
        f"- MRI reads only ROC-AUC: `{fmt(nested(phase1, 'results', 'mri_reads_only', 'roc_auc'))}`\n",
        f"- tabular+ROI/WMH/reads ROC-AUC: `{fmt(roi_combo_auc)}` (descriptive; cohort-probe caveat applies)\n\n",
        "## Method X-Strict 3D CNN\n\n",
        f"- ROC-AUC (5-fold CV): `{fmt(three_d_auc)}`\n",
        f"- external-like ROC-AUC: `{fmt(nested(three_d, 'aggregate', 'external_like', 'roc_auc'))}`\n",
        f"- final_claim_allowed: `{nested(three_d, 'aggregate', 'final_claim_allowed')}`\n",
        f"- ablation best variant: `{nested(ablation, 'best_by_cv_roc_auc', 'variant')}` "
        f"({fmt(nested(ablation, 'best_by_cv_roc_auc', 'roc_auc'))})\n",
        "- note: cross-attention > concat but all variants <0.73; lightweight encoder; EfficientNet3D switch deferred.\n\n",
        "## Method X-Cog (exploratory)\n\n",
        f"- status: `{cog.get('status')}`\n",
        f"- best ROC-AUC: `{fmt(nested(cog, 'best', 'roc_auc'))}`\n",
        f"- train-SF→enrolled collapse: see SF stress tests below.\n\n",
        "## Method X-Cog/SF internal stress tests\n\n",
        f"- status: `{cog_sf.get('status')}`\n",
        f"- pooled Cramér's V: `{fmt(pick_nested(cog_sf, ('cohort_label_coupling', 'pooled_cramers_v')))}`\n",
        f"- SF-inclusive cognitive ROC-AUC: `{fmt(nested(cog_sf, 'scientific_interpretation', 'pooled_sf_inclusive_caution', 'cognitive_roc_auc'))}`\n",
        f"- train-SF→enrolled cognitive ROC-AUC: `{fmt(nested(cog_sf, 'scientific_interpretation', 'internal_decoupling_transfer', 'train_sf_cognitive_roc_auc'))}`\n\n",
        "## Cohort-label coupling audit\n\n",
        f"- before SF Cramér's V: `{fmt(pick_nested(coupling, ('before_sf_inclusion', 'cohort_label_cramers_v'), ('before_sf', 'cohort_label_cramers_v')))}`\n",
        f"- after SF Cramér's V: `{fmt(pick_nested(coupling, ('after_sf_inclusion', 'cohort_label_cramers_v'), ('after_sf', 'cohort_label_cramers_v')))}`\n\n",
        "## Temporal-gap sensitivity (headline CPU models)\n\n",
        f"- status: `{temporal.get('status', 'MISSING')}`\n",
    ]
    for window in temporal.get("windows") or []:
        w = window.get("window_days")
        tab = nested(window, "models", "fixed_tabular") or {}
        if tab.get("status") != "OK":
            continue
        lines.append(
            f"- window≤{w}d n={window.get('n_catalog')}: tabular AUC "
            f"`{fmt_ci(tab.get('roc_auc'), tab.get('roc_auc_ci'))}`\n"
        )
    lines.append("- 3D CNN: full-cohort reference only per window (not retrained).\n\n")

    lines.extend(
        [
            "## Matched covariate subset (sensitivity)\n\n",
            f"- full cohort AUC: `{fmt_ci(nested(matched, 'full_cohort', 'roc_auc'), nested(matched, 'full_cohort', 'roc_auc_ci'))}`\n",
            f"- matched 1:1 AUC (n_pairs={matched.get('n_pairs')}): "
            f"`{fmt_ci(nested(matched, 'matched_subset', 'roc_auc'), nested(matched, 'matched_subset', 'roc_auc_ci'))}`\n",
            f"- 95% CI overlap full vs matched: `{matched_overlap}` — difference ~0.007 is not significant if CIs overlap.\n\n",
            "## Decision curve analysis\n\n",
            f"- status: `{dca.get('status')}`\n",
        ]
    )
    tab_dca = nested(dca, "curves", "method_x_strict_tabular") or {}
    if tab_dca.get("thresholds"):
        lines.append(f"- tabular thresholds: `{tab_dca.get('thresholds')}`\n")
        lines.append(f"- net benefit (model): `{[round(x, 3) for x in tab_dca.get('net_benefit_model', [])]}`\n\n")
    else:
        lines.append("- curves: see `results/tables/decision_curve_analysis.json`\n\n")

    lines.extend(
        [
            "## Method Y PET (exploratory branch)\n\n",
            *method_y_section(method_y, manifest, method_y_incremental, synthetic_policy, augmentation_policy),
            "\n",
        ]
    )
    lines.extend(
        [
            "## Bio-Cog / Method Z (Phase 3; separate branch)\n\n",
            f"- status: `{bio_z.get('status', 'MISSING')}`\n",
        ]
    )
    enrolled = nested(bio_z, "arms", "enrolled_a4_learn", "variants") or {}
    core = enrolled.get("core_blood_only") or {}
    full = enrolled.get("full_blood_panel") or {}
    lines.append(
        f"- core_blood_only AUC: `{fmt_ci(core.get('roc_auc'), core.get('roc_auc_ci'))}`\n"
    )
    lines.append(
        f"- full_blood_panel AUC: `{fmt_ci(full.get('roc_auc'), full.get('roc_auc_ci'))}`\n"
    )
    if bio_ablation.get("drop_one_ablation"):
        full_auc = float(nested(bio_ablation, "full_panel", "roc_auc") or 0)
        top_drop = max(
            bio_ablation["drop_one_ablation"],
            key=lambda x: full_auc - float(x.get("roc_auc") or 0),
        )
        lines.append(
            f"- largest drop-one impact: drop `{top_drop.get('dropped_feature')}` "
            f"→ AUC `{fmt(top_drop.get('roc_auc'))}` (Δ={fmt(full_auc - float(top_drop.get('roc_auc') or 0))})\n"
        )
    lines.append("- separate minimally invasive benchmark; does not support Method X non-invasive claim.\n\n")

    cdr = nested(clinical, "endpoints", "cdr_worsening_event") or {}
    pacc = nested(clinical, "endpoints", "pacc_decline_event") or {}
    lines.extend(
        [
            "## Clinical endpoint (exploratory; serious caveat)\n\n",
            f"- CDR worsening AUC: `{fmt_ci(cdr.get('roc_auc'), cdr.get('roc_auc_ci'))}`\n",
            f"- PACC decline AUC: `{fmt_ci(pacc.get('roc_auc'), pacc.get('roc_auc_ci'))}`\n",
            "- AUC ~0.63–0.66 is modest; progression proxies are not AD diagnosis; confounded by substudy.\n\n",
            "## External-like evaluation\n\n",
            f"- external_holdout_type: `{external.get('external_holdout_type')}`\n",
            f"- is_truly_independent_external: `{external.get('is_truly_independent_external')}`\n\n",
        ]
    )
    if source_warnings:
        lines.append("## Report builder source validation\n\n")
        for warn in source_warnings:
            lines.append(f"- WARNING: {warn}\n")
        lines.append("\n")

    lines.append(
        "## Required wording\n\n"
        "Enrolled A4/LEARN catalog exhibits cohort-label coupling. Pooled binary AUC is descriptive only. "
        "Method X-Strict remains the protected PET-free thesis anchor; extensions have separate claim boundaries. "
        "The title and abstract framing should be early prediction of amyloid biomarker status and preclinical risk, "
        "not clinical Alzheimer diagnosis.\n"
    )

    atomic_write_text(out_dir / "thesis_grade_report.md", "".join(lines))
    print(f"Wrote {out_dir / 'thesis_grade_report.md'}")
    if source_warnings:
        print(json.dumps({"source_warnings": source_warnings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
