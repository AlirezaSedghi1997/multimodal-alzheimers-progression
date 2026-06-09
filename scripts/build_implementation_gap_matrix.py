from __future__ import annotations

import json
import sys
from pathlib import Path
import csv
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.final_claim_policy import spm_is_complete, validity_recommendation
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.pet_preprocess_progress import summarize_pet_preprocess_progress
from src.utils.state_manager import StateManager


SITE_LIKE_TERMS = ("site", "scanner", "manufacturer", "field", "center", "protocol", "vendor", "magnet", "model")


def load_json(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def exists(path: str) -> bool:
    return Path(path).exists()


def master_columns() -> list[str]:
    p = Path("data/processed/master_catalog.csv")
    if not p.exists():
        return []
    header = p.open("r", encoding="utf-8").readline().strip()
    return [c.strip() for c in header.split(",")]


def active_catalog_n() -> int:
    p = Path("data/processed/master_catalog.csv")
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def has_site_scanner_metadata() -> bool:
    cols = [c.lower() for c in master_columns()]
    return any(any(term in col for term in SITE_LIKE_TERMS) for col in cols)


def add(rows: list[dict[str, str]], requirement: str, status: str, artifact: str, note: str) -> None:
    rows.append({"requirement": requirement, "status": status, "artifact": artifact, "note": note})


def nested(obj: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def build_rows() -> list[dict[str, str]]:
    final = load_json("results/final_results/final_analysis/final_analysis_summary.json") or {}
    three_d = load_json("results/final_results/three_d/three_d_metrics.json") or {}
    three_d_ablation = load_json("results/final_results/three_d_ablation/three_d_ablation_summary.json") or {}
    tau_reg = load_json("results/final_results/tau/tau_regression_metrics.json") or {}
    full_preprocess = load_json("results/tables/full_mri_preprocess_toolchain_report.json") or {}
    xai = load_json("results/final_results/xai/xai_summary.json") or {}
    spm_ready = spm_is_complete()
    active_n = active_catalog_n()
    rows: list[dict[str, str]] = []
    add(rows, "Final execution program policy", "DONE" if exists("configs/policy/final_execution_program.json") else "MISSING", "configs/policy/final_execution_program.json", "Locks protected Method X object, branch roles, minimum Phase-2 MVP, and optional/supplementary controls.")
    add(rows, "Phase 0 governance manifest", "DONE" if exists("results/governance/phase0_governance_manifest.json") else "MISSING", "results/governance/phase0_governance_manifest.json", "Generated before modeling; includes aggregate/hash-level governance artifact paths.")
    add(rows, "Label-freeze template", "DONE" if exists("results/governance/label_freeze_template.json") else "MISSING", "results/governance/label_freeze_template.json", "Centiloid cutoff placeholder must be filled and frozen before modeling.")
    freeze = load_json("results/governance/label_freeze.json") or {}
    freeze_done = freeze.get("status") == "FROZEN_BEFORE_MODELING" and freeze.get("centiloid_threshold") not in {None, "", "[INSERT PRESPECIFIED CUTOFF BEFORE MODELING]"}
    add(rows, "Frozen centiloid cutoff", "DONE" if freeze_done else "MISSING", "results/governance/label_freeze.json", "Required before modeling; cutoff must not change after result inspection.")
    add(rows, "Cohort-probe protocol", "DONE" if exists("results/governance/cohort_probe_protocol.json") else "MISSING", "results/governance/cohort_probe_protocol.json", "Linear primary plus nonlinear secondary probe and gap are pre-registered.")
    add(rows, "Deconfounding-scope document", "DONE" if exists("results/governance/deconfounding_scope.json") else "MISSING", "results/governance/deconfounding_scope.json", "Documents valid settings and N/A-degenerate binary A4/LEARN-only setting.")
    add(rows, "Branch feature allowlists", "DONE" if exists("results/governance/feature_allowlist_summary.json") else "MISSING", "results/governance/feature_allowlist_summary.json", "Branch-level allowlist/quarantine summary; final feature matrices still require no-PET tests.")
    branch_manifest = load_json("data/processed/catalogs/branch_catalog_manifest.json") or {}
    add(rows, "Five branch-specific hash-level catalogs", "DONE" if branch_manifest.get("status") == "BRANCH_CATALOGS_LOCKED" else "MISSING", "data/processed/catalogs/branch_catalog_manifest.json", "Method X-Strict, Method X-Cog/SF, MRI ROI/WMH/reads, Bio-Cog/Method Z, Clinical Endpoint.")
    add(rows, "SF internal decoupling governance", "DONE" if exists("governance/sf_internal_decoupling_policy.md") else "MISSING", "governance/sf_internal_decoupling_policy.md", "SF is internal decoupling/stress-test only; never external validation.")
    coupling = load_json("results/governance/cohort_label_coupling_audit.json") or {}
    add(rows, "Cramér's V before/after SF inclusion", "DONE" if coupling.get("status") == "COMPLETED" else "MISSING", "results/governance/cohort_label_coupling_audit.json", "Reports cohort-label coupling before and after SF-inclusive catalog.")
    leakage = load_json("results/governance/leakage_review_report.json") or {}
    add(rows, "Standalone leakage review report", "DONE" if leakage.get("status") in {"PASS", "PARTIAL"} else "MISSING", "results/governance/leakage_review_report.json", "Formal Phase 0 leakage review beyond automated matrix check.")
    phase1 = load_json("results/final_results/phase1_baselines/phase1_baselines_summary.json") or {}
    add(rows, "Phase 1 interpretable branch baselines", "DONE" if phase1.get("status") == "COMPLETED" else "MISSING", "results/final_results/phase1_baselines/phase1_baselines_summary.json", "Logistic regression baselines on tabular, cognitive, ROI, WMH, and MRI reads without leaky surrogates.")
    incr = load_json("results/tables/phase1_incremental_value_table.json") or {}
    add(rows, "Phase 1 incremental-value table", "DONE" if incr.get("status") == "COMPLETED" else "MISSING", "results/tables/phase1_incremental_value_table.json", "Descriptive incremental-value table with cohort-coupling caveat.")
    cohort_probe = load_json("results/tables/cohort_probe/cohort_probe_summary.json") or {}
    add(rows, "Linear + nonlinear cohort-probe", "DONE" if cohort_probe.get("status") == "COMPLETED" else "MISSING", "results/tables/cohort_probe/cohort_probe_summary.json", "Phase 0.6 protocol on 3D MRI latents, tabular, and ROI/WMH/reads.")
    centiloid = load_json("results/final_results/centiloid_regression/centiloid_regression_metrics.json") or {}
    add(rows, "Continuous centiloid within-cohort regression", "DONE" if centiloid.get("status") == "COMPLETED" else "MISSING", "results/final_results/centiloid_regression/centiloid_regression_metrics.json", "Huber/elastic net/GBT on pooled, within-A4, and within-LEARN subsets.")
    phase2_mvp = load_json("results/tables/phase2_mvp/phase2_mvp_status.json") or {}
    add(rows, "Phase 2 Minimum MVP audit", "DONE" if phase2_mvp.get("status") == "MVP_COMPLETE" else "PARTIAL", "results/tables/phase2_mvp/phase2_mvp_status.json", "Tracks Minimum Phase-2 MVP checklist per headline model.")
    matched = load_json("results/tables/matched_propensity_subset.json") or {}
    add(rows, "Matched/propensity-weighted subset analysis", "DONE" if matched.get("status") == "COMPLETED" else "MISSING", "results/tables/matched_propensity_subset.json", "Phase 2 post-MVP sensitivity on age/sex/APOE4/education/MMSE-matched subset.")
    dca = load_json("results/tables/decision_curve_analysis.json") or {}
    add(rows, "Decision-curve analysis", "DONE" if dca.get("status") == "COMPLETED" else "MISSING", "results/tables/decision_curve_analysis.json", "Net-benefit sensitivity for tabular Method X branch.")
    tripod = load_json("results/governance/tripod_probast_checklists.json") or {}
    add(rows, "TRIPOD+AI / PROBAST+AI scaffold", "DONE" if tripod.get("status") == "SCAFFOLD_COMPLETE" else "PARTIAL", "results/governance/tripod_probast_checklists.json", "Governance checklist scaffold; manuscript items remain Phase 4.")
    add(rows, "TRIPOD manuscript scaffold", "DONE" if exists("results/governance/tripod_manuscript_scaffold.json") else "MISSING", "results/governance/tripod_manuscript_scaffold.json", "Phase 4 Methods/Results scaffold text.")
    add(rows, "Domain-ceiling citation list", "DONE" if exists("results/governance/domain_ceiling_citation_list.json") else "MISSING", "results/governance/domain_ceiling_citation_list.json", "Literature interpretive anchors with exact citations.")
    add(rows, "Headline temporal-gap model evaluation", "DONE" if exists("results/tables/headline_temporal_gap_evaluation.json") else "MISSING", "results/tables/headline_temporal_gap_evaluation.json", "Tabular and ROI baselines re-evaluated per gap window; 3D reference-only.")
    add(rows, "Method Y class imbalance policy", "DONE" if exists("configs/policy/method_y_class_imbalance_policy.json") else "MISSING", "governance/method_y_class_imbalance_policy.md", "Frozen exploratory tier primary_feasible_90 and weighting rules.")
    add(rows, "Phase 4 figure manifest", "DONE" if exists("results/governance/phase4_figure_manifest.json") else "MISSING", "results/governance/phase4_figure_manifest.json", "Tracks required thesis figures vs artifacts.")
    method_y_prep = load_json("results/tables/method_y_pet_preprocess_summary.json") or {}
    prep_total = nested(method_y_prep, "link_summary", "linked_work_items") or method_y_prep.get("selected_work_items") or 0
    prep_progress = summarize_pet_preprocess_progress(linked_work_items=int(prep_total or 0))
    prep_done = prep_progress.get("resume_done") or method_y_prep.get("final_done") or method_y_prep.get("status_counts", {}).get("PASS", 0)
    prep_status = "DONE" if prep_total and int(prep_done) >= int(prep_total) else "PARTIAL" if int(prep_done) > 0 else "MISSING"
    add(rows, "Method Y PET SPM/MNI preprocessing", prep_status, "results/logs/method_y_pet_preprocess_resume_fbp_spm_pet_t1_mni.json", f"SPM/MNI PET preprocess progress {prep_done}/{prep_total}; training-ready gate separate.")
    add(rows, "Leaky MRI surrogate quarantine", "DONE" if exists("data/processed/quarantine/README.md") else "PARTIAL", "data/processed/quarantine/", "Retired 15-feature surrogates moved out of headline feature path.")
    sf_stress = load_json("results/final_results/method_x_cog_sf/sf_stress_tests_summary.json") or {}
    add(rows, "Method X-Cog/SF internal stress tests", "DONE" if sf_stress.get("status") == "COMPLETED" else "MISSING", "results/final_results/method_x_cog_sf/sf_stress_tests_summary.json", "SF-only, SF-inclusive, and train-SF/test-enrolled internal decoupling arms; not external validation.")
    bio_z = load_json("results/tables/bio_cog_method_z_readiness.json") or {}
    bio_status = "DONE" if bio_z.get("status") in {
        "FEASIBILITY_ONLY_NOT_TRAINED",
        "DESIGN_READY_AWAITING_PHASE3_TRAINING",
    } else "PARTIAL" if bio_z else "MISSING"
    add(rows, "Bio-Cog/Method Z Phase 3 readiness audit", bio_status, "results/tables/bio_cog_method_z_readiness.json", "CPU feasibility audit for blood-biomarker branch; separate from Method X claim.")
    matrix_check = load_json("results/governance/method_x_feature_matrix_pet_leakage_check.json") or {}
    add(rows, "Method X final feature-matrix no-PET test", "DONE" if matrix_check.get("status") == "PASS" else "MISSING", "results/governance/method_x_feature_matrix_pet_leakage_check.json", "Tests the final feature matrix itself, not just the allowlist.")
    add(rows, "PET is never a model input", "DONE" if final.get("pet_as_model_input") is False else "MISSING", "final_analysis_summary.json", "PET remains label/target only.")
    add(rows, "Allowed tabular schema is fixed", "DONE" if set(master_columns()).issuperset({"age", "sex", "education_years", "apoe4_carrier", "mmse_base"}) else "MISSING", "data/processed/master_catalog.csv", "Only aggregate column-level check; no raw rows exported.")
    add(rows, "Raw IDs/PII excluded from artifacts", "DONE" if final.get("raw_identifiers_used") is False else "MISSING", "public_bundle/PUBLIC_BUNDLE_MANIFEST.json", "Public bundle manifest reports no raw identifiers.")
    add(rows, "Patient-level split locking", "DONE" if exists("splits/split_summary.json") else "MISSING", "splits/split_summary.json", "External holdout remains locked pseudo-external.")
    add(rows, "Baseline tabular and MRI-feature models", "DONE" if exists("results/final_results/baseline/baseline_metrics.json") and exists("results/final_results/mri_feature_baseline/mri_feature_baseline_metrics.json") else "MISSING", "results/final_results/", "Fast-track baselines generated.")
    add(
        rows,
        "Fusion/ablation variants",
        "DONE" if three_d_ablation.get("status") == "COMPLETED" else "PARTIAL",
        "results/final_results/three_d_ablation/three_d_ablation_summary.json",
        "Feature/surrogate fusion exists; DONE requires real 3D concat/cross-attention/alignment ablation.",
    )
    three_d_status = "PARTIAL"
    agg = three_d.get("aggregate", {})
    if not spm_ready:
        three_d_status = "PARTIAL"
    elif agg.get("final_claim_allowed"):
        three_d_status = "DONE"
    elif agg.get("mode") == "FULL_3D_TRAINING":
        three_d_status = "PARTIAL"
    add(
        rows,
        "Real 3D MRI training",
        three_d_status,
        "results/final_results/three_d/three_d_metrics.json",
        "Final claim requires SPM complete, 96^3 cache, 5-fold final phase, and validity audit allowance.",
    )
    spm_status = "DONE" if spm_ready else "PARTIAL"
    add(
        rows,
        "Deep MRI preprocessing (SPM 96^3 complete)",
        spm_status,
        "results/logs/spm_mri_preprocess_v2_resume_wsl.json",
        f"{active_n or 'active catalog'} active catalog SPM outputs with non-empty QC required before final 96^3 cache and 5-fold 3D training.",
    )
    power = Path("results/tables/power_analysis.json")
    add(rows, "Power analysis with actual N", "DONE" if power.exists() else "MISSING", "results/tables/power_analysis.json", "Must use catalog/split N, class balance, CIs, and caveats.")
    add(rows, "Site/scanner drift sensitivity", "MISSING" if has_site_scanner_metadata() else "SKIP_WITH_JUSTIFICATION", "results/tables/site_scanner_drift_sensitivity_report.json", "No site/scanner metadata columns are present in the locked master catalog.")
    add(rows, "Subgroup performance audit", "MISSING" if not exists("results/final_results/subgroup_performance_audit.json") else "DONE", "results/final_results/subgroup_performance_audit.json", "Required for sex, age bins, APOE4, education, and MMSE with CIs.")
    add(
        rows,
        "XAI / interpretability",
        "DONE" if xai.get("status") == "COMPLETED" else "MISSING",
        "results/final_results/xai/xai_summary.json",
        "Grad-CAM MRI and SHAP tabular must be aggregate-only in public outputs.",
    )
    tau_status = "DONE" if tau_reg.get("mode") == "FULL" else "PARTIAL" if tau_reg else "MISSING"
    tau_note = (
        "Tau regression executed with grouped CV; current result is exploratory and negative versus the grouped mean baseline."
        if tau_status == "DONE"
        else "Tau feasibility is FULL but current regression artifact is skipped; rerun is required when DATA_ROOT resolves."
    )
    add(rows, "Tau exploratory branch", tau_status, "results/final_results/tau/tau_regression_metrics.json", tau_note)
    add(rows, "Public reproducibility bundle", "DONE" if (load_json("public_bundle/PUBLIC_BUNDLE_MANIFEST.json") or {}).get("violations") == [] else "MISSING", "public_bundle/PUBLIC_BUNDLE_MANIFEST.json", "Aggregate text only; no raw data/HDF5/checkpoints/images.")
    validity_rec = validity_recommendation()
    thesis_status = "PARTIAL"
    thesis_path = Path("results/final_results/thesis/thesis_grade_report.md")
    if thesis_path.exists() and validity_rec:
        thesis_status = "DONE"
    add(
        rows,
        "Thesis-grade results report",
        thesis_status,
        "results/final_results/thesis/thesis_grade_report.md",
        "Must include validity audit recommendation and SPM progress; strong claims blocked until SPM+96/5-fold complete.",
    )
    return rows


def main() -> int:
    rows = build_rows()
    summary = {status: sum(1 for row in rows if row["status"] == status) for status in ["DONE", "PARTIAL", "MISSING", "SKIP_WITH_JUSTIFICATION"]}
    out = {"schema_version": 1, "summary": summary, "rows": rows}
    atomic_write_json("results/tables/implementation_gap_matrix.json", out)
    lines = ["# Implementation Gap Matrix\n\n", "| requirement | status | artifact | note |\n", "|---|---|---|---|\n"]
    for row in rows:
        lines.append(f"| {row['requirement']} | {row['status']} | `{row['artifact']}` | {row['note']} |\n")
    atomic_write_text("results/tables/implementation_gap_matrix.md", "".join(lines))
    StateManager().append_event("gap_matrix", "COMPLETED", "Formal implementation gap matrix generated.")
    print(json.dumps({"gap_matrix": "COMPLETED", "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
