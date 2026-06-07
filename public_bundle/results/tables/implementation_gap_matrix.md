# Implementation Gap Matrix

| requirement | status | artifact | note |
|---|---|---|---|
| Final execution program policy | DONE | `configs/policy/final_execution_program.json` | Locks protected Method X object, branch roles, minimum Phase-2 MVP, and optional/supplementary controls. |
| Phase 0 governance manifest | DONE | `results/governance/phase0_governance_manifest.json` | Generated before modeling; includes aggregate/hash-level governance artifact paths. |
| Label-freeze template | DONE | `results/governance/label_freeze_template.json` | Centiloid cutoff placeholder must be filled and frozen before modeling. |
| Frozen centiloid cutoff | DONE | `results/governance/label_freeze.json` | Required before modeling; cutoff must not change after result inspection. |
| Cohort-probe protocol | DONE | `results/governance/cohort_probe_protocol.json` | Linear primary plus nonlinear secondary probe and gap are pre-registered. |
| Deconfounding-scope document | DONE | `results/governance/deconfounding_scope.json` | Documents valid settings and N/A-degenerate binary A4/LEARN-only setting. |
| Branch feature allowlists | DONE | `results/governance/feature_allowlist_summary.json` | Branch-level allowlist/quarantine summary; final feature matrices still require no-PET tests. |
| Five branch-specific hash-level catalogs | DONE | `data/processed/catalogs/branch_catalog_manifest.json` | Method X-Strict, Method X-Cog/SF, MRI ROI/WMH/reads, Bio-Cog/Method Z, Clinical Endpoint. |
| SF internal decoupling governance | DONE | `governance/sf_internal_decoupling_policy.md` | SF is internal decoupling/stress-test only; never external validation. |
| Cramér's V before/after SF inclusion | DONE | `results/governance/cohort_label_coupling_audit.json` | Reports cohort-label coupling before and after SF-inclusive catalog. |
| Standalone leakage review report | DONE | `results/governance/leakage_review_report.json` | Formal Phase 0 leakage review beyond automated matrix check. |
| Phase 1 interpretable branch baselines | DONE | `results/final_results/phase1_baselines/phase1_baselines_summary.json` | Logistic regression baselines on tabular, cognitive, ROI, WMH, and MRI reads without leaky surrogates. |
| Phase 1 incremental-value table | DONE | `results/tables/phase1_incremental_value_table.json` | Descriptive incremental-value table with cohort-coupling caveat. |
| Linear + nonlinear cohort-probe | DONE | `results/tables/cohort_probe/cohort_probe_summary.json` | Phase 0.6 protocol on 3D MRI latents, tabular, and ROI/WMH/reads. |
| Continuous centiloid within-cohort regression | DONE | `results/final_results/centiloid_regression/centiloid_regression_metrics.json` | Huber/elastic net/GBT on pooled, within-A4, and within-LEARN subsets. |
| Phase 2 Minimum MVP audit | DONE | `results/tables/phase2_mvp/phase2_mvp_status.json` | Tracks Minimum Phase-2 MVP checklist per headline model. |
| Method X-Cog/SF internal stress tests | DONE | `results/final_results/method_x_cog_sf/sf_stress_tests_summary.json` | SF-only, SF-inclusive, and train-SF/test-enrolled internal decoupling arms; not external validation. |
| Bio-Cog/Method Z Phase 3 readiness audit | DONE | `results/tables/bio_cog_method_z_readiness.json` | CPU feasibility audit for blood-biomarker branch; separate from Method X claim. |
| Method X final feature-matrix no-PET test | DONE | `results/governance/method_x_feature_matrix_pet_leakage_check.json` | Tests the final feature matrix itself, not just the allowlist. |
| PET is never a model input | DONE | `final_analysis_summary.json` | PET remains label/target only. |
| Allowed tabular schema is fixed | DONE | `data/processed/master_catalog.csv` | Only aggregate column-level check; no raw rows exported. |
| Raw IDs/PII excluded from artifacts | DONE | `public_bundle/PUBLIC_BUNDLE_MANIFEST.json` | Public bundle manifest reports no raw identifiers. |
| Patient-level split locking | DONE | `splits/split_summary.json` | External holdout remains locked pseudo-external. |
| Baseline tabular and MRI-feature models | DONE | `results/final_results/` | Fast-track baselines generated. |
| Fusion/ablation variants | DONE | `results/final_results/three_d_ablation/three_d_ablation_summary.json` | Feature/surrogate fusion exists; DONE requires real 3D concat/cross-attention/alignment ablation. |
| Real 3D MRI training | PARTIAL | `results/final_results/three_d/three_d_metrics.json` | Final claim requires SPM complete, 96^3 cache, 5-fold final phase, and validity audit allowance. |
| Deep MRI preprocessing (SPM 96^3 complete) | DONE | `results/logs/spm_mri_preprocess_v2_resume_wsl.json` | 1699 active catalog SPM outputs with non-empty QC required before final 96^3 cache and 5-fold 3D training. |
| Power analysis with actual N | DONE | `results/tables/power_analysis.json` | Must use catalog/split N, class balance, CIs, and caveats. |
| Site/scanner drift sensitivity | SKIP_WITH_JUSTIFICATION | `results/tables/site_scanner_drift_sensitivity_report.json` | No site/scanner metadata columns are present in the locked master catalog. |
| Subgroup performance audit | DONE | `results/final_results/subgroup_performance_audit.json` | Required for sex, age bins, APOE4, education, and MMSE with CIs. |
| XAI / interpretability | DONE | `results/final_results/xai/xai_summary.json` | Grad-CAM MRI and SHAP tabular must be aggregate-only in public outputs. |
| Tau exploratory branch | DONE | `results/final_results/tau/tau_regression_metrics.json` | Tau regression executed with grouped CV; current result is exploratory and negative versus the grouped mean baseline. |
| Public reproducibility bundle | DONE | `public_bundle/PUBLIC_BUNDLE_MANIFEST.json` | Aggregate text only; no raw data/HDF5/checkpoints/images. |
| Thesis-grade results report | DONE | `results/final_results/thesis/thesis_grade_report.md` | Must include validity audit recommendation and SPM progress; strong claims blocked until SPM+96/5-fold complete. |
