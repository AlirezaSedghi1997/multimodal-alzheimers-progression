# TRIPOD+AI / PROBAST+AI Checklists

- TRIPOD+AI done: `12/14`
- PROBAST+AI done: `10/10`
- Manuscript-only pending: `1`

## TRIPOD+AI items

- [DRAFT_IN_METHODS] Title identifies prediction model development/validation — `results/governance/tripod_methods_scaffold.md`
- [PENDING_MANUSCRIPT] Abstract includes biomarker-status wording (not AD diagnosis)
- [DONE] Background states cohort-label coupling limitation — `results/governance/cohort_label_coupling_audit.json`
- [DONE] Methods describe locked pseudo-external evaluation — `splits/split_summary.json`
- [DONE] Methods describe patient-level grouped CV — `results/tables/phase2_mvp/phase2_mvp_status.json`
- [DONE] Calibration reported (Brier/ECE)
- [DONE] Cohort-probe validity metric reported — `results/tables/cohort_probe/cohort_probe_summary.json`
- [DONE] Incremental value ranked with validity flags
- [DONE] Temporal-gap sensitivity with bootstrap CI
- [DONE] Matched subset and DCA sensitivity
- [DONE] Claim policy documented — `docs/final_execution_program.md`
- [DONE] Domain-ceiling citations frozen — `results/governance/domain_ceiling_citation_list.json`
- [DONE] Thesis-grade aggregate report
- [DONE] Aggregate-only reproducibility bundle — `public_bundle/PUBLIC_BUNDLE_MANIFEST.json`

## PROBAST+AI items

- [DONE] Participants: hash-level catalog with inclusion flow — `data/processed/catalogs/branch_catalog_manifest.json`
- [DONE] Predictors: Method X allowlist enforced — `results/governance/method_x_feature_matrix_pet_leakage_check.json`
- [DONE] Outcome: frozen centiloid threshold label — `results/governance/label_freeze.json`
- [DONE] Analysis: label-shuffle and permutation controls — `results/tables/validity_audit/validity_audit_summary.json`
- [DONE] Analysis: cohort-probe for representation validity — `results/tables/cohort_probe/cohort_probe_summary.json`
- [DONE] Risk of bias: cohort coupling documented — `results/tables/cross_cohort_bias_report.json`
- [DONE] Risk of bias: ROI cohort-artifact caveat — `results/tables/phase1_incremental_value_table.json`
- [DONE] Applicability: pseudo-external limitation stated — `results/final_results/thesis/thesis_grade_report.md`
- [DONE] Method Y/PET separate claim boundary — `results/tables/method_y_training_readiness.json`
- [DONE] No subject-level predictions in public JSON — `tests/test_public_bundle_outputs.py`
