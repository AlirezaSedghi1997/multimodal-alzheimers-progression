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
| Matched/propensity-weighted subset analysis | DONE | `results/tables/matched_propensity_subset.json` | Phase 2 post-MVP sensitivity on age/sex/APOE4/education/MMSE-matched subset. |
| Decision-curve analysis | DONE | `results/tables/decision_curve_analysis.json` | Net-benefit sensitivity for tabular Method X branch. |
| TRIPOD+AI / PROBAST+AI scaffold | DONE | `results/governance/tripod_probast_checklists.json` | Governance checklist scaffold; manuscript items remain Phase 4. |
| TRIPOD manuscript scaffold | DONE | `results/governance/tripod_manuscript_scaffold.json` | Phase 4 Methods/Results scaffold text. |
| Domain-ceiling citation list | DONE | `results/governance/domain_ceiling_citation_list.json` | Literature interpretive anchors with exact citations. |
| Headline temporal-gap model evaluation | DONE | `results/tables/headline_temporal_gap_evaluation.json` | Tabular and ROI baselines re-evaluated per gap window; 3D reference-only. |
| Method Y class imbalance policy | DONE | `governance/method_y_class_imbalance_policy.md` | Frozen exploratory tier primary_feasible_90 and weighting rules. |
| Phase 4 figure manifest | DONE | `results/governance/phase4_figure_manifest.json` | Tracks required thesis figures vs artifacts. |
| Method Y PET SPM/MNI preprocessing | DONE | `results/logs/method_y_pet_preprocess_resume_fbp_spm_pet_t1_mni.json` | SPM/MNI PET preprocess progress 266/266; training-ready gate separate. |
| Method Y paired hash list (primary_feasible_90) | DONE | `results/tables/method_y_paired_hash_list.json` | Frozen N=266 paired-subset hashes for Y1–Y6 identical intersection. |
| Method Y Y2 MRI-only paired trainer | DONE | `scripts/train_mri_method_y_paired.py` | Script ready; full training gated until PET preprocess + readiness PASS. |
| Method Y Y3 late-fusion trainer | DONE | `scripts/train_late_fusion_method_y.py` | Script ready; training gated until readiness PASS. |
| Method Y Y4 gated-fusion trainer | DONE | `scripts/train_gated_fusion_method_y.py` | Shared multimodal training module; full training gated until readiness PASS. |
| Method Y Y5 early-concat ablation trainer | DONE | `scripts/train_early_concat_method_y.py` | Ablation only; never primary claim path. |
| Method Y Y6 cross-attention trainer | DONE | `scripts/train_cross_attention_method_y.py` | Cross-attention fusion; training gated until readiness PASS. |
| Method Y Y10 label reproducibility | DONE | `results/tables/method_y_label_reproducibility.json` | Aggregate centiloid vs frozen-label discordance on paired subset. |
| Biomarker hierarchy summary (X/Y/Z) | DONE | `results/tables/biomarker_hierarchy_summary.json` | Three-pillar framing; Method Y metrics populate after training. |
| FLAIR volume index | DONE | `results/tables/flair_volume_index.json` | FLAIR files linked to master catalog: 7006. |
| FLAIR cache readiness audit | DONE | `results/tables/flair_cache_readiness.json` | Cache build prep; T1+FLAIR sensitivity track. |
| FLAIR 96^3 volume cache | DONE | `results/tables/flair_volume_cache_summary.json` | Supplementary FLAIR cache ok=1696. |
| FLAIR T1+FLAIR sensitivity subset | DONE | `results/tables/flair_sensitivity_subset.json` | Intersection of T1+FLAIR cache OK samples. |
| EfficientNet3D-B0 ablation (Method X) | DONE | `results/final_results/efficientnet3d_ablation/efficientnet3d_ablation_summary.json` | Lightweight vs EfficientNet3D controlled comparison. |
| PET-guided KD sensitivity protocol | DONE | `results/tables/pet_guided_kd_sensitivity.json` | Teacher PET+MRI, student MRI-only; supplementary framing. |
| AmyPET QC cross-check audit | DONE | `results/tables/amypet_qc_crosscheck.json` | Aggregate PET QC benchmark row; no AmyPET import. |
| T1+FLAIR sensitivity protocol | DONE | `results/tables/t1_flair_sensitivity_protocol.json` | Supplementary T1 vs T1+FLAIR under locked splits. |
| FLAIR-linked enrolled hash list | DONE | `results/tables/flair_linked_hash_list.json` | Frozen hashes for T1+FLAIR sensitivity; independent of cache build. |
| 3-month parallel execution progress table | DONE | `results/tables/parallel_execution_progress_table.json` | Read-only lane snapshot; no Tier deferral. |
| Dual-pillar framing paragraph | DONE | `results/governance/dual_pillar_framing_summary.json` | Approved Method X / Method Y manuscript framing. |
| Training & evaluation protocol (TRIPOD-aligned) | DONE | `results/governance/training_evaluation_protocol.json` | Maps optimizers, CV, overfitting controls per branch. |
| 3D MRI overfit diagnostics | DONE | `results/tables/three_d_overfit_diagnostics.json` | Train vs val loss gap from early-stopping curves. |
| Method X imaging training summary | DONE | `results/tables/method_x_imaging_training_summary.json` | Consolidated 3D, EfficientNet, T1+FLAIR tracks. |
| PET-guided KD student trainer | DONE | `scripts/run_pet_guided_kd_train.py` | MRI-only student; full KD after Y3 teacher checkpoint. |
| Method Y incremental-value table | DONE | `results/tables/method_y_incremental_value_table.json` | Scaffold; metrics populate after Y1–Y4 training. |
| Method Y DCA three-way workflow | DONE | `results/tables/method_y_dca_workflow_summary.json` | Workflow summary; execution after Y1–Y3 probability exports. |
| Method Y Y9 missing-PET simulation | DONE | `results/tables/method_y_missing_pet_simulation.json` | Protocol scaffold; dropout eval after fusion training. |
| Leaky MRI surrogate quarantine | DONE | `data/processed/quarantine/` | Retired 15-feature surrogates moved out of headline feature path. |
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
