SHELL := /bin/bash
ENV_NAME ?= amyloid-tau
PY := conda run -n $(ENV_NAME) python
DATA_ROOT ?= /mnt/h/A4\ Learn
MNI_TEMPLATE ?=
SPM_MANIFEST ?= results/logs/spm_mri_preprocess_v2_resume_wsl.json
SPM_BATCH_SIZE ?= 10
export WANDB_MODE := offline

.PHONY: test preflight inventory catalog preprocess preprocess-toolchain-check preprocess-spm-toolchain-check preprocess-full-mri preprocess-full-mri-spm preprocess-full-mri-spm-auto spm-completion-guard mri-volume-cache mri-volume-cache-96 mri-volume-cache-smoke mri-features mri-feature-baseline train-3d three-d-ablation xai-summary splits baseline cross-cohort-bias optuna fusion external-eval calibration final-analysis public-bundle tau external-drive-check compute-plan phase0-governance label-freeze method-x-feature-matrix method-x-feature-check phase0-readiness power-analysis site-drift subgroup-audit gap-matrix method-y-pet-inventory method-y-multimodal-design clinical-ad-feasibility run-stage run-next-stage run-all-stages

test:
	$(PY) -m pytest tests/ -q

preflight:
	$(PY) scripts/preflight_check.py --data_root "$(DATA_ROOT)"

external-drive-check:
	$(PY) scripts/external_drive_check.py --data_root "$(DATA_ROOT)"

compute-plan:
	$(PY) scripts/compute_plan.py

phase0-governance:
	$(PY) scripts/write_phase0_governance_artifacts.py

branch-catalogs:
	$(PY) scripts/build_branch_catalogs.py --data_root "$(DATA_ROOT)"

cohort-label-coupling-audit:
	$(PY) scripts/run_cohort_label_coupling_audit.py

leakage-review-report:
	$(PY) scripts/write_leakage_review_report.py

phase0-complete: phase0-governance branch-catalogs cohort-label-coupling-audit leakage-review-report method-x-feature-matrix method-x-feature-check

gpu-preflight:
	$(PY) scripts/gpu_preflight_check.py

phase1-baselines:
	$(PY) scripts/run_branch_baselines.py

phase1-incremental-value:
	$(PY) scripts/build_incremental_value_table.py

extract-3d-embeddings-oof:
	$(PY) scripts/gpu_preflight_check.py
	$(PY) scripts/extract_3d_mri_embeddings_oof.py --cache data/cached/t1_volume_cache_96.h5

cohort-probe:
	$(PY) scripts/run_cohort_probe.py

centiloid-regression:
	$(PY) scripts/run_centiloid_regression.py --data_root "$(DATA_ROOT)"

phase2-mvp-audit:
	$(PY) scripts/run_phase2_mvp_audit.py

method-x-cog-sf-cognitive-features:
	$(PY) scripts/build_cognitive_features.py --catalog data/processed/catalogs/method_x_cog_sf_catalog.csv --out data/processed/cognitive_features_method_x_cog_sf.csv --summary results/tables/cognitive_feature_summary_method_x_cog_sf.json --data_root "$(DATA_ROOT)"

method-x-cog-sf-stress-tests: method-x-cog-sf-cognitive-features
	$(PY) scripts/run_method_x_cog_sf_stress_tests.py

bio-cog-method-z-readiness:
	$(PY) scripts/audit_bio_cog_method_z_readiness.py

bio-cog-blood-column-mapping-audit:
	$(PY) scripts/audit_bio_cog_blood_column_mapping.py

bio-cog-method-z-train:
	$(PY) scripts/train_bio_cog_method_z.py

clinical-endpoint-readiness:
	$(PY) scripts/audit_clinical_endpoint_readiness.py

clinical-endpoint-governance:
	$(PY) scripts/write_clinical_endpoint_governance.py

clinical-endpoint-freeze:
	$(PY) scripts/freeze_clinical_endpoint_governance.py

clinical-endpoint-labels:
	$(PY) scripts/build_clinical_endpoint_labels.py --data_root "$(DATA_ROOT)"

clinical-endpoint-baseline:
	$(PY) scripts/train_clinical_endpoint_baseline.py

bio-cog-blood-ablation:
	$(PY) scripts/run_bio_cog_blood_ablation.py

method-y-status-summary:
	$(PY) scripts/build_method_y_status_summary.py

thesis-grade-report:
	$(PY) scripts/build_thesis_grade_report.py

three-d-ablation:
	$(PY) scripts/gpu_preflight_check.py
	$(PY) scripts/run_3d_ablation.py --cache data/cached/t1_volume_cache_96.h5 --folds 5 --epochs 4 --batch_size 1

label-freeze:
	@test -n "$(CENTILOID_CUTOFF)" || (echo "Set CENTILOID_CUTOFF before freezing the label." && exit 2)
	@test -n "$(LABEL_RATIONALE)" || (echo "Set LABEL_RATIONALE before freezing the label." && exit 2)
	@test -n "$(APPROVED_BY)" || (echo "Set APPROVED_BY before freezing the label." && exit 2)
	$(PY) scripts/freeze_label_definition.py --cutoff "$(CENTILOID_CUTOFF)" --source_rationale "$(LABEL_RATIONALE)" --approved_by "$(APPROVED_BY)"


method-x-feature-matrix:
	$(PY) scripts/build_method_x_strict_feature_matrix.py

method-x-feature-check:
	@test -n "$(FEATURE_MATRIX)" || (echo "Set FEATURE_MATRIX to the final Method X feature matrix CSV." && exit 2)
	$(PY) scripts/check_method_x_feature_matrix.py --matrix "$(FEATURE_MATRIX)" --strict_allowed_features

phase0-readiness:
	$(PY) scripts/phase0_readiness_guard.py --require_label_freeze --require_feature_matrix_check

inventory:
	$(PY) scripts/run_inventory.py --data_root "$(DATA_ROOT)"

catalog:
	$(PY) scripts/build_master_catalog.py --data_root "$(DATA_ROOT)"

preprocess:
	$(PY) scripts/preprocess_mri.py --data_root "$(DATA_ROOT)"
	$(PY) -m pytest tests/test_no_pet_in_hdf5_cache.py tests/test_mri_preprocess_outputs.py -q

preprocess-toolchain-check:
	$(PY) scripts/check_preprocess_toolchain.py --template "$(METHOD_X_MNI_TEMPLATE)" --diagnostic_only

preprocess-spm-toolchain-check:
	$(PY) scripts/check_spm_toolchain.py --matlab "$(MATLAB_BIN)" --spm_dir "$(SPM_DIR)" --diagnostic_only

preprocess-full-mri:
	$(PY) scripts/preprocess_mri_full.py --data_root "$(DATA_ROOT)" --template "$(METHOD_X_MNI_TEMPLATE)" --execute

preprocess-full-mri-spm:
	$(PY) scripts/preprocess_mri_spm.py --data_root "$(DATA_ROOT)" --matlab "$(MATLAB_BIN)" --spm_dir "$(SPM_DIR)" --manifest "$(SPM_MANIFEST)" --execute

preprocess-full-mri-spm-auto:
	$(PY) scripts/preprocess_mri_spm.py --data_root "$(DATA_ROOT)" --matlab "$(MATLAB_BIN)" --spm_dir "$(SPM_DIR)" --manifest "$(SPM_MANIFEST)" --execute --batch_size $(SPM_BATCH_SIZE) --auto_batches

spm-completion-guard:
	$(PY) scripts/spm_completion_guard.py

mri-volume-cache:
	$(PY) scripts/spm_completion_guard.py
	$(PY) scripts/build_mri_volume_cache.py --data_root "$(DATA_ROOT)"

mri-volume-cache-96:
	$(PY) scripts/spm_completion_guard.py
	$(PY) scripts/build_mri_volume_cache.py --preprocessed_dir data/preprocessed_mri_96 --out data/cached/t1_volume_cache_96.h5 --resolution 96

mri-volume-cache-smoke:
	$(PY) scripts/mri_volume_cache_smoke.py --preprocessed_dir data/preprocessed_mri_96 --resolution 96 --device auto

mri-features:
	$(PY) scripts/extract_mri_features.py --data_root "$(DATA_ROOT)"

mri-feature-baseline:
	$(PY) scripts/train_mri_feature_baseline.py

train-3d:
	$(PY) scripts/spm_completion_guard.py
	$(PY) scripts/train_3d_mri.py

three-d-ablation:
	$(PY) scripts/spm_completion_guard.py
	$(PY) scripts/run_3d_ablation.py --cache data/cached/t1_volume_cache_96.h5 --folds 5 --epochs 4 --batch_size 1

xai-summary:
	$(PY) scripts/spm_completion_guard.py
	$(PY) scripts/run_xai_summary.py --cache data/cached/t1_volume_cache_96.h5

splits:
	$(PY) scripts/make_splits.py

baseline:
	$(PY) scripts/train_baseline.py

cross-cohort-bias:
	$(PY) scripts/cross_cohort_bias_test.py

optuna:
	$(PY) scripts/run_optuna.py

fusion:
	$(PY) scripts/spm_completion_guard.py
	$(PY) scripts/run_fusion.py

external-eval:
	$(PY) scripts/spm_completion_guard.py
	$(PY) scripts/run_external_eval.py

calibration:
	$(PY) scripts/spm_completion_guard.py
	$(PY) scripts/run_calibration.py

final-analysis:
	$(PY) scripts/spm_completion_guard.py
	$(PY) scripts/final_analysis.py

public-bundle:
	$(PY) scripts/build_public_bundle.py

tau:
	$(PY) scripts/run_tau_branch.py --data_root "$(DATA_ROOT)"

tau-descriptive:
	$(PY) scripts/tau_descriptive_report.py --data_root "$(DATA_ROOT)"

power-analysis:
	$(PY) scripts/power_analysis.py

site-drift:
	$(PY) scripts/site_scanner_drift_sensitivity.py

subgroup-audit:
	$(PY) scripts/subgroup_performance_audit.py

gap-matrix:
	$(PY) scripts/build_implementation_gap_matrix.py

method-y-pet-inventory:
	$(PY) scripts/method_y_pet_inventory.py --data_root "$(DATA_ROOT)"

method-y-multimodal-design:
	$(PY) scripts/method_y_multimodal_design.py

.PHONY: validity-audit temporal-window-sensitivity optuna-nested external-final cognitive-family-ablation three-d-final-contract thesis-report v36-tests v38-tests v38-method-x-validity v38-pet-gap v38-pet-manifest v38-pet-dry v38-pet-execute-smoke v38-pet-cache v38-pet-readiness v38-pet-train v38-agent-summary method-y-gap-filter method-y-pet-manifest method-y-pet-dry method-y-pet-dry-run method-y-pet-execute method-y-pet-execute-smoke method-y-pet-cache method-y-readiness method-y-pet-train method-y-summary method-y-phase2-tests final-comprehensive-tests final-method-x-sequence final-method-y-pet-sequence final-agent-summary

validity-audit:
	$(PY) scripts/run_validity_audit.py

temporal-window-sensitivity:
	$(PY) scripts/run_temporal_window_sensitivity.py

optuna-nested:
	$(PY) scripts/run_optuna_nested.py

external-final:
	$(PY) scripts/run_external_eval.py --final_locked_config

cognitive-family-ablation:
	$(PY) scripts/run_cognitive_family_ablation.py

three-d-final-contract:
	$(PY) scripts/train_3d_mri.py --phase final --resolution 96 --folds 5 --input_mode mri_plus_tabular --fusion cross_attention --device auto --amp

thesis-report:
	$(PY) scripts/build_thesis_grade_report.py

v36-tests:
	$(PY) -m pytest tests/test_v3_6_policy_contract.py tests/test_validity_audit_outputs.py tests/test_temporal_window_variants.py tests/test_no_raw_ids_in_exports.py tests/test_full_pipeline_smoke_contracts.py -q

v38-tests:
	$(PY) -m pytest tests/test_v3_8_literature_policy_contract.py tests/test_v3_8_pet_manifest_and_gap.py tests/test_v3_8_pet_qc_and_readiness.py -q

v38-method-x-validity:
	$(PY) scripts/run_validity_audit.py
	$(PY) scripts/run_temporal_window_sensitivity.py
	$(PY) scripts/run_external_eval.py --final_locked_config
	$(PY) scripts/build_thesis_grade_report.py

matched-propensity-analysis:
	$(PY) scripts/run_matched_propensity_analysis.py --catalog data/processed/master_catalog.csv

decision-curve-analysis:
	$(PY) scripts/run_decision_curve_analysis.py --catalog data/processed/master_catalog.csv

tripod-probast-checklists:
	$(PY) scripts/write_tripod_probast_checklists.py

v38-pet-gap method-y-gap-filter:
	$(PY) scripts/method_y_gap_filter.py --catalog data/processed/master_catalog.csv

v38-pet-manifest method-y-pet-manifest:
	$(PY) scripts/build_pet_private_manifest.py --data_root "$(DATA_ROOT)" --tracer fbp --catalog data/processed/master_catalog.csv

v38-pet-dry method-y-pet-dry method-y-pet-dry-run:
	$(PY) scripts/preprocess_pet_spatial.py --data_root "$(DATA_ROOT)" --tracer fbp --dry_run --limit 10

v38-pet-execute-smoke method-y-pet-execute method-y-pet-execute-smoke:
	@test -n "$(MNI_TEMPLATE)" || (echo "Set MNI_TEMPLATE before PET execute smoke." && exit 2)
	$(PY) scripts/preprocess_pet_spatial.py --data_root "$(DATA_ROOT)" --tracer fbp --execute --spatial_mode ants_pet_t1_mni --mni_template "$(MNI_TEMPLATE)" --n_jobs 2 --limit 20

v38-pet-cache method-y-pet-cache:
	$(PY) scripts/build_pet_volume_cache.py --tracer fbp

v38-pet-readiness method-y-readiness:
	$(PY) scripts/check_method_y_training_readiness.py --tracer fbp

v38-pet-train method-y-pet-train:
	$(PY) scripts/check_method_y_training_readiness.py --tracer fbp
	$(PY) scripts/train_pet_method_y.py --tracer fbp --device auto --amp

v38-agent-summary method-y-summary final-agent-summary:
	$(PY) scripts/compact_policy_index.py
	$(PY) scripts/build_phase_y_summary.py

method-y-phase2-tests:
	$(PY) -m pytest tests/test_method_y_pet_phase2_contract.py tests/test_pet_manifest_resolution.py tests/test_pet_qc_gates.py tests/test_pet_spatial_preprocess_contract.py -q

final-comprehensive-tests:
	$(PY) -m pytest tests/test_integrated_patch_contract.py tests/test_v3_6_policy_contract.py tests/test_method_y_pet_phase2_contract.py tests/test_pet_manifest_resolution.py tests/test_pet_qc_gates.py tests/test_pet_spatial_preprocess_contract.py -q

final-method-x-sequence:
	$(PY) scripts/run_validity_audit.py
	$(PY) scripts/run_temporal_window_sensitivity.py
	$(PY) scripts/run_external_eval.py --final_locked_config
	$(PY) scripts/run_cognitive_family_ablation.py
	$(PY) scripts/build_thesis_grade_report.py

final-method-y-pet-sequence:
	$(PY) scripts/method_y_gap_filter.py --catalog data/processed/master_catalog.csv
	$(PY) scripts/build_pet_private_manifest.py --data_root "$(DATA_ROOT)" --tracer fbp --catalog data/processed/master_catalog.csv
	$(PY) scripts/preprocess_pet_spatial.py --data_root "$(DATA_ROOT)" --tracer fbp --dry_run --limit 10
	@echo "Review dry-run summary before PET execute smoke, cache, or training."

clinical-ad-feasibility:
	$(PY) scripts/clinical_ad_target_feasibility.py --data_root "$(DATA_ROOT)"

run-stage:
	$(PY) scripts/run_project_stage.py --stage "$(STAGE)" --data_root "$(DATA_ROOT)"

run-next-stage:
	$(PY) scripts/run_project_stage.py --next --data_root "$(DATA_ROOT)"

run-all-stages:
	$(PY) scripts/run_project_stage.py --run_all --data_root "$(DATA_ROOT)"


