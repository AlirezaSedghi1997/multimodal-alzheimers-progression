SHELL := /bin/bash
ENV_NAME ?= amyloid-tau
PY := conda run -n $(ENV_NAME) python
DATA_ROOT ?= /mnt/h/A4\ Learn
export WANDB_MODE := offline

.PHONY: test preflight inventory catalog preprocess preprocess-toolchain-check preprocess-spm-toolchain-check preprocess-full-mri preprocess-full-mri-spm spm-completion-guard mri-volume-cache mri-volume-cache-96 mri-features mri-feature-baseline train-3d three-d-ablation xai-summary splits baseline cross-cohort-bias optuna fusion external-eval calibration final-analysis public-bundle tau external-drive-check compute-plan power-analysis site-drift subgroup-audit gap-matrix method-y-pet-inventory method-y-multimodal-design clinical-ad-feasibility run-stage run-next-stage run-all-stages

test:
	$(PY) -m pytest tests/ -q

preflight:
	$(PY) scripts/preflight_check.py --data_root "$(DATA_ROOT)"

external-drive-check:
	$(PY) scripts/external_drive_check.py --data_root "$(DATA_ROOT)"

compute-plan:
	$(PY) scripts/compute_plan.py

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
	$(PY) scripts/preprocess_mri_spm.py --data_root "$(DATA_ROOT)" --matlab "$(MATLAB_BIN)" --spm_dir "$(SPM_DIR)" --execute

spm-completion-guard:
	$(PY) scripts/spm_completion_guard.py

mri-volume-cache:
	$(PY) scripts/spm_completion_guard.py
	$(PY) scripts/build_mri_volume_cache.py --data_root "$(DATA_ROOT)"

mri-volume-cache-96:
	$(PY) scripts/spm_completion_guard.py
	$(PY) scripts/build_mri_volume_cache.py --preprocessed_dir data/preprocessed_mri_96 --out data/cached/t1_volume_cache_96.h5 --resolution 96

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

clinical-ad-feasibility:
	$(PY) scripts/clinical_ad_target_feasibility.py --data_root "$(DATA_ROOT)"

run-stage:
	$(PY) scripts/run_project_stage.py --stage "$(STAGE)" --data_root "$(DATA_ROOT)"

run-next-stage:
	$(PY) scripts/run_project_stage.py --next --data_root "$(DATA_ROOT)"

run-all-stages:
	$(PY) scripts/run_project_stage.py --run_all --data_root "$(DATA_ROOT)"
