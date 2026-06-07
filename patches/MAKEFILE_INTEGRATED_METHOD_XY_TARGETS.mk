# Integrated Method X v3.6 + Method Y PET targets.
# Append this file to the existing Makefile after reviewing `git diff`.
# Assumes PY := conda run -n $(ENV_NAME) python is already defined.

.PHONY: \
        validity-audit temporal-window-sensitivity optuna-nested external-final \
        cognitive-family-ablation three-d-final-contract thesis-report v36-tests \
        method-y-gap-filter method-y-pet-dry-run method-y-pet-smoke method-y-pet-tests \
        integrated-contract-tests integrated-stage-1-audit integrated-stage-2-pet-smoke integrated-report

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
	$(PY) scripts/train_3d_mri.py --phase final --resolution 96 --folds 5 --input_mode mri_plus_tabular --fusion cross_attention

thesis-report:
	$(PY) scripts/build_thesis_grade_report.py

v36-tests:
	$(PY) -m pytest tests/test_v3_6_policy_contract.py tests/test_validity_audit_outputs.py tests/test_temporal_window_variants.py tests/test_no_raw_ids_in_exports.py tests/test_full_pipeline_smoke_contracts.py -q

method-y-gap-filter:
	$(PY) scripts/method_y_gap_filter.py --catalog data/processed/master_catalog.csv

method-y-pet-dry-run:
	$(PY) scripts/preprocess_pet.py --data_root "$(DATA_ROOT)" --tracer fbp --dry_run

method-y-pet-smoke:
	$(PY) scripts/preprocess_pet.py --data_root "$(DATA_ROOT)" --tracer fbp --execute --limit 10

method-y-pet-tests:
	$(PY) -m pytest tests/test_pet_preprocess_contract.py -q

integrated-contract-tests:
	$(PY) -m pytest tests/test_integrated_patch_contract.py tests/test_v3_6_policy_contract.py tests/test_pet_preprocess_contract.py -q

integrated-stage-1-audit:
	$(PY) scripts/run_validity_audit.py
	$(PY) scripts/run_temporal_window_sensitivity.py
	$(PY) scripts/run_external_eval.py --final_locked_config
	$(PY) scripts/build_thesis_grade_report.py

integrated-stage-2-pet-smoke:
	$(PY) scripts/method_y_gap_filter.py --catalog data/processed/master_catalog.csv
	$(PY) scripts/preprocess_pet.py --data_root "$(DATA_ROOT)" --tracer fbp --dry_run

integrated-report:
	$(PY) scripts/build_thesis_grade_report.py
