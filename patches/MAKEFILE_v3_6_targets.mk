# Add these targets to the existing Makefile.
# They assume PY := conda run -n $(ENV_NAME) python is already defined.

.PHONY: validity-audit temporal-window-sensitivity optuna-nested external-final cognitive-family-ablation three-d-final-contract thesis-report v36-tests

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
