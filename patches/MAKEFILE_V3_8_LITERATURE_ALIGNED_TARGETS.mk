.PHONY: v38-tests v38-method-x-validity v38-pet-gap v38-pet-manifest v38-pet-dry v38-pet-execute-smoke v38-pet-cache v38-pet-readiness v38-agent-summary

v38-tests:
	$(PY) -m pytest tests/test_v3_8_literature_policy_contract.py tests/test_v3_8_pet_manifest_and_gap.py tests/test_v3_8_pet_qc_and_readiness.py -q

v38-method-x-validity:
	$(PY) scripts/run_validity_audit.py
	$(PY) scripts/run_temporal_window_sensitivity.py
	$(PY) scripts/run_external_eval.py --final_locked_config
	$(PY) scripts/build_thesis_grade_report.py

v38-pet-gap:
	$(PY) scripts/method_y_gap_filter.py --catalog data/processed/master_catalog.csv

v38-pet-manifest:
	$(PY) scripts/build_pet_private_manifest.py --data_root "$(DATA_ROOT)" --tracer fbp --catalog data/processed/master_catalog.csv

v38-pet-dry:
	$(PY) scripts/preprocess_pet_spatial.py --data_root "$(DATA_ROOT)" --tracer fbp --dry_run --limit 10

v38-pet-execute-smoke:
	$(PY) scripts/preprocess_pet_spatial.py --data_root "$(DATA_ROOT)" --tracer fbp --execute --spatial_mode ants_pet_t1_mni --mni_template "$(MNI_TEMPLATE)" --n_jobs 2 --limit 20

v38-pet-cache:
	$(PY) scripts/build_pet_volume_cache.py --tracer fbp

v38-pet-readiness:
	$(PY) scripts/check_method_y_training_readiness.py --tracer fbp

v38-agent-summary:
	$(PY) scripts/compact_policy_index.py
	$(PY) scripts/build_phase_y_summary.py
