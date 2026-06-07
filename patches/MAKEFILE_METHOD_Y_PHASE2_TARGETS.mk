.PHONY: method-y-gap-filter method-y-pet-manifest method-y-pet-dry method-y-pet-execute method-y-pet-cache method-y-readiness method-y-summary method-y-phase2-tests

method-y-gap-filter:
	$(PY) scripts/method_y_gap_filter.py --catalog data/processed/master_catalog.csv

method-y-pet-manifest:
	$(PY) scripts/build_pet_private_manifest.py --data_root "$(DATA_ROOT)" --tracer fbp

method-y-pet-dry:
	$(PY) scripts/preprocess_pet_spatial.py --data_root "$(DATA_ROOT)" --tracer fbp --dry_run --limit 10

method-y-pet-execute:
	@test -n "$(MNI_TEMPLATE)" || (echo "Set MNI_TEMPLATE before PET execute smoke." && exit 2)
	$(PY) scripts/preprocess_pet_spatial.py --data_root "$(DATA_ROOT)" --tracer fbp --execute --spatial_mode ants_pet_t1_mni --mni_template "$(MNI_TEMPLATE)" --n_jobs 2 --limit 20

method-y-pet-cache:
	$(PY) scripts/build_pet_volume_cache.py --tracer fbp

method-y-readiness:
	$(PY) scripts/check_method_y_training_readiness.py --tracer fbp

method-y-summary:
	$(PY) scripts/compact_policy_index.py
	$(PY) scripts/build_phase_y_summary.py

method-y-phase2-tests:
	$(PY) -m pytest tests/test_method_y_pet_phase2_contract.py tests/test_pet_manifest_resolution.py tests/test_pet_qc_gates.py tests/test_pet_spatial_preprocess_contract.py -q
