# Agent Execution Plan - Method X/Y v3.8 Literature-Aligned

## Mission
Apply and execute the v3.8 patch without contaminating Method X with PET-derived inputs and without starting PET training before readiness gates pass.

## Read first
1. `README_V3_8_LITERATURE_ALIGNED_PATCH.md`
2. `configs/model/final_method_x_y_v36_pet_phase2_policy.yaml`
3. `configs/model/method_y_spatial_preprocess_policy.yaml`
4. `configs/model/method_y_suvr_reference_policy.yaml`
5. `docs/method_y_phase2_modeling_design.md`

## Do not read unless needed
- raw NIfTI files
- full master catalog
- subject-level PET QC rows
- raw path manifests

Prefer summaries:
- `results/tables/method_y_gap_filter_report.json`
- `results/tables/method_y_pet_manifest_summary.json`
- `results/tables/method_y_pet_preprocess_summary.json`
- `results/tables/method_y_training_readiness.json`
- `results/final_results/method_y/phase_y_summary.md`

## Stage A — policy tests
```bash
python -m pytest tests/test_v3_8_literature_policy_contract.py -q
```
Stop if policy tests fail.

## Stage B — Method X validity
Do not modify Method Y during this phase; Stage C/D are separate gated Method Y dry-run/smoke phases.

```bash
python scripts/run_validity_audit.py
python scripts/run_temporal_window_sensitivity.py
python scripts/run_external_eval.py --final_locked_config
python scripts/build_thesis_grade_report.py
```

## Stage C — Method Y PET dry-run only
```bash
python scripts/method_y_gap_filter.py --catalog data/processed/master_catalog.csv
python scripts/build_pet_private_manifest.py --data_root "$DATA_ROOT" --tracer fbp --catalog data/processed/master_catalog.csv
python scripts/preprocess_pet_spatial.py --data_root "$DATA_ROOT" --tracer fbp --dry_run --limit 10
```

## Stage D — PET execute smoke only after human review
```bash
python scripts/preprocess_pet_spatial.py --data_root "$DATA_ROOT" --tracer fbp --execute --spatial_mode ants_pet_t1_mni --mni_template /path/to/MNI152_T1_1mm.nii.gz --n_jobs 2 --limit 20
```

## Stop conditions
- Stop if `selected_work_items == 0`.
- Stop if private manifest used regex fallback for training-ready run.
- Stop if `method_y_training_readiness.status != READY_FOR_EXPLORATORY_METHOD_Y_TRAINING`.
- Stop if any PET-derived string appears in Method X cache.

## Allowed edits
- `scripts/build_pet_private_manifest.py`
- `scripts/preprocess_pet_spatial.py`
- `scripts/check_method_y_training_readiness.py`
- `src/utils/pet_manifest.py`
- `src/utils/pet_spatial_qc.py`
- `configs/model/method_y_spatial_preprocess_policy.yaml`
- `configs/model/method_y_suvr_reference_policy.yaml`

## Forbidden edits without human approval
- `data/processed/master_catalog.csv`
- `data/cached/**`
- Method X model input schema
- public bundle contents
Do not edit these forbidden items without human approval.
