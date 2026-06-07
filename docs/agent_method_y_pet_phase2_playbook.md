# Agent Playbook — Method Y PET Phase 2

Goal: minimize token use and prevent unsafe broad edits.

## Agent operating rules

1. Read compact summaries first:
   - `.agent/CONTEXT_METHOD_Y_PET.md` if present
   - `results/tables/method_y_gap_filter_report.json`
   - `results/tables/method_y_pet_preprocess_summary.json`
   - `results/tables/method_y_training_readiness.json`

2. Do not read raw CSVs/NIfTIs unless a test or stage fails.

3. Do not edit Method X files unless the task explicitly says so.

4. Never run `git add .`.

5. Always run dry-run before execute.

## Task cards

### Phase 2a — Build manifest

Read only:
- `configs/model/method_y_spatial_preprocess_policy.yaml`
- `scripts/build_pet_private_manifest.py`

Command:
```bash
python scripts/build_pet_private_manifest.py --data_root "$DATA_ROOT" --tracer fbp
```

Stop condition:
- `data/preprocessed_pet/_private/manifests/fbp_private_manifest.csv` exists
- `results/tables/method_y_pet_manifest_summary.json` exists
- no raw paths in `results/tables`

### Phase 2b — Dry-run spatial preprocessing

Read only:
- `scripts/preprocess_pet_spatial.py`
- `src/utils/pet_manifest.py`
- `src/utils/pet_spatial_qc.py`

Command:
```bash
python scripts/preprocess_pet_spatial.py --data_root "$DATA_ROOT" --tracer fbp --dry_run --limit 10
```

Stop condition:
- selected work items > 0
- no raw paths in exportable outputs
- readiness still not training-ready unless spatial execute completed

### Phase 2c — Execute spatial preprocessing smoke

Command:
```bash
python scripts/preprocess_pet_spatial.py --data_root "$DATA_ROOT" --tracer fbp --execute --spatial_mode ants_rigid_pet_to_t1 --n_jobs 2 --limit 20
```

Fallback:
- use `precomputed_petprep_output` if PETPrep outputs exist
- use `identity_for_tests` only in tests with `--allow_identity_for_tests`

### Phase 2d — Readiness

Command:
```bash
python scripts/check_method_y_training_readiness.py --tracer fbp
```

Only proceed to modeling if status is `READY_FOR_EXPLORATORY_METHOD_Y_TRAINING`.
