# Method Y PET Phase 2 Spatial+QC Patch

This patch upgrades the previous Method Y PET scaffold into a **manifest-first, spatial-alignment-gated, QC-gated, resumable and agent-efficient** preprocessing package.

It does **not** change Method X. PET images, SUVR, centiloid, tracer terms, scan-date terms, and PET-derived values remain forbidden as Method X / Method X-Cog model inputs.

## What this patch fixes

| Risk from previous PET scaffold | Fix in this package |
|---|---|
| PET preprocessing could run without spatial alignment | `method_y_spatial_preprocess_policy.yaml` makes coregistration/resampling mandatory for training-ready outputs |
| SUVR/reference region was undefined | `method_y_suvr_reference_policy.yaml` freezes reference-region policy and blocks scalar SUVR/Centiloid inputs |
| Filename regex could miss many PET files | `build_pet_private_manifest.py` and `src/utils/pet_manifest.py` make manifest-first linking mandatory; regex is fallback only |
| Multiple PET scans per patient could leak | one-PET-per-patient selection is enforced for training candidates |
| No resume/checkpoint | `preprocess_pet_spatial.py` writes `results/logs/method_y_pet_preprocess_resume_<tracer>.json` after each item |
| No parallel execution | `--n_jobs` supports controlled parallelism with conservative defaults |
| QC was descriptive only | `src/utils/pet_spatial_qc.py` implements QC gates and failure statuses |
| Error logs could leak raw paths | error text is scrubbed before JSON/MD outputs |
| Agent had to reread too many files | compact summaries, task cards, and policy index script are included |

## Status after applying

The target state is:

```text
PET spatial-preprocessing ready, QC-gated, resumable, stage-runner integrated
```

It is **not** a strong clinical-claim training package. Method Y remains exploratory and must be reported separately from Method X.

## Quick apply

```bash
unzip method_y_pet_phase2_spatial_qc_patch_pack.zip
cd method_y_pet_phase2_spatial_qc_patch
bash apply_method_y_pet_phase2_patch.sh /path/to/amyloid-tau-prediction-a4learn
cd /path/to/amyloid-tau-prediction-a4learn
cat patches/MAKEFILE_METHOD_Y_PHASE2_TARGETS.mk >> Makefile
python -m pytest tests/test_method_y_pet_phase2_contract.py -q
```

## Recommended execution order

```bash
# 0) Validate policies and tests
python -m pytest tests/test_method_y_pet_phase2_contract.py tests/test_pet_manifest_resolution.py tests/test_pet_qc_gates.py -q

# 1) Build or refresh private manifest
python scripts/build_pet_private_manifest.py --data_root "$DATA_ROOT" --tracer fbp

# 2) Gap + cohort feasibility report
python scripts/method_y_gap_filter.py --catalog data/processed/master_catalog.csv

# 3) Dry run spatial preprocessing
python scripts/preprocess_pet_spatial.py --data_root "$DATA_ROOT" --tracer fbp --dry_run --limit 10

# 4) Execute only if spatial method is available
python scripts/preprocess_pet_spatial.py \
  --data_root "$DATA_ROOT" \
  --tracer fbp \
  --execute \
  --spatial_mode ants_rigid_pet_to_t1 \
  --n_jobs 2 \
  --limit 20

# 5) Build Method Y PET cache only after QC pass
python scripts/build_pet_volume_cache.py --tracer fbp

# 6) Check training readiness
python scripts/check_method_y_training_readiness.py --tracer fbp

# 7) Build compact report for agents/users
python scripts/build_phase_y_summary.py
```

## Important scientific boundary

- Method X: T1 MRI ± fixed tabular/cognitive; PET is label/target only.
- Method Y: PET-assisted exploratory; PET image can be input only here.
- PET-only baseline = biomarker reproduction / reader-surrogate context, not Alzheimer diagnosis.
- MRI+PET = exploratory PET-assisted fusion behavior, not a non-invasive Method X claim.
