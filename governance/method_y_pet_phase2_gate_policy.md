# Method Y PET Phase-2 Gate Policy

Version: 2026-05-29  
Status: proposed patch

This policy defines the minimum gates required to move Method Y-PET/MM from `DESIGN_READY_NOT_TRAINED` to `PHASE_2_RUNNING`.

## Gate 1 — Preregistration frozen

Required file:

```text
docs/method_y_preregistration.md
```

The file must freeze:

- target wording;
- temporal gap tiers;
- sample-selection tier for primary analysis;
- modeling scenarios;
- metrics;
- statistical comparison rules;
- public-output restrictions.

## Gate 2 — Gap and cohort policy frozen

Required file:

```text
configs/model/method_y_gap_cohort_eval_policy.yaml
```

The policy must state:

- PET-T1 gap is not a model feature;
- primary strict tier is `<=90 days` unless amended with a version bump;
- `<=120 days` is sensitivity;
- `>365 days` is aggregate-only/excluded from training;
- cohort-stratified AUC is reported only where both classes are present.

## Gate 3 — PET preprocessing root is separate

Allowed examples:

```text
data/preprocessed_pet/
data/method_y/preprocessed_pet/
```

Forbidden examples:

```text
data/cached/
data/cached/method_x/
data/processed/
```

## Gate 4 — Method X loader regression tests pass

After adding PET code, the following must pass:

```text
pytest tests/test_no_pet_in_hdf5_cache.py tests/test_method_x_feature_contract.py tests/test_pet_preprocess_contract.py -q
```

If Method X loaders contain PET tensors, PET-derived scalar features, gap variables, raw PET paths, or tracer metadata, Method Y must be blocked.

## Gate 5 — Aggregate-only public output

The public bundle must not contain:

- raw IDs;
- raw paths;
- PET images;
- MRI images;
- NIfTI files;
- HDF5 caches;
- checkpoints;
- subject-level PET tables.

## Gate 6 — Downgrade language enforced

Until an independent clinical target exists, reports must use:

> Aβ-positive vs Aβ-negative biomarker status

and must not use:

> Alzheimer disease diagnosis

## Gate 7 — Training readiness packet

Before any PET-image training, generate:

```text
results/tables/method_y_gap_filter_report.md
results/tables/method_y_pet_preprocess_summary.md
results/logs/method_y_pet_preprocess_manifest.json
```

The packet must include counts by label, cohort, and temporal-gap tier.
