# Method Y PET Phase 2 Spatial+QC Governance Policy

This policy upgrades Method Y PET from scaffold to controlled spatial preprocessing. It does not modify Method X.

## Non-negotiable

- PET is never an input to Method X or Method X-Cog.
- PET preprocessing outputs must live outside Method X caches.
- PET spatial alignment is mandatory for training-ready Method Y outputs.
- SUVR/Centiloid scalar values are forbidden as Method X inputs.
- SUVR/Centiloid scalar values are not Method Y training features in this patch; they are private QC/sensitivity only.
- Public bundle remains aggregate-only.
- Execute stage must be resumable and QC-gated.

## Required artifacts

```text
data/preprocessed_pet/_private/manifests/<tracer>_private_manifest.csv
results/tables/method_y_pet_manifest_summary.json
results/tables/method_y_pet_preprocess_summary.json
results/tables/method_y_training_readiness.json
results/logs/method_y_pet_preprocess_resume_<tracer>.json
```

## Training readiness

Training readiness is blocked unless:

1. spatial preprocessing has completed;
2. QC pass rate is reported;
3. match rate is above configured block threshold;
4. output roots are Method Y-only;
5. Method X cache scan is clean;
6. preregistration has been updated with counts and policy hashes.

## Allowed stage labels

```text
method-y-gap-filter
method-y-pet-manifest
method-y-pet-preprocess-dry
method-y-pet-preprocess-execute
method-y-pet-cache
method-y-training-readiness
method-y-phase-summary
```
