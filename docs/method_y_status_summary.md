# Method Y PET-Assisted Branch — Status Summary

- training readiness: `NOT_TRAINING_READY`
- multimodal design: `DESIGN_READY_NOT_TRAINED`
- PET inventory: `COMPLETED`
- phase Y summary: `MISSING`
- method_x_claim_unchanged: `True`
- pet_as_method_x_input: `False`

## Claim boundary

Exploratory PET-assisted branch (Method Y). PET never enters Method X-Strict. Training has not started; design and inventory artifacts only.

## Blockers

- `REGEX_FALLBACK_MANIFEST_NOT_TRAINING_READY`
- `PRIVATE_MANIFEST_NOT_TRAINING_READY`
- `PRIMARY_TIER_MIN_CLASS_BELOW_BLOCK_THRESHOLD`
- `PREPROCESS_NOT_EXECUTED`
- `NO_QC_PASS_PET_OUTPUTS`

## Warnings

- `COHORT_LABEL_AUDIT_POSSIBLE_STRONG_COUPLING`
- `TRACER_LABEL_AUDIT_SINGLE_ROW_OR_COLUMN_REVIEW`
- `GAP_LABEL_AUDIT_POSSIBLE_STRONG_COUPLING`
- `PET_CACHE_NOT_BUILT_YET`

## Required before training

- PET-only baseline plan frozen
- MRI-only baseline on Method Y split plan frozen
- calibration/ECE/Brier/CI plan frozen
- patient-level split and one-PET-per-patient verified
- cohort/tracer/gap confounding audits reviewed
