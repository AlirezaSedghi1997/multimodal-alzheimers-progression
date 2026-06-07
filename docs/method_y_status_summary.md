# Method Y PET-Assisted Branch — Design & Readiness Summary

- overall training readiness: **`NOT_TRAINING_READY`**
- multimodal design: `DESIGN_READY_NOT_TRAINED`
- PET inventory: `COMPLETED`
- phase Y summary: `MISSING`
- training started: `False`
- PET preprocessing started: `False`
- method_x_claim_unchanged: `True`
- pet_as_method_x_input: `False`
- current primary target: `AB-positive vs AB-negative biomarker status`

## Claim boundary

Exploratory PET-assisted branch (Method Y). PET never enters Method X-Strict or Method X-Cog. Multimodal PET/MRI training has not started; only design, inventory, and readiness audits exist.

## Design availability (aggregate)

| resource | count |
|---|---:|
| locked catalog rows | 1700 |
| dev rows | 1360 |
| external rows | 340 |
| amyloid FBP image files | 5661 |
| amyloid locked patient matches | 1698 |
| tau FTP image files | 1576 |
| tau locked patient matches | 447 |

## Modality scenarios (design-ready)

| scenario | PET input | Method X compliant | status |
|---|---|---|---|
| mri_only | False | True | Method X baseline/final-comparison path |
| pet_only | True | False | Method Y only |
| mri_pet | True | False | Method Y multimodal only |
| mri_pet_cognitive | True | False | Method Y multimodal+cognitive only |

## Training blockers

- **`REGEX_FALLBACK_MANIFEST_NOT_TRAINING_READY`** — PET file manifest still relies on regex fallback; training-grade private manifest is required.
- **`PRIVATE_MANIFEST_NOT_TRAINING_READY`** — Hash-level private PET manifest is incomplete or not validated for one-PET-per-patient discipline.
- **`PRIMARY_TIER_MIN_CLASS_BELOW_BLOCK_THRESHOLD`** — Primary-tier Method Y split has insufficient minority-class support for stable training.
- **`PREPROCESS_NOT_EXECUTED`** — Method Y PET spatial preprocessing / cache build has not been executed.
- **`NO_QC_PASS_PET_OUTPUTS`** — No PET preprocessing outputs passed spatial QC gates.

## Warnings

- **`COHORT_LABEL_AUDIT_POSSIBLE_STRONG_COUPLING`** — Amyloid label may remain coupled to cohort/tracer in Method Y splits.
- **`TRACER_LABEL_AUDIT_SINGLE_ROW_OR_COLUMN_REVIEW`** — Tracer vs label contingency needs manual review.
- **`GAP_LABEL_AUDIT_POSSIBLE_STRONG_COUPLING`** — PET–T1 temporal gap bins may correlate with label.
- **`PET_CACHE_NOT_BUILT_YET`** — Method Y PET volume cache has not been built.

## Required before training

- PET-only baseline plan frozen
- MRI-only baseline on Method Y split plan frozen
- calibration/ECE/Brier/CI plan frozen
- patient-level split and one-PET-per-patient verified
- cohort/tracer/gap confounding audits reviewed

## Resolution checklist

- Build and validate private PET manifest (no regex fallback).
- Execute Method Y PET spatial preprocessing and pass QC gates.
- Build PET volume cache separate from Method X T1 cache.
- Confirm patient-level split with one amyloid PET per patient.
- Run cohort/tracer/gap confounding audits on Method Y split plan.
- Keep calibration/ECE/Brier/CI plan train-fold-only.

## Execution gates (from design artifact)

- Create separate Method Y PET preprocessing root outside Method X caches.
- Create no-PET-in-Method-X regression test after adding PET code.
- Confirm PET images are never loaded by Method X or Method X-Cog dataloaders.
- Define whether the target is biomarker abnormality or an independent clinical diagnosis.
- If clinical AD output is desired, build and validate a separate clinical-label table before training.
- Keep public outputs aggregate-only unless a separate privacy review approves hash-level artifacts.
