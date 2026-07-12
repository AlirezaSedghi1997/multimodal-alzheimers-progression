# MRI 128^3 Data Augmentation Policy

## Canonical training cache

- Final path: `data/cached/t1_volume_cache_128.h5`
- Immutable real-only source: `data/cached/archive/t1_volume_cache_128_real_20260712T203222Z_26fff22aa7de.h5`
- Source last-write time (UTC): `2026-07-11T18:44:08.956207+00:00`
- Source manifest SHA-256: `26fff22aa7de1fa5f3b0799ddcbfde7b83e7daa8e3f6397516ac40efc31d1ccb`
- Final rows: 6,213 (1,669 real QC-pass rows and 4,544 materialized augmented rows)
- Excluded source rows: 30 (28 HOLD and 2 FAIL)

The canonical cache is derived only from the July-11 real cache. Each augmented
row stores its parent sample/patient hash, source fold, deterministic seed, and
transform parameters.

## Leakage controls

Augmentation is materialized separately for the five locked outer folds. A
parent is eligible only when it is in that fold's outer-training and inner-fit
partition. Outer-test, inner-validation, locked-holdout, external-test, and
non-QC-pass parents are excluded. Training additionally filters supplementary
rows by their parent hash, so an augmented derivative cannot enter a validation
or test partition.

## Transform policy

No axis flip is permitted. The source HDF5 tensor does not encode a reliable
patient-left/right affine contract, so left-right, anterior-posterior, and
superior-inferior flips are all excluded.

The builder uses the deterministic
`mri_128_signed_zscore_adaptive_background_v3` profile:

- Images with an explicit zero background may receive a small 3-D rotation,
  translation, and scale perturbation; brain-mask-constrained contrast/offset,
  smooth bias field, and low-amplitude noise are also allowed. Background is
  restored to zero after transformation.
- Full-field z-scored images have no recoverable zero-background mask. They
  therefore receive only smooth multiplicative intensity perturbations; affine,
  additive offset, and noise operations are disabled.
- Every generated volume is checked for shape, finite values, non-degenerate
  variance, plausible range, and background preservation where applicable.

The online MONAI policy is intentionally limited to small random affine and bias
field perturbations. Final runners disable online augmentation when consuming
the already materialized canonical cache, preventing double augmentation.

## Validation evidence

- Full cache validation: 6,213/6,213 images passed.
- Geometry audit: all 1,699 preprocessed source NIfTI files are 128^3, 1.5-mm,
  LAS-oriented, and share one affine; no orientation or affine outlier exists.
- Final verification: `results/tables/final_mri_augmented_cache_verification.json`
- Build summary: `results/tables/fold_local_mri_augmentation_summary.json`
- Geometry audit: `results/tables/mri_128_geometry_audit.json`
- Cache cleanup manifest: `results/tables/cache_cleanup_manifest.json`

This augmentation expands training exposure but does not by itself establish a
scientific performance gain. That claim requires fold-locked ablation against
the real-only baseline.
