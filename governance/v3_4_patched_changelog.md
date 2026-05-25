# v3.4-PATCHED Changelog

## Kept Unchanged

- Method X.
- PET is label/target only.
- T1 MRI and frozen tabular inputs remain the allowed model inputs.
- Tau remains exploratory.
- External-like holdout is locked before training and never used for tuning.

## Mandatory Fixes Included In This Scaffold

- Conda-run Makefile contract.
- Real A4 Learn schema correction: `Date_DAYS_T0`.
- External drive readiness checks.
- GPU/CPU compute policy.
- Stage execution policy.
- Calibration fit-scope policy.
- Fusion queue requiring cross-attention, convolutional fusion, and latent alignment ablation.
- no-PET-in-HDF5 test placeholder and implementation target.

## Explicitly Rejected

- PET as model input.
- PET-derived features as model input.
- Site/scanner as model input.
- Method X-Plus renaming.
- Mandatory Docker.
- Any public bundle containing raw data or identifiers.

