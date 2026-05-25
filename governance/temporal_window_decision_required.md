# Temporal Window Decision Required

## Finding

The current hashed master catalog has both AB-positive and AB-negative classes. After using `SV.csv` visit timing for T1 files, the strict 120-day PET-T1 subset also contains both classes.

## Consequence

Gate 1 binary model training may proceed after split locking, but the subset is imbalanced and must be reported with class counts and confidence intervals.

## Valid Paths

1. Freeze `SV.csv` as the primary T1 timing source when T1 filename scan number matches `VISITCD`.
2. Use volumetric MRI timing as secondary support.
3. Report 60/90/120-day sensitivity.

## Current Default

Proceed after the timing hierarchy is frozen in policy and splits are locked.
