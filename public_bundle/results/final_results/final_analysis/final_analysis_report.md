# Final Analysis Report

## Summary

Method X pipeline execution reached Gate 4 and calibration using T1 MRI-derived features plus allowed tabular variables. PET was used only as label/target source and never as a model input.

## Key Results

- Catalog rows: `1699`
- Dev rows: `1359`
- Pseudo-external rows: `340`
- Best fusion ROC-AUC: `0.7544`
- External-like ROC-AUC: `0.7505`
- 3D training mode: `FULL_3D_TRAINING`
- 3D ROC-AUC: `0.7222`
- 3D external-like ROC-AUC: `0.7236`
- 3D ablation status: `PASSED`
- 3D ablation best ROC-AUC: `0.7173`
- CV calibrated Brier: `0.1832`
- CV calibrated ECE: `0.0441`
- Tau feasibility mode: `FULL`
- Tau analysis mode: `FULL`
- Tau regression R2: `-0.0446`

- Tau rows used: `281`
- Tau RMSE: `0.0671`
- Tau RMSE reduction vs mean baseline: `-0.0011`
- Tau interpretation: `No out-of-fold predictive signal over the grouped mean baseline; report as a negative exploratory result.`

- Power analysis status: `DESCRIPTIVE_POWER_ANALYSIS`
- Full MRI preprocessing status: `COMPLETED`
- Site/scanner drift status: `SKIP_WITH_JUSTIFICATION`
- Subgroup audit status: `PASSED`
- XAI status: `PASSED`
- Method X-Cog status: `PASSED`
- Method X-Cog best ROC-AUC: `0.7992`
- Method X-Cog AUC gain over fixed tabular: `0.0678`
- Method X-Cog best non-MRI variant: `fixed_tabular_plus_all_cognitive`
- Method X-Cog best non-MRI ROC-AUC: `0.7935`
- Method X-Cog/SF stress-test status: `PASSED`
- Phase 2 minimum MVP status: `PASSED`
- Cohort-probe (3D latent linear AUC): `0.5031`
- Centiloid within-A4 Spearman: `NA`
- Centiloid within-LEARN Spearman: `NA`
- Method Y PET inventory status: `AGGREGATE_ONLY`
- Method Y amyloid PET image files counted: `5661`
- Method Y tau PET image files counted: `1576`
- Method Y multimodal design status: `DESIGN_READY_NOT_TRAINED`
- Method Y clinical Alzheimer output allowed now: `False`
- Implementation gap summary: `{'DONE': 41, 'MISSING': 0, 'PARTIAL': 2, 'SKIP_WITH_JUSTIFICATION': 1}`

## Required Caveat

The local catalog has complete cohort-label coupling. Report performance as Method X engineering evidence under a pseudo-external holdout, not as proof of pathology-only generalization to an independent cohort.

## Compliance

- PET as model input: `False`
- PET as Method X-Cog input: `False`
- Method Y PET inventory is aggregate-only: `True`
- Method Y multimodal training started: `False`
- Alzheimer diagnosis output requires independent clinical label: `True`
- Raw identifiers used: `False`
- External used for calibration fit: `False`
- External holdout type: `locked_pseudo_external_random_patient_holdout`

## Cohort-probe validity notes

- `method_x_3d_mri_latent_oof`: linear AUC `0.5031`, nonlinear AUC `0.5102`, gap `0.0070`
- `method_x_strict_tabular`: linear AUC `0.7313`, nonlinear AUC `0.7050`, gap `-0.0263`
- `mri_roi_wmh_reads`: linear AUC `0.7621`, nonlinear AUC `0.9920`, gap `0.2299`
- ROI/WMH/read branch shows strong nonlinear cohort encoding (~0.99). Treat as cohort/protocol artifact risk, not successful deconfounding.
