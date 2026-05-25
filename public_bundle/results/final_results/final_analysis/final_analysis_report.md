# Final Analysis Report

## Summary

Method X pipeline execution reached Gate 4 and calibration using T1 MRI-derived features plus allowed tabular variables. PET was used only as label/target source and never as a model input.

## Key Results

- Catalog rows: `1700`
- Dev rows: `1360`
- Pseudo-external rows: `340`
- Best fusion ROC-AUC: `0.7544`
- External-like ROC-AUC: `0.7527`
- 3D training mode: `FULL_3D_TRAINING`
- 3D ROC-AUC: `0.7121`
- 3D external-like ROC-AUC: `0.7316`
- 3D ablation status: `PENDING`
- 3D ablation best ROC-AUC: `NA`
- CV calibrated Brier: `0.1847`
- CV calibrated ECE: `0.0525`
- Tau feasibility mode: `FULL`
- Tau analysis mode: `FULL`
- Tau regression R2: `-0.0446`

- Tau rows used: `281`
- Tau RMSE: `0.0671`
- Tau RMSE reduction vs mean baseline: `-0.0011`
- Tau interpretation: `No out-of-fold predictive signal over the grouped mean baseline; report as a negative exploratory result.`

- Power analysis status: `DESCRIPTIVE_POWER_ANALYSIS`
- Full MRI preprocessing status: `BATCH_COMPLETED`
- Site/scanner drift status: `SKIP_WITH_JUSTIFICATION`
- Subgroup audit status: `PASSED`
- XAI status: `PASSED`
- Method X-Cog status: `PASSED`
- Method X-Cog best ROC-AUC: `0.7871`
- Method X-Cog AUC gain over fixed tabular: `0.0536`
- Method X-Cog best non-MRI variant: `fixed_tabular_plus_all_cognitive`
- Method X-Cog best non-MRI ROC-AUC: `0.7851`
- Method Y PET inventory status: `AGGREGATE_ONLY`
- Method Y amyloid PET image files counted: `5661`
- Method Y tau PET image files counted: `1576`
- Method Y multimodal design status: `DESIGN_READY_NOT_TRAINED`
- Method Y clinical Alzheimer output allowed now: `False`
- Implementation gap summary: `{'DONE': 10, 'MISSING': 2, 'PARTIAL': 2, 'SKIP_WITH_JUSTIFICATION': 1}`

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
