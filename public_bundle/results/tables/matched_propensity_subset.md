# Matched / Propensity-Weighted Subset Analysis

- status: `COMPLETED`
- method: `greedy_1to1_covariate_nearest_neighbor`
- n_pairs: `536`

## Full cohort vs matched 1:1

- full cohort ROC-AUC: `0.7314427432912822` [0.7082529099985883, 0.7562719773873539]
- matched subset ROC-AUC: `0.7243017654265983` [0.6921678338716863, 0.7559864216696369]
- point difference (full − matched): `0.0071`
- 95% CI overlap: `True`

## Interpretation

Matched analysis is a covariate-balance sensitivity check only. A ~0.007 AUC difference with overlapping CIs does **not** support a meaningful performance shift; pseudo-external holdout rules are unchanged.
