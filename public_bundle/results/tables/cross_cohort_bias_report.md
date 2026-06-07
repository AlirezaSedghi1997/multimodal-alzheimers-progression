# Cross-Cohort Bias Report

- risk_level: `HIGH`
- baseline_roc_auc: `0.731167791947987`
- cohort_label_cramers_v: `1.0000`
- all_cohorts_single_label: `True`
- label_by_substudy: `{'A4': {'1': 1163}, 'LEARN': {'0': 536}}`
- raw_identifiers_used: `False`

## Interpretation

Do not interpret current AUC as pathology-only generalization. Report cohort-label confounding, keep external holdout labeled pseudo-external, and require sensitivity analyses before final claims.

## Required Reporting Language

The local A4/LEARN catalog has strong cohort-label coupling; the locked holdout is pseudo-external, not an independent external validation cohort.
