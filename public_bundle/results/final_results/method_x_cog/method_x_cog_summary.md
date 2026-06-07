# Method X-Cog Summary

- status: `COMPLETED`
- best_variant: `mri_feature_surrogate_plus_cognitive`
- best_C: `3.0`
- n: `1359`
- feature_count: `44`
- roc_auc: `0.7992`
- pr_auc: `0.8910`
- brier: `0.1848`
- improvement_over_fixed_tabular_auc: `0.06777953229566136`
- best_non_mri_variant: `fixed_tabular_plus_all_cognitive`
- best_non_mri_roc_auc: `0.7935`
- best_non_mri_pr_auc: `0.8841`
- non_mri_improvement_over_fixed_tabular_auc: `0.0621525427977041`
- best_overall_may_use_mri_surrogate_features: `True`
- uses_pet_as_input: `False`
- uses_raw_identifiers: `False`
- method_x_claim_unchanged: `True`
- method_x_cog_exploratory: `True`
- cohort_label_confound_warning: `True`

## Best By Variant

| variant | status | n | feature_count | roc_auc | pr_auc | brier |
|---|---|---:|---:|---:|---:|---:|
| cognitive_core_only | OK | 1359 | 5 | 0.5762 | 0.7469 | 0.2455 |
| fixed_tabular_only | OK | 1359 | 5 | 0.7314 | 0.8512 | 0.2099 |
| fixed_tabular_plus_all_cognitive | OK | 1359 | 30 | 0.7935 | 0.8841 | 0.1866 |
| fixed_tabular_plus_clinical_severity | OK | 1359 | 13 | 0.7400 | 0.8574 | 0.2081 |
| fixed_tabular_plus_cognitive_core | OK | 1359 | 10 | 0.7318 | 0.8503 | 0.2100 |
| fixed_tabular_plus_cogstate | OK | 1359 | 11 | 0.7553 | 0.8662 | 0.2028 |
| fixed_tabular_plus_complaints | OK | 1359 | 7 | 0.7376 | 0.8556 | 0.2083 |
| fixed_tabular_plus_mood_anxiety | OK | 1359 | 9 | 0.7624 | 0.8645 | 0.1997 |
| mri_feature_surrogate_plus_cognitive | OK | 1359 | 44 | 0.7992 | 0.8910 | 0.1848 |
