# Method X-Cog Summary

- status: `COMPLETED`
- best_variant: `fixed_tabular_plus_mood_anxiety`
- best_C: `0.03`
- n: `1360`
- feature_count: `9`
- roc_auc: `0.7434`
- pr_auc: `0.8496`
- brier: `0.1852`
- improvement_over_fixed_tabular_auc: `0.023227056764191012`
- uses_pet_as_input: `False`
- uses_raw_identifiers: `False`
- method_x_claim_unchanged: `True`
- method_x_cog_exploratory: `True`
- cohort_label_confound_warning: `True`

## Best By Variant

| variant | status | n | feature_count | roc_auc | pr_auc | brier |
|---|---|---:|---:|---:|---:|---:|
| cognitive_core_only | OK | 1360 | 5 | 0.5702 | 0.7387 | 0.2174 |
| fixed_tabular_only | OK | 1360 | 5 | 0.7201 | 0.8422 | 0.1895 |
| fixed_tabular_plus_all_cognitive | OK | 1360 | 30 | 0.7102 | 0.8142 | 0.4475 |
| fixed_tabular_plus_clinical_severity | OK | 1360 | 13 | 0.6844 | 0.8025 | 0.4841 |
| fixed_tabular_plus_cognitive_core | OK | 1360 | 10 | 0.7065 | 0.8236 | 0.1969 |
| fixed_tabular_plus_cogstate | OK | 1360 | 11 | 0.7196 | 0.8254 | 0.1969 |
| fixed_tabular_plus_complaints | OK | 1360 | 7 | 0.7106 | 0.8292 | 0.1971 |
| fixed_tabular_plus_mood_anxiety | OK | 1360 | 9 | 0.7434 | 0.8496 | 0.1852 |
| mri_feature_surrogate_plus_cognitive | OK | 1360 | 44 | 0.6593 | 0.7883 | 0.4482 |
