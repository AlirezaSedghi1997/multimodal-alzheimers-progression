# Method X-Cog Summary

- status: `COMPLETED`
- best_variant: `mri_feature_surrogate_plus_cognitive`
- best_C: `1.0`
- n: `1360`
- feature_count: `44`
- roc_auc: `0.7871`
- pr_auc: `0.8732`
- brier: `0.1882`
- improvement_over_fixed_tabular_auc: `0.05357214303575897`
- best_non_mri_variant: `fixed_tabular_plus_all_cognitive`
- best_non_mri_roc_auc: `0.7851`
- best_non_mri_pr_auc: `0.8771`
- non_mri_improvement_over_fixed_tabular_auc: `0.05159164791197801`
- best_overall_may_use_mri_surrogate_features: `True`
- uses_pet_as_input: `False`
- uses_raw_identifiers: `False`
- method_x_claim_unchanged: `True`
- method_x_cog_exploratory: `True`
- cohort_label_confound_warning: `True`

## Best By Variant

| variant | status | n | feature_count | roc_auc | pr_auc | brier |
|---|---|---:|---:|---:|---:|---:|
| cognitive_core_only | OK | 1360 | 5 | 0.5723 | 0.7442 | 0.2454 |
| fixed_tabular_only | OK | 1360 | 5 | 0.7335 | 0.8538 | 0.2092 |
| fixed_tabular_plus_all_cognitive | OK | 1360 | 30 | 0.7851 | 0.8771 | 0.1896 |
| fixed_tabular_plus_clinical_severity | OK | 1360 | 13 | 0.7393 | 0.8516 | 0.2080 |
| fixed_tabular_plus_cognitive_core | OK | 1360 | 10 | 0.7322 | 0.8514 | 0.2095 |
| fixed_tabular_plus_cogstate | OK | 1360 | 11 | 0.7540 | 0.8656 | 0.2030 |
| fixed_tabular_plus_complaints | OK | 1360 | 7 | 0.7386 | 0.8563 | 0.2080 |
| fixed_tabular_plus_mood_anxiety | OK | 1360 | 9 | 0.7617 | 0.8618 | 0.2000 |
| mri_feature_surrogate_plus_cognitive | OK | 1360 | 44 | 0.7871 | 0.8732 | 0.1882 |
