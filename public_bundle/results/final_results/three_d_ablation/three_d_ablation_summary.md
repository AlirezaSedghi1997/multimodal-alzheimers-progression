# 3D Fusion Ablation Summary

| variant | mode | folds | resolution | roc_auc | pr_auc | brier | external_like_roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|
| concat | FULL_3D_TRAINING | 5 | 96 | 0.6855753565430984 | 0.8102096273080849 | 0.23399824938027894 | 0.6955196341903654 |
| cross_attention | FULL_3D_TRAINING | 5 | 96 | 0.7215931022382636 | 0.8395955560866111 | 0.21077718555140285 | 0.719746500340941 |
| cross_attention+latent_alignment_0.15 | FULL_3D_TRAINING | 5 | 96 | 0.7165801939995489 | 0.8332365551321125 | 0.22241471082066433 | 0.7244795635955237 |

- best_by_cv_roc_auc: `{'variant': 'cross_attention', 'mode': 'FULL_3D_TRAINING', 'folds': 5, 'cache_resolution': 96, 'roc_auc': 0.7215931022382636, 'pr_auc': 0.8395955560866111, 'brier': 0.21077718555140285, 'external_like_roc_auc': 0.719746500340941, 'uses_pet_as_input': False, 'uses_raw_identifiers': False, 'final_claim_allowed': False}`
- contains_pet_input: `False`
- contains_raw_identifiers: `False`
- public_output_granularity: `aggregate_variant_level`
