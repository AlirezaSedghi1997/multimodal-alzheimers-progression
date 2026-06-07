# 3D Fusion Ablation Summary

| variant | mode | folds | resolution | roc_auc | pr_auc | brier | external_like_roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|
| concat | FULL_3D_TRAINING | 5 | 96 | 0.6932050028824223 | 0.8184680772618702 | 0.22868499040315612 | 0.7108820344149853 |
| cross_attention | FULL_3D_TRAINING | 5 | 96 | 0.7164749229265358 | 0.8446536115484701 | 0.2051277199778236 | 0.7282499699169709 |
| cross_attention+latent_alignment_0.15 | FULL_3D_TRAINING | 5 | 96 | 0.7173120786024012 | 0.8332888445983319 | 0.21921396907630766 | 0.7164975331916088 |

- best_by_cv_roc_auc: `{'variant': 'cross_attention+latent_alignment_0.15', 'mode': 'FULL_3D_TRAINING', 'folds': 5, 'cache_resolution': 96, 'roc_auc': 0.7173120786024012, 'pr_auc': 0.8332888445983319, 'brier': 0.21921396907630766, 'external_like_roc_auc': 0.7164975331916088, 'uses_pet_as_input': False, 'uses_raw_identifiers': False, 'final_claim_allowed': False, 'phase': 'final', 'requested_resolution': 96}`
- contains_pet_input: `False`
- contains_raw_identifiers: `False`
- public_output_granularity: `aggregate_variant_level`
