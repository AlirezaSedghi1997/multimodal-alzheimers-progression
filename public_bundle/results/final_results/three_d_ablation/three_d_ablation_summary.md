# 3D Fusion Ablation Summary

| variant | mode | folds | resolution | roc_auc | pr_auc | brier | external_like_roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|
| concat | FULL_3D_TRAINING | 5 | 96 | 0.7049076371657017 | 0.8227867993199433 | 0.2266999288423956 | None |
| cross_attention | FULL_3D_TRAINING | 5 | 96 | 0.7206882723011755 | 0.8346241781623293 | 0.22412379434280275 | None |
| cross_attention_latent_alignment_0_15 | FULL_3D_TRAINING | 5 | 96 | 0.7195353034062711 | 0.838495953963556 | 0.22185283817706195 | None |

- best_by_cv_roc_auc: `{'variant': 'cross_attention', 'mode': 'FULL_3D_TRAINING', 'folds': 5, 'cache_resolution': 96, 'roc_auc': 0.7206882723011755, 'pr_auc': 0.8346241781623293, 'brier': 0.22412379434280275, 'external_like_roc_auc': None, 'uses_pet_as_input': False, 'uses_raw_identifiers': False, 'final_claim_allowed': False, 'phase': 'final', 'requested_resolution': 96}`
- contains_pet_input: `False`
- contains_raw_identifiers: `False`
- public_output_granularity: `aggregate_variant_level`
