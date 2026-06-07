# Method X-Cog/SF Internal Stress Tests

- status: `COMPLETED`
- branch: `method_x_cog_sf`
- confound_controlled_cv: `REQUIRED`
- disclaimer: Internal decoupling / stress-test only. SF is not external validation and must never be described as independent or quasi-external validation.

## Catalog counts

| substudy | n | positive_rate |
|---|---:|---:|
| A4 | 1163 | 1.0 |
| LEARN | 536 | 0.0 |
| SF | 2785 | 0.069659 |

- pooled Cramér's V (substudy vs label): `0.8995985490049206`
- SF-only Cramér's V: `None`
- enrolled-only Cramér's V: `1.0`

## SF primary vs sensitivity label discrepancy

- SF rows: `2785`
- primary positive rate: `0.069659`
- sensitivity positive rate: `0.057451`
- agree/disagree: `2749` / `36`

## Scientific interpretation (required reading)

- CV protocol: sf_inclusive_confound_controlled_5fold_cv uses patient-grouped 5-fold CV on the SF-inclusive catalog. It is confound-aware evaluation, not a claim of confound-free accuracy.
- SF-only: Within SF-only screening, cognitive features add little beyond fixed tabular.
- Pooled SF-inclusive: High pooled AUC on SF-inclusive data must not be read as biological proof; residual cohort-label coupling remains (Cramér's V≈0.90).
- Internal decoupling: Train-SF / test-enrolled cognitive transfer collapses (~0.51 AUC), indicating cognitive features do not generalize across substudy contexts and must not support Method X-Strict claims.

## Stress-test arms

### sf_only_5fold_cv (cv)

| variant | status | n | roc_auc | pr_auc | brier |
|---|---|---:|---:|---:|---:|
| fixed_tabular_only | OK | 2785 | 0.7152832763690332 | 0.16351345607882284 | 0.2111925939002174 |
| fixed_tabular_plus_all_cognitive | OK | 2785 | 0.7017550840140534 | 0.15399625120296587 | 0.20652147012201813 |

### sf_inclusive_confound_controlled_5fold_cv (cv)

| variant | status | n | roc_auc | pr_auc | brier |
|---|---|---:|---:|---:|---:|
| fixed_tabular_only | OK | 4484 | 0.7118228121769201 | 0.5043671496083556 | 0.2144466559412682 |
| fixed_tabular_plus_all_cognitive | OK | 4484 | 0.8307394247784586 | 0.7259858848459966 | 0.16166526827804018 |

### train_sf_test_enrolled_internal_decoupling (train_sf_test_enrolled)

| variant | status | n | roc_auc | pr_auc | brier |
|---|---|---:|---:|---:|---:|
| fixed_tabular_only | OK | 1699 | 0.7310240499993583 | 0.8476059672555291 | 0.21712359684553764 |
| fixed_tabular_plus_all_cognitive | OK | 1699 | 0.5149301857009022 | 0.7100741583718183 | 0.3962894348418715 |

### train_enrolled_test_sf_internal_decoupling (train_enrolled_test_sf)

| variant | status | n | roc_auc | pr_auc | brier |
|---|---|---:|---:|---:|---:|
| fixed_tabular_only | OK | 2785 | 0.7172398906603747 | 0.16922646639217703 | 0.22444030190695358 |
| fixed_tabular_plus_all_cognitive | OK | 2785 | 0.6970062906094451 | 0.1594964516769602 | 0.15768111120685244 |

