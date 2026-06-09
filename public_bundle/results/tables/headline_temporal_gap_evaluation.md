# Headline Temporal-Gap Evaluation

CPU baselines on gap-filtered catalogs. 3D CNN: full-cohort reference only (not retrained per window).

- gap source: `abs_pet_t1_gap_days`

| window | n | tabular AUC [95% CI] | ROI AUC [95% CI] | 3D overlap n |
|---:|---:|---|---|---:|
| 0 | 1699 | 0.7289 [0.702, 0.756] | 0.7535 [0.724, 0.780] | 1699 |
| 60 | 154 | 0.7969 [0.703, 0.878] | 0.8028 [0.728, 0.873] | 154 |
| 90 | 266 | 0.7435 [0.645, 0.836] | 0.7880 [0.727, 0.845] | 266 |
| 120 | 330 | 0.7585 [0.673, 0.829] | 0.7710 [0.722, 0.823] | 330 |

## Interpretation

Window≤60d (n≈154) may show higher point AUC than full cohort, but bootstrap CIs often overlap baseline. Do not interpret subset AUC without CI and cohort descriptives.
