# Decision Curve Analysis

- status: `COMPLETED`
- claim boundary: Exploratory clinical utility framing only; not Alzheimer diagnosis claim.

## Method X-Strict fixed tabular

- n: `1699`
- thresholds: `[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]`

| threshold | net benefit (model) | treat all | treat none |
|---:|---:|---:|---:|
| 0.05 | 0.6679 | 0.6679 | 0.0000 |
| 0.10 | 0.6495 | 0.6495 | 0.0000 |
| 0.15 | 0.6288 | 0.6288 | 0.0000 |
| 0.20 | 0.6009 | 0.6057 | 0.0000 |
| 0.25 | 0.5529 | 0.5794 | 0.0000 |
| 0.30 | 0.5134 | 0.5493 | 0.0000 |
| 0.35 | 0.4701 | 0.5146 | 0.0000 |
| 0.40 | 0.4208 | 0.4742 | 0.0000 |
| 0.45 | 0.3919 | 0.4264 | 0.0000 |
| 0.50 | 0.3584 | 0.3690 | 0.0000 |

## Method Z

- status: `METRICS_ONLY` — `Method Z core blood (CV metrics only; DCA curve requires stored OOF)`
- enrolled ROC-AUC (metrics only): `0.8085818422796553`
