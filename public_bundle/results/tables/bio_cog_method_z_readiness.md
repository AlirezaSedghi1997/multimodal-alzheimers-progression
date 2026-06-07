# Bio-Cog / Method Z Readiness (Phase 3)

- status: `FEASIBILITY_ONLY_NOT_TRAINED`
- rows: `3010`
- claim boundary: Minimally invasive biomarker-assisted branch; never supports the non-invasive Method X claim.

## Feature coverage

| feature | present | coverage_rate |
|---|---:|---:|
| pTau217 | 1483 | 0.492691 |
| GFAP | 2337 | 0.776412 |
| NfL | 0 | 0.0 |
| pTau181 | 0 | 0.0 |
| Ab42 | 0 | 0.0 |
| Ab40 | 0 | 0.0 |
| Ab42_40_ratio | 0 | 0.0 |

## Blockers

- `NFL_COLUMN_EMPTY_IN_LOCKED_CATALOG`
- `PTAU181_COLUMN_EMPTY_IN_LOCKED_CATALOG`
- `AB42_40_RATIO_NOT_POPULATED_IN_LOCKED_CATALOG`

## Required before training

- Frozen blood-feature allowlist and no-PET-input test
- SF-aware internal stress-test protocol for blood branch
- Locked pseudo-external evaluation plan separate from Method X
- Permutation/ablation for blood biomarker families
