# Clinical Endpoint Branch — Readiness Audit

- status: **`EXPLORATORY_FEASIBILITY_ONLY`**
- total rows: `4484`
- enrolled (A4/LEARN): `1699`
- SF rows: `2785`
- cohort-label Cramér's V (catalog): `0.8995985490049206`

## Claim boundary

Exploratory clinical/cognitive progression proxy only. Do not call Alzheimer diagnosis or conversion-to-AD without independent validated label.

Enrolled A4/LEARN rows show structural feasibility flags for progression endpoints, but endpoint labels are not frozen and no models are trained. This branch must not be used to support Method X amyloid-status claims.

## Substudy feasibility

| substudy | rows | CDR worsening feasible | PACC decline feasible | has CDR series | has PACC series |
|---|---:|---:|---:|---:|---:|
| A4 | 1163 | 1163 | 1163 | 1163 | 1163 |
| LEARN | 536 | 536 | 536 | 536 | 536 |
| SF | 2785 | 0 | 0 | 0 | 0 |

## Blockers

- **`LONGITUDINAL_ENDPOINT_LABELS_NOT_FROZEN`** — CDR-worsening and PACC-decline endpoint definitions are not yet frozen in governance.
- **`PROGRESSION_MODELING_NOT_STARTED`** — No progression models trained; branch remains feasibility-only.
- **`SF_LONGITUDINAL_COVERAGE_LIMITED`** — SF rows lack the same longitudinal endpoint coverage as enrolled A4/LEARN MRI sample.

## Required before modeling

- Freeze CDR-worsening and PACC-decline endpoint definitions in governance
- Use baseline-only predictors with non-future visit discipline
- Keep separate claim boundary from Method X-Strict amyloid biomarker status
- Report SF longitudinal limitation explicitly if SF rows are included

## Forbidden claims

- Alzheimer disease diagnosis
- conversion-to-AD
- clinical AD classification
