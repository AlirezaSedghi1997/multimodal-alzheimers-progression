# Method X Operating Contract

## Non-Negotiable Rules

- Method: `X`.
- Primary task: `AB_POS_vs_AB_NEG`.
- Secondary task: `TAU_BURDEN_REGRESSION`, exploratory only.
- PET amyloid and PET tau are label/target sources only.
- PET images, PET-derived volumes, PET-derived regions, and site/scanner IDs are forbidden model inputs.
- Raw data under `<LOCAL_PATH_REDACTED> Learn` is read-only.
- Raw identifiers must not appear in logs, reports, public bundles, or final tables.

## Allowed Inputs

- Imaging: structural T1 MRI only.
- Tabular: `age`, `sex`, `education_years`, `apoe4_carrier`, `mmse_base`.

## Required Outputs

- Aggregate inventory.
- Master catalog with hashed identifiers.
- Frozen governance policies.
- Gate approval packets.
- Aggregate metrics, figures, and final reports.

