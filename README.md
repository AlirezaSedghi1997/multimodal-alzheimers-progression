# amyloid-tau-prediction-a4learn

Method X implementation scaffold for A4/LEARN amyloid status prediction.

Core contract:
- PET is label/target only, never model input.
- T1 MRI is the only imaging input.
- Tabular inputs are limited to age, sex, education_years, apoe4_carrier, and mmse_base.
- Raw identifiers and raw patient data must not be printed in logs or public outputs.

Exploratory extensions:
- `Method X-Cog` may add cognitive and psychological test summaries as a separate extension. It does not replace the Method X claim.
- `Method Y` may inspect PET availability or QC as exploratory support only. PET remains prohibited as a Method X model input.

Default local data root:
- Windows: `H:\A4 Learn`
- WSL: `/mnt/h/A4 Learn`

