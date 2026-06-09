# Data Schema Policy

## Confirmed Local Data Root

- Windows path: `<LOCAL_PATH_REDACTED> Learn`
- Expected WSL path: `<LOCAL_PATH_REDACTED> Learn`

## Confirmed Required Sources

- AB label: `Clinical/External Data/imaging_PET_VA.csv`
- Tau SUVR: `Clinical/External Data/imaging_SUVR_tau.csv`
- T1 timing and volumetric metadata: `Clinical/External Data/imaging_volumetric_mri.csv`
- Subject tabular source: `Clinical/Derived Data/SUBJINFO.csv`
- Visit fallback source: `Clinical/Derived Data/SV.csv`
- T1 MRI directory: `T1`

## Confirmed Columns And Timing Hierarchy

- AB label column: `overall_score`
- PET day column: `scan_date_DAYS_T0`
- Tau region column: `brain_region`
- Tau target candidates: `suvr_cer`, `suvr_persi`, `suvr_crus`
- Primary T1 timing: match T1 filename scan number to `SV.csv` column `VISITCD`, then use `SVUSEDTC_DAYS_T0` with fallback `SVSTDTC_DAYS_T0`
- Secondary T1 timing: `imaging_volumetric_mri.csv` column `Date_DAYS_T0`
- T1 join keys: `SUBSTUDY`, `<RAW_ID_FIELD>`, `VISCODE`
- Visit fallback day columns: `SVUSEDTC_DAYS_T0`, `SVSTDTC_DAYS_T0`
- Tabular mapping:
  - `age <- AGEYR`
  - `sex <- SEX`
  - `education_years <- EDCCNTU`
  - `apoe4_carrier <- APOEGN`
  - `mmse_base <- MMSETSV6`, fallback `MMSETSV1`

## Privacy Handling

Only hashed identifiers may be written to processed artifacts. Raw `<RAW_ID_FIELD>`, `<RAW_ID_FIELD>`, `<RAW_ID_FIELD>`, and raw image paths must not be printed in logs.
