# v3.6-validity-fixes Governance Policy

This policy adds validity-first safeguards to Method X. It does not rename Method X and does not allow PET as model input.

## Non-negotiable constraints

- Method X imaging input remains T1 MRI only.
- PET amyloid/tau is label/target only.
- PET-derived values such as SUVR and centiloid are forbidden as Method X / Method X-Cog model inputs.
- The current external-like holdout must be reported as a locked pseudo-external holdout, not a truly independent external cohort.
- DeLong testing is blocked until a truly independent binary external set exists.
- Calibration fitting is forbidden on test or external-like data.
- Public bundle remains aggregate-only.

## Required new stages

1. `validity-audit` before final claim.
2. `temporal-window-sensitivity` before deciding whether all aligned rows or a strict PET–T1 window should be the mainline.
3. `optuna-nested` for thesis-grade HPO if the model is used for final claims.
4. `external-final` with pseudo-external terminology and DeLong blocked.
5. `xai-final` only after the final selected 96/3D model.
6. `thesis-report` after all required caveats and audit outputs exist.

## Claim rule

A strong Method X claim is allowed only if validity audit supports measurable MRI signal and does not show dominant cohort/protocol artifact. Otherwise, the final thesis must frame the result as a reproducible, governance-safe, pseudo-external biomarker-status pipeline under strong cohort-label confounding.
