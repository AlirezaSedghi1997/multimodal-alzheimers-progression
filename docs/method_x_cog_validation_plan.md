# Method X-Cog Validation Plan

Method X-Cog is an exploratory extension of Method X. It evaluates whether cognitive and psychological test summaries add predictive signal beyond the fixed Method X inputs.

Method X remains the primary thesis method and is not replaced by this branch.

## Inputs

Allowed Method X-Cog inputs:

- Method X fixed tabular variables: age, sex, education_years, apoe4_carrier, mmse_base.
- Cognitive core: PACC, FCTOTAL96, LDELTOTAL, DIGITTOTAL, MMSCORE.
- Clinical severity: CDGLOBAL, CDRSB, CDR domain scores.
- Cognitive complaints: CFIPTTOTAL, CFSPTTOTAL.
- Optional computerized cognition: C3Comp, LearnWMComp, OCLONBComp, PsychAttnComp, AttnComp, C3AbrComp.
- Optional mood/anxiety covariates: GDTOTAL, STAITOTAL, CADCNCRN, IESCORE.

Forbidden inputs:

- PET images.
- PET SUVR or amyloid centiloid columns.
- PET-derived regions or volumes.
- Raw identifiers.
- Raw image paths.
- Future visits after the PET/T1 target time.

## Alignment

Clinical and cognitive rows are aligned to each locked Method X catalog sample by subject and visit time. The selected row must be non-future relative to the target time when timing is available. If visit timing is unavailable, exact visit-code matching may be used and must be reported.

## Ablations

Required model comparisons:

- fixed_tabular_only.
- cognitive_core_only.
- fixed_tabular_plus_cognitive_core.
- fixed_tabular_plus_all_cognitive.
- mri_feature_surrogate_plus_cognitive, when MRI feature summaries are available.

All outputs must remain aggregate or hash-level only.
