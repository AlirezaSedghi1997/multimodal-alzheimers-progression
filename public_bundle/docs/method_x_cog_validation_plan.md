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

## Execution Status 2026-05-25

The non-MRI Method X-Cog stage has been completed on branch `method-x-cog` at commit `cf3f83d Run parallel Method X-Cog analysis`.

Aggregate feature coverage:

- cognitive feature rows: `1700`
- core cognitive complete: `1700/1700`
- clinical severity complete: `1700/1700`
- complaints complete: `1696/1700`
- Cogstate complete: `1657/1700`
- mood/anxiety complete: `1661/1700`

Current exploratory ablation refresh:

- fixed tabular reference in Method X-Cog ablation: ROC-AUC `0.7335`, PR-AUC `0.8538`
- best non-MRI variant: `fixed_tabular_plus_all_cognitive`
- best non-MRI ROC-AUC: `0.7851`
- best non-MRI PR-AUC: `0.8771`
- best non-MRI Brier: `0.1896`
- best non-MRI AUC gain over Method X-Cog fixed tabular: `+0.0516`
- best overall variant with MRI-feature surrogate: `mri_feature_surrogate_plus_cognitive`
- best overall ROC-AUC: `0.7871`
- best overall PR-AUC: `0.8732`

Remaining MRI-dependent step: final `MRI 96/3D + cognitive` fusion after SPM/MRI preprocessing is complete.

## Additional Required Comparisons

The executed ablation table also reports clinical severity, complaints, Cogstate, and mood/anxiety families separately. These are exploratory family-level comparisons and must not be mixed with the primary Method X claim.
