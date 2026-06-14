# Final Write-Up: Method X-Cog and Method Y-PET/MM

Status date: 2026-06-14.

This document is the current final write-up block for the exploratory cognitive extension and the PET-assisted multimodal design branch. It does not replace the primary Method X thesis claim.

## Primary Boundary

Method X remains the main thesis method. Its allowed inputs are T1 MRI plus the fixed baseline tabular variables: age, sex, education_years, apoe4_carrier, and mmse_base. PET is never a Method X model input. Amyloid PET is used only as the label or target source for A-beta positive vs A-beta negative biomarker status.

The current outputs should therefore be described as **early prediction of amyloid biomarker status and preclinical risk**. They must not be worded as definitive clinical Alzheimer disease diagnosis unless an independent clinical diagnosis or conversion label is built and validated separately.

## Method X-Cog Write-Up

Method X-Cog was implemented as an exploratory extension to test whether cognitive and psychological summaries add signal beyond the fixed Method X tabular inputs. The extension preserves the Method X input boundary: PET images, PET SUVR, centiloid values, PET-derived regions, raw identifiers, raw paths, and future visits are forbidden model inputs.

Cognitive feature construction produced 1700 rows for the locked catalog. Core cognitive features were complete for 1700/1700 rows, clinical severity for 1700/1700, complaints for 1696/1700, Cogstate for 1657/1700, and mood/anxiety for 1661/1700. Alignment used locked catalog membership and non-future visit selection when timing information was available, reducing the risk of future-information leakage.

The refreshed ablation table evaluated cognitive-only, fixed-tabular-only, fixed-tabular plus cognitive families, fixed-tabular plus all cognitive features, and an MRI-feature-surrogate plus cognitive variant. The primary Method X fixed-tabular baseline table reports ROC-AUC 0.7312 and PR-AUC 0.8526. Within the refreshed Method X-Cog ablation run, the fixed-tabular reference is ROC-AUC 0.7335 and PR-AUC 0.8538; the X-Cog gain should be read against this ablation reference.

The best non-MRI Method X-Cog variant is `fixed_tabular_plus_all_cognitive`, with ROC-AUC 0.7851, PR-AUC 0.8771, Brier 0.1896, and AUC gain +0.0516 over the Method X-Cog fixed-tabular reference. The best overall exploratory variant is `mri_feature_surrogate_plus_cognitive`, with ROC-AUC 0.7871 and PR-AUC 0.8732. Because this overall variant uses MRI-feature surrogate summaries, it must be reported separately from the non-MRI cognitive ablation.

Interpretation: these results support the presence of additional catalog-specific predictive signal in cognitive and psychological summaries. They are exploratory and should not be presented as final clinical validation. The locked holdout is pseudo-external, not a fully independent external cohort, and A4/LEARN cohort-label coupling remains a major limitation.

## Method Y-PET Inventory and Training Write-Up

Method Y is a separate PET-assisted exploratory branch. PET preprocessing has been completed, and Method Y training has been executed with all incremental modalities.

The locked catalog contains 1700 rows. Aggregate PET inventory counted 5661 amyloid FBP image files, with 1698 locked-patient matches in memory. It also counted 1576 tau FTP image files, with 447 locked-patient matches in memory. These counts are availability and QC evidence only. They are not model inputs and do not alter the Method X non-invasive claim.

Public Method Y artifacts are aggregate-only. Raw identifiers, raw paths, raw clinical rows, PET/MRI images, HDF5/cache files, checkpoints, and subject-level predictions are not public artifacts.

## Method Y Training Results

Method Y has been trained on the paired PET-cache hash intersection (N=184 effective subjects). The incremental value table shows:

| scenario | ROC-AUC | accuracy | sensitivity | specificity | F1 | status |
|---|---|---|---|---|---|---|
| PET-only reference | 0.633 | 0.871 | 0.981 | 0.029 | 0.929 | COMPLETED |
| MRI-only (paired) | 0.571 | 0.881 | 1.000 | 0.000 | 0.936 | COMPLETED |
| Late fusion MRI+PET | 0.533 | 0.881 | 1.000 | 0.000 | 0.936 | COMPLETED |
| Gated fusion MRI+PET | 0.470 | 0.886 | 0.974 | 0.143 | 0.937 | COMPLETED |
| MRI+PET+cognitive | 0.633 | 0.871 | 0.981 | 0.029 | 0.929 | COMPLETED |

**Critical caveat:** Method Y results have PET-label tautology risk because PET is both label source and model input. These results are exploratory only and must not be used for clinical claims.

## Method Y Design Reporting Scenarios

The design separates four reporting scenarios:

| scenario | inputs | reporting boundary |
|---|---|---|
| MRI-only | T1 MRI | Method X-compatible only when PET is not an input |
| PET-only | amyloid PET image and optional tau PET image | Method Y only |
| MRI+PET | T1 MRI plus PET image inputs | Method Y only |
| MRI+PET+cognitive | T1 MRI, PET image inputs, cognitive summaries | Method Y only |

The currently approved target is A-beta positive vs A-beta negative biomarker status. A clinical Alzheimer disease output is blocked until an independent clinical label is defined, such as clinical diagnosis at visit, CN/MCI/AD dementia category, conversion to MCI or AD dementia on follow-up, or adjudicated clinical diagnosis.

## Approved Reporting Language

Use:

- Method X predicts A-beta biomarker positivity from T1 MRI plus approved fixed tabular variables.
- Method X-Cog is an exploratory cognitive extension and does not replace the Method X claim.
- Method Y is PET-assisted exploratory with PET-label tautology caveat; results are not for clinical claims.
- Method Y MRI+PET+cognitive is the best Method Y modality but still exploratory.
- Synthetic MRI-to-PET is supplementary, fold-local, train-only, and never counted as real subjects.
- Augmentation is train-only exposure multiplication, not real-N expansion.

Avoid:

- Claiming that AB-positive vs AB-negative is definitive Alzheimer disease diagnosis.
- Calling PET-assisted models non-invasive Method X.
- Merging PET-only, MRI+PET, or MRI+PET+cognitive results into the primary Method X claim.
- Reporting subject-level PET availability, raw identifiers, raw paths, image files, HDF5/cache files, or checkpoints in public artifacts.
- Using clinical Alzheimer diagnosis framing; use "early prediction of amyloid biomarker status and preclinical risk" instead.

## Next Executable Gate

Method Y training is complete with all incremental modalities. The current focus is on:

1. Final reporting document refresh to ensure consistent framing as "early prediction of amyloid biomarker status and preclinical risk."
2. Public bundle rebuild after report changes.
3. Full pytest verification.
4. Documentation updates for SESSION_HANDOFF.md and PROGRESS.yaml.

Method Y clinical Alzheimer output remains blocked until an independent clinical label is available.
