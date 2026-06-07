# Final Write-Up: Method X-Cog and Method Y-PET/MM

Status date: 2026-05-25.

This document is the current final write-up block for the exploratory cognitive extension and the PET-assisted multimodal design branch. It does not replace the primary Method X thesis claim.

## Primary Boundary

Method X remains the main thesis method. Its allowed inputs are T1 MRI plus the fixed baseline tabular variables: age, sex, education_years, apoe4_carrier, and mmse_base. PET is never a Method X model input. Amyloid PET is used only as the label or target source for A-beta positive vs A-beta negative biomarker status.

The current outputs should therefore be described as biomarker-status prediction. They must not be worded as definitive clinical Alzheimer disease diagnosis unless an independent clinical diagnosis or conversion label is built and validated separately.

## Method X-Cog Write-Up

Method X-Cog was implemented as an exploratory extension to test whether cognitive and psychological summaries add signal beyond the fixed Method X tabular inputs. The extension preserves the Method X input boundary: PET images, PET SUVR, centiloid values, PET-derived regions, raw identifiers, raw paths, and future visits are forbidden model inputs.

Cognitive feature construction produced 1700 rows for the locked catalog. Core cognitive features were complete for 1700/1700 rows, clinical severity for 1700/1700, complaints for 1696/1700, Cogstate for 1657/1700, and mood/anxiety for 1661/1700. Alignment used locked catalog membership and non-future visit selection when timing information was available, reducing the risk of future-information leakage.

The refreshed ablation table evaluated cognitive-only, fixed-tabular-only, fixed-tabular plus cognitive families, fixed-tabular plus all cognitive features, and an MRI-feature-surrogate plus cognitive variant. The primary Method X fixed-tabular baseline table reports ROC-AUC 0.7312 and PR-AUC 0.8526. Within the refreshed Method X-Cog ablation run, the fixed-tabular reference is ROC-AUC 0.7335 and PR-AUC 0.8538; the X-Cog gain should be read against this ablation reference.

The best non-MRI Method X-Cog variant is `fixed_tabular_plus_all_cognitive`, with ROC-AUC 0.7851, PR-AUC 0.8771, Brier 0.1896, and AUC gain +0.0516 over the Method X-Cog fixed-tabular reference. The best overall exploratory variant is `mri_feature_surrogate_plus_cognitive`, with ROC-AUC 0.7871 and PR-AUC 0.8732. Because this overall variant uses MRI-feature surrogate summaries, it must be reported separately from the non-MRI cognitive ablation.

Interpretation: these results support the presence of additional catalog-specific predictive signal in cognitive and psychological summaries. They are exploratory and should not be presented as final clinical validation. The locked holdout is pseudo-external, not a fully independent external cohort, and A4/LEARN cohort-label coupling remains a major limitation.

## Method Y-PET Inventory Write-Up

Method Y is a separate PET-assisted exploratory branch. Phase 0 has run only as aggregate inventory and QC. It did not start PET preprocessing, did not train a PET model, and did not feed PET images, SUVR, centiloid, tau, or PET-derived features into Method X or Method X-Cog.

The locked catalog contains 1700 rows. Aggregate PET inventory counted 5661 amyloid FBP image files, with 1698 locked-patient matches in memory. It also counted 1576 tau FTP image files, with 447 locked-patient matches in memory. These counts are availability and QC evidence only. They are not model inputs and do not alter the Method X non-invasive claim.

Public Method Y artifacts are aggregate-only. Raw identifiers, raw paths, raw clinical rows, PET/MRI images, HDF5/cache files, checkpoints, and subject-level predictions are not public artifacts.

## Method Y-PET/MM Design Write-Up

Method Y-PET/MM is design-ready but not trained. Its status is `DESIGN_READY_NOT_TRAINED`; PET preprocessing has not started and model training has not started.

The design separates four reporting scenarios:

| scenario | inputs | reporting boundary |
|---|---|---|
| MRI-only | T1 MRI | Method X-compatible only when PET is not an input |
| PET-only | amyloid PET image and optional tau PET image | Method Y only |
| MRI+PET | T1 MRI plus PET image inputs | Method Y only |
| MRI+PET+cognitive | T1 MRI, PET image inputs, cognitive summaries | Method Y only |

The currently approved target is A-beta positive vs A-beta negative biomarker status. A clinical Alzheimer disease output is blocked until an independent clinical label is defined, such as clinical diagnosis at visit, CN/MCI/AD dementia category, conversion to MCI or AD dementia on follow-up, or adjudicated clinical diagnosis.

Before any Method Y-PET/MM training, the project must create a separate PET preprocessing root, add regression tests proving that Method X loaders do not load PET tensors or PET-derived tabular values, define the target policy, and keep public outputs aggregate-only unless a separate privacy review approves hash-level release.

## Approved Reporting Language

Use:

- Method X predicts A-beta biomarker positivity from T1 MRI plus approved fixed tabular variables.
- Method X-Cog is an exploratory cognitive extension and does not replace the Method X claim.
- Method Y PET inventory is aggregate-only and does not use PET as a Method X input.
- Method Y-PET/MM is design-ready, not trained, and separate from Method X.

Avoid:

- Claiming that AB-positive vs AB-negative is definitive Alzheimer disease diagnosis.
- Calling PET-assisted models non-invasive Method X.
- Merging PET-only, MRI+PET, or MRI+PET+cognitive results into the primary Method X claim.
- Reporting subject-level PET availability, raw identifiers, raw paths, image files, HDF5/cache files, or checkpoints in public artifacts.

## Next Executable Gate

The next thesis-critical execution remains the MRI/SPM-dependent stage: final MRI 96/3D cache, 5-fold 3D ablation, final MRI 96/3D plus cognitive fusion, XAI refresh, final analysis refresh, gate matrix refresh, and public bundle refresh. Before SPM reaches 1700 completed files, only aggregate-safe docs, tests, orchestration updates, and the explicit MRI volume cache smoke check are allowed; the smoke check is not a training cache and must not be used for claims. Method Y clinical Alzheimer output remains blocked until an independent clinical label is available.
