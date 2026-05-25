# Research Methods, Limitations, Ethics/Privacy, and Statistics Addendum

This addendum is the current thesis/proposal text block for the non-MRI work that can be completed while SPM/MRI preprocessing continues.

## Method X Core Boundary

The primary thesis method remains Method X: amyloid PET status is the label/target source, not a model input. The primary imaging input is T1 structural MRI. The fixed tabular inputs are age, sex, education years, APOE4 carrier status, and baseline MMSE. PET images, PET SUVR, centiloid, amyloid-derived values, tau-derived values, raw image paths, and raw identifiers are forbidden as Method X model inputs.

## Method X-Cog Extension

Method X-Cog is an exploratory extension that evaluates incremental value from cognitive and psychological test summaries. It does not replace the Method X claim. The current completed non-MRI comparisons include fixed-tabular, cognitive-core-only, fixed-tabular plus cognitive core, fixed-tabular plus all cognitive features, clinical severity, complaints, Cogstate, mood/anxiety, and MRI-feature-surrogate plus cognitive when MRI feature summaries are available.

The current aggregate result is exploratory: within the Method X-Cog ablation, the fixed-tabular reference has ROC-AUC 0.7335 and PR-AUC 0.8538. The best non-MRI variant is fixed_tabular_plus_all_cognitive with ROC-AUC 0.7851, PR-AUC 0.8771, Brier 0.1896, and AUC gain +0.0516 over the Method X-Cog fixed-tabular reference. The best overall variant, mri_feature_surrogate_plus_cognitive, has ROC-AUC 0.7871 and PR-AUC 0.8732. These results must be reported as catalog/split-specific exploratory evidence only, not as a final thesis claim until MRI 96/3D fusion is complete.

## Method Y PET Exploratory Boundary

Method Y can inspect PET availability, PET QC readiness, label reproducibility, and aggregate sensitivity. Method Y cannot be described as Method X, cannot feed PET images or PET-derived values into Method X/Method X-Cog, and cannot publish subject-level PET outputs.

A separate PET-image model is technically possible, but it would answer a different research question and must be labeled as a separate Method Y/PET-supervised or PET-assisted model. It cannot be used to support the non-invasive Method X claim because PET would no longer be only a target source.

## Statistical Reporting

All model comparisons must report ROC-AUC, PR-AUC, Brier score, and confidence intervals when available. Cross-validation results must not use DeLong tests as if folds were independent. The locked holdout is pseudo-external because it is derived from the same A4/LEARN source, so it must not be described as a fully independent external cohort.

Cohort-label confounding is severe in the locked catalog: A4 and LEARN are strongly coupled to amyloid label. Reports must preserve this caveat and avoid clinical overclaiming.

## Ethics and Privacy

Raw identifiers may be used transiently in memory for joins but must not be written to processed feature tables, logs, public bundles, model outputs, or reports. Public artifacts must be aggregate-only or hash-level only. No raw clinical rows, raw MRI/PET paths, NIfTI/HDF5/cache files, checkpoints, or subject-level predictions are allowed in public outputs.

## Pending After MRI/SPM

Final thesis-level interpretation remains pending until the final MRI 96/3D cache, 5-fold 3D ablation, MRI-only vs MRI+cognitive comparison, XAI refresh, final analysis, gate matrix, and public bundle refresh are complete.
