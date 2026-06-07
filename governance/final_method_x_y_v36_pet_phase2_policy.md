# Final Method X/Y v3.8 Literature-Aligned Governance Policy

## Scope
This policy supersedes the PET portions of the prior v3.6/v3.7 packages while preserving the Method X validity fixes.

## Method X firewall
PET images, PET-derived scalars, SUVR, Centiloid, FBP/FBB/FTP tracer features, PET-T1 gap, scan dates, and PET-derived labels as features are forbidden in Method X and Method X-Cog. PET remains label/target source only.

## Method Y PET branch
Method Y may process PET images and may train PET-assisted exploratory models only after readiness gates pass. It must be reported separately from Method X.

## Literature-aligned PET requirements
1. Temporal tiers must include gap30, gap90, gap120, and report-only relaxed tiers.
2. Primary Method Y training excludes PET-T1 gaps greater than 90 days.
3. Training-ready PET preprocessing requires PET→T1→MNI spatial alignment or QC-passed precomputed PETPrep/petBrain/ADprep-style outputs.
4. Whole cerebellum is the primary cross-sectional amyloid reference; cerebellar gray is sensitivity.
5. FBP and FBB tracer-specific SUVR thresholds and Centiloid equations must be documented.
6. Visual or numeric QC is required before cache construction.
7. One PET per patient is required for training.
8. Cohort-label, tracer-label, and gap-label audits are required before training.
9. PET-only and MRI-only baselines are required before fusion claims.
10. Calibration and bootstrap CI are required for reporting.

## PET-label tautology caveat
PET-only and MRI+PET models in Method Y are evaluated against PET-derived biomarker labels. These models may reproduce PET quantification rather than discover independent clinical diagnosis. They must be described as PET-assisted biomarker-status modeling, not Alzheimer disease diagnosis.

## External terminology
A locked internal holdout from A4/LEARN is pseudo-external, not independent external validation. Independent external validation requires a distinct cohort/institution/scanner environment.

## TRIPOD+AI compliance
Before thesis reporting, create a checklist appendix covering sample size, preprocessing, missing data, model specification, calibration, uncertainty, external/pseudo-external terminology, limitations, and generalisability.
