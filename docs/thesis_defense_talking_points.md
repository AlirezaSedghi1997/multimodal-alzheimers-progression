# Thesis Defense Talking Points — v3.6-validity-fixes

## What the project predicts

The output is **Aβ-positive vs Aβ-negative biomarker status**. It is not an Alzheimer disease diagnosis.

## Why PET is used

PET is used only as label/target source in Method X. PET image, SUVR, centiloid, tau PET and other PET-derived values are not Method X inputs.

## Why the external result is called pseudo-external

The holdout is locked and not used for fitting or tuning, but it still comes from the same A4/LEARN source. Therefore it is external-like/pseudo-external rather than a truly independent external cohort.

## Main scientific risk

The A4/LEARN catalog has strong cohort-label coupling. A model may learn cohort/protocol artifacts rather than amyloid biology. v3.6 fixes this by requiring validity audit: MRI permutation, modality ablation, substudy prediction, global label shuffle and random feature controls.

## Why temporal window sensitivity matters

The PET label and T1 MRI can be separated by a large time gap. v3.6 creates all/60/90/120-day catalog variants and reports how N, label balance and performance change.

## Method X-Cog interpretation

Method X-Cog is exploratory. Cognitive and mood/anxiety variables may improve prediction, but family-level ablation is required before attributing the gain to core cognition.

## Tau result

Tau regression is exploratory. A negative R² should be reported honestly as a negative finding, not reframed as a successful result.

## Final framing if validity audit is mixed

The scientifically defensible claim is: a reproducible, privacy-safe, PET-free Method X pipeline for Aβ biomarker-status prediction under strong cohort-label confounding and pseudo-external evaluation, with caveats and sensitivity analyses.
