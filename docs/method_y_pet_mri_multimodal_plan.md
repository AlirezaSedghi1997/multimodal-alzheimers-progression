# Method Y PET/MRI Multimodal Modeling Plan

Status: design-ready, not trained.

## Scientific Boundary

Method X remains the non-invasive thesis path: T1 MRI plus allowed fixed tabular variables, with PET used only as label/target source. Any model that receives PET images or PET-derived values is Method Y and must be reported separately.

## Modality-Flexible Scenarios

The proposed Method Y model family may support these input patterns:

- `MRI-only`: T1 MRI, reported under Method X only when PET is not an input.
- `PET-only`: amyloid PET image and optional tau PET image, reported only as Method Y.
- `MRI+PET`: T1 MRI plus PET image inputs, reported only as Method Y.
- `MRI+PET+cognitive`: T1 MRI plus PET image inputs plus cognitive/psychological summaries, reported only as Method Y exploratory multimodal analysis.

The deployment interface may accept whichever modalities are present, but the training and evaluation tables must report each input pattern separately.

## Target Policy

The current approved target is `AB-positive vs AB-negative biomarker status`. This must not be described as a definitive Alzheimer disease diagnosis.

A clinical Alzheimer output is allowed only after creating an independent clinical-label table, such as cognitively normal vs MCI vs AD dementia, clinical diagnosis at visit, conversion to MCI/AD on follow-up, or adjudicated clinical diagnosis. Without that label, the model may report biomarker abnormality risk, not clinical Alzheimer status.

## Leakage Controls

- If the target is PET-derived amyloid positivity, PET images must not be used in the same Method X claim.
- If PET images are used as Method Y inputs, the report must say the model is PET-assisted and not non-invasive.
- PET preprocessing outputs must live outside Method X MRI caches.
- No PET image, SUVR, centiloid, raw path, raw identifier, or subject-level prediction may enter the public bundle.

## Required Pretraining Gates

1. Create a separate PET preprocessing root and cache naming convention.
2. Add regression tests proving Method X dataloaders do not load PET tensors.
3. Define the target as biomarker abnormality or clinical diagnosis before training.
4. If using clinical Alzheimer diagnosis, validate the clinical-label source and missingness.
5. Report PET-only, MRI-only, MRI+PET, and MRI+PET+cognitive separately.
6. Keep public artifacts aggregate-only unless a separate privacy review approves hash-level release.

## Current Status

Phase 0 aggregate PET inventory is complete. Training has not started, PET preprocessing has not started, and Method X remains unchanged.
