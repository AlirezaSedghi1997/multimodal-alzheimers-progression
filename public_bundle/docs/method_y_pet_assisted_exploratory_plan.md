# Method Y PET-Assisted Exploratory Plan

Status: Phase 0 aggregate inventory/QC is allowed and can run in parallel. PET preprocessing or PET-image modeling remains blocked until Method X MRI/SPM is stabilized.

Do not start PET preprocessing, PET model training, or PET-MRI multimodal training until final Method X SPM preprocessing, 96-resolution cache, 5-fold 3D ablation, and final XAI refresh are complete. Aggregate-only inventory/QC may run now because it does not feed PET into Method X or Method X-Cog.


## Phase 0: Aggregate Inventory/QC While MRI Runs

Allowed now:

- Count aggregate PET image availability in FBP/FTP folders.
- Count aggregate PET label-source rows and label reproducibility signals.
- Count aggregate tau PET source availability.
- Match locked catalog membership only in memory through hashed linkage.
- Export aggregate reports only.

Disallowed now:

- PET image preprocessing for model input.
- PET image training.
- PET-derived features in Method X or Method X-Cog.
- Raw IDs, raw paths, raw rows, or subject-level PET tables in public artifacts.

## Phase 1: PET Availability Inventory

For the 1700 locked Method X catalog samples:

- Check whether a PET image or PET-derived imaging record is available.
- Match only through existing catalog-safe hashed linkage.
- Report aggregate counts only.
- Do not export raw IDs, raw paths, raw CSV rows, or image files.

## Phase 2: PET QC Summary

If PET images are available and access is approved:

- Run PET-specific preprocessing in a separate output root.
- Keep all artifacts out of Method X model caches.
- Produce aggregate QC metrics only.

## Phase 3: Exploratory Sensitivity Analysis

Allowed:

- Label reproducibility checks.
- Aggregate QC.
- Aggregate sensitivity analysis.

Disallowed:

- PET as a Method X input.
- PET-MRI multimodal training under the Method X claim.
- Public subject-level outputs.

## Phase 4: Method Y-PET/MM Modality-Flexible Modeling

This phase is design-ready but not trained. It is allowed only as a separate Method Y branch.

Planned modality patterns:

- `MRI-only`: reference comparison; Method X-compatible only when PET is absent from inputs.
- `PET-only`: PET-assisted biomarker model; Method Y only.
- `MRI+PET`: multimodal image fusion; Method Y only.
- `MRI+PET+cognitive`: multimodal plus cognitive/psychological covariates; Method Y only.

Current approved output wording: probability of `A-beta-positive vs A-beta-negative biomarker status`. Do not call this Alzheimer diagnosis. A clinical Alzheimer output requires a separate independent clinical diagnosis/conversion label.