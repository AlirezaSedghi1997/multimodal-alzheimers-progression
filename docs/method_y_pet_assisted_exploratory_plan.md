# Method Y PET-Assisted Exploratory Plan

Status: planned.

Do not start this branch until final Method X SPM preprocessing, 96-resolution cache, 5-fold 3D ablation, and final XAI refresh are complete.

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
