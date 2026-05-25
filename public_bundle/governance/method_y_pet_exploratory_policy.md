# Method Y PET Exploratory Policy

PET image preprocessing may be added only as an exploratory Method Y branch for QC, label reproducibility, and aggregate sensitivity analyses.

PET remains prohibited as a Method X model input.

Method X inputs remain limited to T1 MRI plus age, sex, education_years, apoe4_carrier, and mmse_base.

Method Y outputs must not publish raw identifiers, raw CSV rows, raw PET images, MRI images, DICOM/NIfTI files, or subject-level public tables.

Public or shareable Method Y artifacts must be aggregate-only or hash-level only.

## Parallel Aggregate Inventory Exception

Aggregate-only PET availability/QC inventory may be run before final MRI/SPM completion if all of the following hold:

- PET is not loaded into Method X or Method X-Cog model tensors.
- PET-derived values are not joined into Method X or Method X-Cog feature vectors.
- Raw identifiers and raw paths are used only transiently in memory and are not written.
- Public outputs are aggregate-only.
- The output explicitly states that Method X remains PET-as-label-only.

## PET-Image Modeling Boundary

A PET-image model is technically possible only as a separate Method Y experiment. It must not be described as non-invasive Method X, and it must not be used to answer a clinical Alzheimer diagnosis question unless a separate clinical target definition is approved. AB-positive vs AB-negative is a biomarker-label task, not equivalent to diagnosing Alzheimer disease.
## Method Y-PET/MM Approved Design Direction

The approved exploratory design separates four modality patterns:

- MRI-only: Method X-compatible only when PET is not an input.
- PET-only: Method Y only.
- MRI+PET: Method Y only.
- MRI+PET+cognitive: Method Y only.

The current approved target remains biomarker status (`AB-positive vs AB-negative`). Any output phrased as clinical Alzheimer disease requires an independent clinical diagnosis or conversion label and a separate target-validation report.