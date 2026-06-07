# Method Y Multimodal Operating Contract

Method Y-PET/MM is an exploratory PET-assisted branch. It is not Method X.

## Allowed Inputs

Method Y may use:

- PET image inputs in a separate PET-only or PET-assisted model.
- T1 MRI inputs in MRI+PET models.
- Cognitive/psychological summaries in MRI+PET+cognitive models.

## Forbidden Claims

- Do not call PET-assisted models Method X.
- Do not claim non-invasive screening when PET is an input.
- Do not call AB-positive vs AB-negative a definitive Alzheimer disease diagnosis.
- Do not use PET-derived target values as both target and hidden tabular input.

## Target Contract

Current approved target: biomarker status (`AB-positive vs AB-negative`).

Clinical Alzheimer classification requires a separate, independently defined clinical diagnosis/conversion label. Until then, outputs must be worded as biomarker abnormality probability or risk.

## Reporting Contract

Report each modality pattern separately:

- MRI-only.
- PET-only.
- MRI+PET.
- MRI+PET+cognitive.

## Privacy Contract

Public Method Y outputs must be aggregate-only. Raw IDs, raw paths, raw PET/MRI images, DICOM/NIfTI/HDF5/cache files, checkpoints, and subject-level predictions are not public artifacts.
