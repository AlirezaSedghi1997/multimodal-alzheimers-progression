# Method Y PET/MRI Multimodal Design

- status: `DESIGN_READY_NOT_TRAINED`
- training_started: `False`
- PET preprocessing started: `False`
- Method X claim unchanged: `True`
- PET input allowed in Method X: `False`
- PET input allowed only in Method Y: `True`

## Availability

- locked catalog rows: `1699`
- amyloid FBP image files: `5661`
- amyloid FBP locked-patient matches: `1697`
- tau FTP image files: `1576`
- tau FTP locked-patient matches: `447`
- cognitive feature rows: `1700`

## Modality Scenarios

| scenario | inputs | reporting status | Method X compliant | PET input |
|---|---|---|---:|---:|
| mri_only | T1 MRI | Method X baseline/final-comparison path | True | False |
| pet_only | amyloid PET image, optional tau PET image | Method Y only | False | True |
| mri_pet | T1 MRI, amyloid PET image, optional tau PET image | Method Y multimodal only | False | True |
| mri_pet_cognitive | T1 MRI, PET image, cognitive/psychological summaries | Method Y multimodal+cognitive only | False | True |

## Target Policy

Current output should be named `AB-positive vs AB-negative biomarker status`, not Alzheimer disease diagnosis. A clinical Alzheimer output is blocked until an independent clinical diagnosis/conversion label is defined and validated.

## Gates Before Training

- Create separate Method Y PET preprocessing root outside Method X caches.
- Create no-PET-in-Method-X regression test after adding PET code.
- Confirm PET images are never loaded by Method X or Method X-Cog dataloaders.
- Define whether the target is biomarker abnormality or an independent clinical diagnosis.
- If clinical AD output is desired, build and validate a separate clinical-label table before training.
- Keep public outputs aggregate-only unless a separate privacy review approves hash-level artifacts.

## Reporting Boundary

PET-only, MRI+PET, and MRI+PET+cognitive are Method Y results. They must not be merged into the Method X non-invasive claim.
