# Integrated Method X v3.6 + Method Y PET Patch Policy

This policy accompanies the unified patch pack that combines:

1. **Method X v3.6 validity fixes**: validity audit, temporal window sensitivity, nested HPO, pseudo-external reporting, 3D final contract, Method X-Cog family ablation, privacy scans, and thesis-grade reporting.
2. **Method Y PET Phase-2 preparation**: PET preprocessing scaffold, PET-T1 gap filter, Method Y preregistration template, PET gap/cohort evaluation policy, PET Phase-2 gate policy, and PET preprocessing contract tests.

## Non-negotiable separation

- Method X remains the primary non-invasive thesis path.
- PET is never an input to Method X or Method X-Cog.
- PET images, SUVR, centiloid, tau PET values, PET-derived features, and PET-T1 gap must not enter Method X feature vectors.
- Method Y is a separate exploratory PET-assisted branch.
- Any PET-containing scenario must be reported as Method Y, not Method X.

## Required order of work

1. Run Method X v3.6 validity audit before making stronger Method X claims.
2. Run temporal PET-T1 sensitivity as a reporting/audit layer; do not overwrite the main catalog silently.
3. Run pseudo-external evaluation with explicit terminology and DeLong blocked status.
4. Complete Method Y preregistration before PET training.
5. Run PET dry-run/preprocessing only into `data/preprocessed_pet`, never into Method X caches.
6. Report Method X and Method Y in separate tables.

## Safe merge rule

This integrated package may be applied to a single repository branch for convenience, but scientific reporting must preserve two lanes:

- **Lane A / Method X**: T1 MRI ± fixed tabular ± cognitive exploratory.
- **Lane B / Method Y**: PET-only, MRI+PET, MRI+PET+cognitive exploratory.

## Public export rule

Public bundle must remain aggregate-only. The following are forbidden in public outputs:

- raw IDs
- raw file paths
- NIfTI/DICOM/PET images
- HDF5 caches
- checkpoints
- subject-level PET outputs
