# Synthetic MRI-to-PET Supplementary Branch

- status: `DESIGN_READY_SUPPLEMENTARY_ONLY`
- generator fit scope: `train_fold_only`
- synthetic added to: `same_train_fold_only`
- validation real only: `True`
- test real only: `True`
- Method X input allowed: `False`
- synthetic counted as real subjects: `False`

## Leakage Guards

- Fit MRI-to-PET generator independently inside each train fold.
- Do not train generator on validation, test, or holdout hashes.
- Do not add synthetic PET to validation, test, holdout, or Method X.
- Report real N separately from synthetic train exposures.

## Claim Boundary

Supplementary augmentation sensitivity only; not evidence of more real patients, not a non-invasive Method X input, and not a clinical Alzheimer diagnosis model.
