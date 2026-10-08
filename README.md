# Multimodal neuroimaging and progression research

Research software and aggregate evidence for an internal, post-hoc study of MRI,
amyloid/tau PET, clinical/cognitive predictors and available blood measures.
This is a **private release candidate**, not a clinical tool or a validated
Alzheimer disease diagnostic system. No external validation cohort was available.

## Study components

| Component | Question and reporting scope |
|---|---|
| Method X | PET-free MRI/tabular biomarker-status prediction; technical evidence only because cohort and label are confounded |
| Method Y | Future progression from FBP amyloid or FTP tau; regional and image analyses are separate populations |
| Prognostic fusion | Baseline MRI, amyloid and clinical/cognitive predictors with ancestry-safe cross-fitting; blood and tau comparisons use their own matched subsets |
| Post-fusion analyses | Censored metrics, calibration, paired uncertainty, scanner sensitivity, longitudinal PACC and regional evidence |
| FP32 spatial sensitivity | Separate precision/head-refit analysis in progress at packaging; no result from it is included |

MRI/FBP-image/tau/blood do **not** form one validated four-way model. The four-way
analysis was not estimable under the recorded support rules. Negative comparisons,
failed numerical attribution controls and absent external validation remain part
of the evidence. Brain maps describe model attribution, not proven disease lesions.

## Read first

1. [Study map](docs/STUDY_MAP.md): active methods, compatibility dependencies and exclusions.
2. [Evidence guide](docs/EVIDENCE_GUIDE.md): source hashes, aggregate projections and claim limits.
3. [Reproduction guide](docs/REPRODUCING.md): installation, protected inputs and execution boundaries.
4. [Data access](docs/DATA_ACCESS.md) and [license status](docs/LICENSING.md).
5. [Release gaps](docs/RELEASE_GAPS.md): unresolved steps that are not concealed by a passing integrity check.

## Safe offline check

```powershell
python -B check_package.py
```

This checks the listed file bytes. It does not download data, access a GPU, train
models, rewrite the study or establish scientific validity. No private repository
history, participant-level records, clinical images or model weights are bundled.

## Layout

`scripts/` and `src/` retain the computational layout; `configs/` records selected
settings; `tests/` holds selected regression tests; `environment/` records installed
versions; `results/aggregate/` contains derived aggregate evidence; `docs/` supplies
the versioned route, source and result registries. Filename versions inside the
implementation are compatibility identifiers, not invitations to run retired analyses.

No author identity, institutional approval, open-source license or permission to
redistribute data is inferred. Publication remains subject to those checks.
