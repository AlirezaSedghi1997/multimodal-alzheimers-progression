# Multimodal Alzheimer's Progression Research

Research software for studying future cognitive progression with structural MRI,
amyloid and tau PET, and clinical/cognitive predictors. The project combines
cross-fitted survival models with matched comparisons, calibration, scanner audits,
longitudinal analysis and spatial explanations.

The primary prognostic question is whether imaging adds information about future
sustained CDR progression beyond baseline clinical and cognitive measures.
Longitudinal PACC change provides a secondary outcome. Analyses use A4/LEARN data
under controlled access; each comparison retains its own eligible population and
prediction time origin.

This private research release contains selected code, configurations, regression
tests and aggregate evidence. Findings are **internal and exploratory**. No external
validation cohort was available, and clinical diagnostic use is not supported.

## Study components

| Component | Question and reporting scope |
|---|---|
| Method X | PET-free MRI/tabular biomarker-status prediction; technical evidence only because cohort and label are confounded |
| Method Y | Future progression from FBP amyloid or FTP tau; regional and image analyses are separate populations |
| Prognostic fusion | Baseline MRI, amyloid and clinical/cognitive predictors with ancestry-safe cross-fitting; blood and tau comparisons use their own matched subsets |
| Post-fusion analyses | Censored metrics, calibration, paired uncertainty, scanner sensitivity, longitudinal PACC and regional evidence |
| FP32 spatial sensitivity | Completed precision/head-refit sensitivity; reviewed aggregate comparisons and numerical map-control findings |

MRI/FBP-image/tau/blood do **not** form one validated four-way model. The four-way
analysis was not estimable under the recorded support rules. Negative comparisons,
failed numerical attribution controls and absent external validation remain part
of the evidence. Brain maps describe model attribution, not proven disease lesions.

## Latest evidence update

The [FP32 sensitivity review](docs/FP32_SENSITIVITY.md) includes matched MRI
comparisons in 1,059 participants and FBP-image comparisons in 544 participants.
The paired confidence intervals did not establish incremental value in either
comparison. Of 66 spatial baselines, 36 passed the numerical criteria; attribution
remains incompletely validated because the randomized-network control was
degenerate and label-permutation training was not performed.

These findings accompany the parent fusion analysis. Model precision and refitted
heads define a separate sensitivity analysis, so its metrics are not interchangeable
with the parent results.

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

## Access and citation

Data access is governed by the original study agreements. This repository excludes
participant-level data, individual brain images and model weights. See
[data access](docs/DATA_ACCESS.md) for the required inputs and
[licensing](docs/LICENSING.md) for the current code and third-party status.
Software citation metadata will be finalized with the research publication.
