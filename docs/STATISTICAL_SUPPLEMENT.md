# Statistical supplement — 11 October 2026

This supplement addresses methodological questions raised during thesis review.
It adds descriptive audits, stronger comparators, and sensitivity analyses to the
existing prognostic study. The analyses were specified after prior results had
been inspected. They are **post-hoc, internal and exploratory**; neither the
revision nor multiplicity adjustment creates prospective confirmation.

The main cohort is 1,059 amyloid-positive A4 participants with 239 sustained CDR
progression events. LEARN is relevant to historical MRI architecture experiments,
not the main prognostic population. CDR progression is not an Alzheimer's
dementia diagnosis.

## Analysis and result map

| Question | Code in `scripts/thesis_statistics_v1/` | Aggregate evidence |
|---|---|---|
| Who entered each analysis, and were available-data subsets different? | `demographics.py`, `sample_size.py` | [Cohort, flow, SMD and fixed-grid KM](../results/aggregate/10_statistical_supplement/cohort_summary.json) |
| Was tau measured before the individual follow-up origin? | `demographics.py` | Same cohort summary; 358/358 measurements at or before origin |
| Do death records establish an adjudicated competing event? | `disposition.py` | [Disposition, APOE coding and QC](../results/aggregate/10_statistical_supplement/disposition_and_qc.json) |
| Does imaging improve on a null model or a stronger clinical baseline? | `prediction_sensitivity.py` | [Baseline and IPA](../results/aggregate/10_statistical_supplement/baseline_and_ipa.json) |
| Are all three tau contrasts reported with adjusted uncertainty? | `prediction_sensitivity.py` | [Tau contrasts and calibration](../results/aggregate/10_statistical_supplement/tau_multiplicity.json) |
| Does the risk–PACC association remain without cognitive inputs in the risk model? | `prediction_sensitivity.py` | [Noncognitive risk sensitivity](../results/aggregate/10_statistical_supplement/noncognitive_pacc.json) |
| How much information supports the regression heads? | `sample_size.py`, `planning.py` | [Riley scenarios](../results/aggregate/10_statistical_supplement/riley_scenarios.json), [tau head support](../results/aggregate/10_statistical_supplement/tau_training_support.json) |

The main statistical record is the [supplement report](../reports/07_statistical_supplement/REPORT.md).
Machine-readable values retain precision for reproduction; displayed tables round
them to an appropriate number of digits.

## Statistical definitions

**Null comparator and IPA.** A Kaplan–Meier curve is fitted only to the outer
training participants and gives the same prediction to every held-out participant
in that fold. The original IPCW time grid and censoring reference are retained.
Fold-size-weighted IBS values are combined first, then
`IPA = 1 - IBS(model) / IBS(null)` is calculated. This is not an unweighted mean
of fold-specific ratios.

**Stronger baseline.** Existing nested tabular predictions fitted on the entire
outer training set are evaluated alongside the matched fusion-head baseline.
Ridge values 0.1, 1 and 10 were selected within three inner folds using the
recorded one-standard-error rule. The supplement reuses those locked predictions;
it does not select a new baseline using outer-test performance.

**Paired uncertainty.** Participant pairs are resampled within each outer fold,
with 2,000 draws and seed 20261011. Differences are augmented minus base: negative
IBS and positive Uno differences are favorable. Bonferroni percentile limits use
`alpha/(2K)` and `1-alpha/(2K)`, with K=3 tau contrasts or K=2 stronger-baseline
contrasts, **separately for each metric**. No joint family coverage across both
metrics is claimed. Intervals condition on fitted models and training censoring
distributions and exclude development/selection uncertainty. Invalid Uno draws
and their actual valid counts remain visible.

**Noncognitive risk sensitivity.** The 45 fusion heads were refitted without
baseline PACC or MMSE. Original folds, frozen MRI experts, remaining predictors
and ridge penalty 10 were retained. Frozen CNN inference was performed where
the full outer-fold predictions were not already available; CNN weights were
not retrained. The longitudinal model retains its original baseline-cognition
adjustment and exactly the same 1,049 people and 8,557 observations. It uses ML,
an unstructured random intercept/slope covariance and a centered random-time
basis. Optimizer agreement, gradient and covariance checks are recorded. This
is an association conditional on a generated predictor, not independent cognitive
validation or a causal effect.

**Sample-size scenarios.** Riley's shrinkage, optimism and overall-risk precision
criteria are evaluated using assumed Cox–Snell R² values 0.05–0.20 and 7, 8 or 12
parameters. The constant-hazard approximation is used only for this retrospective
planning calculation. The corrected null likelihood is used, and the maximum of
the three criteria is reported. These scenarios do not establish adequate CNN
training size. See [Riley et al.](https://doi.org/10.1002/sim.7992) and the
[published correction](https://doi.org/10.1002/sim.8409).

## Reproduction with authorized inputs

Inspecting aggregates and running the focused tests require no participant data:

```bash
python -B -m unittest discover -s tests/thesis_statistics_v1 -v
python -B check_package.py
```

Full recalculation requires the separately authorized original research tree:

- `results/xyz_fusion_v2/private/`: locked contract, splits, tabular and MRI
  prediction jobs, and selected frozen model weights;
- `results/xyz_fusion_postall_v2/private/`: eligibility, tau evaluations and
  matched longitudinal PACC table;
- `results/xyz_mri_baseline_fast_v1/private/`: intake, MRI QC lineage and cache;
- the earlier endpoint artifact under `results/method_y_revival_v2/endpoints/`;
- provider-authorized A4/LEARN clinical CSVs and dictionaries.

From this checkout, execute the following stages in order, using actual local
paths in place of the placeholders. `PRIVATE_OUTPUT` must be outside this Git
checkout and outside the source project's locked `results/` tree.

```bash
python -B -m scripts.thesis_statistics_v1.demographics --project-root RESEARCH_ROOT --clinical-root CLINICAL_ROOT --output PRIVATE_OUTPUT
python -B -m scripts.thesis_statistics_v1.disposition --project-root RESEARCH_ROOT --clinical-root CLINICAL_ROOT --output PRIVATE_OUTPUT
python -B -m scripts.thesis_statistics_v1.sample_size --project-root RESEARCH_ROOT --clinical-root CLINICAL_ROOT --output PRIVATE_OUTPUT
python -B -m scripts.thesis_statistics_v1.prediction_sensitivity --project-root RESEARCH_ROOT --clinical-root CLINICAL_ROOT --output PRIVATE_OUTPUT --allow-frozen-inference
python -B -m scripts.thesis_statistics_v1.export_aggregate --source PRIVATE_OUTPUT --destination AGGREGATE_REVIEW_DIRECTORY
```

The first stage requires an empty output directory and records input/code hashes.
Later stages accept the same context; changed inputs or code require a new output
directory. The prediction stage resumes completed local calculations. Its explicit
frozen-inference flag is required before the 45-component sensitivity stage and
requires the recorded PyTorch/CUDA environment and sufficient memory for the
held-out MRI bank. Parent scientific modules match the original analysis source;
the wrappers make roots/output paths explicit. Original-script hashes are in
[`original_analysis_lineage.json`](../scripts/thesis_statistics_v1/original_analysis_lineage.json).

The repository does not contain the restricted inputs. No clean-machine full
reconstruction is claimed. All source-run outputs remain unchanged by these
commands; new participant-level intermediates stay in the private output folder.

## Privacy, scope and unresolved issues

Only allowlisted aggregates are included. Individual predictions, subject/sample
hashes, exact participant event-time arrays, raw scans, weights, signed thesis
documents and source dictionaries are excluded. KM is projected onto fixed yearly
points rather than exposing every observed event time. Descriptive covariates for
the very small excluded groups are omitted. This is data minimization, not a
formal differential-privacy guarantee or new permission to redistribute study
data. Provider terms still govern use, including in a private GitHub repository.

All 358 tau measurements precede or coincide with origin. Death/disposition
records do not adjudicate the unknown event-free interval after the last CDR:
no fabricated competing-risk endpoint is introduced. Full visual review of every
processed MRI, external validation and stable tau calibration remain unresolved.

The baseline/tau/noncognitive calculations here use the **original AMP parent**.
They must not be pooled with or silently replace the separate
[FP32 sensitivity analysis](FP32_SENSITIVITY.md). All three tau contrasts and
unfavorable or uncertain results are retained.
