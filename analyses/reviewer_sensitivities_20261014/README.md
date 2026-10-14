# Post-hoc statistical analyses and OOF sensitivity — 14 October 2026

This is the cumulative **research-code and aggregate-evidence** update from the
first scientific review through the latest statistical additions. It preserves
the original fitted-model results and adds the completed analyses below.
The files contain no manuscript prose, editorial revisions or review letters.
Frozen import closures keep numerical work separate from later source changes.

## Analysis map

| Executable | Completed work | Aggregate evidence |
|---|---|---|
| `revision_analyses/selected_analyses.py` | 40,000-draw paired tau contrasts; conditional calibration; IPA; longitudinal PACC mixed models | `tau_bootstrap`, `calibration`, `ipa`, `pacc` |
| `revision_analyses/cohort_and_censoring.py` | Cohort and availability reconciliation; measured-covariate censoring sensitivity | `cohort_audit`, `censoring_sensitivity` |
| `revision_analyses/tabular_sensitivities.py` | Nested training-only clinical comparators; MMSE omission; hippocampus-only and compact-tau models | `tabular_sensitivities` |
| `revision_analyses/reviewer2_analyses.py` | Baseline biomarker denominators; PET exclusion reconciliation; expert event support; all 20 three-year Brier contrasts; absolute IBS/IPA intervals; calibration bands | `baseline_biomarkers`, `pet_selection`, `expert_support`, `brier_3year`, `table2_precision`, `calibration_curves` |
| `oof_stacking_extension/scripts/run_xyz_oof_stacking_v1.py` | Completed pooled OOF experiment using frozen FP32 features, 180 heads and all seven locked contrasts | `oof_aggregate_results` |
| `oof_stacking_extension/oof_distribution_diagnostic.py` | Between/within-expert decomposition of pooled training-score variation | `oof_score_distribution_diagnostic` |
| `revision_analyses/absolute_concordance.py` | 2,000-draw absolute Uno-C intervals for 13 primary rows and 16 OOF/comparator model entries | `absolute_concordance` |
| `revision_analyses/export_training_logs.py` | Export of 405 original model/epoch histories without participant membership or weights | `original_training_logs` |

Evidence names in the last column refer to JSON files in `aggregate_results/`.
Earlier sample-size, demographic, noncognitive-PACC and regional evidence is
retained in `prior_aggregate_results/` and the root statistical supplement.
`figure_sources/aggregate_evidence.json` is a numeric comparison index used by
the statistical scripts; article-layout and graphical-abstract scripts are excluded.

The CNN/Jacobian source and its nonnumerical no-go decision are separate in the
[pilot guide](../../docs/CNN_JACOBIAN_PILOT.md). Its full-cohort experiment was not run.

## Reading the findings

- The revised **40,000-draw** tau intervals supersede historical 2,000-draw
  intervals. All three multiplicity-adjusted incremental IBS intervals include
  zero. The two Cox-based tau concordance signals remain positive under the
  recorded 40-contrast sensitivity, without establishing probability accuracy
  or clinical usefulness.
- All 20 multiplicity-adjusted three-year Brier contrasts include zero.
- All seven adjusted pooled-OOF IBS intervals include zero. Pooling did not
  establish additional imaging benefit. Large between-expert score differences
  mean this is not a clean experiment on head-training sample size alone.
- Absolute concordance intervals describe uncertainty for individual models;
  they are nominal, conditional intervals, not paired OOF incremental tests.
  All 2,000 primary-model draws were estimable. The four tau OOF arms had 1,969
  estimable draws; a draw is omitted when a fold lacks IPCW comparable pairs.
- Recorded training histories document execution. They do not demonstrate
  independent reproducibility or learning of a specific biological signal.

Resampling holds fitted predictions, folds and training censoring distributions
fixed. It does not include refitting/model-selection uncertainty or correct for
all historical development decisions. This remains post-hoc internal research
in an already explored cohort, with no independent external validation.
Original AMP and matched FP32/OOF comparisons remain separate.

## Inspect or reproduce

Use Python 3.11. `requirements-analysis.txt` records the analysis environment;
PyTorch must be installed with the appropriate official CPU/CUDA build.
`requirements-preprocessing.txt` additionally specifies ANTs/image dependencies.
A fresh installation on every platform has not been tested.

From this directory, safe checks without controlled participant data are:

```text
python -B synthetic_check.py
python -B replay_aggregates.py --output PATH_OUTSIDE_THIS_CHECKOUT/replay
```

The synthetic check exercises metric identities, tied risks/times, leakage
guards, calibration and a synthetic 128-cubed image forward pass. Aggregate
replay verifies file hashes and arithmetic, not the original participant data.
Reproduction creates new outputs; use an isolated copy and recheck manifests
before treating a modified directory as the exact distributed release.

With authorized prepared inputs, the statistical commands are:

```text
python -B revision_analyses/selected_analyses.py --repo AUTHORIZED_ROOT --output NEW_OUTPUT/results --task all
python -B revision_analyses/tabular_sensitivities.py --repo AUTHORIZED_ROOT --output NEW_OUTPUT/results
python -B revision_analyses/cohort_and_censoring.py --repo AUTHORIZED_ROOT --raw-clinical AUTHORIZED_CLINICAL --output NEW_OUTPUT/results --task all
python -B revision_analyses/reviewer2_analyses.py --repo AUTHORIZED_ROOT --output NEW_OUTPUT/results --task descriptive
python -B revision_analyses/reviewer2_analyses.py --repo AUTHORIZED_ROOT --output NEW_OUTPUT/results --task brier --resamples 40000
python -B revision_analyses/reviewer2_analyses.py --repo AUTHORIZED_ROOT --output NEW_OUTPUT/results --task precision
python -B revision_analyses/absolute_concordance.py --repo AUTHORIZED_ROOT --output NEW_OUTPUT/results
python -B revision_analyses/export_training_logs.py --repo AUTHORIZED_ROOT --output NEW_OUTPUT/results/original_training_logs.json
```

For `reviewer2_analyses.py --task curves`, first copy the distributed
`aggregate_results/calibration.json` into the new output results directory;
its parameters are checked against a fresh fit. The absolute-concordance OOF
calculation requires the completed restricted OOF cells. See the
[input contract](REPRODUCING_RESTRICTED_DATA.md) and
[OOF guide](oof_stacking_extension/README.md) for reconstruction requirements.
The script-generated sibling `private/` directory must remain local.

`run_models.py` provides the original prepared-data model reconstruction CLI,
with validation as its default action. It does not download restricted data or
reconstruct missing provenance. Original source/default-path and distributed
hashes are retained; the publication changes only machine-path defaults.
`checks/` retains earlier authorized statistical replay receipts (2,311 and
2,846 numerical values matched exactly) and tau-model reconstruction evidence.
New publication checks are reported separately in the root release audit.

## Data and distribution

No participant-level data, pseudonymous keys, individual predictions, scans,
trained weights or secrets are supplied. A4/LEARN input access is controlled
separately from this repository. Public code and aggregate evidence do not
confer data redistribution rights or a new software license. Read
[distribution terms](DISTRIBUTION_TERMS.md) and the root data-access guide.
