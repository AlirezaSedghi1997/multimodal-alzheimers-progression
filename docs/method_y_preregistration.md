# Method Y-PET/MM Preregistration Template

**Project:** amyloid-tau-prediction-a4learn  
**Branch:** Method Y-PET/MM exploratory  
**Version:** YYYY-MM-DD  
**Status:** DRAFT — must be frozen before any PET-image training  
**Related policy:** `configs/model/method_y_gap_cohort_eval_policy.yaml`

## 1. Non-negotiable scope

Method Y is a PET-assisted exploratory branch. It is not Method X and must not be used to support the non-invasive Method X claim.

- Method X input remains T1 MRI ± approved tabular/cognitive features only.
- PET images, SUVR images, centiloid values, PET-derived scalar features, and PET-derived embeddings are forbidden in Method X and Method X-Cog.
- Any scenario containing PET image input is reported only under Method Y.
- The currently approved target is **Aβ-positive vs Aβ-negative biomarker status**, not clinical Alzheimer disease diagnosis.
- Clinical Alzheimer disease, MCI conversion, or dementia diagnosis output requires a separately validated independent clinical target.

## 2. Primary objective

Estimate whether adding amyloid PET image information to a fixed MRI-only baseline improves prediction of Aβ biomarker status on the same eligible sample set, while preserving strict separation from Method X.

Primary comparison:

```text
MRI+PET vs MRI-only_same_sample_set
```

Primary estimand:

```text
ΔROC-AUC = ROC-AUC(MRI+PET) - ROC-AUC(MRI-only_same_sample_set)
```

A result is considered scientifically interesting only if all conditions hold:

1. Both compared models are evaluated on the exact same subject/sample set.
2. The evaluation set has both classes.
3. Bootstrap 95% CI for ΔROC-AUC is reported.
4. The result remains directionally consistent in gap-tier and cohort-aware sensitivity checks.
5. The report explicitly states that the task is biomarker-status prediction, not Alzheimer diagnosis.

## 3. Secondary objectives

1. Train and evaluate a PET-only baseline to contextualize fusion results.
2. Compare MRI-only, PET-only, and MRI+PET on identical sample-selection tiers.
3. Run MRI+PET+cognitive only after MRI 96/3D and cognitive contracts are final.
4. Quantify how PET-T1 temporal gap affects results.
5. Quantify how cohort-label coupling affects results.
6. Run permutation/shuffle controls to test whether fusion improves because of real modality signal or because of cohort/protocol artifacts.

## 4. Data eligibility

### 4.1 Required inputs

A sample is eligible for Method Y-PET/MM only if all are available:

- locked Method X catalog row;
- Aβ label from official PET label source;
- T1 MRI reference selected by existing Method X alignment logic;
- amyloid PET image discoverable and processable;
- `abs_pet_t1_gap_days` available;
- no raw identifier or raw path is exported to public outputs.

### 4.2 Temporal gap tiers

The following tiers are frozen before training:

| Tier | Rule | Use |
|---|---:|---|
| primary strict | `abs_pet_t1_gap_days <= 90` | primary Method Y training/evaluation if feasible |
| secondary window | `<= 120` | sensitivity analysis |
| relaxed exploratory | `<= 365` | aggregate or non-primary sensitivity only |
| excluded from training | `> 365` or missing | aggregate inventory only |

The temporal gap must **never** be used as a model input feature. It is a sample-selection and reporting variable only.

## 5. Preprocessing plan

PET preprocessing outputs must live under a separate Method Y root, for example:

```text
data/preprocessed_pet/
  fbp/
  ftp/
  qc/
  _scratch_pet/
```

They must not be written into Method X MRI caches.

Minimum preprocessing stages:

1. Discover PET files from tracer-specific roots without exporting raw paths publicly.
2. Resolve PET files to locked catalog samples using in-memory raw identifiers and hash-level outputs.
3. Reorient PET volume to canonical orientation.
4. Optionally coregister PET to subject T1 if the required local toolchain is available.
5. Apply intensity cleanup and normalization defined in the frozen config.
6. Write hash-named private PET outputs only.
7. Write aggregate QC summary and hash-level private index.

Important boundary: if SUVR-like normalization is performed, it must be described as Method Y PET-image preprocessing and must not create scalar SUVR/centiloid model features for Method X or Method X-Cog.

## 6. Modeling scenarios

All scenario names and reporting labels are frozen before training.

| Scenario | Inputs | Report under | Required? |
|---|---|---|---|
| MRI-only_same_sample_set | T1 MRI only | Method Y baseline table; Method X-compatible | yes |
| PET-only | amyloid PET image | Method Y only | yes |
| MRI+PET | T1 MRI + amyloid PET image | Method Y only | yes |
| MRI+PET+cognitive | T1 MRI + PET + cognitive | Method Y only | optional |

Tau PET is not part of the primary Method Y model. It may be reported only as an exploratory subset because coverage is limited.

## 7. Model-selection rules

- Hyperparameters are selected only inside development folds.
- The locked pseudo-external holdout is never used for fitting, tuning, threshold selection, calibration fitting, or model selection.
- Calibration parameters are fit only on validation data inside the fold.
- If a truly independent external cohort is not available, the report must use the wording `locked pseudo-external holdout`.

## 8. Metrics

Report all of the following when both classes are present:

- ROC-AUC with bootstrap 95% CI;
- PR-AUC with bootstrap 95% CI;
- Brier score;
- ECE;
- reliability diagram;
- predicted-probability distribution by label, cohort, and gap tier.

If a stratum is single-class:

- do not report ROC-AUC;
- do not report PR-AUC;
- report Brier/ECE only if meaningful;
- report predicted-probability distribution;
- mark as `SINGLE_CLASS_NO_DISCRIMINATION_METRIC`.

## 9. Statistical comparison

Default comparison uses bootstrap CI for paired metric differences on the same sample set.

DeLong testing is allowed only if all are true:

1. evaluation set is truly independent external, not pseudo-external;
2. both classes are present;
3. both models are evaluated on the same subjects;
4. governance context explicitly marks `external_is_truly_independent=True`.

DeLong is forbidden on CV folds, pseudo-external holdout, and single-class strata.

## 10. Cohort-confounding checks

Because A4/LEARN may be strongly coupled with Aβ label, every Method Y report must include:

1. label distribution by cohort;
2. prediction distribution by cohort;
3. per-cohort metrics only if both classes exist in that cohort;
4. descriptive single-class reports where per-cohort AUC is invalid;
5. optional cohort-prediction probe to test whether embeddings encode cohort;
6. shuffle/permutation control for PET and/or MRI embeddings.

Any claim is downgraded to exploratory if performance is high overall but cannot be verified within cohorts.

## 11. Permutation and negative controls

Run at least one of the following before final reporting:

- PET embedding shuffle within split;
- label permutation within cohort where feasible;
- modality dropout control;
- PET-only vs PET-shuffled comparison;
- MRI+PET vs MRI+PET-shuffled comparison.

If shuffled controls perform similarly to the real model, the fusion claim must be rejected or downgraded.

## 12. Required outputs

Private/hash-level outputs:

```text
data/preprocessed_pet/<tracer>/
results/final_results/method_y/
results/tables/method_y_pet_preprocess_summary.json
results/tables/method_y_pet_preprocess_summary.md
```

Public/shareable outputs:

```text
public_bundle/results/final_results/method_y/*.md
public_bundle/results/tables/*method_y*.md
```

Public outputs must be aggregate-only. No raw IDs, raw paths, NIfTI files, HDF5 caches, checkpoints, or subject-level tables may enter the public bundle.

## 13. Stopping and downgrade rules

Stop or downgrade Method Y training if any condition holds:

- PET preprocessing root is not separate from Method X cache;
- PET tensors appear in Method X loaders;
- gap policy is not applied before training;
- public bundle contains raw paths, raw IDs, NIfTI, HDF5, or checkpoints;
- primary strict tier has insufficient class counts and no defensible sensitivity tier exists;
- results are explainable primarily by cohort-label confounding;
- no independent clinical target exists but outputs are phrased as Alzheimer diagnosis.

## 14. Approved wording for thesis/report

Use:

> Method Y is an exploratory PET-assisted biomarker-status modeling branch. It evaluates whether PET-image information and MRI/PET fusion can reproduce or improve prediction of Aβ-positive vs Aβ-negative biomarker status under strict governance and temporal-gap controls. These analyses are separate from the non-invasive Method X claim and do not constitute clinical Alzheimer disease diagnosis.

Do not use:

- "Method X with PET input";
- "non-invasive MRI+PET model";
- "Alzheimer diagnosis";
- "external validation" unless a truly independent cohort is used.

## 15. Freeze checklist

Before training, fill and sign off:

- [ ] policy version frozen;
- [ ] temporal gap tier selected;
- [ ] sample counts by label/cohort/gap reported;
- [ ] PET preprocessing root verified separate;
- [ ] no-PET-in-Method-X regression tests pass;
- [ ] metrics and statistical comparisons frozen;
- [ ] public-output privacy rules verified;
- [ ] limitation wording approved.
