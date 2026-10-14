> Execution update, 14 October 2026: the full frozen-feature OOF experiment completed (180 heads; 327 execution units verified). The protocol below is retained as written before evaluation. Later absolute concordance intervals and a training-score heterogeneity diagnostic are separate post-hoc additions; see the parent bundle README.

# Pooled OOF survival stacking: version 1

Protocol date: 14 October 2026. Status: implementation and bounded local validation; the real full experiment must be run explicitly. Existing article results are not replaced by this implementation.

## Question and estimand

The original fusion fits a separate Cox head on approximately one third of the outer-training participants for each frozen expert, then averages survival curves. This experiment asks whether a **shared head per outer fold and seed**, fitted on all eligible cross-fitted rows, changes probability accuracy. A secondary question remains whether imaging adds value over matched clinical/biomarker predictors after this pooling.

This is a post-hoc internal sensitivity experiment. Earlier outcome summaries were already available when its protocol was written. The protocol is fixed before evaluating the new pooled heads; it is not a preregistered confirmation of an imaging benefit.

## Fixed inputs and lineage

- Parent: `results/xyz_fusion_v2`, five outer folds, three inner components and seeds 13/29/47.
- Scores: `results/xyz_fp32_spatial_sensitivity_v1`. All 270 stored FP32 meta/outer score receipts are reused. No images are reread, no new CNN inference is required and no CNN weights are fitted.
- Ninety expert components contain 135 modality-specific checkpoints. Receipt contracts, payload digests, checkpoint hashes, expert/selection ancestry, component membership and score order are checked on each invocation.
- Clinical variables and outcomes come directly from the immutable parent manifest. The tau frame must exactly match the parent's recorded tau-frame digest; only its verified tau feature is joined to the parent covariates.
- MRI, PET, plasma and tau retain their own eligibility. The observed input sizes are 1,059, 544, 696 and 358, respectively. They are not one four-modality cohort.
- The original source outputs are accessed read-only. New models, private predictions and evaluation journals are written under a separate versioned output directory on the project drive.

The FP32 route is selected because complete, consistently computed scores already exist for both meta and outer subjects. Using original AMP meta scores with FP32 outer scores would change the feature distribution and is rejected. The direct pooling comparisons use the **existing FP32 split-head** predictions, not the article's AMP predictions.

## Fitting and prediction

For outer fold f and seed s, component c supplies a three-horizon vector

`z_i^(f,s) = [-log S_c(t_1|x_i), -log S_c(t_2|x_i), -log S_c(t_3|x_i)]`

only for participants outside that expert's training and model-selection ancestry. Component holdout sets must be disjoint and their union must be the complete outer-training set. Each eligible participant contributes exactly one row per seed. The prediction times are the unchanged parent times, 547.875, 821.71875 and 1095.75 days.

The pooled clinical/image Cox model uses the parent clinical variables, optional biomarkers and these common time-specific cumulative-hazard features. Median imputation and standardization are fitted solely on the eligible outer-training OOF rows. Ridge alpha is fixed at 10, as in the parent. No selection of alpha or features from the new outer outcomes is allowed.

**There is no naive inner cross-validation over the already pooled feature matrix.** Although each row is OOF for its own CNN, CNNs producing other rows can have used a proposed inner-validation participant during training. Such tuning would require another nested generation of base predictions. Fixing alpha avoids presenting that shortcut as an independent validation procedure.

For each outer participant, apply the same pooled head separately to each of the three frozen component experts' scores. Average the resulting survival curves, then average survival curves across the three seeds:

`S_hat_f(t|x) = (1/3) sum_s [(1/3) sum_c S_head(f,s)(t|clinical(x), z_(f,s,c)(x))]`.

Feature vectors are not averaged before the head. Latent neural representations are not pooled. The design assumes that time-specific cumulative hazards are sufficiently comparable across expert fits; pooling does not eliminate that modeling assumption. Base networks still use two thirds of the outer-training participants, rather than being refitted on the full outer-training set.

All 180 heads are locked before any new outer performance is calculated. The source score files contain no outcomes passed to CNN prediction. Outer outcomes are also removed from the head-prediction frame. Evaluation is a separate committed step.

## Models and seven fixed comparisons

| Route | Pooled models | Principal comparisons |
|---|---|---|
| MRI | Clinical + amyloid; add MRI | Incremental MRI; pooled versus original FP32 augmented head |
| PET | Clinical + amyloid; add MRI; add PET | Incremental PET beyond clinical + MRI + amyloid; pooled versus original FP32 augmented head |
| Plasma | Clinical + amyloid; add plasma; add MRI + plasma | Incremental MRI beyond the plasma comparator |
| Tau | Clinical + amyloid; add tau; add MRI; add MRI + tau | Incremental tau without/with MRI |

All members of each comparison use the same held participants and the same eligible outer-training censoring distribution. Each individual appears once in the five outer test folds. Seeds and component experts are not counted as independent observations.

## Evaluation and uncertainty

Primary metric: unchanged IPCW integrated Brier score on the parent nine-point grid from 365 to 1095.75 days. The same administrative cutoff and established censoring conventions are retained. Uno's C and the existing fold-level calibration/dynamic-AUC summaries are descriptive secondary results. This version does not generate a new Uno-C confidence interval.

Paired participant bootstrap uses 40,000 resamples **within outer folds**, holding fitted models and training censoring distributions fixed. The point estimate weights individuals, not folds equally. Nominal 95% and Bonferroni-adjusted percentile intervals are both retained. Fixed families are: imaging (2 contrasts), biomarker extension (3), and pooling design (2). These are separate post-hoc questions, not multiplicity control over all previous model-development choices. Intervals do not incorporate CNN fitting, model selection or a new cohort's variability.

Every contrast is reported regardless of direction. No best-seed or best-model selection is made. A completed run cannot establish clinical utility, external validation, causal effects or a statistically guaranteed benefit.

## Execution and privacy

The CLI defaults to `preflight`, which validates inputs without fitting. `smoke` fits two real MRI heads for one fold/seed, creates predictions and checks resume/integrity, without calculating performance metrics. `all` fits the full experiment, evaluates, writes all seven contrasts and verifies artifacts. `fit` and `report` separate fitting/evaluation from bootstrap reporting. Rerunning the same command resumes committed units. Changing code, settings, runtime or inputs requires a new output version; never bypass the contract mismatch guard.

The public subfolder contains aggregate JSON and a Markdown report. The private subfolder contains participant membership, model artifacts and individual predictions; it must remain local. No GitHub publication is performed by any mode. No installation, data download or CNN retraining is performed by the runner.

## Method references

The general OOF training principle and the risk of fitting a stacker on in-sample base predictions are documented in [scikit-learn's stacking reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html). This project implements a custom survival head with ancestry-bound inputs; it does not treat censored outcomes as ordinary regression labels. Ridge Cox fitting and Breslow ties use [scikit-survival's CoxPHSurvivalAnalysis](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html). The locally installed versions are recorded in each run contract.
