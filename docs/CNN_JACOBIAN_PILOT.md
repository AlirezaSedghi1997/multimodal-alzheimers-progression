# Baseline T1 / deformation-aware compact-CNN pilot

Protocol frozen before pilot performance was inspected. The initial scan-preparation version is `configs/cnn_jacobian_pilot_v1.json`; the executable version is `configs/cnn_jacobian_pilot_v1_2.json`. A pre-fitting boundary fix holds the Cox cumulative baseline hazard constant beyond the last fitting observation to cover the last neural bin; all 1–3-year evaluation times remain inside observed support. A separate output and source snapshot preserve the original preparation; no result-driven amendment was made. Date: 2026-10-14. This is development research in a historically explored cohort, not confirmatory validation. Existing OOF, primary article results, model weights and source images remain read-only. The source and a nonnumerical execution decision are published; restricted pilot artifacts remain local.

## Scientific question and decision

Does explicitly retaining baseline nonlinear deformation information improve a small, matched image model beyond the same model receiving registered T1 alone, while retaining the clinical comparator? The first review correctly identifies a weak tested encoder and an untested preprocessing explanation; it does not establish that MRI has no useful information or that nonlinear registration removes all atrophy. The OOF experiment changes the fusion stage, not representation learning. Its raw-score heterogeneity further prevents attributing the result to fusion sample size alone.

The pilot changes two things relative to historical modeling—network size and the imaging evaluation design—but isolates **Jacobian input** by pairing equal-parameter compact networks. It is not a direct, matched comparison against historical CNN performance. The two paired arms both have two input channels: T1 with a zero second channel, versus T1 with log-Jacobian. This prevents parameter count from explaining the comparison. No broad architecture sweep is authorized by this protocol.

## Source-driven choices

1. **ANTs geometry and Jacobian:** [official warp/Jacobian documentation](https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images,-pointsets-and-Jacobians) defines the fixed-space output and distinguishes image resampling from point transformation. It also records a 2025 issue in a non-geometric Jacobian implementation; the geometric calculation avoids that reported issue. We use `geom=True` and independently test identity, translation and known isotropic scales on an anisotropic, oblique physical grid. The nonlinear forward warp is the field used; its determinant excludes the separate affine transform. Positive/negative log values must not be casually labeled disease or atrophy, and no voxelwise localization claim follows.

2. **Disease information and longitudinal leakage:** [Regional Deep Atrophy, authors' paper](https://arxiv.org/abs/2304.04673) studies deformation-based information in longitudinal MRI. It motivates inspecting deformation information, but its task is different. Here only the baseline scan-to-fixed-template warp is admissible. A future-to-baseline warp would leak follow-up information and is forbidden.

3. **Sample efficiency:** [MedicalNet official repository](https://github.com/Tencent/MedicalNet) and [Med3D paper](https://arxiv.org/abs/1904.00625) provide pretrained 3D residual encoders. Their existence does not establish compatibility or benefit for this endpoint. The published code environment is old; dataset provenance, exact architecture, preprocessing and weight loading would need explicit checks. The 23-dataset weights are not imported into this first paired pilot.

4. **Brain-specific pretraining:** [BrainIAC paper](https://www.nature.com/articles/s41593-026-02202-6) supports evaluating brain-specific pretraining in low-data tasks; its official [repository](https://github.com/AIM-KannLab/BrainIAC) identifies a noncommercial academic license. The [Hugging Face conversion](https://huggingface.co/eugenehp/brainiac) is a third-party repackaging, not independent validation of model fidelity or participant nonoverlap. We do not use its example `strict=False` loading without checking every missing/unexpected key. This pilot uses no imported weights. A later pretrained arm requires original-source weights, license compliance, input parity and pretraining participant/cohort overlap review. No participant images are sent to a hosted model.

5. **Split integrity:** [Wen et al., reproducible Alzheimer's CNN evaluation](https://pubmed.ncbi.nlm.nih.gov/32417716/) motivates participant-level separation and reproducible preprocessing comparisons. No scan-, crop- or patch-level random split is used. All pilot subjects come from original outer-fold-0 training. Its 212 held-out participants are excluded from this pilot.

6. **Uncertainty:** [Varoquaux, cross-validation failure](https://pubmed.ncbi.nlm.nih.gov/28655633/) cautions against treating small-sample performance as precise. The pilot bootstrap is conditional on the fitted models, not a distribution over retraining. An 80% interval is a resource-allocation signal, explicitly not a conventional significance claim. The absolute and relative thresholds below are analyst-selected engineering gates, not established clinically important differences.

7. **Survival objective:** [Gensheimer and Narasimhan](https://pubmed.ncbi.nlm.nih.gov/30701130/) supplies the discrete-hazard neural-survival framework. Our right-closed event/censor convention remains the project's existing, separately tested implementation. No uncensored binary classification substitutes for the time-to-event endpoint.

## Geometry, data and compute

The source is the frozen baseline MRI cohort, receipt-bound `registration_1Warp.nii.gz`, corresponding affine and `tensor.npy`. The fixed template and mask hashes are verified. No registration is rerun and no source is overwritten. Finite positive determinants within the brain mask are mandatory; failures stop execution. Log-Jacobian is clipped at fixed [-2,2], with the clipped fraction reported. It is **not** individually z-scored, which would discard between-person scale information. Its common scalar mean/SD is learned from pilot fitting images only.

Aligned 2×2×2 block averaging reduces 128³ to 64³ for this computational pilot. This resolution choice may obscure fine structures; it is not equivalent to the original input. The masked template grid is used; no new skull-stripping success is claimed. Affine determinant is recorded for QC but excluded from the channel, so the input represents nonlinear residual deformation rather than global head-size change.

The pilot has 512 participants: 320 fitting, 64 epoch selection and 128 audit. Sampling and partitions are fixed by seed and stratified by early-event/later-event/censor category, without outcome-dependent resampling of a favorable split. This is about half the small cohort, chosen to retain a minimal number of three-year events; a few dozen participants can test software but cannot support a meaningful prognostic gate. Six compact fits replace the much larger historical multi-expert training workload. The original outer test is not evaluated. Nothing from the pilot audit enters weight fitting, imputation, normalization, penalty selection or early stopping.

The compact encoder uses stride-2 3D convolution blocks with 8/16/32 channels, GroupNorm, SiLU, mathematically equivalent deterministic block averaging to spatial2³ and a 64-unit hidden layer; fewer than 100,000 parameters. Two age/hippocampal-volume regressors (one per input arm, seed13) test learnability of known anatomy. A real eight-subject overfit check must reduce standardized MSE by at least75% before substantive fitting. This checks optimization, not generalization.

Four survival fits pair T1 and T1+log-Jacobian at seeds13/29. They learn nine discrete-hazard logit adjustments to a clinical ridge-Cox baseline fitted only on the same320 people. The image head starts at zero so its initial prediction equals the clinical baseline at bin boundaries. The clinical variables are age, sex, education, APOE4, baseline PACC, baseline MMSE, treatment arm and amyloid Centiloid. This is a residual multimodal survival model, not an imaging-only CNN. Both arms have identical training budgets, FP32 computation, optimizer and selection rule. Fixed learning rate0.001, weight decay0.001, batch8, maximum24epochs, minimum8, patience6; maximum one hour of model fitting per invocation. All six models are locked before any audit performance is computed.

Benchmarks on the identical fitting/audit split: clinical Cox; clinical Cox plus three regional volumes; clinical Cox plus64 fixed spatially averaged Jacobian features. These are reported regardless of direction. They are not post hoc candidates substituted for the primary paired CNN comparison.

## Gate for considering a full experiment

All conditions must pass: at least10 audit events by three years and30 still at risk; audit R²>0 for both age and hippocampal-volume prediction with T1+Jacobian; both survival seeds favor Jacobian on IBS; seed-averaged improvement at least0.0004 and1% relative to T1; upper bound of the conditional paired80% bootstrap interval below zero; lower IBS than the matched clinical comparator; no concordance reduction greater than0.02. All failures and seeds remain in the record. No silent re-seeding or threshold relaxation is allowed.

A positive gate warrants more evaluation, not proof of scientific/clinical superiority. Full-cohort execution needs a new locked protocol: the same small candidate set and selection budget applied only inside each outer-training partition, complete participant OOF predictions, fold-training censoring estimates, paired bootstrap and calibration, fixed comparison families, matched clinical/regional baselines, scanner/site sensitivity, and complete numeric/visual QC. The pilot architecture choice remains a historical development choice even with this nesting; the historical fivefold dataset cannot become an untouched confirmatory sample. Independent-cohort evaluation remains necessary for external-generalization claims. A negative gate stops scaling of this candidate rather than implying MRI or Jacobian is uninformative.


## Execution outcome and implementation amendments

The pilot completed but did not meet its locked advancement gate. No full-cohort
CNN/Jacobian run was launched, and no efficacy result from this pilot replaces
the completed parent or OOF analyses. All pilot outcomes, unsuccessful technical
attempts and source snapshots remain in the restricted research archive. This
release distributes the decision and source, not numerical pilot performance.

Two implementation amendments preceded any successful optimizer step or audit
evaluation. Version 1.1 handles the last Cox baseline-hazard step without extending
the 1–3-year metric horizon; version 1.2 replaces unsupported deterministic CUDA
adaptive pooling with mathematically equivalent nonoverlapping block means.
All three configurations are retained, and the runner defaults to version 1.2.

For a separate authorized reproduction tree, review its input contract and use:

```text
python -B -m pytest tests/test_cnn_jacobian_pilot_v1.py -q -p no:cacheprovider
python -B scripts/run_cnn_jacobian_pilot_v1.py --mode prepare
python -B scripts/run_cnn_jacobian_pilot_v1.py --mode fit
python -B scripts/run_cnn_jacobian_pilot_v1.py --mode evaluate
```

The code resolves the research root from the runner's location. Reproduction
requires the authorized parent contract, accepted baseline scans/warps, exact
template assets and dependency versions. Do not run in a live source tree or
overwrite historical outputs. Changing source or configuration requires a new
versioned output directory. `full_dataset_run_automatic` remains false.
The preparation stage can reuse verified version-1.1 scans when available; it
recomputes them if the preceding cache is absent. Source/config hashes bind the
run. No restricted inputs, trained weights or participant predictions are included.
