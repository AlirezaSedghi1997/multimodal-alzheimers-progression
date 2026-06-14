# Literature Loop Cards

Status: project memory generated from `docs/literature_backbone_evidence_2023_2026.yaml`.
Use these cards before re-reading large paper batches. Fields marked "not recorded"
mean the seed registry did not contain that detail and the paper must be checked
before manuscript citation.

## LIT001 - T1w MRI amyloid prediction domain ceiling reference

- Problem: estimate a realistic ceiling for MRI-only amyloid prediction.
- Data: not recorded in seed registry.
- Modalities: T1w MRI.
- Architecture: not recorded in seed registry.
- Leakage risk: overusing this as proof of clinical diagnosis.
- Validation: not recorded in seed registry.
- Metric: MRI-only amyloid AUC context around 0.60-0.75.
- Claim boundary: ceiling context only, not project validation.
- Code/repo: none recorded.
- Project decision: use to calibrate Method X expectations.

## LIT002 - EfficientNet3D T1+FLAIR amyloid classification

- Problem: test whether FLAIR adds value beyond T1.
- Data: not recorded in seed registry.
- Modalities: T1w MRI and FLAIR.
- Architecture: EfficientNet3D.
- Leakage risk: do not imply FLAIR is a Method X-Strict input.
- Validation: not recorded in seed registry.
- Metric: seed registry notes about +0.05 AUC vs T1 alone.
- Claim boundary: FLAIR sensitivity track only.
- Code/repo: none recorded.
- Project decision: keep FLAIR as controlled sensitivity extension.

## LIT003 - External validation gap in AD ML

- Problem: external validation claims are often overstated.
- Data: narrative review context; specific data not recorded.
- Modalities: multimodal AD ML, not a project input recipe.
- Architecture: not applicable.
- Leakage risk: relabeling pseudo-external holdout as independent validation.
- Validation: emphasizes independent external validation gap.
- Metric: not recorded in seed registry.
- Claim boundary: governance support only.
- Code/repo: none recorded.
- Project decision: keep locked holdout wording as pseudo-external.

## LIT004 - Decision curve analysis for amyloid screening

- Problem: evaluate clinical utility beyond AUC.
- Data: amyloid screening context; specific data not recorded.
- Modalities: not recorded.
- Architecture: not applicable.
- Leakage risk: DCA must use out-of-sample predictions only.
- Validation: not recorded in seed registry.
- Metric: net benefit by threshold.
- Claim boundary: utility framing, not diagnosis.
- Code/repo: none recorded.
- Project decision: supports Method Y and Method X DCA reporting.

## LIT005 - Multimodal fusion small-N caution

- Problem: fusion models can overfit at small N.
- Data: not recorded in seed registry.
- Modalities: multimodal neuroimaging.
- Architecture: not recorded.
- Leakage risk: ViT or fusion headline at Method Y N=184.
- Validation: not recorded in seed registry.
- Metric: report effective N and uncertainty.
- Claim boundary: cautionary discussion only.
- Code/repo: none recorded.
- Project decision: avoid strong Method Y architecture claims.

## LIT006 - PET-guided MRI amyloid detection KD

- Problem: transfer PET information into MRI-only inference.
- Data: OASIS-3 context in registry.
- Modalities: PET teacher, MRI student.
- Architecture: cross-modal knowledge distillation.
- Leakage risk: PET must not be available at Method X inference.
- Validation: not recorded in seed registry.
- Metric: T1w AUC about 0.73; multi-contrast up to about 0.74.
- Claim boundary: sensitivity analysis only.
- Code/repo: github.com/FrancescoChiumento/pet-guided-mri-amyloid-detection.
- Project decision: implemented as `scripts/run_pet_guided_kd_train.py`.

## LIT007 - DiaMond multimodal ViT MRI+PET

- Problem: multimodal MRI+PET representation learning.
- Data: not recorded in seed registry.
- Modalities: MRI and PET.
- Architecture: multimodal ViT.
- Leakage risk: transformer overclaim at small Method Y N.
- Validation: not recorded in seed registry.
- Metric: not recorded in seed registry.
- Claim boundary: Method Y reference pattern only.
- Code/repo: github.com/ai-med/DiaMond.
- Project decision: adopt paired HDF5 cache pattern, not ViT headline.

## LIT008 - TransMF_AD transformer fusion

- Problem: MRI+PET multimodal fusion.
- Data: not recorded in seed registry.
- Modalities: structural MRI and PET.
- Architecture: transformer / cross-attention fusion.
- Leakage risk: unablated transformer claim.
- Validation: not recorded in seed registry.
- Metric: not recorded in seed registry.
- Claim boundary: comparison row only.
- Code/repo: github.com/Kateridge/TransMF_AD.
- Project decision: keep as planned Method Y fusion reference.

## LIT009 - MultimodalAD Swin-UNETR + tabular

- Problem: large multimodal AD modeling.
- Data: ADNI-scale context, seed registry notes n=1147.
- Modalities: imaging plus tabular.
- Architecture: Swin-UNETR and TabPFN.
- Leakage risk: copying ADNI-scale architecture into N=184 subset.
- Validation: not recorded in seed registry.
- Metric: not recorded in seed registry.
- Claim boundary: small-N caution only.
- Code/repo: github.com/huangshuo343/multimodalAD.
- Project decision: reference architecture but avoid headline use.

## LIT010 - AI in AD diagnosis using PET-MRI review

- Problem: PET-MRI AI evidence and validation limitations.
- Data: narrative review.
- Modalities: PET and MRI.
- Architecture: multiple AI methods.
- Leakage risk: external validation and label leakage warnings.
- Validation: review-level evidence only.
- Metric: not recorded in seed registry.
- Claim boundary: discussion, not project proof.
- Code/repo: none recorded.
- Project decision: use for leakage and validation caveats.

## LIT011 - DL for early AD diagnosis using multimodal neuroimaging

- Problem: contextualize multimodal fusion gains.
- Data: review-level evidence.
- Modalities: multimodal neuroimaging.
- Architecture: deep learning survey.
- Leakage risk: importing diagnosis wording into biomarker thesis.
- Validation: review-level evidence only.
- Metric: fusion gains are context-dependent.
- Claim boundary: discussion only.
- Code/repo: none recorded.
- Project decision: cite only with biomarker-status wording.

## LIT012 - Plasma p-tau217 pre-screening for Amyloid-PET

- Problem: blood biomarker pre-screening for amyloid PET.
- Data: ADNI context in registry.
- Modalities: plasma p-tau217 and amyloid PET.
- Architecture: not recorded in seed registry.
- Leakage risk: blood biomarkers must stay out of Method X.
- Validation: not recorded in seed registry.
- Metric: DCA framing.
- Claim boundary: Method Z contrast only.
- Code/repo: none recorded.
- Project decision: support Method Z and DCA discussion.

## LIT013 - Radiomics vs SUVr for AV45 PET amyloid classification

- Problem: compare PET radiomics and SUVr for amyloid labels.
- Data: AV45 PET context.
- Modalities: amyloid PET.
- Architecture: radiomics model.
- Leakage risk: PET input on PET-derived label is tautological.
- Validation: not recorded in seed registry.
- Metric: DCA and amyloid classification metrics.
- Claim boundary: Method Y PET-only supplementary.
- Code/repo: none recorded.
- Project decision: always attach PET-label tautology caveat.

## LIT014 - LLM extraction from brain MRI reports

- Problem: extract MRI-read variables from reports.
- Data: brain MRI reports; exact cohort not recorded.
- Modalities: text reports derived from MRI reads.
- Architecture: LLM extraction.
- Leakage risk: report-extraction branch can encode site/cohort practice.
- Validation: balanced accuracy reported for MTA/Fazekas/microbleeds.
- Metric: MTA about 90-96 percent balanced accuracy; Fazekas about 94 percent.
- Claim boundary: ROI/reads sensitivity branch only.
- Code/repo: none recorded.
- Project decision: planned sensitivity support for `mri_roi_wmh_reads`.

## LIT015 - GNOVA trajectory modeling

- Problem: model irregular cognitive and clinical trajectories.
- Data: ADNI n=1727 in registry.
- Modalities: routine clinical/cognitive variables.
- Architecture: GRU, Neural ODE, VAE uncertainty.
- Leakage risk: future visits must not leak into baseline prediction.
- Validation: CI coverage reported in registry.
- Metric: MAE CDR-SB 1.35, MMSE 2.28; 85 percent CI coverage.
- Claim boundary: clinical endpoint proxy, not AD diagnosis.
- Code/repo: none recorded.
- Project decision: optional Method X-Cog clinical endpoint stretch.

## LIT016 - MMS-PRM worst-dimension optimization

- Problem: prevent one strong metric from hiding weak dimensions.
- Data: not project-specific.
- Modalities: multimodal reasoning setting.
- Architecture: worst-dimension / Chebyshev scalarization.
- Leakage risk: using scalar AUC alone despite calibration or cohort failure.
- Validation: not recorded in seed registry.
- Metric: augmented Chebyshev score over multiple dimensions.
- Claim boundary: audit only, not full MCTS training.
- Code/repo: github.com/leibniz-Man/MMS-PRM.
- Project decision: implemented as `src/evaluation/worst_dimension_audit.py`.

## LIT017 - Rationale SFT hurts ADRD prediction

- Problem: rationale-style supervision may harm prediction.
- Data: ADRD prediction context; exact data not recorded.
- Modalities: tabular or text reasoning context.
- Architecture: rationale SFT.
- Leakage risk: chain-of-thought features contaminating tabular Method X.
- Validation: not recorded in seed registry.
- Metric: not recorded in seed registry.
- Claim boundary: anti-pattern only.
- Code/repo: none recorded.
- Project decision: excluded; do not add rationale LLM to Method X.

## LIT018 - MFSCA multifractal MRI signatures

- Problem: quantify MRI structural heterogeneity with multifractals.
- Data: dementia MRI context; exact data not recorded.
- Modalities: MRI.
- Architecture: Hilbert ordering plus MFDFA features.
- Leakage risk: texture branch must be evaluated out-of-fold and not overclaimed.
- Validation: not recorded in seed registry.
- Metric: H, alpha width, asymmetry slice profiles.
- Claim boundary: R3 texture branch, not primary Method X-Strict claim.
- Code/repo: none recorded.
- Project decision: implemented as `src/features/mfsca_texture.py`.

## LIT019 - SynIB synergistic information bottleneck

- Problem: reduce shortcut learning and encourage multimodal synergy.
- Data: multimodal learning context; exact data not recorded.
- Modalities: generic multimodal; project maps MRI plus tabular/PET branches.
- Architecture: information bottleneck with modality masking.
- Leakage risk: synergy loss cannot justify leaky PET in Method X.
- Validation: not recorded in seed registry.
- Metric: PID-inspired synergy objective plus downstream metrics.
- Claim boundary: training objective, not guaranteed AUC gain.
- Code/repo: none recorded.
- Project decision: implemented as `src/models/losses/synib_synergy_loss.py`.

## LIT020 - Glucosamine and AD risk epidemiology news

- Problem: epidemiologic association context.
- Data: news summary; primary paper not recorded.
- Modalities: non-imaging exposure.
- Architecture: not applicable.
- Leakage risk: using epidemiology news as modeling evidence.
- Validation: not recorded in seed registry.
- Metric: not recorded in seed registry.
- Claim boundary: discussion only.
- Code/repo: none recorded.
- Project decision: do not use as model input.

## LIT021 - Mislinked LLM/chatbot philosophy draft

- Problem: likely unrelated to amyloid imaging.
- Data: not applicable until verified.
- Modalities: not applicable.
- Architecture: not applicable.
- Leakage risk: miscitation.
- Validation: not applicable.
- Metric: not applicable.
- Claim boundary: excluded pending user confirmation.
- Code/repo: none recorded.
- Project decision: do not cite or implement.

## LIT022 - Optimal multimodal information bottleneck

- Problem: balance multimodal information bottleneck pressure.
- Data: not recorded in seed registry.
- Modalities: multimodal.
- Architecture: dynamic lambda information bottleneck.
- Leakage risk: planned extension must not alter Method X input allowlist.
- Validation: not recorded in seed registry.
- Metric: not recorded in seed registry.
- Claim boundary: future SynIB extension.
- Code/repo: none recorded.
- Project decision: planned only.

## LIT023 - MRdIB representation-disentangled information bottleneck

- Problem: disentangle unique, redundant, and synergistic information.
- Data: not recorded in seed registry.
- Modalities: multimodal.
- Architecture: representation-disentangled IB.
- Leakage risk: do not treat disentanglement as deconfounding proof.
- Validation: not recorded in seed registry.
- Metric: not recorded in seed registry.
- Claim boundary: Method Y fusion idea, not current claim.
- Code/repo: none recorded.
- Project decision: planned only.

## LIT024 - PID-guided tau-PET synthesis from T1+FLAIR

- Problem: synthesize tau-PET related latent factors from structural inputs.
- Data: not recorded in seed registry.
- Modalities: T1, FLAIR, tau PET.
- Architecture: PID-guided latent factor model.
- Leakage risk: synthetic PET must never enter Method X and must remain fold-local.
- Validation: not recorded in seed registry.
- Metric: not recorded in seed registry.
- Claim boundary: supplementary synthesis policy only.
- Code/repo: none recorded.
- Project decision: complements synthetic MRI-to-PET and FLAIR sensitivity controls.

## LIT025 - OMIB duplicate anchor

- Problem: duplicate literature anchor.
- Data: same as LIT022.
- Modalities: same as LIT022.
- Architecture: same as LIT022.
- Leakage risk: duplicate citation inflation.
- Validation: same as LIT022.
- Metric: same as LIT022.
- Claim boundary: consolidated under LIT022.
- Code/repo: none recorded.
- Project decision: do not count as a separate evidence source.

## LIT026 - PID in multimodal learning

- Problem: theoretical foundation for partial information decomposition.
- Data: not recorded in seed registry.
- Modalities: multimodal.
- Architecture: PID analysis rather than project code architecture.
- Leakage risk: theory citation must not imply empirical validation.
- Validation: not recorded in seed registry.
- Metric: post-hoc synergy quantification.
- Claim boundary: theoretical support for IP1/IP5.
- Code/repo: none recorded.
- Project decision: use to explain the SynIB/PID rationale after verification.

## LIT027 - Strategic Decision Support for AI Agents

- Problem: How to optimally decide when AI agents should seek external support.
- Data: Synthetic and real-world decision tasks.
- Modalities: Text, structured data.
- Architecture: Threshold-based policy with calibration-on-the-fly.
- Leakage risk: None (agent management, not model training).
- Validation: Cross-validation with error control.
- Metric: Support usage reduction while controlling missed-support error.
- Claim boundary: Agent management framework, not clinical model.
- Code/repo: not recorded.
- Project decision: Apply threshold-based subagent spawning strategy for token optimization.

## LIT028 - MSFL for Brain Disorder Detection (arxiv:2603.24603)

- Problem: Combining amplitude and phase of fMRI signals for brain disorder detection.
- Data: ABIDE I (autism), REST-meta-MDD (depression).
- Modalities: fMRI (amplitude + phase).
- Architecture: Multi-scale fusion learning (MSFL) with SWC and PS features.
- Leakage risk: Must not imply fMRI is Method X input.
- Validation: Cross-validation with SHAP explanation.
- Metric: Classification accuracy, feature importance.
- Claim boundary: Fusion methodology reference, not project validation.
- Code/repo: not recorded.
- Project decision: Consider phase synchronization features for MRI branch enhancement.

## LIT029 - Oncomorphic Neural Agent Populations (arxiv:2503.12743)

- Problem: Distributed AI under sequential task exposure and uneven compute.
- Data: Synthetic nonlinear classification.
- Modalities: Not applicable (agent framework).
- Architecture: Cancer-inspired multi-agent with replication/mutation/migration.
- Leakage risk: None (agent management).
- Validation: Multi-task competence evaluation.
- Metric: Local accuracy, multi-task competence, entropy analysis.
- Claim boundary: Agent management inspiration, not clinical model.
- Code/repo: not recorded.
- Project decision: Inspiration for evolutionary architecture search and resource management.

## LIT030 - Flow Matching for Brain Dynamics (arxiv:2606.11833)

- Problem: Generating realistic fMRI brain dynamics for unseen cognitive tasks.
- Data: fMRI task-based datasets.
- Modalities: fMRI, language (text descriptions).
- Architecture: Per-timestep conditioned diffusion transformer.
- Leakage risk: Must not use for clinical diagnosis generation.
- Validation: Cross-task generalization, held-out task conditions.
- Metric: Region-specific recruitment recovery, zero-shot generation.
- Claim boundary: Synthetic data generation reference, not clinical model.
- Code/repo: Available at paper URL.
- Project decision: Apply flow matching for MRI-to-PET synthesis improvement.

## LIT031 - UniTok Time Series Tokenizer (arxiv:2606.09861)

- Problem: Transforming time series into discrete tokens for LLM processing.
- Data: Multiple time series benchmarks.
- Modalities: Time series (any).
- Architecture: Vector-quantized autoencoder + LLM (NTP).
- Leakage risk: None (tokenization method).
- Validation: Zero-shot and few-shot forecasting/classification.
- Metric: Forecasting accuracy, classification accuracy.
- Claim boundary: Tokenization methodology, not clinical model.
- Code/repo: not recorded.
- Project decision: Apply to longitudinal biomarker trajectory tokenization for temporal modeling.

## LIT032 - Personalized Graph-Based Dynamical Model for AD (arxiv:2511.08847)

- Problem: Forecasting cortical atrophy cascade at individual level from longitudinal MRI/PET.
- Data: 1,891 participants from ADNI.
- Modalities: Longitudinal MRI and PET.
- Architecture: Personalized graph-based dynamical model with individualized brain graphs.
- Leakage risk: Must not imply longitudinal data is available in A4/LEARN cross-sectional subset.
- Validation: Prediction of AD biomarkers (amyloid-beta, tau, neurodegeneration, cognition).
- Metric: Outperforms clinical and neuroimaging benchmarks.
- Claim boundary: Digital twin framework reference, not project validation.
- Code/repo: not recorded.
- Project decision: Inspiration for longitudinal trajectory prediction if multi-timepoint data becomes available.

## LIT033 - Operator Learning for AD Progression (arxiv:2507.16148)

- Problem: Learning patient-specific disease operators for AD progression modeling.
- Data: Longitudinal multimodal imaging, biomarker, and clinical data.
- Modalities: Longitudinal MRI, PET, biomarkers, clinical data.
- Architecture: Geometry-aware neural operators with Laplacian eigenfunction bases.
- Leakage risk: Must not imply multi-timepoint data is available in A4/LEARN.
- Validation: Prediction accuracy exceeding 90% across multiple biomarkers.
- Metric: High prediction accuracy, personalized predictions.
- Claim boundary: Digital twin paradigm reference, not project validation.
- Code/repo: not recorded.
- Project decision: Inspiration for personalized modeling if longitudinal data becomes available.

## LIT034 - SSL-AD: Spatiotemporal Self-Supervised Learning (arxiv:2509.10453)

- Problem: Self-supervised learning for Alzheimer's prediction with variable-length inputs.
- Data: 4 publicly available datasets, 3,161 patients.
- Modalities: 3D brain MRI (longitudinal).
- Architecture: Temporal SSL with order prediction and contrastive learning.
- Leakage risk: Must not imply longitudinal data is available in A4/LEARN.
- Validation: 7 downstream tasks including diagnosis, conversion detection.
- Metric: Outperforms supervised learning on 6/7 tasks.
- Claim boundary: SSL methodology reference, not project validation.
- Code/repo: github.com/emilykaczmarek/SSL-AD
- Project decision: Inspiration for self-supervised pre-training if larger unlabeled dataset becomes available.

## LIT035 - FOMO25: Brain MRI Foundation Models (arxiv:2604.11679)

- Problem: Foundation models for clinical brain MRI analysis with domain shift.
- Data: FOMO60K pretraining dataset, clinical workflow data.
- Modalities: Brain MRI (T1, FLAIR).
- Architecture: Self-supervised pretraining (MAE, contrastive).
- Leakage risk: Must not imply foundation model pre-training is feasible on A4/LEARN size.
- Validation: Few-shot and out-of-domain settings.
- Metric: SSL pretraining improves generalization on clinical data.
- Claim boundary: Foundation model methodology reference, not project validation.
- Code/repo: not recorded.
- Project decision: Inspiration for self-supervised pre-training if larger dataset becomes available.

## LIT036 - Conformal Prediction Pitfalls in Medical Imaging (arxiv:2506.18162)

- Problem: Pitfalls and limitations of conformal predictions in medical imaging.
- Data: Dermatology and histopathology examples.
- Modalities: Medical images.
- Architecture: Conformal prediction framework analysis.
- Leakage risk: None (methodology paper).
- Validation: Analysis of reliability under distributional shifts.
- Metric: Coverage guarantees and practical utility.
- Claim boundary: Methodological caution, not project validation.
- Code/repo: not recorded.
- Project decision: Use to guide conformal prediction implementation with awareness of limitations.

## LIT038 - COMPASS: Robust Feature Conformal Prediction (arxiv:2509.22240)

- Problem: Metric-based conformal prediction for medical segmentation.
- Data: Skin lesions and anatomical structures.
- Modalities: Medical images.
- Architecture: Feature-space perturbation with learned importance weights.
- Leakage risk: None (methodology paper).
- Validation: Valid marginal coverage under exchangeability.
- Metric: Tighter intervals than traditional CP baselines.
- Claim boundary: Conformal prediction methodology reference.
- Code/repo: not recorded.
- Project decision: Inspiration for feature-space conformal prediction in neuroimaging.

## LIT039 - Conformal Prediction Pitfalls in Medical Imaging (arxiv:2506.18162)

- Problem: Pitfalls and limitations of conformal predictions in medical imaging.
- Data: Dermatology and histopathology examples.
- Modalities: Medical images.
- Architecture: Conformal prediction framework analysis.
- Leakage risk: None (methodology paper).
- Validation: Analysis of reliability under distributional shifts.
- Metric: Coverage guarantees and practical utility.
- Claim boundary: Methodological caution, not project validation.
- Code/repo: not recorded.
- Project decision: Use to guide conformal prediction implementation with awareness of limitations.

## LIT040 - Human-AI Collaborative Uncertainty Quantification (arxiv:2510.23476)

- Problem: Human-AI collaboration within uncertainty quantification.
- Data: Image classification, regression, text-based medical decision making.
- Modalities: Multiple.
- Architecture: Two-threshold structure over score function.
- Leakage risk: None (framework paper).
- Validation: Collaborative prediction sets outperform either agent alone.
- Metric: Higher coverage, smaller set sizes.
- Claim boundary: Human-AI collaboration framework reference.
- Code/repo: not recorded.
- Project decision: Inspiration for human-in-the-loop uncertainty quantification.
