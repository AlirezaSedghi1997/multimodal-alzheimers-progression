# Decisions

## 2026-05-25 - Add Method X-Cog as Separate Cognitive/Clinical Fusion Extension

- Decision: Add `Method X-Cog` as an exploratory extension that can combine fixed Method X tabular inputs with cognitive and psychological test summaries.
- Reason: Cognitive tests may add clinically meaningful signal, but they change the scientific input contract and must not silently replace Method X.
- Alternatives: Add all cognitive tests directly into Method X, or avoid cognitive testing entirely.
- Impact: Method X remains the primary T1 MRI plus fixed-tabular method. Method X-Cog is optional, separately staged, and must report leakage controls, temporal alignment, missingness, and cohort-label confounding.

## 2026-05-25 - Define PET Processing as Separate Method Y Exploratory Branch

- Decision: PET image preprocessing may be added only as an exploratory `Method Y` branch for QC, label reproducibility, and aggregate sensitivity analyses.
- Reason: In Method X, PET is the label/target source and must not become a model input. Adding PET images to the Method X model would change the scientific claim and create leakage risk.
- Alternatives: Add PET images as an additional Method X model input, or ignore PET image availability entirely.
- Impact: Method X remains restricted to T1 MRI plus fixed tabular variables. Any PET-assisted work must be documented, tested, and reported separately as exploratory, with public outputs limited to aggregate or hash-level artifacts.
## 2026-05-18 - Add SPM Backend as FSL Alternative for Full MRI Preprocessing

- Decision: Add an optional SPM/MATLAB full-preprocessing backend alongside the ANTs/FSL path.
- Reason: FSL installation is unreliable under the user's current network, while the user already has SPM installed in MATLAB.
- Alternatives: Keep waiting for FSL installation, or downgrade to Python-only preprocessing.
- Impact: Full preprocessing may proceed through SPM segmentation, bias-corrected brain masking, and MNI normalization if `MATLAB_BIN` and `SPM_DIR` pass the new checker. The final output remains hashed `sample_hash.nii.gz` and feeds the same 96-volume cache path.

## 2026-05-17 - Prepare Final 96/5-Fold 3D Validation Workflow

- Decision: Add explicit support for smoke-safe full preprocessing, preprocessed 96-volume cache construction, configurable 3D fusion/latent-alignment training, real 3D ablation summarization, and aggregate-only XAI.
- Reason: Final thesis-grade validation must not treat a limited smoke run as completed, and the 96-resolution model must train from full-preprocessed MRI outputs rather than raw T1 discovery.
- Alternatives: Reuse the existing 64-resolution cache/training path, or report only the surrogate fusion table.
- Impact: Remaining execution order is toolchain READY check, limited preprocessing smoke test, full resumable preprocessing, 96 cache build, 5-fold 3D ablation, XAI summary, then final refresh and WSL pytest.

## 2026-05-17 - Accept Interim Aggregate XAI for Current 64-Resolution 3D Checkpoint

- Decision: Accept the current XAI artifact as an interim aggregate interpretability result and mark `gate_xai=PASSED` for the current validation package.
- Reason: WSL2 Conda execution completed MRI-only Grad-CAM and tabular-only exact Shapley summaries with no PET input, no raw identifiers, no images, no per-subject outputs, and full tests passing.
- Alternatives: Defer all XAI until after final 96-resolution 5-fold training.
- Impact: Reporting must label this XAI as `current_3d_model_pending_final_96_refresh`; it must be rerun after final 96/5-fold model selection.

## 2026-05-17 - Reopen Project for Thesis-Grade Validation

- Decision: Change the active project state from completed implementation to `RESEARCH_COMPLETION_IN_PROGRESS`.
- Reason: Core Method X and real CUDA 3D training are complete, but the proposal/patch-level validation package still lacks full preprocessing, final 96/5-fold training, 3D ablation, subgroup audit, XAI, and tau consistency.
- Alternatives: Keep the project marked complete and only write the thesis from fast-track artifacts.
- Impact: Reporting must distinguish completed core implementation from in-progress thesis-grade validation.

## 2026-05-17 - Add Resume-Safe Long-Stage Infrastructure

- Decision: Add a hashed per-item resumable manifest utility and a specialized full MRI preprocessing stage that blocks unless the required toolchain/template is available.
- Reason: Full preprocessing and 96-resolution 5-fold training can run for hours or days and must survive terminal exits or interrupted execution without corrupt partial artifacts.
- Alternatives: Run long scripts monolithically, or rely only on final output existence.
- Impact: Long stages can resume from completed hashed items, and missing specialized tools are reported explicitly before heavy execution.

## 2026-05-17 - Require Compatible Registration Tool for Full MRI Preprocessing

- Decision: Restrict the full MRI preprocessing registration command check to `antsRegistrationSyNQuick.sh`.
- Reason: The implemented preprocessing command uses the SyNQuick interface; accepting `antsApplyTransforms` as a substitute would make the toolchain appear ready but fail at execution.
- Alternatives: Implement separate command builders for multiple registration interfaces.
- Impact: The blocker is stricter but more accurate. Future support for other registration commands should add explicit command templates and tests.

## 2026-05-17 - Replace Placeholder Power Analysis and Add Gap Matrix

- Decision: Generate realized-N power/precision reporting and an implementation gap matrix, then refresh final analysis and public bundle.
- Reason: `power_analysis.md` was a placeholder and the project needed a formal mapping from proposal/patch requirements to actual artifacts.
- Alternatives: Defer these artifacts until after final 96 training.
- Impact: Current gaps are now explicit: full MRI preprocessing, final 96/5-fold 3D training, subgroup audit, XAI, and tau rerun remain open.

## 2026-05-17 - Skip Site/Scanner Drift With Justification

- Decision: Generate an official `SKIP_WITH_JUSTIFICATION` site/scanner drift artifact.
- Reason: The locked master catalog contains no site/scanner metadata columns, and site/scanner values must not be invented or used as model inputs.
- Alternatives: Search raw clinical documents further, or infer site/scanner from file paths.
- Impact: Site/scanner drift is documented as unavailable for fold-wise sensitivity unless future metadata is provided.

## 2026-05-17 - Add Subgroup Audit as Aggregate Reporting Stage

- Decision: Add a subgroup performance audit for sex, age bins, APOE4, education, and MMSE using aggregate metrics and bootstrap CIs.
- Reason: Patch v3.4 requires subgroup reporting, but not as a hard fairness gate.
- Alternatives: Defer subgroup analysis until after final 96/5-fold training.
- Impact: Current surrogate/fusion subgroup audit can run in WSL2 Conda; final 3D subgroup reporting should be refreshed after final model selection.

## 2026-05-17 - Complete Current Subgroup Audit

- Decision: Accept the current aggregate subgroup audit into the refreshed validation package.
- Reason: WSL2 Conda execution completed with `n_cv=1360`, `n_external=340`, public bundle validation had no violations, and the full test suite reported `53 passed in 7.05s`.
- Alternatives: Wait until final 96/5-fold 3D training before reporting any subgroup results.
- Impact: Current subgroup reporting is available for the surrogate/fusion package, with the caveat that it is not a hard fairness certification and should be refreshed after final 3D model selection.

## 2026-05-17 - Resolve Tau Branch as Negative Exploratory Result

- Decision: Accept the tau rerun as `FULL` but scientifically negative.
- Reason: WSL2 Conda execution used 281 grouped-CV rows and produced R2 `-0.0446`, RMSE `0.0671`, and worse RMSE than the grouped mean baseline.
- Alternatives: Keep tau skipped, or tune tau models further before reporting.
- Impact: The tau inconsistency is resolved. Reporting must state that tau is secondary/exploratory and did not show out-of-fold predictive signal over the grouped mean baseline.

## 2026-05-17 - Mark Primary Method X Implementation Complete

- Decision: Mark the primary implementation package as completed after final WSL2 Conda validation passed.
- Reason: Real CUDA 3D training completed with `FULL_3D_TRAINING`, final analysis/public bundle were refreshed, `gate_3d_training` is `PASSED`, public bundle validation reports no violations, and the full test suite reports `40 passed in 10.01s`.
- Alternatives: Keep the project open until tau regression is rerun, or until a truly independent external cohort is added.
- Impact: Thesis/report writing can proceed from the refreshed artifacts. Tau remains exploratory and skipped with justification; external performance must be described as locked pseudo-external, not truly independent external validation.

## 2026-05-17 - Accept GPU 3D Training Artifact Into Final Reporting

- Decision: Refresh final analysis and public bundle after successful `FULL_3D_TRAINING` on CUDA.
- Reason: The 3D training artifact reports `final_claim_allowed=true`, Method X compliance, no PET input, no raw identifiers, and external-like inference without external fit.
- Alternatives: Keep the older surrogate-only final package, or wait for another 3D rerun before reporting.
- Impact: `gate_3d_training` is now `PASSED` in the final gate matrix. Scientific reporting must still preserve the locked pseudo-external holdout and cohort-label confounding caveats.

## 2026-05-17 - Align 3D Cross-Attention Fusion API

- Decision: Update `BottleneckCrossAttention` to support the v3.4 lightweight fusion arguments used by `MethodX3DClassifier` while preserving the older `out_dim`/`heads` calling style.
- Reason: `train-3d` failed before training because `MethodX3DClassifier` passed `hidden_dim`, `bottleneck_dim`, `num_heads`, and `dropout`, but the fusion module only accepted `out_dim` and `heads`.
- Alternatives: Change only the 3D model call site, or bypass cross-attention with concat fusion.
- Impact: The real 3D training stage can instantiate the intended cross-attention model, and the contract test now covers this constructor path.

## 2026-05-13 - Add T1 Visibility Diagnostics for 3D Cache

- Decision: Add fail-fast diagnostics to the 3D cache builder for `indexed_t1`, catalog T1 hash count, hash overlap, and T1 directory visibility; make NIfTI extension matching case-insensitive.
- Reason: A WSL run reported `ok=0` and `missing=1700`, while the Windows-visible data root has full hash overlap between the locked catalog and T1 files. This indicates a WSL mount/path visibility problem or file discovery mismatch, not a catalog problem.
- Alternatives: Continue writing all-zero HDF5 rows, or ask the user to manually inspect raw file names.
- Impact: The next cache run will either build normally or stop quickly with aggregate diagnostics only, without exposing raw IDs or filenames.

## 2026-05-12 - Harden 3D MRI Cache Publication

- Decision: Treat the 3D MRI HDF5 cache as complete only if it can be opened and contains the required Method X keys; publish new cache files atomically via a temporary file before replacing the final artifact.
- Reason: The previous `t1_volume_cache_64.h5` could be left partially written or corrupt, while the stage runner only checked for file existence and then advanced to `train-3d`.
- Alternatives: Manually delete the corrupt cache before every rerun, or let `train_3d_mri.py` fail with the raw HDF5 error.
- Impact: The next `--next`/`--run_all` execution will rebuild `mri-volume-cache` if the current HDF5 file is unreadable, then proceed to real 3D training only after the cache validates.

## 2026-05-12 - Start Real 3D MRI Training Phase

- Decision: Add Phase 2 stages for voxel-level T1 caching and real lightweight 3D MRI training with tabular fusion.
- Reason: The previous completed pipeline was a fast-track feature/surrogate implementation and did not satisfy the user's requirement for real 3D model training.
- Alternatives: Keep the feature-based surrogate only, or replace the whole repo with a new deep-learning scaffold.
- Impact: The project is reopened as `IN_PROGRESS`; `run_project_stage.py --run_all` now resumes at `mri-volume-cache`, then runs `train-3d`, final analysis, and public bundle refresh.

## 2026-05-11 - Tau Regression Reported as Skipped

- Decision: Keep exploratory tau regression in `SKIP_WITH_JUSTIFICATION` mode for the final package.
- Reason: The current tau regression artifact reports the tau target file as unavailable at stage execution time, while Method X primary AB pipeline, calibration, final analysis, public bundle, and tests completed successfully.
- Alternatives: Block the whole project until the tau target file is revalidated, or report only the feasibility table without a tau regression artifact.
- Impact: Primary Method X results are unaffected. Tau remains exploratory and must be described as skipped in the final reporting unless the tau target file is made available and `scripts/run_project_stage.py --run_all --data_root ...` is rerun.

## 2026-06-14 - Blood Biomarker Integration

- Decision: Add pTau217, GFAP, AB42/40 ratio, APOE4 as blood biomarker features.
- Reason: pTau217 is the strongest single blood AD biomarker (AUC > 0.90 in literature). Integrating blood features provides a minimally invasive benchmark.
- Implementation: `scripts/build_blood_features.py` extracts features, `scripts/train_baseline_blood.py` trains models.
- Impact: Blood baseline AUC=0.999 (with pTau217-based labels). 1083 subjects with blood biomarkers.

## 2026-06-14 - FLAIR WMH Feature Integration

- Decision: Extract white matter hyperintensity features from FLAIR data.
- Reason: WMH correlates with AD pathology and provides structural complement to T1 MRI.
- Implementation: `scripts/extract_flair_wmh_features.py` extracts WMH volume, ICV, corrected, ratio.
- Impact: FLAIR baseline AUC=0.543 (modest signal). 1754 subjects with FLAIR features.

## 2026-06-14 - Method Y PET Training Completed

- Decision: Train PET-only model on Method Y branch.
- Reason: Complete the exploratory PET-assisted branch as specified in preregistration.
- Implementation: `scripts/train_pet_method_y.py` with 233 PET samples, 3-fold CV, 10 epochs.
- Impact: PET-only AUC=0.819. Tautology caveat applies (PET-derived labels used for training).

## 2026-06-14 - Bug Fixes Applied

- Decision: Fix critical bugs in SynIB loss, conformal prediction, cross-attention, scheduler.
- Reason: These bugs affected training dynamics and evaluation validity.
- Changes: pos_weight in SynIB, inverted conformal sets, multi-query cross-attention, SequentialLR scheduler.
- Impact: All 235 tests pass. Training dynamics improved.

## 2026-06-14 - Unified Execution Program Created

- Decision: Create consolidated execution program in `docs/UNIFIED_EXECUTION_PROGRAM.md`.
- Reason: Multiple documents (proposal, patches, execution programs) needed consolidation.
- Impact: Single source of truth for project state and next steps.

