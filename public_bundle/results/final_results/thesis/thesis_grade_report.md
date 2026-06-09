# Thesis-grade Method X Report — execution program v3.9

## Required caveats

1. Output is **Aβ biomarker status**, not Alzheimer disease diagnosis.
2. PET is label/target only; never a Method X predictor.
3. Evaluation is locked **pseudo-external**, not independent external validation.
4. A4/LEARN cohort-label coupling (Cramér's V≈1.0 enrolled-only) limits pooled binary interpretation.
5. SF is internal decoupling only — never external validation.
6. ROI/WMH nonlinear cohort-probe ≈0.99 → high pooled AUC on ROI+tabular is **descriptive**, not biological proof.
7. WMH/reads alone ≈0.50 AUC; signal in MRI branch is driven by ROI volumes, not WMH/reads.
8. 3D CNN AUC (~0.72) below tabular/ROI descriptive AUC reflects architecture/cohort coupling; `final_claim_allowed=false`.

## Phase 2 Minimum MVP

- overall status: `MVP_COMPLETE`
- method_x_strict_tabular: `True`
- method_x_strict_3d_cnn: `True`
- mri_roi_wmh_reads_branch: `True`
- method_x_cog_sf_stress_tests: `DONE`

## Validity audit summary

- status: `CLAIM_MUST_BE_LIMITED_OR_REFRAMED`

## Cohort-probe (headline validity metric)

- 3D MRI latent linear AUC: `0.5031`
- 3D MRI latent nonlinear AUC: `0.5102`
- tabular linear AUC: `0.7313`
- ROI/WMH/reads nonlinear AUC: `0.9920`
- reading: 3D latent ≈0.50 (cohort not encoded); ROI branch ≈0.99 (cohort artifact).

## Incremental value vs validity (critical)

- highest **descriptive** pooled ROC-AUC step: `fixed_tabular_plus_roi_wmh_reads` (0.8207)
- validity-safe headline anchor: `fixed_tabular_plus_cognitive` (tabular 0.7289; 3D 0.7222 with cohort-probe ≈0.50)
- ROI nonlinear cohort-probe: `0.9920` — do **not** rank ROI+tabular as best biological model.

## Continuous centiloid (target only; tabular predictors)

- pooled enrolled best Spearman r: `0.4430`
- within-A4 best Spearman r: `0.3078`
- within-LEARN best Spearman r: `0.0751`
- centiloid_as_predictor: `False`
- interpretation: pooled descriptive; within-LEARN signal weak (~0.08).

## Phase 1 interpretable baselines

- status: `COMPLETED`
- fixed tabular ROC-AUC: `0.7289`
- tabular+cognitive ROC-AUC: `0.7932`
- ROI volumes only ROC-AUC: `0.7534`
- WMH only ROC-AUC: `0.4982` (single official ICV-normalized WMH column)
- MRI reads only ROC-AUC: `0.5028`
- tabular+ROI/WMH/reads ROC-AUC: `0.8207` (descriptive; cohort-probe caveat applies)

## Method X-Strict 3D CNN

- ROC-AUC (5-fold CV): `0.7222`
- external-like ROC-AUC: `0.7236`
- final_claim_allowed: `False`
- ablation best variant: `cross_attention+latent_alignment_0.15` (0.7173)
- note: cross-attention > concat but all variants <0.73; lightweight encoder; EfficientNet3D switch deferred.

## Method X-Cog (exploratory)

- status: `COMPLETED`
- best ROC-AUC: `0.7992`
- train-SF→enrolled collapse: see SF stress tests below.

## Method X-Cog/SF internal stress tests

- status: `COMPLETED`
- pooled Cramér's V: `0.8996`
- SF-inclusive cognitive ROC-AUC: `0.8307`
- train-SF→enrolled cognitive ROC-AUC: `0.5149`

## Cohort-label coupling audit

- before SF Cramér's V: `1.0000`
- after SF Cramér's V: `0.8996`

## Temporal-gap sensitivity (headline CPU models)

- status: `COMPLETED`
- window≤0d n=1699: tabular AUC `0.7289 [0.7025, 0.7546]`
- window≤60d n=154: tabular AUC `0.7969 [0.7006, 0.8883]`
- window≤90d n=266: tabular AUC `0.7435 [0.6486, 0.8243]`
- window≤120d n=330: tabular AUC `0.7585 [0.6725, 0.8290]`
- 3D CNN: full-cohort reference only per window (not retrained).

## Matched covariate subset (sensitivity)

- full cohort AUC: `0.7314 [0.7083, 0.7563]`
- matched 1:1 AUC (n_pairs=536): `0.7243 [0.6922, 0.7560]`
- 95% CI overlap full vs matched: `True` — difference ~0.007 is not significant if CIs overlap.

## Decision curve analysis

- status: `COMPLETED`
- tabular thresholds: `[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]`
- net benefit (model): `[0.668, 0.649, 0.629, 0.601, 0.553, 0.513, 0.47, 0.421, 0.392, 0.358]`

## Method Y PET (exploratory branch)

- design/training readiness status: `DESIGN_READY_EXECUTION_GATED`
- approved tier: `primary_feasible_90`
- design blockers: `[]`
- execution blockers (preprocess/cache): `['PET_PREPROCESS_INCOMPLETE', 'PET_CACHE_NOT_BUILT_YET']`
- warnings: `['PRIMARY_FEASIBLE_90_MIN_CLASS_BELOW_WARN_THRESHOLD', 'PRIMARY_STRICT_TIER_TOO_IMBALANCED_FOR_SOLE_PRIMARY_TRAINING', 'GOLD_STANDARD_30_FOR<RAW_ID_FIELD>DEN_FOR_PRIMARY_TRAINING_MIN_NEG_VIOLATION', 'COHORT_LABEL_AUDIT_POSSIBLE_STRONG_COUPLING', 'TRACER_LABEL_AUDIT_SINGLE_ROW_OR_COLUMN_REVIEW', 'GAP_LABEL_AUDIT_POSSIBLE_STRONG_COUPLING']`
- PET preprocess progress: `110/266` PASS
- regex_fallback_used (manifest): `False`
- project_inventory_used: `True`
- note: exploratory **design** may pass while **execution** (266/266 preprocess + PET cache) remains gated.

## Bio-Cog / Method Z (Phase 3; separate branch)

- status: `COMPLETED`
- core_blood_only AUC: `0.8086 [0.7878, 0.8286]`
- full_blood_panel AUC: `0.8691 [0.8508, 0.8865]`
- largest drop-one impact: drop `Ab42_40_ratio` → AUC `0.8254` (Δ=0.0437)
- separate minimally invasive benchmark; does not support Method X non-invasive claim.

## Clinical endpoint (exploratory; serious caveat)

- CDR worsening AUC: `0.6566 [0.6290, 0.6834]`
- PACC decline AUC: `0.6295 [0.6036, 0.6549]`
- AUC ~0.63–0.66 is modest; progression proxies are not AD diagnosis; confounded by substudy.

## External-like evaluation

- external_holdout_type: `locked_pseudo_external_random_patient_holdout`
- is_truly_independent_external: `False`

## Required wording

Enrolled A4/LEARN catalog exhibits cohort-label coupling. Pooled binary AUC is descriptive only. Method X-Strict remains the protected thesis anchor; extensions have separate claim boundaries.
