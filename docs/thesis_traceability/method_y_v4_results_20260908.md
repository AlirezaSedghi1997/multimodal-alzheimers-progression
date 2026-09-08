# Method Y v4 Image Factorial — Thesis Results (append-only, 2026-09-08)

> Status: `FINALIZED` internal exploratory. Ceiling `POST_HOC_INTERNAL_EXPLORATORY`.
> `positive_claim_allowed=false`, `external_claim_allowed=false`.
> This file does NOT mutate any finalized checkpoint, metric, or locked contract.
> It only compiles locked evidence into thesis-ready prose + tables.
> Skills applied: `tdd` (verifier-as-spec), `statistical-analysis` (NI + bootstrap reading),
> `scientific-writing` (evidence binding, no fabrication), plus independent strict
> supervisor audits Stage-1 (completion) and Stage-2 (numbers) — both PASS
> (Stage-2 with one precision correction on scanner AUC range, incorporated below).

## 1. Execution verdict (thesis § Methods — provenance)

- **Training COMPLETED by operator, no rerun required.**
  `results/method_y_revival_v4_image_factorial_v1/orchestration_state.json`:
  `stage=FINALIZED`, `outer_cells_finalized=50`, `run_contract_sha256=969ebd0f61d9…7743e`,
  `inner_only_selected_arms={amyloid:A4, tau:A1}`.
- `run_contract.json` digest identical (`969ebd0f…`). Schema 4.0, seed 20260714.
- Verifier (CPU, read-only, re-ran 2026-09-08):
  `--level final` → `PASS` (`expected_outer_cells=50`, `valid_outer_cells=50`, `failures=[]`);
  `--level readiness` → `PASS` (amyloid 560/110, tau 306/75, coverage 560/306, 5 outer × 3 inner).
- `reports/factorial_benchmark_summary.json`: `status=COMPLETED`, `analysis_role=POST_HOC_INTERNAL_EXPLORATORY`.
- Both branches `BRANCH_OUTER_CV_COMPLETE` (25+25 cells). `development/**/*.pt` = 1000
  (factorization 5 arms × 2 branches × 5 outer × 20; 20 = 2 final + 18 inner
  [3 candidates × 3 inner folds × 2 files]). 50 `reports/*_public.json` + 50
  `orchestration/*/winner_*_outer_*.json` present.
- All `runtime/logs/*.err` = 0 bytes. Sole non-zero stderr (158 B,
  `production_20260901_124531`) is a PowerShell launcher quoting error, superseded by
  the successful `124557` stream. No training traceback.
- **Terminal snippet in the user prompt is NOT a failure.** It shows the correct
  closed-gate tail: per-pair `claim.positive_claim_allowed=false`,
  `hard_blockers=[POST_HOC_OUTER_TEST_PREVIOUSLY_EXPOSED, QUANTITATIVE_EQUIVALENCE_NOT_ESTABLISHED, SCANNER_ROBUSTNESS_NOT_ESTABLISHED, LOCKED_EXTERNAL_VALIDATION_MISSING]`,
  per-pair `privacy_scan=PASS`, aggregate `science_gate_blockers=[3 items]`,
  aggregate `privacy_scan=FAIL` on `reports/factorial_benchmark_summary.json:contains:H:\`.
  That FAIL is absolute-path disclosure in 10 `scanned_root` fields (10 occurrences
  of `H:\`), NOT a row-level subject leak. Remediated append-only:
  `reports/factorial_benchmark_summary.sanitized.json` (0 `H:\`, 10 `PROJECT_ROOT/`)
  + `reports/factorial_benchmark_summary.sanitized.transaction.json`
  (`APPLIED_APPEND_ONLY`, 10 replacements, original preserved byte-identical,
  `claim_ceiling=POST_HOC_INTERNAL_EXPLORATORY`). Per-pair privacy 10/10 PASS.
  `post_eval/post_eval_report.json` privacy `PASS` (`hits=[]`).
- Science blockers are **by-design closed gates**, not errors:
  `QUANTITATIVE_EQUIVALENCE_NOT_ESTABLISHED`, `SCANNER_ROBUSTNESS_NOT_ESTABLISHED`,
  `LOCKED_EXTERNAL_VALIDATION_MISSING` (+ post-hoc exposure). No rerun.

## 2. Locked design (thesis § Methods — design paragraph, copy-paste safe)

- Factorial arms (matched, `run_contract.json:factorial_arms`):
  A0 = GROUPNORM_RESIDUAL_TINY_3D / NONE; A1 = same + PET_SAFE_V1;
  A2 = RESNET10_3D_GROUPNORM (w=0.25) / NONE; A3 = R10 + PET_SAFE_V1;
  A4 = RESNET18_3D_GROUPNORM (w=0.25) + PET_SAFE_V1 (+ outcome-blind w=0.5
  resource-feasibility probe, `role=RESOURCE_FEASIBILITY_ONLY_UNLESS_NEW_VERSIONED_ARM_IS_LOCKED`).
- Augmentation PET_SAFE_V1: max rotation 1.0°, max translation 0.5 vox, prob 1.0,
  `exposure_scope=TRAIN_ONLY`, `validation_and_test_are_unaugmented=true`,
  scale locked 1.0, count-noise DISABLED (no raw counts), PSF DISABLED (no scanner model),
  image+mask share transform, seed unit SUBJECT×BRANCH×OUTER×INNER×EPOCH.
  QC contract `provenance/pet_augmentation_qc_contract.json` (Pearson/Dice p5 ≈0.992;
  2° envelope FAILED; scale/PSF/noise disabled without evidence).
- Survival head: DISCRETE_TIME_3D_SURVIVAL, base_channels 8, embedding 64,
  tabular_hidden 32, dropout 0.25, batch 2 / grad-accum 4 (effective 8), AMP bfloat16,
  max_epochs 40, patience 8, checkpoint every epoch (atomic). HPO candidates:
  `lr1e4_wd1e3` (rank 0), `lr1e4_wd1e4` (rank 1), `lr3e4_wd1e4` (rank 2).
  Selection metric IPCW_INTEGRATED_BRIER_SCORE, rule ONE_STANDARD_ERROR_THEN_LOWEST_COMPLEXITY.
  Inner censoring from INNER_TRAIN_ONLY; admin cutoff ≈1630–1638 d;
  grids [365,730,1095,1460] (training) and post-eval horizons [547.875,1095.75,1640.0]
  (primary 1095.75). Bootstrap 2000, seed 20260714 (paired-subject over locked OOF);
  post-eval NI margins dIBS=0.01, dUno=0.02, permutation P=2000.
- Splits/membership (locked fingerprints in `run_contract.json:input_fingerprints`):
  amyloid membership `c02a9374…` (147391 B), splits `110ddcfe…` (930273 B),
  cache `31981b33…` (1543490958 B); tau membership `b511ffc2…` (67677 B),
  splits `b0d0652c…` (214973 B), cache `4095fcbc…` (695953377 B).
  Subjects/events: amyloid 560/110, tau 306/75; outer coverage 560/306.
  Augmentation creates no new subjects (`augmentation_creates_new_subjects=false`,
  `real_subject_count_unchanged=true`). Outer test exactly-once, unaugmented.
- Determinism: CUBLAS `:4096:8`, cudnn deterministic, benchmark false, TF32 false,
  `deterministic_algorithms=true`. Toolchain: torch 2.5.1+cu121, CUDA 12.1,
  RTX 4060 Laptop (CC 8.9, 8585216000 B), monai 1.5.2, scikit-survival 0.27.0,
  numpy 2.3.5, scipy 1.15.3, pandas 2.3.3, nibabel 5.3.3, h5py 3.16.0.
  Config `3e6d6607…`, software contract `74854b5a…`, bundle `0f172c43…`.
- Pair kinds (prespecified): A0_vs_A1 TINY_AUGMENTATION_EFFECT;
  A0_vs_A2 ARCHITECTURE_EFFECT_WITHOUT_AUGMENTATION;
  A1_vs_A3 ARCHITECTURE_EFFECT_WITH_PET_SAFE_AUGMENTATION;
  A2_vs_A3 RESNET10_AUGMENTATION_EFFECT; A3_vs_A4 DEPTH_EFFECT_WITH_PET_SAFE_AUGMENTATION.
  Inner selection locked before outer (`locked_before_outer_evaluation=true`,
  `outer_test_consulted=false`).

## 3. Inner-only selection (thesis § Results Table 1)

| Branch | Best-mean arm | Selected | Mean IBS | SE | 1-SE threshold | Eligible |
|---|---|---|---|---|---|---|
| Amyloid | A4 | **A4** | 0.09393381 | 0.00074674 | 0.09468055 | [A4] only |
| Tau | A1 | **A1** | 0.11416670 | 0.00109409 | 0.11526078 | [A1] only |

Source: `orchestration/{amyloid,tau}/inner_only_arm_selection.json` mirrored in
`reports/factorial_benchmark_summary.json:inner_only_arm_selections`.
Amyloid arm means: A0 0.09472356 (SE 0.00094083), A1 0.09505275 (0.00108133),
A2 0.09537701 (0.00145823), A3 0.09591921 (0.00144783), A4 0.09393381 (0.00074674).
Tau means: A0 0.11604275 (0.00117889), A1 0.11416670 (0.00109409),
A2 0.11880276 (0.00223819), A3 0.11681911 (0.00214312), A4 0.11810400 (0.00174811).
Outer partition mean-inner-IBS vectors (5 values each) are in the summary JSON;
report them as supplementary, not as outer-test evidence.

## 4. Outer stability (thesis § Results Table 2 — locked outer test, unaugmented)

Means recomputed from 50 `reports/*_public.json` match `stability` block exactly.

| Branch-Arm | IBS mean | IBS SD | IBS range | Uno mean | Uno SD | Uno range |
|---|---|---|---|---|---|---|
| Amyloid A0 | 0.09583 | 0.00979 | [0.08570, 0.11077] | 0.6974 | 0.05908 | [0.6392, 0.7861] |
| Amyloid A1 | 0.09585 | 0.01016 | [0.08752, 0.11283] | 0.6954 | 0.04750 | [0.6549, 0.7720] |
| Amyloid A2 | 0.09746 | 0.01244 | [0.08071, 0.11520] | 0.6980 | 0.05439 | [0.6139, 0.7596] |
| Amyloid A3 | 0.09664 | 0.01221 | [0.07977, 0.11365] | 0.6982 | 0.05671 | [0.6272, 0.7826] |
| Amyloid A4 | 0.09784 | 0.01141 | [0.08255, 0.11294] | 0.6735 | 0.03998 | [0.6345, 0.7316] |
| Tau A0 | 0.12211 | 0.00864 | [0.10843, 0.12964] | 0.7169 | 0.02603 | [0.6750, 0.7403] |
| Tau A1 | 0.12260 | 0.00936 | [0.10884, 0.13138] | 0.7087 | 0.03152 | [0.6616, 0.7370] |
| Tau A2 | 0.11683 | 0.00664 | [0.10864, 0.12491] | 0.7060 | 0.04752 | [0.6609, 0.7854] |
| Tau A3 | 0.11576 | 0.00679 | [0.10843, 0.12509] | 0.7096 | 0.05650 | [0.6676, 0.8062] |
| Tau A4 | 0.13118 | 0.02106 | [0.10820, 0.15733] | 0.6874 | 0.05570 | [0.6473, 0.7855] |

Reading: amyloid outer IBS ≈0.096–0.098, Uno ≈0.67–0.70; tau IBS ≈0.116–0.131,
Uno ≈0.69–0.72. Tau A4 is the clear outlier (worst mean + largest SD/range).
Sample fold anchors: amyloid A4 fold_0 (n_test 109, 21 events): IBS 0.10066,
Uno 0.64891, dynAUC 0.7097 @547.875 (at-risk 92), 0.6491 @1095.75 (at-risk 72),
1640 NOT_ESTIMABLE (at-risk 0). Tau A1 fold_0 (n_test 60, 14 events): IBS 0.11735,
Uno 0.72258, dynAUC 0.8105 @547.875 (51), 0.8297 @1095.75 (43), 1640 NOT_ESTIMABLE.

## 5. Paired contrasts — primary thesis table (thesis § Results Table 3)

Paired-subject bootstrap (2000 iters, seed 20260714) over locked OOF; NI margins
dIBS 0.01 / dUno 0.02 (`post_eval/post_eval_report.json`). Positive deltas = challenger worse.

| Pair | dIBS mean | dIBS 95% CI | NI-IBS | dUno 95% CI | NI-Uno | Acceptance |
|---|---|---|---|---|---|---|
| Amyloid A0→A1 | −0.000084 | [−0.00245, 0.00229] | RETAIN_NEGATIVE | [−0.01952, 0.01475] | RETAIN_NEGATIVE |Exploratory, NOT accepted |
| Amyloid A0→A2 | +0.001963 | [−0.00360, 0.00753] | RETAIN_NEGATIVE | [−0.03530, 0.03802] | INCONCLUSIVE_UNDERPOWERED | NOT accepted |
| Amyloid A1→A3 | +0.001012 | [−0.00387, 0.00589] | RETAIN_NEGATIVE | [−0.02936, 0.03834] | INCONCLUSIVE_UNDERPOWERED | NOT accepted |
| Amyloid A2→A3 | −0.000869 | [−0.00270, 0.00096] | RETAIN_NEGATIVE | [−0.01533, 0.01638] | RETAIN_NEGATIVE | NOT accepted |
| Amyloid A3→A4 | +0.001189 | [−0.00298, 0.00536] | RETAIN_NEGATIVE | [−0.05608, 0.00538] | INCONCLUSIVE_UNDERPOWERED | NOT accepted |
| Tau A0→A1 | +0.000461 | [−0.00293, 0.00385] | RETAIN_NEGATIVE | [−0.02710, 0.00841] | INCONCLUSIVE_UNDERPOWERED | NOT accepted |
| Tau A0→A2 | −0.005252 | [−0.01310, 0.00260] | RETAIN_NEGATIVE | [−0.05079, 0.02753] | INCONCLUSIVE_UNDERPOWERED | NOT accepted |
| Tau A1→A3 | −0.007010 | [−0.01566, 0.00164] | RETAIN_NEGATIVE | [−0.04012, 0.04302] | INCONCLUSIVE_UNDERPOWERED | NOT accepted |
| Tau A2→A3 | −0.001067 | [−0.00460, 0.00247] | RETAIN_NEGATIVE | [−0.00932, 0.01716] | RETAIN_NEGATIVE | NOT accepted |
| Tau A3→A4 | +0.016040 | [+0.00590, +0.02618] | INCONCLUSIVE_UNDERPOWERED | [−0.06762, 0.01863] | INCONCLUSIVE_UNDERPOWERED | NOT accepted |

Fold dIBS vectors (5 folds each, `acceptance_gate.fold_delta_ibs`):
Amyloid A0vsA1 [0.00164, 0.00206, −0.00281, 0.00182, −0.00262];
A0vsA2 [0.00319, 0.00443, 0.00515, −0.00499, 0.00037];
A1vsA3 [0.00293, 0.00081, 0.00476, −0.00775, 0.00320];
A2vsA3 [0.00138, −0.00155, −0.00319, −0.00094, 0.00022];
A3vsA4 [0.00578, −0.00071, −0.00282, 0.00278, 0.00100].
Tau A0vsA1 [−0.00165, −0.00048, 0.00041, 0.00047, 0.00369];
A0vsA2 [0.00591, −0.01071, 0.00020, −0.01430, −0.00753];
A1vsA3 [0.00775, −0.02073, 0.00102, −0.00938, −0.01285];
A2vsA3 [0.00018, −0.01049, 0.00123, 0.00539, −0.00164];
A3vsA4 [−0.00540, 0.01286, −0.00166, 0.04044, 0.03088] (folds 3–4 drive the signal).
- IBS: 9/10 RETAIN_NEGATIVE; only tau depth (A3→A4) is INCONCLUSIVE_UNDERPOWERED
  with CI entirely on the worsening side (challenger A4 worse by ≈0.016).
- Uno: 3/10 RETAIN_NEGATIVE (amyloid A0vsA1, amyloid A2vsA3, tau A2vsA3);
  7/10 INCONCLUSIVE_UNDERPOWERED — honestly underpowered, not equivalence.
- `exploratory_operational_acceptance=false` for all 10; `selected_epoch_nll_gap_delta`
  ranges −0.060 → +0.002 (largest train/val NLL gaps in tau A0vsA1/A0vsA2, still gated).
- Omnibus (5 arms): stat 7.76, **p=0.100776… (≈0.101)** — no arm-level difference.
- Thesis sentence (locked wording): 10 arms, 10 paired contrasts (IBS negatives
  RETAIN_NEGATIVE everywhere except tau depth; Uno mostly INCONCLUSIVE_UNDERPOWERED),
  omnibus p=0.101; see `post_eval/post_eval_report.json:thesis.markdown`.

## 6. Calibration / subgroups / scanner (thesis § Results continued)

- Calibration (`analysis_pairs/*/reports/*_calibration.json`, horizons 547.875/1095.75/1640.0):
  1640.0 is NOT_ESTIMABLE everywhere (effective rows 0; public reports confirm
  `at_risk=0`, `OUTSIDE_COMMON_SUPPORT_OR_INSUFFICIENT_AT_RISK`). Primary horizon
  1095.75 + early 547.875 are estimable with 10-bin predicted-vs-observed tables
  (report as supplementary, not as non-inferiority proof).
  `calibration_not_worse`: amyloid true only for A2vsA3 (1/5); tau true for
  A0vsA1, A0vsA2, A1vsA3, A2vsA3 (4/5), false for A3vsA4. Global calibration
  non-inferiority NOT demonstrated → acceptance blocker in 6/10 pairs.
- Subgroups: `multiplicity_status=DESCRIPTIVE_NO_POSITIVE_SUBGROUP_CLAIMS` (10/10);
  every `positive_subgroup_claim_allowed=false` (exhaustive walk). Age-group example
  (amyloid A3vsA4): g0 ΔIBS +0.00239 [−0.00232, 0.00723] (n=248, 36 events);
  tau A3vsA4 g1 ΔUno −0.15891 [−0.31663, −0.01525] but ΔIBS +0.02477 — wide CIs,
  no claim. All subgroup CIs are supplementary.
- Scanner (`*_scanner_comparison.json`): support amyloid GE 157/33, PHILIPS 58/14,
  SIEMENS 303/57, TOSHIBA 5/0, UNKNOWN 37/6; tau GE 42/9, PHILIPS 38/9,
  SIEMENS 180/50, UNKNOWN 46/7. OOF scanner probe (locked-model embeddings,
  within-fold StandardScaler, 5 folds): **macro AUC range 0.43–0.58 across 20 arms**
  (amyloid 0.4288–0.5183; tau 0.5104–0.5775; supervisor-corrected — do NOT use the
  narrower 0.49–0.52 draft range), permutation null ≈0.50–0.55,
  `scanner_is_primary_predictor=false` (20/20), convergence warnings 0.
  `leave_manufacturer_out_performance.status=NOT_ESTIMABLE_NOT_EXECUTED_AS_LOCKED_TRAINING_DESIGN`
  everywhere; `leave_manufacturer_out_performance_established=false` (10/10);
  TOSHIBA not estimable; tau GE/PHILIPS ineligible (9 events each — second-manufacturer
  support insufficient by the locked rule). `scanner_robustness_passed=false`,
  `status=COMPLETED_WITH_CLAIM_GATE_UNRESOLVED`. Scanner robustness NOT established.

## 7. Claim policy + privacy (thesis § Methods — guardrails; § Results — ceiling)

- Ceiling: `POST_HOC_INTERNAL_EXPLORATORY` (summary `analysis_role`,
  post-eval `ceiling`, every CLAIM_POLICY/acceptance `claim_status`).
  `positive_claim_allowed=false` globally + per-pair (10/10);
  `external_claim_allowed=false`; `exploratory_operational_acceptance=false` (10/10);
  `official_centiloid_equivalence_claim_allowed=false`,
  `official_centaur_equivalence_claim_allowed=false`;
  `quantitative_equivalence_proven=false`, `locked_external_validation_passed=false`.
- Hard blockers (4, all pairs): POST_HOC_OUTER_TEST_PREVIOUSLY_EXPOSED,
  QUANTITATIVE_EQUIVALENCE_NOT_ESTABLISHED, SCANNER_ROBUSTNESS_NOT_ESTABLISHED,
  LOCKED_EXTERNAL_VALIDATION_MISSING. Science blockers (3): the latter three.
  Per-pair operational blockers as in §5 table
  (UNO_CINDEX_NOT_NONINFERIOR, CALIBRATION_NOT_WORSE_NOT_DEMONSTRATED,
  SCANNER_PROBE_NOT_WORSE_NOT_DEMONSTRATED, LEAVE_MANUFACTURER_OUT_PERFORMANCE_NOT_ESTABLISHED,
  plus IBS_NOT_NONINFERIOR + IBS_NONINFERIORITY_NOT_STABLE_IN_ALL_FOLDS for tau A3vsA4 only).
- Privacy: 10/10 per-pair scans PASS (`violations=[]`); post-eval privacy PASS
  (`hits=[]`); aggregate sanitized twin PASS (0 `H:\`). No subject rows, identifiers,
  private membership, individual predictions, checkpoints, or images in public reports
  (`no_subject_rows_calibration_absent` + `no_external_intake_registry` legs are
  NOT_MEASURABLE with recorded evidence — no claim made, per post-eval legs
  LEG-EVAL-03/12).
- Allowed thesis language: "internal exploratory prognostic association in the analyzed
  A4 setting; paired negative results retained; tau depth worsening is internal only."
  Forbidden: AD diagnosis, causal Amyloid/Tau effects, GAAIN-CTX or CenTauR equivalence,
  scanner robustness, external generalization, image-CNN/XAI claims, DLBS-as-validation
  (not part of v4).

## 8. Files a thesis examiner can open (evidence pointers)

- `orchestration_state.json`, `run_contract.json` (digests above)
- `reports/factorial_benchmark_summary.json` + `.sanitized.json` + `.sanitized.transaction.json`
- `reports/{amyloid,tau}_{A0..A4}_fold_{0..4}_public.json` (50)
- `analysis_pairs/{A0_vs_A1,…,A3_vs_A4}/reports/{amyloid,tau}_{CALIBRATION,calibration,paired_oof_bootstrap,scanner_comparison,subgroups,factorial_acceptance,pair_identity}.json` (70)
- `orchestration/{amyloid,tau}/{branch_complete,inner_only_arm_selection,master_contract,winner_*}.json` (54)
- `post_eval/{eval_preflight_pass,post_eval_report}.json` (PASS/PASS, omnibus p=0.101)
- `provenance/contract_migration_v4_*.json` + `pet_augmentation_qc_contract.json`;
  `governance/method_y_v4_contract_migration_654f_to_969e_20260904.json`,
  `governance/method_y_post_eval_v4_{program,amendment_A2,amendment_R2}_*.json`
- Verifier: `scripts/verify_method_y_image_factorial_v4.py --level readiness|final`
