# Method Y v4 — Thesis Discussion, Limitations, Ethics, Statistics (append-only, 2026-09-08)

> Companion to `method_y_v4_results_20260908.md`. Same ceiling:
> `POST_HOC_INTERNAL_EXPLORATORY`, no positive/external claim.
> Supervisor Stage-2 verdict incorporated (scanner AUC range corrected to 0.43–0.58).

## 1. What the results mean (thesis § Discussion ¶1–3)

- **No arm wins.** Inner-only selection picked the deepest amyloid arm (A4) and the
  augmented tiny tau arm (A1) on inner IBS alone, but outer paired tests do not
  support superiority: IBS RETAIN_NEGATIVE in 9/10 contrasts (i.e., failed to show
  non-inferiority-margin-crossing improvement; CIs centred near zero), Uno
  INCONCLUSIVE_UNDERPOWERED in 7/10, omnibus p=0.101. The honest headline is
  **negative/internal**: neither PET-safe augmentation (1°/0.5-vox, train-only) nor
  ResNet depth (Tiny→R10→R18 at w=0.25) demonstrated outer-test benefit on IPCW-IBS
  or Uno C in this A4 setting.
- **The one directional signal is a worsening, not a win.** Tau A3→A4 (depth under
  augmentation): dIBS +0.0160 [+0.00590, +0.02618], folds [−0.005, +0.013, −0.002,
  +0.040, +0.031] — folds 3–4 drive it; outer IBS mean jumps 0.11576 (A3) → 0.13118
  (A4) with SD doubling (0.0068 → 0.0211, range up to 0.157). NI-IBS is
  INCONCLUSIVE_UNDERPOWERED (not RETAIN), Uno also inconclusive, calibration_not_worse
  false, and both IBS-non-inferiority blockers fire. Report as "internal depth
  degradation signal, exploratory, requires versioned replication" — never as proof
  that depth harms tau prognosis generally.
- **Tau vs amyloid absolute levels are not comparable as claims.** Outer IBS ≈0.096
  (amyloid) vs ≈0.116–0.131 (tau) and Uno ≈0.69–0.70 vs ≈0.69–0.72 reflect different
  cohorts (560/110 vs 306/75), event rates, censoring, and grids — describe, do not rank.

## 2. Limitations that must stay in the thesis (thesis § Limitations — do not soften)

1. **Post-hoc outer-test exposure.** Hard blocker `POST_HOC_OUTER_TEST_PREVIOUSLY_EXPOSED`
   caps everything at exploratory. No confirmatory or clinical reading is permitted.
2. **No quantitative equivalence.** `QUANTITATIVE_EQUIVALENCE_NOT_ESTABLISHED`:
   A4-region definition ≠ official GAAIN CTX (amyloid) and PERSI-SUVR ≠ CenTauR (tau);
   image-derived↔source-regional agreement unproven. No Centiloid/Centaur wording.
3. **No scanner robustness.** Probe AUCs (0.43–0.58 vs null 0.50–0.55,
   `scanner_is_primary_predictor=false`) show embeddings are not trivially scanner-driven,
   but LMO performance is NOT_ESTIMABLE (TOSHIBA 0 events; tau GE/PHILIPS 9 events each;
   UNKNOWN 37/46 subjects handled by excluded-sensitivity only). Robustness NOT established.
4. **No external validation.** `LOCKED_EXTERNAL_VALIDATION_MISSING`; v4 has no external
   cohort, no transport, no refit. All CIs are internal OOF bootstrap (2000, seed 20260714).
5. **Calibration not proven non-inferior** (amyloid 1/5 true; tau 4/5 true; tau depth false).
   1640-d horizon NOT_ESTIMABLE (at-risk 0) — tail claims are NOT_MEASURABLE.
6. **Subgroups descriptive only** (`DESCRIPTIVE_NO_POSITIVE_SUBGROUP_CLAIMS`; CIs wide;
   e.g., tau A3vsA4 age-g1 ΔUno −0.159 [−0.317, −0.015] with ΔIBS +0.025 — no claim).
7. **Small-event tau arms.** 75 tau events across 5 folds (≈15/fold, ≈60/fold test n=60–61)
   → Uno underpowered by design (7/10 inconclusive). Power was not retro-fitted;
   observed-power circularity was avoided per `statistical-analysis` integrity rules.
8. **Outcome-blind width probe (A4 w=0.5) is feasibility-only**, not a sixth arm.
9. **Single-site pipeline, single seed family** (20260714 + deterministic CUDA).
   No multi-seed or multi-site variance is claimed.

## 3. Statistics note (thesis § Methods — analysis paragraph)

- Estimands: paired dIBS and dUno-C (challenger − reference) on matched locked OOF;
  95% CIs from paired-subject bootstrap (2000 successful / 0 failed, `CACHE COMPUTED_AND_LOCKED`).
  NI margins prespecified (post-eval program `governance/method_y_post_eval_v4_program_20260906.json`):
  dIBS 0.01, dUno 0.02. Decisions: RETAIN_NEGATIVE (failed to demonstrate NI-margin
  improvement) vs INCONCLUSIVE_UNDERPOWERED (CI too wide to decide) — neither is
  equivalence proof; non-significance was never presented as no-effect.
- Multiplicity: 10 contrasts + omnibus reported without family-wise correction;
  subgroups explicitly descriptive with `positive_subgroup_claim_allowed=false` everywhere.
  Omnibus (5 arms, stat 7.76, p=0.101) is global, not a substitute for pairwise NI.
- Assumptions/diagnostics: censoring from INNER_TRAIN_ONLY, admin cutoffs recorded per
  inner fold; IPCW grids fixed; convergence warnings 0 in scanner probes; no listwise-deletion
  beyond locked membership; seeds/versions recorded (§1 of results file).
- Reproducibility: run digest `969ebd0f…`, config `3e6d6607…`, contract `74854b5a…`,
  bundle `0f172c43…`; verifier `--level final` PASS (50/50, `failures=[]`).

## 4. Ethics / governance / privacy (thesis § Ethics + Data)

- No raw identifiers, paths (sanitized), membership, individual predictions, checkpoints,
  or images in public reports. Per-pair privacy PASS (10/10); post-eval privacy PASS;
  aggregate path disclosure remediated via sanitized twin + transaction (original preserved).
  `LEG-EVAL-03/12` are NOT_MEASURABLE with evidence — absence reported as such, never PASS.
- Leaky-history rule honored: no push of `method-y-post-revival-sanitized-20260812` /
  `methody/pet-branch-implementation`; publishable SOT remains `methody/pretraining-sanitized-v2`.
  Method X files left unstaged; no `--no_resume`; no finalized checkpoint/metric edited.
- Human review: independent human confirmation is NOT a completion condition for this
  internal exploratory report (five machine-verifiable layers + two supervisor audits);
  any clinical use would require new versioned protocol, ethics approval, and locked
  external validation — explicitly out of scope.

## 5. Future work that the thesis may promise (and may not)

- May promise (versioned, not in-place): locked quantitative image pipeline + CTX/CenTauR
  agreement study; pre-registered LMO with adequately supported manufacturers;
  compatible external cohort with locked mapping + no-refit committed evaluation;
  multi-seed replication of the tau-depth signal; calibration-focused study at 1095.75 d.
- May NOT promise: clinical deployment, diagnostic use, causal interpretation,
  scanner-robust or generalizable claim from v4 alone.

## 6. One-paragraph thesis abstract insert (allowed wording)

> In a locked post-hoc internal exploratory image-factorial programme (amyloid n=560/110
> events; tau n=306/75; 5 matched arms × 5 outer × 3 inner folds; inner IPCW-IBS one-SE
> selection; outer exactly-once unaugmented test; 2000-replicate paired bootstrap),
> inner selection retained amyloid-A4 and tau-A1, but outer paired non-inferiority
> (margins dIBS 0.01, dUno 0.02) showed no operational acceptance in any of 10 contrasts
> (IBS RETAIN_NEGATIVE 9/10; tau depth A3→A4 +0.016 [+0.006, +0.026] INCONCLUSIVE_UNDERPOWERED
> with outer-mean degradation 0.116→0.131; Uno INCONCLUSIVE_UNDERPOWERED 7/10;
> omnibus p=0.101). Calibration non-inferiority, scanner robustness
> (LMO NOT_ESTIMABLE; probe AUC 0.43–0.58 vs null), quantitative equivalence, and locked
> external validation were not established. All findings are internal exploratory;
> no positive, diagnostic, causal, or generalizable claim is made (run `969ebd0f…`,
> verifier final PASS 50/50).
