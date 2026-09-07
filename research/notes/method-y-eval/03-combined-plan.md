> FINAL-COMBINED 2026-09-06 — reconciles 01-initial-plan + supervisor defects (C1–C6, H1–H8, M1–M6) + knowledge pack 02. Basis for implementation (Phase 4). Changes vs 01 are marked [FIX].

# 03 — Combined Post-Training Evaluation Program (v4 image factorial)

## Locked global parameters (config `method_y_post_eval_v4.yaml`)
- Horizons (from v4 config, verbatim): [547.875, 1095.75, 1640.0] days. Primary horizon H = min(1095.75, tau*).
- tau* rule [FIX H1]: largest t with train-fold KM G(t) ≥ 0.15 AND ≥10 test events beyond median; computed at runtime, recorded with diagnostics; tail beyond → NOT_MEASURABLE.
- IPCW: marginal Kaplan–Meier fit on INNER-TRAIN ONLY, frozen per outer partition; weights truncated at cap 10 [FIX H1/H8]; estimator choice locked (no conditional-Cox variant in this program).
- Risk score def: `1 - S(tau|X)` from locked survival curves; ties count 0.5; recorded.
- Bootstrap: B=2000, numpy RandomState(seed=13) + recorded scipy/sklearn/survival lib versions; fold-stratified paired resampling (same indices both arms); primary = percentile CI; BCa sensitivity co-reported (unstable → NOT_MEASURABLE, never silent fallback) [FIX H2]; permutation P=2000, unit = arm labels within subject, censoring fixed, p secondary [FIX H2].
- NI margin dIBS = 0.01 absolute; Uno margin 0.02 (both reused from locked v4 factorial margins — consistency, not new science).
- Subgroups (prespecified, E8): sex, age-below/above-median (median from LOCKED baseline table, recorded), treatment arm, baseline cognition below/above-median, QC pass/fail, cohort. CI bar: ≥25 events else descriptive-only; <10 events → NOT_ESTIMABLE + k-suppressed from public bundle [FIX H3/M3]. Interaction-first; BH across arm contrasts [FIX H3].
- Multiplicity: prespecified hierarchy omnibus → Holm reference contrasts; rest exploratory [FIX B-negatives/H3].
- PRNG: Philox-equivalent via numpy RandomState seeded per analysis + SHA-seeded substreams; thread count 1 for bootstrap loops; rerun-identical proof required [FIX M2].

## E1. Winner/final/outer-exactly-once — status-quo locked flow (no redesign).
- Plus outer-firewall log assertion: access journals show zero outer_result reads before lock; else EXACTLY_ONCE_AT_RISK flag [FIX M5].

## E2. Discrimination (research: Uno primary, Harrell sensitivity, C/D AUC + mean, I/D exploratory)
- Per-cell: Uno-C(H) + Harrell-C (delta shown as censoring-bias diagnostic) + cumulative/dynamic AUC at H + mean over common grid; Antolini Ctd exploratory.
- 5-arm factorial contrasts: paired ΔUno + ΔIBS with CIs (E5 machinery); reference = A0; matched pairs A0/A1, A2/A3, A0/A2, A1/A3 + secondary A3/A4 (from locked config).

## E3. Calibration (descriptive, never gate input) [FIX C4]
- IPCW-logistic slope/intercept at H (+ optional earlier horizon) with CIs + calibration plot + prediction rug; ICI/E50 numeric secondary; Emax + D-calibration exploratory with exact bin counts. Link: logit; horizons/bins/seed prespecified in config.

## E4. Risk tables + KM (pooled one-row-per-subject, fold-stratified variance) [FIX H7]
- At-risk/event counts per horizon; KM overlays descriptive only. Dedup assertion + key schema (patient_hash, outer_fold, run_digest) on every E-output [FIX H7].

## E5. Paired bootstrap engine (shared by E2/E10)
- As global params; variance-exclusion statement auto-attached (test noise only, not training/tuning variance).

## E6. Manufacturer: STRATIFIED_MANUFACTURER_REPORT (descriptive, never gate) [FIX C1]
- No refit-LMO anywhere in this program (forbidden). Tau GE/Philips (9 events) → NOT_MEASURABLE with evidence. Pooled GE+Philips FORBIDDEN.

## E7. Probes: audit-then-freeze [FIX C2/M6]
- Implementation step 1 audits existing OOF scanner probe: if probe weights fit on outer-derived data → PROBE_REUSE_VIOLATION → replace with inner-frozen variant (fit inner-train only, frozen solver/seed/C) or mark section NOT_MEASURABLE. E7-O2 min-n rule applies (≥25 events).

## E8. Subgroups per global params. E9. PACC MixedLM DESCRIPTIVE ONLY [FIX H4]
- statsmodels MixedLM, random intercept (+trend iff ≥3 obs/subject median), prespecified formula, locked convergence criteria (maxiter, tol); singular/non-converged → NOT_MEASURABLE (no model switch); MAR stated + dropout/missingness table; treatment×time reported WITHOUT causal reading; GEE dropped.

## E10. Negatives with NI margin [FIX C3]
- Paired dIBS + permutation p + full Δ/CI tables for all prespecified contrasts; retain-negative ONLY if upper CI < 0.01 NI margin with identical IPCW/splits; else INCONCLUSIVE_UNDERPOWERED (never "preserved" by wide CI); null arms retained; no retesting.

## E11. CAM (existing build; H6 locks enforced)
- eval-mode/mask-grid/baseline/seed/background-digest frozen (already); missing lock → section NOT_MEASURABLE. Randomized + fidelity gates unchanged.

## E12. External intake: static schema lint ONLY [FIX H5]
- No fixtures through eval path, no synthetic rows, horizons redacted from public registry, quarantine rule recorded, mapping lock assert-absent.

## E13. Checklists: machine items enforced in code; judgement = 2 attesters + quote + digest, capped ATTESTED_ONLY, dissent verbatim [FIX C5].

## E14. Privacy: recursive scan + k-suppression threshold (≥10 subjects AND ≥... events rule in config) before any public artifact [FIX M3].

## E15. Claims compiler: UNKNOWN/NOT_MEASURABLE propagates; exploratory stamp never overrides; positive/external FALSE [FIX M4].

## E16. Thesis writer [FIX C6]
- Allowlist tokens ONLY (gate verdicts + numbers + ceiling strings); sentence-level JSON-pointer provenance; ceiling-string check vs verifier; NOT_MEASURABLE → fixed boilerplate; output under docs/thesis_traceability/method_y_post_training/.

## Legs table (orchestrator gates, each PASS/BLOCKED/NOT_MEASURABLE/TIMEOUT)
LEG-EVAL-01 locked-inputs (verifier PASS + finals + splits + horizons) · 02 discrimination · 03 calibration · 04 risk tables · 05 paired contrasts · 06 manufacturer-stratified · 07 probes (post-audit) · 08 subgroups · 09 PACC-descriptive · 10 negatives/NI · 11 CAM-gates-read · 12 intake-lint · 13 checklists · 14 privacy+bundle · 15 claims · 16 thesis-write. Production `run` refuses unless LEG-01 PASS; any BLOCKED halts with reason (no override).

## Out of scope (reaffirmed)
Retraining/refit (incl. refit-LMO); outer reuse; outer-driven selection/thresholds; PET-in-Method-X; diagnosis/causal/external claims; synthetic external; bound-source edits while v4 trains (implementation uses NEW files only — verified against software_contract_v4.SOURCE_PATHS: 25 bound paths).

## Amendment A1 — reviewer round-2 dispositions (binding)
- H1: DROP the "≥10 test events beyond median" clause (outer-driven). tau* = largest t with train G(t)≥0.15 AND t ≤ max locked horizon; tail rule purely from weight diagnostics (% truncated at cap 10, min G). [FIX]
- C2/H6/M6: probe replacement = refit → FORBIDDEN. Audit existing probe: outer-fit weights → PROBE_REUSE_VIOLATION → section NOT_MEASURABLE. Inner-frozen variant ONLY in a new versioned program. [FIX]
- C3: apply NI logic symmetrically to Uno (retain-negative iff lower CI > −0.02 margin). [FIX]
- H2: pairing = within-contrast (same subjects across the two arms of each locked pair), never "both arms" globally. [FIX]
- H3: subgroup descriptive rule completed: <10 events NOT_ESTIMABLE+k-suppressed; 10–24 events descriptive point-estimate only; ≥25 events CI allowed. [FIX]
- H4: MixedLM prespecified NOW: formula `pacc ~ time*treatment + (1|subject)`, estimator ML, maxiter=200, tol=1e-6; singular/non-converged → NOT_MEASURABLE; GEE dropped. [FIX]
- M1: omnibus→Holm tagged as the M1 fix (documentation). [FIX]
- M2: MT19937 correction accepted — numpy RandomState + SeedSequence spawning, versions recorded, rerun-identical proof. No Philox claim. [FIX]
- M3: k-suppression = ≥10 subjects AND ≥10 events per public cell. [FIX]
- M5: firewall verdict added as LEG-EVAL-00 (input integrity incl. firewall log). [FIX]
- H7: key schema += arm + branch/modality + split digest. [FIX]
- H8: paired Δ computed WITHIN each outer partition under its frozen IPCW, then meaned; bootstrap stratified by partition. [FIX]
- NUMBERS: recorded expectation — tau arm-cells (~3 events) and 1640d tail will be NOT_MEASURABLE/descriptive in most cells. [FIX]
- CONTRA: locked pairs complete per v4 config (A0/A1, A2/A3, A0/A2, A1/A3 + secondary A3/A4). Continuation rule: BLOCKED halts; NOT_MEASURABLE/TIMEOUT recorded, pipeline continues. [FIX]
- REUSE: L12 medians bound to baseline-table file+hash pinned in config at build; pooled-contrast IPCW per H8 rule. [FIX]
- M4: accepted as specified. [FIX]

## Amendment A2 — implementation (2026-09-06, RED-first, 35/35 GREEN + dry-run PASS)
- Support rule hardened: test times AND grid AND tau STRICTLY below train max (sksurv half-open); per-fold shared cross-arm mask (pairing preserved) + NOT_MEASURABLE records + n_dropped counts.
- I/D AUC explicitly NOT_IMPLEMENTED (unstable at 75 events; documented scope cut).
- Antolini Ctd implemented exploratory (perfect-order fixture gives 1.0).
- Only fits in program: IPCW-GLM recalibration + MixedLM, both DESCRIPTIVE_ONLY + fail-closed (audited: no training/launcher/H5 imports).
- Production run-mode executes full battery loop over parameterized root (fixture-tested); refuses pre-verifier (proven).


