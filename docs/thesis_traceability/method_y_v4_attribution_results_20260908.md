# Method Y v4 — PET Attribution Production Results (append-only, 2026-09-08)

> Production executed on explicit operator order ("full on GPU") after versioned
> config promotion DRAFT→FINAL. Execution was CPU-single-process (the launcher has
> no device path; GPU was verified idle but unused — documented limitation, not a
> validity issue; determinism ledger effective-true throughout).
> Ceiling `POST_HOC_INTERNAL_EXPLORATORY`. Supervisor F audited outputs (1 BLOCKER
> remediated, 2 HIGHs carried as disclaimers below). Allowed-sentences regime:
> everything below is either an allowed sentence or a direct number citation.

## 1. Run facts

- 50/50 locked production cells via frozen-chain provenance (exact bytes that generated
  the locked outer metrics; per-cell bit-identical LayerCAM rerun gates).
- 4330 LayerCAM/GradCAM map rows (amyloid 2800 + tau 1530 = 866 firewall subjects × 5 arms).
- Report `pet_attribution_report_v4.json`: status PASS, legs 8 evaluated, ban/privacy PASS.
  PASS means table-valid (no BLOCKED status, every NOT_MEASURABLE evidenced) —
  it does NOT mean all gates passed.

## 2. Findings (with mandatory disclaimers)

- Attribution was run post-hoc on 50 locked production cells (4330 map rows) under
  ceiling POST_HOC_INTERNAL_EXPLORATORY; internal exploratory evidence only, carrying no
  diagnostic, causal, biomarker-quantified, or externally validated claim.
- The legs table evaluated 8 legs without BLOCKED status; LEG-04 (agreement) and LEG-07
  (PET-quant consistency) are NOT_MEASURABLE with recorded evidence, so no convergence
  and no PET-quant consistency is claimed.
- LayerCAM/Score-CAM agreement did NOT converge on the 20-row pilot subset (Spearman
  rho 0.085, 95% CI [-0.177, 0.334], n=64 blocks, margin 0.7; top-decile Dice 0.145,
  margin 0.6; null 97.5th pct 0.262): maps are method-dependent and must not be presented
  as cross-method validated localization.
- Image-vs-tabular share averaged 0.875 across the 45 evaluated cells (floor 0.5;
  min 0.385); 4 evaluated cells fell below the floor — amyloid/A3/o0, amyloid/A4/o2,
  amyloid/A4/o4, tau/A4/o4 (PARTIAL_EXPLANATION) — and 5 tau/o0 cells were unevaluated
  for share/fidelity: image-dominance holds on average only, cell-conditionally, with
  unevaluated cells NOT_MEASURABLE.
- Occlusion fidelity passed in 15 of 45 evaluated cells, BLOCKED in 30 (5 unevaluated):
  fidelity-passing cells only support exploratory overlay reading with bootstrap intervals.
- Score-CAM full-cohort was NOT run (pilot: 20 subjects / 110 mapped rows / agreement on
  first 20 rows per locked cap; 6.54 s/volume ≈ 7.9 h projected < 24 h budget).
- Risk-scalar sensitivity (pilot-subset volume, logit_sum vs mortality_last voxelwise
  Spearman strongly negative over 128³ voxels): map appearance depends on the risk
  definition, fixed to unsaturated logit_sum as primary.
- Determinism enforced and re-verified (per-cell bit-identical rerun; effective
  cublas/python-hash flags); 0.0 GB VRAM peak is the vacuous CPU-host value, not a GPU
  measurement.

## 3. Privacy posture

- Subject-level rows (`cell_progress.json`, tier PRIVATE_SCRATCH_SUBJECT_ROWS) and the
  pilot subject list are H-only access-controlled scratch, excluded from every shared
  artifact. The pilot list was re-tiered post-run (F-PRIV-05): replaced by count (20) +
  list-sha256 (`8f1eb880…`; list deterministically recomputable as sorted-first-20 of
  firewall subjects). All other evidence JSONs verified zero subject-ID occurrences.
- LEG-07 stays NOT_MEASURABLE (default CIRCULARITY_CONFOUNDED): no PET-quant
  consistency claim, ever, without a pre-registered derivation DAG.

## 4. Production incidents honest log

- `FINAL_DIGEST_MISMATCH` (embedded 20c7… vs live 969e…): root-caused to by-design
  weight-preserving migrations; fixed via frozen-chain provenance binding (exact bytes +
  self-hashed FINALIZED outer state), the same lesson as the v3.2 verifier.
- Wrong-cohort enumeration (v2 splits 323 vs locked 109): fixed to development-role ∩
  splits mirroring the training launcher (fold_0 109 == locked n_test; coverage 560/306).
- Raw-tabular object-dtype crash: fixed with the locked FoldTabularTransformer fit on
  each cell's outer-train frame.
- Pilot OOM (1.72 GiB): agreement capped at locked pilot budget (20 rows), float32 stack.
- One preflight invocation stalled transiently with no output; rerun with log file PASSed
  (no code hang found; GPU confirmed idle throughout).
