# Method Y post-training evaluation program — registry (2026-09-06)

Live twin: `governance/method_y_post_eval_v4_program_20260906.json`.
Plans: `research/notes/method-y-eval/01-initial-plan.md` (draft) →
`03-combined-plan.md` (FINAL + Amendment A1). Knowledge: `02-knowledge-pack.md`.
Implementation: `src/method_y_revival/post_eval_v4.py`,
`scripts/run_method_y_post_eval_v4.py`, `configs/method_y_post_eval_v4.yaml`,
`tests/method_y_revival/test_post_eval_v4.py` (20 passed, CPU, synthetic).

## Execution order (post verifier-PASS only)
1. `run_method_y_post_eval_v4.py --mode preflight` → PASS required.
2. `--mode run` (fail-closed chain: preflight → config SHA → verifier → digest).
3. Legs LEG-EVAL-00..16 in order; BLOCKED halts, NOT_MEASURABLE/TIMEOUT recorded.
4. Thesis writer → `docs/thesis_traceability/method_y_post_training/` + limitations.
5. Privacy scan + k-suppression → public bundle → claims compiler (positive/external FALSE).

## Prespecified locks (config SHA pinned at build; any change = new versioned run)
- Horizons [547.875, 1095.75, 1640.0], primary 1095.75; tau* rule (train G≥0.15, cap 10).
- Bootstrap B=2000 seed=13 fold-stratified paired; percentile primary + BCa sensitivity; permutation P=2000.
- NI margins dIBS 0.01 / dUno 0.02; Uno primary + Harrell sensitivity + C/D AUC + mean.
- Subgroups [sex, age, treatment, cognition, QC, cohort]; CI≥25 events; BH multiplicity.
- LMO renamed STRATIFIED_DESCRIPTIVE_ONLY; pooled GE+Philips forbidden.
- PACC MixedLM descriptive (`pacc ~ time*treatment + (1|subject)`, ML, maxiter 200, tol 1e-6).
- Probes: audit-then-freeze (outer-fit → PROBE_REUSE_VIOLATION → NOT_MEASURABLE).
- External: intake-lint only. Claims: exploratory stamp, UNKNOWN propagates.
- Fusion keys: patient_hash + arm + branch + outer_fold + split digest, dedup asserted.

## Supervision trail
- Round 1 (Phase 1): 6 CRITICAL + 8 HIGH + 6 MEDIUM → all resolved in 03 + A1.
- Round 2 (Phase 3): adjudicated fixes verified present.
- Round 3 (implementation): 20/20 tests + dry-run PASS + refusal proof; re-audit: all PASS, no blockers.
- R-C subagent cancelled ×2 (over-broad scope + flaky hosts) → executed inline with 6-query budget; incident + fix recorded here.
