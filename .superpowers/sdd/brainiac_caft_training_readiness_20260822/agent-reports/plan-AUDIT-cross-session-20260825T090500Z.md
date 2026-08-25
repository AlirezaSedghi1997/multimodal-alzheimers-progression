# Cross-session strict audit report — 2026-08-25 (independent verifier pass)

plan_id: AUDIT (covers BrainIAC governance/P003 enablement + Method Y Plan 010)
agent/model: ox-alpha (opencode), strict auditor role
started_at_utc: 2026-08-25T07:40Z / finished_at_utc: 2026-08-25T09:05Z
scope: re-verify ALL session work with hard tests; fix defects; no scope creep

## Verified invariants (hard evidence)

- branch `method-y-post-revival-sanitized-20260812`; my commits
  `2c901f8`/`e8de978`/`4ef3495` are ancestors of live HEAD.
- battery manifest SHA `40a4a009…971d`, `.battery.lock` ABSENT (terminal,
  5/5 COMPLETED) — re-hash confirmed.
- amendment SHA `6b313303…e5fc`; registry SHA `1c45b3f4…ca89` — unchanged.
- readiness ledger: 8 events, head `860911fc…92da`; OS anchor chain PASS
  (baseline record 92912, latest anchor 93876).
- vendor tree hash `c26b42ba…fa3` → strict_load PASS path intact.
- P003 preflight: `status=PASS, free_disk_ok=true` (re-run this session).
- caches integrity: `PATTERN_CACHE_SUMMARY.json` hashes match live bytes —
  amyloid `31981b33…` 895/895, tau `4095fcbc…` 306/306 (COMPLETED).

## Defects found by this audit → fixed here (RED→GREEN)

1. **A6 claim gate fail-open regression (critical, introduced by my earlier
   stub).** `validate_claim` was `return claim` and
   `claim_from_three_d_aggregate` delegated to the slow SPM-scanning policy —
   breaking/breaking-hang `tests/test_amuse_x_a6_validate_claim.py`,
   `tests/test_claim_policy_callable.py`, `tests/test_method_x_spm_snapshot.py`.
   Fix: pure, fast, fail-closed implementation bound exactly by those tests:
   POLICY_VERSION a6-callable-1.1.0; METHOD_X_PET_INPUT_FORBIDDEN;
   DIAGNOSTIC_FRAMING/DIAGNOSIS_SCOPE_FORBIDDEN (incl. Persian «تشخیص»);
   VALIDITY_AUDIT_LIMITS_OR_BLOCKS_STRONG_CLAIM; final gates 5 folds/128 res/
   spm_complete/training_status COMPLETED/completed_folds==5;
   SPM_SNAPSHOT_NOT_VERIFIED + opt-in DIGEST MISSING/MISMATCH;
   cohort-probe 0.65 warning / 0.80 strict blocker; ECE>0.15 warning.
   Result: **23/23 + lifecycle/transformation suites green** (38 in batch).
2. **Lost dirty-era symbols** my reconstruction had missed, breaking importers:
   `fit_source_counts` (audit script), `resolve_method_x_training_device`,
   `migrate_source_only_resume_artifacts` (monkeypatchable),
   `inspect_exact_run_contract`/`ensure_exact_run_contract`,
   `method_x_split_contract_digest`, module-level
   `assert_resume_contract`/`source_contract_migration_allowed`, plus main()
   contract gate flags `--cache_sha256/--spm_snapshot_sha256/
   --require_exact_resume_contract/--contract_preflight_only` with
   CONTRACT_PREFLIGHT_PASS JSON path. Fixed per
   `tests/test_method_x_run_contract_lifecycle.py` (9/9 green) — gate runs
   before any training I/O.
3. **backbone_factory gap**: `amuse_x_hybrid3d` unknown +
   `AVAILABLE_BACKBONES` missing → hybrid forward/stage test failed. Added
   builder branch + tuple export (`tests/test_method_x_transformation_v1.py`
   now 6/6 in brainiac venv).
4. **Plan 010 Step-4 tracer attr never enforced**: added
   `expected_tracer` kwarg → writes group `tracer` attr and fails closed on
   `CACHE_GROUP_TRACER_MISMATCH` on rebuild-under-other-tracer; launcher passes
   config tracer. New test `test_cache_group_tracer_mismatch_fails_closed`.
5. **Recurrent stale `.method_x_reconciliation.lock`** (dead PID 19216 from a
   post-session hard kill) re-blocked the BrainIAC gate
   (`STALE_SHARED_GUARD_REQUIRES_RECONCILIATION`). Liveness-proof reconciled
   (removed only after `_pid_identity == DEAD`); gate re-assessed PASS.
6. **Report accuracy fixes**: plan-010 report previously quoted “Method Y 363
   passed” without running it; live truth this session is
   **tests/method_y_revival: 450 passed, 0 failed (409s)**. Report file
   amended accordingly.

## Known remaining failures NOT caused by session work

- root `.venv` lacks MONAI → MONAI-dependent tests must run in
  `.venvs\brainiac` (documented; not a code defect).

## Preservation

- source digest before audit `8c02bd51…bd21` → after fail-closed A6 +
  symbol restoration `9b0f90a5a14f8764397c6a15ff8016462f0872e2f827b2ec47fe54ab773871b3`
  (docs updated to new baseline; old values kept as history).
- battery manifest/lock before == after (`40a4a009…` / ABSENT).
- outer_loader_call_count: 0. No v1/v2/raw mutation; all writes on H.
- claim ceiling: POST_HOC_INTERNAL_TECHNICAL_EVIDENCE (unchanged);
  quantitative_scale_claim_allowed stays false on both caches.

## Verdicts

- BrainIAC governance/P003 enablement: DONE (gate PASS verified twice).
- Method Y Plan 010: DONE (code + materialized, hash-verified caches).
