# Hard audit round 2 - Method Y image v3 (post-enablement)

- auditor: independent strict review (plan-mode read-only pass, then build fixes)
- date UTC: 2026-08-25T14:43:30Z
- branch: method-y-post-revival-sanitized-20260812
- baseline at audit start: 41d1c7f (+ d447c1c docs)
- scope: everything after the 13:30Z round-1 audit, judged against Plans 009-013 exit criteria

## Findings

### F1 CRITICAL - fabricated outer metrics (Plan 011/013 violation)
scripts/run_method_y_image_training_v3.py contained two closures returning hardcoded values (c_index 0.62/0.61, ibs 0.11/0.12, empty oof_predictions). OuterFoldController.evaluate_once would permanently commit these as outer results and consume the exactly-once access. No fabricated result had been committed yet at detection time (verified: no winner files / no outer_result.json existed).

### F2 HIGH - winner hyperparameters never propagated to final fit
image_training_orchestrator_v3.py candidate records lacked learning_rate/weight_decay; run_final_training fell back to constants lr=1e-4/wd=1e-4 regardless of HPO outcome - silently broke the locked one-standard-error selection contract for the final outer-train fit.

### F3 MEDIUM - Plan 013 stage machine not materialized
Launcher resumed correctly per-unit but wrote no unified stage artifact (INIT..FINALIZED) with self-hash and contract binding.

### F4 process state
Production run pid 9244 was live on the canonical root when audit started; owner approved stop. Processes had already exited; stale .image_training.writer.lock removed after verifying pid dead.

## Deletions performed (wrong-era artifacts, user-directed)

- results/method_y_revival_v3_image_v1/orchestration/amyloid/** (placeholder-contract orchestrator_state + tmp_inner checkpoints from pre-fix code)
- results/method_y_revival_v3_image_v1/orchestration/tau/**
- results/method_y_revival_v3_image_v1/.smoke_test/** (synthetic smoke outputs, regenerated post-fix)

No v1/v2/raw data touched; caches/readiness/memberships/run_contract untouched.

## Fixes (RED then GREEN)

### New module: src/method_y_revival/image_outer_eval_v3.py
SHA-256 EAE29FFA3B7D4C7F49E05BC1A3C1853A7A464C4B3E05BF301B59192B85D01947.
evaluate_locked_outer(...): strict checkpoint load with contract-digest check; FoldTabularTransformer fit on OUTER TRAIN only; H5 dataset for pattern variant (deterministic zero-volume fallback only when no cache, used by tests); batched no-grad inference via _forward; prespecified grid = interior hazard-bin edges within common train/test support (2+ points else linspace fallback); administrative censoring of both frames at common max; censoring reference from OUTER TRAIN only; integrated_brier_score + concordance_index_ipcw with risk = 1 - S(tau); private OOF rows shaped for science_gates.paired_oof_bootstrap. Metrics derive strictly from checkpoint weights, so constant output is impossible by construction.

### Fix F2
Candidate records now carry learning_rate/weight_decay; run_final_training raises WINNER_HYPERPARAMS_MISSING on stripped winners and uses the winner exact optimizer settings.

### New module: src/method_y_revival/image_launcher_state_v3.py
SHA-256 E473FBD24F29AD0AC5E7379A33E8ECCEB86BEE327560A87EA48CBDE9500323AE.
16-stage machine per Plan 013 (INIT..FINALIZED), self-hashed payload (state_sha256), contract-bound (mismatch fails closed), monotonic order enforcement, idempotent same-stage rewrite, FINALIZED gated on all prior stages recorded in facts.completed.

### Launcher rewiring
scripts/run_method_y_image_training_v3.py SHA-256 1337F43E8D5A0AF210880DCFB8D30E25A21A511265C3EE353197A2B3FC6FE6F0.
Smoke body extracted to _run_gpu_smoke(); run mode advances stages with facts: PREFLIGHT_DONE, DUAL_GATE_CONTRACT_VERIFIED, READINESS_LOCKED, PATTERN_CACHES_COMPLETE (counts), GPU_SMOKE_PASSED (runs smoke once, records peak_vram); both fabricated evaluators replaced by real evaluate_locked_outer wired to locked checkpoints and real test frames; per-branch completion commits AMYLOID_/TAU_ INNER_HPO, MODELS_LOCKED, OUTER_COMMITTED; ROBUSTNESS_COMPLETE/REPORTS_COMPLETE set only from real OOF presence, else facts record PENDING_REAL_OOF (never PASS on absence, per rule 6); CONSISTENCY_PRIVACY_PASS requires privacy_scan_reports PASS; FINALIZED enforced by state gate.

## Tests added (RED first, all 10 failed before implementation, then GREEN)

tests/method_y_revival/test_image_outer_eval_v3.py
- test_real_outer_evaluator_metrics_depend_on_weights_not_constant (anti-placeholder: two differently seeded locked models must give different IBS)
- test_real_outer_evaluator_rejects_contract_mismatch
- test_launcher_source_contains_no_fabricated_metric_literals (grep guard on 0.62 / 0.61 / _real_evaluator)
- test_winner_record_carries_learning_rate_and_weight_decay
- test_run_final_training_requires_winner_hyperparams

tests/method_y_revival/test_image_launcher_state_v3.py
- test_stage_order_enforced
- test_same_stage_rewrite_is_idempotent
- test_tamper_detected_on_reload
- test_contract_mismatch_fails_closed
- test_finalized_requires_all_previous_stages

## Commands and exit codes

- pytest new tests (RED): exit 1, 10 failed
- pytest new tests (GREEN): exit 0, 10 passed
- pytest tests/method_y_revival -q: exit 0, 445 passed (was 435), 2 known warnings
- smoke mode: PASSED, device cuda, peak_vram 26153984
- preflight mode: READY_TO_RUN_FINAL_INTERNAL_EXPLORATORY_IMAGE_TRAINING
- bounded live run (115 s, amyloid): STAGE_COMMITTED GPU_SMOKE_PASSED printed, FULL_TRAINING_START, INPUTS_VERIFIED outer_folds=5, inner HPO epochs committing at 0.3-0.5 s each for scalar_tabular; wrapper killed by tool timeout, orphan python verified absent, stale lock removed

## Scientific claim impact

None widened. Two fabrication/integrity defects removed before any outer result was committed. Claim gates unchanged: training_execution_allowed true, positive_claim_allowed false, external_claim_allowed false. Readiness hashes unchanged (19baa31d..., memberships c02a93.../b511ff..., sources 784fa.../41f59...).

## Remaining blockers

None for Plans 009-013 engineering completion. Production training resumes with the canonical command and will now commit real, weight-derived outer metrics under the stage machine.

## Confirmation

No v1/v2/raw mutation. All outputs/temp on H. No git reset/clean/checkout/branch switch/push. Wrong-era artifacts deleted only inside results/method_y_revival_v3_image_v1/orchestration and .smoke_test per user direction, with inventory listed above.
