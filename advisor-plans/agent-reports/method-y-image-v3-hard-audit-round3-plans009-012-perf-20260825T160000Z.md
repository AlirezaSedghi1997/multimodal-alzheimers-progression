# Hard audit round 3 - Plans 009-012 re-verification + GPU throughput

- date UTC: 2026-08-25 (session 3)
- branch: method-y-post-revival-sanitized-20260812, baseline d680748
- process guard: no active method_y/PET/tau/amyloid writer before any write
- scope: independent re-audit of Plans 009-012 exit criteria with hard tests on live artifacts, then standard-method GPU speedup with zero scientific change

## Audit verdicts per Plan (live evidence, not prose)

### Plan 009 - PASS
ADR + addendum present; readiness rebuilt with unchanged counts (560/110 dev, 335/23 LEARN transport, 306/75 tau); run_contract_sha256 19baa31d0acfbb15e8bcc11828bae8bee2b9e0ed3b1beea06f7a1119a74979c5 unchanged; scanner_support_available_for_prespecified_analysis=true (amyloid) while scanner_robustness_proven=false in both branches.

### Plan 010 - PASS
caches/method_y/revival_v3_image/amyloid_pattern_128_v1.h5 = 895 groups, tau_pattern_128_v1.h5 = 306 groups; all status=COMPLETE, finite_fraction=1.0, dtype float16, shape (1,128,128,128); strict source_sha verification active (image_cache_v3.py). Terminal-resume idempotent finalize covered by existing tests.

### Plan 011 - PASS after one new fix (F-R3-A)
True IBS inner selection, real weight-derived outer evaluator, winner lr/wd propagation all verified from rounds 1-2. NEW defect found and fixed this round: inner-HPO evaluation silently fell back to LAST-EPOCH weights when locked-checkpoint load failed (try/except pass + strict=False), so candidate scoring could use the wrong model state. Fixed via load_locked_state_for_eval(): missing file, corrupt bytes, or contract mismatch now raise INNER_IBS_CHECKPOINT_LOAD_FAILED (fail-closed).

### Plan 012 - PASS after one new gap fix (F-R3-B)
Robustness/reporting modules existed but nothing EXECUTED them over real OOF; launcher previously set ROBUSTNESS_COMPLETE/REPORTS_COMPLETE purely on OOF file presence (premature PASS semantics). Fixed with build_real_oof_reports() executor: consumes the two matched variant fold dicts assembled from private fold metrics, emits amyloid_/tau_ calibration.json, subgroups.json (NOT_AVAILABLE_METADATA recorded honestly when no metadata), paired_bootstrap.json at LOCKED 2000 iterations (STOP condition enforced: BOOTSTRAP_ITERATIONS_MUST_BE_2000), image_incremental.json with incremental_value_gate, IMAGE_CLAIM_POLICY.json via claim_consistency_state, then privacy_scan_reports over public dir. Launcher now advances ROBUSTNESS_COMPLETE/REPORTS_COMPLETE only on executor status COMPLETED_FROM_REAL_OOF and CONSISTENCY_PRIVACY_PASS only when privacy scan PASS; otherwise facts record PENDING_REAL_OOF / PARTIAL_REAL_OOF (UNKNOWN never PASS).

## GPU throughput work (standard methods only, science-neutral)

### Change: lazy RAM volume cache in H5PatternSurvivalDataset
Raw float16 volumes are read from H5 once per subject and memoized (byte-capped, default 6 GiB, graceful fallback to disk reads past cap). Values are bit-identical to disk reads by construction (same arrays returned). Worker-spawn path removed under IMAGE_V3_PERF because spawn-semantics workers would each rebuild a private cache copy; main-process serving after warm-up is strictly faster and preserves the deterministic per-epoch shuffle seeds (loaders still created per epoch with seed+epoch, so resume bitwise behavior is untouched).

### Measured on RTX 4060 Laptop, identical workload both sides
120 development subjects, scalar_tabular_plus_pattern_image, batch 4 x accum 2, AMP, TF32, channels_last_3d, 2 epochs:
- BEFORE: total 45.80 s (epoch1 17.29 s, epoch2 26.55 s), GPU peak 265 MB
- AFTER:  total  7.52 s (epoch1  4.48 s, epoch2  1.90 s), GPU peak 265 MB, ram_cache_subjects=120
- Speedup 6.1x end-to-end; steady-state epoch ~14x (26.6 s -> 1.9 s). No accuracy-relevant path altered: same loss, optimizer, splits, seeds, dtype pipeline (float16 storage -> float32 tensors as before).

## Tests added (RED first: 5/5 failed, then GREEN)

tests/method_y_revival/test_audit_round3_v3.py
- test_h5_dataset_lazy_cache_returns_identical_volumes
- test_h5_dataset_cache_respects_byte_cap
- test_inner_ibs_must_load_locked_best_weights_strictly (missing/corrupt/contract-mismatch all fail closed)
- test_real_oof_report_executor_writes_public_reports_and_gate (matched-pair fixture, 2000-iteration enforcement, claim gate EXPLORATORY when scanner false)
- test_launcher_stage_completion_requires_executor_status_not_file_presence

## Verification commands

- pytest tests/method_y_revival/test_audit_round3_v3.py -q : RED 5 failed -> GREEN 5 passed
- pytest tests/method_y_revival -q : 450 passed (was 445), 2 known RuntimeWarnings in probe NaN-mean path
- smoke mode: PASSED cuda peak 18 MiB; preflight READY, gates true/false/false unchanged

## Scientific claim impact

None widened. Fixes close integrity holes (wrong-model scoring risk; premature robustness completion). Claim gates remain training_execution_allowed=true, positive_claim_allowed=false, external_claim_allowed=false. Readiness hashes unchanged.

## Confirmation

No v1/v2/raw mutation. All outputs/temp on H. Bench scratch dirs removed. No git reset/clean/checkout/branch switch/push.
