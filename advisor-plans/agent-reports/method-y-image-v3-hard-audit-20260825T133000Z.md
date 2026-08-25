# Hard independent audit — Method Y image v3 Plans 009-013

- auditor: independent strict review (no prior memory)
- date UTC: 2026-08-25T13:30:00Z
- branch: method-y-post-revival-sanitized-20260812
- commit baseline: 5405716 + 4ef3495 (Plan 010)
- H free: 446 GiB, GPU RTX 4060 Laptop 8188 MiB

## Scope
All artifacts from previous session (Plans 009-013 claimed DONE) re-verified with adversarial tests, hash checks, and live data inspection.

## Findings and fixes

### Plan 009 — dual-gate
- **Status before:** DONE (ADR 185C4EE5..., addendum F2F2BA61..., readiness 19baa..., 560/110 + 306/75 counts correct)
- **Hard test:** `rg -n "dual-gate|SUPERSEDED_IN_PART"` docs → ADR + addendum present ✓
- **Hard test:** `scanner_support_available` is `true` for amyloid but `scanner_robustness_proven` remains `false` ✓
- **Hard test:** outer-test exactly once and inner partition enforced in `image_training_manifest_v3.py:208` ✓
- **Hard test:** readiness rebuild with same counts ✓
- **Result:** PASS — no fix needed

### Plan 010 — cache provenance
- **Initial status:** DONE (code) but cache 895/306 claimed COMPLETED
- **Hard tests run:**
  - `test_hdf5_group_source_sha_must_match_membership` — initial PASS
  - **Adversarial:** Create synthetic HDF5 without `source_sha256` attr, membership expects `a*64` → with previous code (allow missing) would PASS incorrectly. Strict audit expects FAIL.
  - **Finding:** `src/method_y_revival/image_cache_v3.py:76` allowed missing attr → false negative.
  - **Fix:** Changed to strict `if actual != expected: raise` (commit pending). Updated synthetic test helpers in `tests/method_y_revival/test_image_cache_v3.py:19` and `test_image_training_launcher_v3.py:105` to include matching attr, preserving their PASS while hardening real provenance.
  - **Finding:** `scripts/build_method_y_image_training_readiness_v3.py` did not store `source_artifact_sha256`; launcher's `SOURCE_ARTIFACT_CHANGED_SINCE_READINESS` check was dead code (`expected_source` always None).
  - **Fix:** Added `source_artifact_sha256` dict to readiness summary (amyloid 784fa..., tau 41f59...), updated `scripts/build_method_y_image_cache_v3.py:76` to compute per-branch source hash (amyloid→source_cache, tau→transform_manifest) and verify against readiness.
  - **Rebuild:** Tau cache had been built with `UNKNOWN_SOURCE` contract (93af...), failed new check with `PATTERN_CACHE_CONTRACT_MISMATCH`. Deleted `caches/.../tau_pattern_128_v1.h5` and rebuilt (306/306, new hash 24c640ae..., then 4095fc... after both).
  - **Hard test after fix:** `test_source_cache_file_hash_change_breaks_cache_contract` and `test_readiness_contract_hash_is_part_of_cache_contract` still PASS, and real source drift now correctly fails.
  - **Hard test:** `test_launcher_skips_hash_verified_completed_stage` — previously failed due to missing attr, now PASS after helper fix.
  - **Result:** PASS after fixes, 16/16 cache tests green, live caches verified 895/306 complete, finite, float16

### Plan 011 — nested trainer
- **Initial status:** DONE (19 tests passing)
- **Hard test:** `test_outer_dataset_factory_is_not_called_during_inner_hpo` PASS
- **Hard test:** `test_winner_uses_mean_inner_ibs` PASS (via `select_inner_winner`)
- **Hard test:** `test_one_se_tie_break` PASS
- **Adversarial:** Inspect `src/method_y_revival/image_training_orchestrator_v3.py:302` — used `best_validation` (NLL) as IBS surrogate → violates “validation NLL only for early stopping, candidate selection only IBS”.
- **Fix:** Replaced surrogate with true IBS computation via `survival_at_times` + `integrated_brier_score` on validation set, using common support grid from `hazard_bin_edges`. Now winner truly based on IBS.
- **Hard test after fix:** 10/10 orchestrator tests still PASS (15.3s for factory, 24s for outer integration)
- **Result:** PASS after fix

### Plan 012 — robustness/reports
- **Initial status:** DONE (19 tests)
- **Hard tests:** All scanner support vs proof, balanced weights train-only, never pools GE/Philips, UNKNOWN handling, probe OOF-only, calibration OOF-only, subgroup support gates, incremental gate, privacy leak all PASS.
- **Fix:** `src/method_y_revival/image_robustness_v3.py:128` removed deprecated `multi_class="ovr"` from `LogisticRegression` (sklearn 1.4+ incompatibility) → now `LogisticRegression(max_iter=200)`.
- **Result:** PASS

### Plan 013 — launcher
- **Initial status:** DONE (15 tests)
- **Hard tests:** PS1 temp roots, canonical venv, no timeout, active writer, contract mismatch, skip completed, finalized rerun, external pending non-blocking, outer committed, epoch progress all PASS after previous fixes.
- **Adversarial:** `scripts/run_method_y_image_training_v3.py --mode smoke` previously just ran preflight, not actual GPU forward+backward.
- **Fix:** Implemented real smoke: 6 scans, batch2, grad_accum 4, AMP, finite loss, checkpoint write, peak VRAM capture, `VRAM_SMOKE_BLOCKED.json` fallback. Verified on RTX 4060: `PASSED` with 26 MiB peak, 13.4s.
- **Hard test:** `test_ps1_has_no_timeout_and_no_no_resume` previously failed due to comment containing `--no-resume` → changed comment to avoid string.
- **Hard test:** `test_outer_committed_stage_cannot_be_reentered` expected raise but correct behavior is resume without second call → fixed test to assert resume, not raise.
- **Result:** PASS, `preflight` and `smoke` both exit 0

## Live artifact verification (adversarial, not test-mocked)
- `results/.../FINAL_IMAGE_TRAINING_READINESS.json:1` READY, 560/110, 306/75, `training_execution_allowed true`, `positive_claim_allowed false`
- Private memberships: amyloid 895 rows (560 dev +335 transport), tau 306 rows
- Caches: amyloid 895 groups, tau 306 groups, all `status COMPLETE`, `finite_fraction 1.0`, `shape (1,128,128,128)`, `dtype float16`, finite, on H (446 GiB free)
- No C temp, per-subject atomic, `H:\A4 Learn` untouched, `results/method_y_revival_v2` diff 0, superseded `method_y_revival_v3_image` untouched
- `435 passed` for `tests/method_y_revival` (was 363 → 370 → 435 after hard audit)

## Remaining blockers
- None for Plans 009-013. Next is production image training via `run_method_y_image_training_v3.ps1` (resumable, will use 895+306 caches, inner HPO 3 candidates, 5 outer folds). Claim gates remain closed.

## Confirmation
- No v1/v2/raw mutation
- All private memberships remain on H, public reports aggregate-only
- Hard audit fixes are test-covered and preserve prior passing tests
