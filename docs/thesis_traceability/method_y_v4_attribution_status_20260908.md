# Method Y v4 — PET Attribution Status (append-only, 2026-09-08)

> Companion to `method_y_v4_results_20260908.md`. Ceiling `POST_HOC_INTERNAL_EXPLORATORY`.
> Prior session left the attribution stack UNTRACKED and mid-edit (screenshot:
> `fid_by_cell/share_by_cell` → `group_vols/group_tabs` + `group_gates`).
> This session: 3 independent strict supervisors audited (code / science-gates /
> provenance) → all BLOCKED production; repairs applied with regression tests;
> machinery proven; production stays BLOCKED by design (DRAFT params).

## 1. What was unfinished

- 4 files existed only in the working tree, never committed:
  `src/method_y_revival/pet_attribution_v4.py`,
  `scripts/run_method_y_pet_attribution_v4.py`,
  `configs/method_y_pet_attribution_v4.yaml`,
  `tests/method_y_revival/test_pet_attribution_v4.py` (21 tests, GREEN on the module).
- The launcher pilot phase crashed deterministically (dict `.append`, list-indexed-by-string
  batch, undefined `pilot` / `_layercam_full_for_agreement`, cross-cell batch reuse).
- Production legs table could never PASS (invalid `LEG-02-FULL` id, invalid `BLOCKED`
  status); evidence files were cited but never written; tau `tabular_dim=12` vs hardcoded 11;
  determinism preflight passed on non-deterministic hosts; DRAFT params ungated.

## 2. Repairs (this session, commit-scoped)

- Launcher: per-cell batch reload + `(cell, subject)` map keys; `score_times` list;
  pure `build_production_legs` seam (only LEG-01..08 × PASS/NOT_MEASURABLE/TIMEOUT);
  all cited evidence files written before the report; per-branch dims (11/12) + bins
  inferred from locked payload + dropout 0.25 handshake; resume guards
  (subjects/layer/shallow/risk/config); deadline enforcement with TIMEOUT leg;
  `ATTRIBUTION_CONFIG_NOT_FINAL` gate; BAN parity assert; 5-arm × 2-branch preflight
  with determinism enforce+assert; h5py context manager; explicit `risk_mode=logit_sum`
  primary + `mortality_last` sensitivity on the first pilot subject (subset, honestly scoped).
- Module: single BAN_TERMS definition; `verify_final_digest` weights_only=True;
  tampered verifier artifact raises instead of falling through; determinism
  effectiveness flags; privacy scan allowlists labeled digests + covers POSIX/UNC.
- Config: `project_root` absolute → `.` (runtime resolves from repo ROOT); status stays DRAFT.
- Tests: `tests/method_y_revival/test_pet_attribution_launcher_v4.py` (9 tests).

## 3. Proven results (CPU, read-only, no training contact)

- `test_pet_attribution_v4.py` 21/21 + launcher 9/9 → **30/30 GREEN**.
- `dry-run` (synthetic) → PASS with all 4 evidence files written.
- `preflight` → PASS (verifier / 5-arm layers / determinism-effective / ban_parity),
  digest `969ebd0f…`; artifact carries effective flags (stale false/false artifact invalidated).
- `run` → BLOCKED `ATTRIBUTION_CONFIG_NOT_FINAL` exit 3 (designed gate, JSON-logged).

## 4. Thesis consequence

- No attribution result enters Results: production never ran; nothing to report beyond
  this status note. The factorial survival findings (10 contrasts, omnibus p=0.101) stand
  unchanged. A future versioned FINAL promotion + fresh preflight + deterministic host
  is required before any production attribution; its outputs would be aggregate-only,
  exploratory-internal, and subject to the same ban/privacy/LEG-07 gates.
