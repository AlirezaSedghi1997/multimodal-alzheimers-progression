# Method Y Image v3 — Supervisor-Plan Execution & Handover Report

- date UTC: 2026-08-26
- branch: `method-y-post-revival-sanitized-20260812`
- base at session start: `762543d` (after supervisor's uncommitted repairs discovered in working tree)
- commits this session:
  - **fix:** supervisor-plan critical repairs + v2 root migration (v1 INVALIDATED) — 24 files, +2268/−243
  - **docs:** this report + README status rows + memory supersession + ledger append
- scope rule honored: only Method-Y-v3-related files staged; unrelated dirty tree untouched; no push, no reset/clean/checkout.

## 1) What was DONE (verified)

### A. Supervisor repairs verified and committed (were uncommitted in working tree)
| Program finding | Fix evidence (file:anchor) | Status |
|---|---|---|
| 1. NLL fallback / CUDA device mismatch in inner-HPO | `INNER_IBS_NOT_ESTIMABLE`, range guard `[0,1]`, eval loop uses `_forward(model,batch,device_object)` | FIXED+TESTED |
| 2. Outer evaluator outside exactly-once closure | launcher factory passes `test_frame_factory`; materialization inside `controller.evaluate_once` | FIXED |
| 3. Fold errors skipped → false branch completion | `FAILED_INCOMPLETE_OUTER_CV` guards; all-20-cells requirement | FIXED |
| 4. Contract not bound to code/runtime | new `software_contract_v3.py` (source bundle + versions + policy); `_build_run_contract` consumes it via config | FIXED+TESTED |
| 5. Row-level leak in public fold report | recursive privacy scanner; row-level `censoring_reference/outcome` keys rejected; per-branch claim files | FIXED+TESTED |
| 6. Scanner/LMO/subgroup not wired real data | executor joins membership metadata (manufacturer/sex/age/treatment/age_group); embeddings from locked model feed `oof_scanner_probe`; LMO eligibility computed; insufficient support = NOT_ESTIMABLE | WIRED (runs when OOF exists) |
| 7. Superficial verifier | `--level readiness|final`, INVALIDATED handling | FIXED+TESTED |

### B. This-session additions (beyond supervisor work)
1. `build_aggregate_claim_policy()` — AND-semantics aggregate `IMAGE_CLAIM_POLICY.json` over per-branch policies + blocker index; wired into launcher (replaced ad-hoc inline dict).
2. `_apply_deterministic_env(config)` in launcher + called from `_preflight`: direct `.py` runs now enforce CUBLAS=:4096:8 / IMAGE_V3_PERF=0 without PS1.
3. New adversarial tests (`test_audit_round4_supervisor_plan.py`, 9): aggregate AND semantics; launcher wiring (inline-dict forbidden); preflight env driver (subprocess, clean env); **CUDA** inner-IBS real metric no-fallback; **CUDA** interrupt/resume bitwise weights+history; one-byte source mutation changes contract bundle digest (+ missing-source fail-closed); verifier final structural 20-cell check; verifier-on-v1 INVALIDATED with exact 4 reasons; per-branch claim file.
4. Stale tests reconciled: round-3 executor test → per-branch claim filename; launcher ps1 test → v3_2 yaml; external-pending test → v2-pending skip semantics; synthetic-fixture tests → explicit `allow_synthetic_fixture=True` (production still fails closed on missing cache).

### C. v2 bootstrap artifacts created
- `configs/method_y_image_training_v3_2.yaml` (output_root=v2, deterministic_cuda, cudnn_benchmark=false, allow_tf32=false, CUBLAS=:4096:8, synthetic_image_fallback_allowed=false)
- `results/method_y_revival_v3_image_v2/run_contract.json` digest **`7fa020331f213bda69582ff225ae81087add73805be7b779cb6581b044a3b238`**
- v2 readiness READY (gates true/false/false; memberships byte-identical to v1 hashes: amyloid c02a93…, tau b511ff…)
- INVALIDATION mirror tracked at `governance/method_y_revival_v3_image_v1_invalidation_20260826.json`

### D. Verification ladder results
| Step | Result |
|---|---|
| Focused v3 suites (repairs, outer_eval, launcher_state, round3, round4, policy, manifest, readiness) | **all green** (round-3 stale test updated) |
| `pytest tests\method_y_revival -q` | **467 passed, 1 skipped**, 0 failed (~5 min) |
| preflight (v3_2→v2) | READY, true/false/false |
| smoke (v3_2→v2, deterministic mode) | PASSED, cuda, peak 76 MB |
| verifier --level readiness (v2) | PASS |
| verifier --level final (--output-root v1) | **INVALIDATED** + 4 reason codes + exposure |
| full repo `tests -q` | 3 MX collection ERRORS (dirty `scripts/train_3d_mri.py`, out-of-scope, untouched). With those ignored: **1946 passed / 120 failed / 1 skipped** — failures are pre-existing out-of-scope modules (e.g., `test_redesign_stage_extensions*`, legacy MX), NOT introduced by this session; root-caused list not produced (see Not-Done #4) |

## 2) What is NOT done (honest handover)

1. **Production v2 training has not run.** Per program §4: after PASS ladder, only preflight+smoke were executed; the long resumable run is left as the manual command below. Expected first-run behavior: creates v2 pattern caches? No — v3_2 reuses hash-valid v1 caches (reuse allowed); it will run inner-HPO from scratch under corrected scoring for both branches (20 outer cells).
2. **verifier --level final on v2 currently FAILs with `ORCHESTRATOR_STATE_MISSING`** — expected until the 20 cells commit; final PASS is achievable only post-run.
3. **Root-cause triage of the 120 out-of-scope repo-suite failures was aborted by operator** (second characterization run cancelled). They pre-date this session and live outside Method-Y-v3 scope; a dedicated MX/redesign repair pass is required. Do not attribute them to image-v3 changes.
4. **Global Codex memory store is outside this repository** — the dual-gate clarification note could only be recorded repo-side (ledger append + memory supersession block). If an external Codex memory exists, a human/agent with access must paste the ledger paragraph there.
5. **README/memory/plan rows are set to the program-mandated intermediate statuses** (009/010 VERIFIED_DONE; 011/012 REVISE_REQUIRED; 013 IN_PROGRESS_INVALIDATED_V1). Flipping 011/012 to revised-DONE and 013 FINALIZED requires the post-run verifier outputs, per program §5.
6. **Embeddings exist only for the pattern variant** (scalar_tabular folds carry none by design); scanner probe therefore reports OOF_EMBEDDINGS_UNAVAILABLE for scalar-only comparisons — recorded honestly as NOT_ESTIMABLE.
7. **No XAI, no external cohort, no DUA action** (out of program scope).

## 3) Canonical commands for next agent/operator

```powershell
# resume corrected production (resumable at every epoch/fold/candidate/outer-cell)
powershell -NoProfile -ExecutionPolicy Bypass -File 'H:\New project\scripts\run_method_y_image_training_v3.ps1'

# verification
& 'H:\New project\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider tests\method_y_revival -q
& 'H:\New project\.venv\Scripts\python.exe' 'H:\New project\scripts\verify_method_y_image_training_v3.py' --level readiness
& 'H:\New project\.venv\Scripts\python.exe' 'H:\New project\scripts\verify_method_y_image_training_v3.py' --level final            # v2 (FAIL until 20/20)
& 'H:\New project\.venv\Scripts\python.exe' 'H:\New project\scripts\verify_method_y_image_training_v3.py' --level final --output-root 'H:\New project\results\method_y_revival_v3_image_v1'   # v1 -> INVALIDATED
```

## 4) Scientific boundary (unchanged)

v1 = protocol-violation evidence only; fold_0 historical exposure banned from aggregation; all results exploratory; FBP≠Centiloid, FTP≠CenTauR; LEARN is internal transport; no diagnosis/causality/external claims; completion = automated three-layer contract/hash + statistical invariants + privacy/adversarial consistency; external status stays EXTERNAL_PENDING.

## 5) Key artifact pointers

- Invalidation: `results/method_y_revival_v3_image_v1/provenance/INVALIDATION_20260826.json` (mirror in `governance/`)
- Active config/root: `configs/method_y_image_training_v3_2.yaml` → `results/method_y_revival_v3_image_v2`
- Software contract module: `src/method_y_revival/software_contract_v3.py`
- Aggregate/per-branch claim policies: written under `<root>/reports/` post-OOF
- Memory supersession: top of `docs/PROJECT_MEMORY_INDEX.md`
- Ledger append: `docs/plans/agent_session_ledger_20260815.md` (2026-08-26 block)
