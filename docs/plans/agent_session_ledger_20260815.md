# Agent session ledger — 2026-08-15 (Method Y defect repair)

Read this before continuing Method Y or assuming the earlier closeout was valid.
Machine twin: `results/method_y_revival_v2/journals/session_ledger_20260815.json`.
Method X twin (untouched): `results/method_x_registry/SESSION_NOTE_METHOD_Y_DEFECT_REPAIR_20260815.json`.

This file does **not** start GPU training, Method Y HPO, or a git push.

## One-line status

Method X battery resume was **not edited**. Method Y v2 pretraining **code** is on the live tree and on `methody/pretraining-sanitized-v2`, but Task 4 is **not** `COMMITTED`, `training_allowed=false`, and live git history is **not publishable**.

## Git (do not push leaky refs)

| Ref | SHA | Meaning |
|---|---|---|
| Live HEAD (`method-y-post-revival-sanitized-20260812`) | `b1b99df32f4ab0a2a7cc4d4cc887eb0e6638fcbb` | Method-Y-only defect-repair commit. Branch **name is false**: history still reaches the private parquet blob. |
| Previous live port | `9fcf5c105cede77b61805e5cc1518aa2ff398678` | Codex Method Y contracts copied onto leaky history. Overclaimed P0/P4/P7. |
| Method X receipt pin | `a0161d69e4a326e0347f5d07320e1986d95a4072` | Do not reset live HEAD back to this. Do not `filter-repo` this checkout. |
| Publishable Method Y code SOT | `methody/pretraining-sanitized-v2` = `46520bfd058cf3c81f79581e3f242e4d9e496944` | Parquet-free. Contains the same repair as live `b1b99df`. |
| Merge-base of the fork | `ceeed03` | Sanitized line was replayed from here, not from `a0161d6`. |
| Private blob | `2be6257e1d7f946805bc6eff09d42ea47983a152` | `results/method_y_revival_v2/provenance/tau_scan_source_identity_audit_v2.parquet`. Reachable from live HEAD and from `methody/pet-branch-implementation` (ahead 4 of origin). **Not** on remotes. Working-tree file absent. |

Do **not** push `method-y-post-revival-sanitized-20260812` or `methody/pet-branch-implementation`.
Live Method X trainer files remain **dirty and unstaged** (`scripts/train_3d_mri.py`, backbones, eval, `device_policy.yaml`, …). Leave them unstaged.

## Method X — this session did not change the battery

No Method X source, YAML, checkpoint, or registry science file was rewritten for MX-02.
Do not re-run A6 `write_bundle`. Never `--no_resume`. PET never enters Method X.

Live battery (unchanged programme):

- ResNet18 `PAUSED_RESUMABLE_2_OF_5` — resume fold 2 from **C staging optimizer step 1534** (H `fold_2_progress.pt` stale at 1321)
- ResNet34 `PAUSED_RESUMABLE_3_OF_5` — fold 3 step 668
- ResNet50 / EfficientNet-B0 `PROTOCOL_EXCLUDED_NOT_STARTED`

`docs/method_x_live_progress_report.md` is **stale** (still says models `NOT_STARTED` and commit `a0161d6`). Use `docs/plans/method_x_project_roadmap.md` and `results/method_x_registry/project_roadmap.json` instead.

Operator command (agents do not start this as a long GPU job):

```powershell
Set-Location -LiteralPath 'H:\New project'
.\.venv\Scripts\python.exe scripts\run_method_x_claim_benchmark.py `
  --mode resume-backbone `
  --backbone resnet18_3d `
  --output-root 'H:\New project\results\final_results\method_x_model_battery_v3_depth_and_diversity_20260729'
```

## Method Y — what actually happened this calendar day

Two passes. The first overclaimed. The second demoted the overclaim and repaired gates. **No A4 training.**

### Pass 1 (retracted as a green closeout)

- Ported Codex P0–P6 modules onto live `src/method_y_revival/`.
- Method-Y-only commit `9fcf5c1`.
- Wrote `PRETRAINING_READINESS_V2` as `READY_FOR_SOURCE_REGIONAL_TRAINING` / `training_allowed=true`.
- Promoted Task 4 to `COMMITTED` with **schema 2.0 PASS stubs** (no junit, no rebuild proof, no review findings).
- Inverted the git-history test so leaky HEAD stayed green.
- That status is **invalid**. Do not resume from it.

### Pass 2 (defect repair — current)

Live commit `b1b99df`. Sanitized twin `46520bf`.

Code/tests/config:

- Evidence schema **2.1** in `src/method_y_revival/review_receipt.py` (pytest_args, junit_sha256, hash_compare before/after, spec_ids / finding ids). Schema 2.0 stubs fail `REVIEW_EVIDENCE_BINDING_INVALID`.
- History tests no longer require the leak; `git ls-files "*.parquet"` must be empty; sanitized ref must stay parquet-free.
- `--preflight-only` on `scripts/run_method_y_training_v2.py`. PowerShell launcher refuses default `PRETRAINING_READINESS_V2.json` as an execute receipt. Timeout restored to 90s.
- CLI Tau auditor uses the same git-ancestor private-output guard as the module.
- Long-table Tau path rejects `centaur` names and requires PERSI primary allow-list.
- Paired negative-result `conclusion_sha256` binds `conclusion` **and** artifact file SHA. Public paired contract rebuilt; v1 bootstrap bytes not edited.
- AMYLCENT locked contract **not mutated**. Amendment: `results/method_y_revival_v2/feature_contracts/amyloid_source_regional_centiloid_claim_amendment_v2.json` (documented A4 composite, not official CTX).
- `configs/method_y_training_control_v2.yaml` `execution_mode: synthetic_smoke_only`. Runtime refuses `test_only is not True`.

Live gates (do not invent COMMITTED):

| Artifact | Status |
|---|---|
| `results/method_y_revival_v2/readiness/PRETRAINING_READINESS_V2.json` | `PRETRAINING_CONTRACTS_PORTED_RECEIPT_NOT_BOUND`, `training_allowed=false` |
| `results/method_y_revival_v2/pretraining_contracts/pretraining_validation_state.json` | `AWAITING_AGENT_MULTI_PASS_QC` |
| Adversarial layer | SPEC `PASS`, QUALITY `FAIL` — `finalize_task4_agent_review` **not** called |
| `results/method_y_revival_v2/training_control/TRAINING_CONTROL_PREFLIGHT_V2.json` | `PREFLIGHT_BLOCKED_TASK4_NOT_COMMITTED` |
| Tau public audit | still `BLOCKED` / `NO_CLAIM` |
| Source-regional `development_allowed` | true (amyloid / tau / paired) |
| Image / outer / claim / external | all false |

Targeted Method Y v2 suite: **150 passed** (junit SHA-256 `20f689611c88f2c050d82a14f12f40938e23227bc1de1e9be58307a506c0714f`).
Full `pytest tests`: **1547 passed / 36 failed / 1 skipped**. Failures are **out of scope** (Method X dirty APIs, legacy Method Y external/image ImportErrors, missing thesis reports). Do not “fix” Method X to green that log. Record: `results/method_y_revival_v2/reports/FULL_PYTEST_STATUS_20260815.json`.

## Orchestration / memory map (what to trust)

| Kind | Path | Trust |
|---|---|---|
| This ledger | this file + JSON twin | **current Method Y session** |
| Method Y work log | `docs/plans/method_y_work_log.md` | append-only Method Y chronology |
| Method X work log | `docs/plans/method_x_work_log.md` | append-only Method X chronology; last entry is isolation |
| Method Y remaining work | `docs/plans/method_y_remaining_work_20260815.md` | current inventory |
| Method Y post-training (store-only) | `docs/plans/method_y_post_training_execution_memory_20260815.md` + `docs/METHOD_Y_V2_POSTTRAINING_EXECUTION_CONTRACT.md` | not executed |
| Method Y P1 Tau review | `docs/plans/method_y_p1_tau_cf0fc23_a0161d6_review_20260815.md` | SPEC/QUALITY on `cf0fc23`/`a0161d6`; Tau still BLOCKED |
| Method X roadmap | `docs/plans/method_x_project_roadmap.md` + `results/method_x_registry/project_roadmap.json` | current Method X battery |
| Method X remaining work | `docs/plans/method_x_remaining_work_20260815.md` | MX-02 next; Method Y not a next card |
| July synthesis snapshot | `supervisor_synthesis_5agents/08_final_outputs/01_CURRENT_PROJECT_STATE.md` | **STALE** (HEAD `480752d`, 2026-07-18) |
| Method X live progress report | `docs/method_x_live_progress_report.md` | **STALE** vs battery |
| Global MODEL_READINESS_V2 | `results/method_y_revival_v2/reports/MODEL_READINESS_V2.json` | **audit-only** conjunction; do not use as training gate |
| `configs/agent_memory/` | AutoMem training harness | **not** this project’s session log |

## Next agent — Method Y (CPU, no training unless gates bind)

1. Do not start `--execute-training` or A4 HPO.
2. Do not rewrite live git history; do not push leaky refs.
3. Do not edit Method X trainer files.
4. Remaining before a honest `training_allowed=true`: Task 4 proof-bearing `COMMITTED` (QUALITY must be PASS, not a stub), then a real `PREFLIGHT_ONLY_TRAINING_NOT_STARTED` report, and Method X not owning the GPU.
5. Quality leftovers if you continue QC: disable/guard `scripts/run_method_y_revival.ps1` train flags; pin `--project-root` to this checkout; optional rubric doc for `METHOD_Y_TASK4_SPEC_QUALITY_V1`.

Operator preflight only:

```powershell
.\.venv\Scripts\python.exe scripts\run_method_y_training_v2.py --preflight-only
```
