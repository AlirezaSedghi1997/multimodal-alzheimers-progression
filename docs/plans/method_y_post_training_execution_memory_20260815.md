# Post-training execution memory — Method Y revival v2

Store-only until Method X releases the GPU and a later operator run actually
starts source-regional training. This file does **not** start Method Y training,
does not report metrics, and does not claim a science-gate pass.

Canonical contract: `docs/METHOD_Y_V2_POSTTRAINING_EXECUTION_CONTRACT.md`
(`PRESPECIFIED_NOT_EXECUTED`).

Pretraining twin: `results/method_y_revival_v2/readiness/PRETRAINING_READINESS_V2.json`.
Remaining-work inventory: `docs/plans/method_y_remaining_work_20260815.md`.
Session ledger: `docs/plans/agent_session_ledger_20260815.md`.

Do not invent outer C-index, IBS, calibration, or incremental-value numbers here.
Those artifacts do not exist until Stages A–E run.

## Operator commands (later — not this closeout)

Preflight only, any time after this closeout:

```powershell
Set-Location -LiteralPath 'H:\New project'
.\.venv\Scripts\python.exe scripts\run_method_y_training_v2.py --preflight-only
```

Execute training only after all of these remain true:

- PRETRAINING_READINESS status is `READY_FOR_SOURCE_REGIONAL_TRAINING`
- `training_allowed=true` and `training_started=false`
- Task 4 state is proof-bearing `COMMITTED` (schema 2.1 evidence, adversarial quality PASS)
- source-regional `development_allowed=true`
- image / outer / claim / external remain false
- Method X is not using the GPU
- a later receipt binds `training_control_contract_sha256` from that preflight plus a `training_executor` dict with `test_only=true`

As of the 2026-08-15 defect repair, Task 4 is `AWAITING_AGENT_MULTI_PASS_QC` (adversarial quality FAIL), live preflight is `PREFLIGHT_BLOCKED_TASK4_NOT_COMMITTED`, and `training_allowed=false`. Do not start training from this file.


```powershell
# .\scripts\run_method_y_training_v2.ps1 -ExecuteTraining
```

Never `--execute-training` against live A4 from this closeout. Never image-CNN.
Never pool GE+Philips to manufacture Tau manufacturer support.

## Ordered stages after training is actually started

Each stage requires the previous committed artifact. One failed outer fold is
`FAILED_INCOMPLETE_OUTER_CV`; remaining folds are not presented as a complete
five-fold result.

1. Winner lock per branch (inner IBS + 1-SE).
2. Final fit on each outer-train partition only.
3. Outer evaluation exactly once.
4. Uno C-index, IBS, dynamic AUC.
5. Calibration slope/intercept + plots.
6. At-risk / event counts per horizon.
7. Paired bootstrap CI for PET incremental value (2,000 subject-level replicates).
8. Amyloid leave-manufacturer/site-out (prespecified; UNKNOWN stays UNKNOWN; no GE+Philips pool).
9. Scanner/cohort probes (audit-only; never predictors).
10. Subgroup CIs (sex, age, treatment, cognition, QC, cohort).
11. PACC mixed-effects secondary.
12. Keep locked negative Tau-beyond-Amyloid / paired A/T unless new outer evidence says otherwise.
13–16. Image XAI only if that branch’s gates pass; `strict=True` load; Grad-CAM++ / IG / Occlusion; fidelity + permutation sanity.
17–18. External intake/mapping lock; no-refit, exactly-once.
19–21. TRIPOD+AI / PROBAST+AI / CLAIM rebuild; public bundle privacy scan; claim bounded by internal/external gates.

Detail, required inputs, and failure modes live in the contract above. Do not
copy that contract into a second competing checklist.

## What this closeout did not do

- No A4 HPO
- No outer-test access
- No external outcome access
- No Method Y GPU use
- No live git history rewrite
- No push
