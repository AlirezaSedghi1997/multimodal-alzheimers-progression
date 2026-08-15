# Method Y remaining work — 2026-08-15 (defect repair)

Session ledger (what the 2026-08-15 agents actually did):
`docs/plans/agent_session_ledger_20260815.md`.
Append-only Method Y work log: `docs/plans/method_y_work_log.md`.

Canonical revival programme is v2 source-regional survival on `H:\New project`.
This file does **not** start Method Y training. Method X currently owns the GPU.

Live readiness: `results/method_y_revival_v2/readiness/PRETRAINING_READINESS_V2.json`.
Post-training memory: `docs/plans/method_y_post_training_execution_memory_20260815.md`.
Contract: `docs/METHOD_Y_V2_POSTTRAINING_EXECUTION_CONTRACT.md`.

Do not push `method-y-post-revival-sanitized-20260812` or
`methody/pet-branch-implementation`. Live HEAD history still reaches private
blob `2be6257e1d7f946805bc6eff09d42ea47983a152`. Publish only from
`methody/pretraining-sanitized-v2`.

The 2026-08-15 closeout **overclaimed** P0, Task 4 QC, and P7. That table is
retracted. Status below is the live repair, not a second greenwash.

## Programme status after defect repair (do not treat as training clearance)

| ID | Item | Status |
|---|---|---|
| P0 | Isolation, no live rewrite, no leaky push | **PARTIAL** — leak contained (working-tree parquet absent; remotes lack blob). Live HEAD is still `NOT_PUBLISHABLE`. Sanitized ref is Method Y code SOT, not the live checkout. |
| P1 | Tau field rename, module git-ancestor, hashed aggregates | Code PASS; Tau still `BLOCKED` / `NO_CLAIM`. Independent `cf0fc23`/`a0161d6` review is a separate note. |
| P2 | Six-branch readiness flags | KEEP — source-regional development isolated from image CTX |
| P3 | AMYLCENT / source SUVR / paired contracts | KEEP bytes; AMYLCENT is documented A4 composite not official CTX (amendment). Paired conclusion bind now includes artifact SHA. |
| P4–P5 | Endpoints, A4 split, scanner, external embargo | Artifacts locked; Task 4 receipt **demoted** from theater `COMMITTED` until proof-bearing QC |
| P6 | CoxNet/RSF/GBSA adapters, 1-SE, preflight launcher | KEEP — default remains preflight; `--execute-training` fail-closed |
| P7 | PRETRAINING_READINESS / tests | Previous `training_allowed=true` / `READY_FOR_SOURCE_REGIONAL_TRAINING` was an **overclaim**. Live status is not a training clearance. |

Source-regional `development_allowed=true` for amyloid, Tau, and paired A/T.
Image, outer, claim, and external remain `false`. Global `training_allowed=false`.

## Do not start from this file

- Method Y `--execute-training` / A4 HPO / outer evaluation
- Image-CNN, DeepSurv image, transform rerun
- Method X files (`scripts/train_3d_mri.py`, `src/training/method_x_*`)
- `git filter-repo` of the live Method X checkout
- Push of leaky refs (`method-y-post-revival-sanitized-20260812`, `methody/pet-branch-implementation`)

## After Method X releases the GPU

1. Confirm Method X is not using the GPU.
2. Run preflight only: `.venv/Scripts/python.exe scripts/run_method_y_training_v2.py --preflight-only`
3. Start training only if PRETRAINING_READINESS says `training_allowed=true`, Task 4 is proof-bearing `COMMITTED`, `training_started=false`, and image/outer/claim/external remain false.
4. Then follow `docs/plans/method_y_post_training_execution_memory_20260815.md`.

## Still blocked scientifically (not “run training”)

- Tau exact scan identity, reference-pipeline equivalence, calibration
- Amyloid image CTX mapping / scale (official Centiloid CTX not proven)
- Tau second-manufacturer event support (claim, not internal exploratory development)
- External DUA / NACC / ADNI / OASIS-3 data on H
