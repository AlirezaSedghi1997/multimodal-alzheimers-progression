# Agent session ledger — 2026-08-15 (Method Y defect repair)

> **Append 2026-09-02 (afternoon) — v4 RUN_CONTRACT_MISMATCH root-caused & fixed (contract HEAD-independence).**
> - User-reported: `run_method_y_image_factorial_v4.ps1` blocked with RUN_CONTRACT_MISMATCH
>   after the v3-verifier commits. Diagnosis: `software_contract_v4.contract_sha256`
>   included the LIVE git commit in the resume-binding digest — a docs-only commit
>   changed HEAD and fail-closed an in-flight run despite ZERO bound-source byte changes.
> - Fix (`bebad7b`): digest-bound core = bound-source bytes + toolchain versions + CUDA
>   runtime/hardware + locked policies ONLY. `git_commit`/`git_provenance_status`/
>   `scoped_dirty_digest` demoted to provenance metadata outside the digest (mirrors the
>   frozen-contract semantics proven in the v3.2 verifier fix `211dd49`). Byte change in
>   any bound source still flips the digest (adversarial test).
> - Migration tool hardened: committed plans no longer shadow fresh ones after a digest
>   FORMULA change (supersedes chain, new auditable provenance file). Applied
>   `d521fa67 -> a2013821` over 8 checkpoints / 22 files / 1 master / run_contract with
>   semantic fingerprints verified — weights untouched, zero outer exposure.
> - Proof: 5 new binding tests + 42/42 v4 suites green; live run resumed under the new
>   contract (A1/outer_0/inner_1 committing epochs). v4 run_contract digest is now
>   `a2013821a158305b7061131da8c2e8be4706c951924f314b223669260842beb5`.
> - Operator skill review: no dedicated contract-migration skill exists in the shared
>   catalog; used repo's own proven atomic-migration pattern + TDD, consistent with
>   available testing/debugging skills.

> **Append 2026-09-02 — v3.2 VERIFY PASS (frozen-contract verifier) + v4 factorial handover.**
> - v3.2 (root `results/method_y_revival_v3_image_v2`) production run COMPLETED by
>   user: 20/20 outer cells committed, per-cell access exactly-once=1, both branches
>   BRANCH_OUTER_CV_COMPLETE + LAUNCHER FINALIZED; OOF reports built from REAL
>   locked-checkpoint metrics (science_gate FAIL kept honest => EXPLORATORY).
>   Headline (internal, exploratory): amyloid aug vs base dIBS mean -0.00141
>   [-0.00752, 0.00477] (CI includes zero -> non-superiority recorded); tau
>   dIBS -0.02402 [-0.03882, -0.00959], dC +0.089 [0.009, 0.167] (internal
>   improvement only; scanner/quantitative/external gates closed).
> - Defect found this session & fixed (`211dd49`): verifier compared software
>   contract against LIVE HEAD and recomputed per-fold digests with today's
>   code -> any post-run commit retroactively FAILed finalized runs. Now:
>   readiness binds to the FROZEN contract inside run_contract.json; final
>   recovers each fold digest from its SELF-HASHED state.json (frozen-chain);
>   audit_post_execution_drift is INFORMATIONAL_ONLY (changed/added/removed
>   bound sources reported, never blocks). +4 adversarial tests, 19/19 green.
> - v3.2 verification evidence: `--level readiness` PASS (drift reported:
>   3 shared sources changed post-run); `--level final` PASS with
>   valid_outer_cells=20, outer_access_count_per_cell=1, privacy PASS,
>   inner_metric IPCW_INTEGRATED_BRIER_SCORE.
> - v4 factorial benchmark (Codex implementation, reviewed): 5 matched arms
>   (A0 baseline / A1 +aug / A2 ResNet10-GN / A3 +aug+R10 / A4 ResNet18-GN
>   w=0.25), PET-safe augmentation 1deg/0.5voxel train-only (QC-locked
>   Pearson/Dice p5 ~= 0.992; 2deg envelope FAILED; scale/PSF/noise disabled
>   without evidence), deterministic subject x fold x epoch seeds, inner
>   IPCW-IBS-only selection, one-SE across 5 outer-training partitions,
>   outer exactly-once, 50-cell verifier. Production v4 stopped by operator at
>   amyloid/A0/outer_0/inner_1/candidate-3/epoch-13 (atomic per-epoch commits;
>   same command resumes). Telemetry ~12-15 s/epoch (A0) -> full 5x2x5x3x3
>   programme estimated ~18-28 h. Device-mismatch resume fix + auditable
>   contract migration (digests only; weights/optimizer/history untouched)
>   verified via `provenance/contract_migration_v4_20260901.json`.

> **Append 2026-08-26 — supervisor-plan execution: v1 INVALIDATED, v2 root staged.**
> Independent supervisor audit found 7 critical gaps; repairs (mostly present as
> uncommitted working-tree edits) verified, completed, tested and committed.
> - v1 `results/method_y_revival_v3_image_v1` INVALIDATED (4 reason codes incl.
>   NLL-fallback HPO scoring, missing software contract, outer wrapping violation,
>   public censoring_reference leak). fold_0 exposure recorded; aggregation banned.
> - Active root: `results/method_y_revival_v3_image_v2` + config
>   `method_y_image_training_v3_2.yaml`; run_contract `7fa02033...`;
>   deterministic CUDA (CUBLAS=:4096:8, benchmark=false, tf32=false).
> - New/verified: strict IPCW-IBS inner selection (fail-closed), evaluator inside
>   exactly-once closure, all-20-cells completion guards, software source+runtime
>   contract binding, recursive privacy scanner (row-level leak blocked),
>   per-branch + aggregate IMAGE_CLAIM_POLICY, real metadata join for
>   scanner/LMO/subgroup executor, two-level verifier (--level readiness|final;
>   v1 final = INVALIDATED).
> - Verification: tests/method_y_revival **467 passed / 1 skipped**; focused v3
>   green incl. CUDA no-fallback + GPU bitwise interrupt/resume; preflight(v2)
>   READY true/false/false; smoke(v2) PASSED cuda 76 MB; full-repo suite
>   1946 passed / 120 failed (pre-existing out-of-scope MX/redesign failures,
>   untouched) + 3 MX collection errors (dirty train_3d_mri, out of scope).
> - Commits: fix commit (supervisor repairs + aggregate/env additions + tests +
>   invalidation mirror) and docs commit (this ledger, memory supersession,
>   README status rows, handover report).
> - Handover report: `advisor-plans/agent-reports/method-y-image-v3-supervisor-plan-handover-20260826.md`.
> - Dual-gate note (repo-side record): dual gate permits internal exploratory
>   image training; quantitative/scanner/external gates still cap all claims.
>     The global Codex memory store is outside this repository; agents must rely
>     on this ledger + PROJECT_MEMORY_INDEX supersession block.

> **Append 2026-08-25T17:30Z — resume STAGE_ORDER crash fix.**
> User-reported: `BLOCKED STAGE_ORDER:cannot_move_backwards:GPU_SMOKE_PASSED->PREFLIGHT_DONE` on second launcher run.
> - Cause: preflight gates (INIT..GPU_SMOKE_PASSED) are re-verified every launch, but the monotonic machine rejected re-setting them once pointer was past.
> - Fix (`7a50e84`): `_REVERIFIABLE` set in `image_launcher_state_v3.py` — these gates refresh facts on re-set without regressing the progress pointer; training/report/finalized stages remain strictly backwards-fail-closed. Launcher also skips per-branch stage commits already present in facts.completed (--branch both after amyloid done).
> - Tests: +2 RED-first (`test_preflight_stages_are_reverifiable_on_resume`, `test_progress_stages_cannot_move_backwards`); 7/7 state-machine green; bounded live run resumed mid-HPO, selected winner lr3e4_wd1e4 (mean_ibs 0.689), entered outer_0 final fit; orphan cleaned.

> **Append 2026-08-25T16:30Z — audit round 3 (Plans 009-012) + 6.1x GPU throughput.**
> Baseline `d680748`. Process guard clean before writes.
> - **F-R3-A fixed:** inner-HPO IBS silently evaluated LAST-EPOCH weights on locked-load failure; now `load_locked_state_for_eval()` fails closed (`INNER_IBS_CHECKPOINT_LOAD_FAILED`) on missing/corrupt/contract-mismatch.
> - **F-R3-B fixed:** Plans 012 analyses had no real executor; launcher marked robustness/reports COMPLETE on file presence alone. New `build_real_oof_reports()` consumes matched variant folds from private metrics and emits calibration/subgroups/paired-bootstrap(LOCKED 2000)/incremental-gate/IMAGE_CLAIM_POLICY + privacy scan; launcher stages advance only on executor status, else PENDING_REAL_OOF/PARTIAL_REAL_OOF facts.
> - **Perf (science-neutral):** lazy RAM volume cache in `H5PatternSurvivalDataset` (float16 passthrough, byte-capped default 6 GiB, graceful disk fallback); worker-spawn path removed under IMAGE_V3_PERF (spawn workers would duplicate caches). Measured identical workload 120 subj x2 epochs pattern-variant: 45.80 s -> 7.52 s total (**6.1x**), steady-state epoch 26.6 s -> 1.9 s (~14x), GPU peak unchanged 265 MB. Same loss/optimizer/splits/seeds/dtype pipeline; per-epoch shuffle seeds intact so resume semantics untouched.
> - **Verification:** RED 5/5 then GREEN 5/5 new tests (`tests/method_y_revival/test_audit_round3_v3.py`); full suite **450 passed**; smoke PASSED cuda 18 MiB; preflight READY; gates true/false/false.
> - Report: `advisor-plans/agent-reports/method-y-image-v3-hard-audit-round3-plans009-012-perf-20260825T160000Z.md`.

> **Append 2026-08-25T14:43Z — hard audit round 2: two critical fixes + Plan 013 stage machine.**
> Scope: only work after the 16:00Z-perf append below. Baseline `41d1c7f`/`d447c1c`.
> - **F1 CRITICAL fixed:** launcher outer evaluators were hardcoded placeholders (`c_index 0.62/0.61`) that would have committed fabricated results and burned exactly-once access. Replaced with real evaluator module `src/method_y_revival/image_outer_eval_v3.py` (`evaluate_locked_outer`: strict contract-checked checkpoint load, train-only transformer, H5 inference, prespecified grid, administrative censoring, IBS + IPCW Uno C from weights; OOF rows bootstrap-shaped). Anti-placeholder regression test asserts two different locked models give different metrics.
> - **F2 HIGH fixed:** winner `learning_rate`/`weight_decay` now carried in candidate records and consumed by `run_final_training` (raises `WINNER_HYPERPARAMS_MISSING` if absent) — final fit previously always used lr=1e-4/wd=1e-4 regardless of HPO outcome.
> - **F3 MEDIUM fixed:** Plan 013 16-stage machine materialized in `src/method_y_revival/image_launcher_state_v3.py` (self-hashed, contract-bound, order-enforced, FINALIZED gated). Launcher advances PREFLIGHT_DONE..GPU_SMOKE_PASSED with facts, commits per-branch stages on completion, and sets ROBUSTNESS/REPORTS/PRIVACY only from real OOF presence else records `PENDING_REAL_OOF` honestly.
> - **Wrong-era artifacts deleted (user-directed):** `orchestration/{amyloid,tau}/**` and `.smoke_test/**` under canonical root; stale writer lock removed after verifying pid dead. No fabricated outer result had been committed at detection time (verified).
> - **Verification:** RED 10 failed then GREEN 10 passed; full `tests/method_y_revival` 445 passed; smoke PASSED cuda 26 MiB; preflight READY; bounded live run printed STAGE_COMMITTED/GPU_SMOKE_PASSED and entered amyloid inner HPO (scalar_tabular epochs ~0.35 s).
> - Report: `advisor-plans/agent-reports/method-y-image-v3-hard-audit-round2-20260825T144330Z.md`. Claim gates unchanged and closed.

> **Append 2026-08-25T16:00Z — image v3 full-training enablement + GPU perf + 10× feasibility research.**
> Covers ONLY work after the 13:30Z hard-audit append below (commits `7c30be5`..`41d1c7f`). Plans 009-013 audit is NOT repeated here.
>
> **Commits (all on `method-y-post-revival-sanitized-20260812`, no push, no history rewrite):**
> - `cd6a65f` feat(method-y): enable full image training — real H5 cache + launcher full nested-CV loop
>   - `src/method_y_revival/image_training_orchestrator_v3.py`: new `h5_cache_path` param → `H5PatternSurvivalDataset` over real `caches/method_y/revival_v3_image/{amyloid,tau}_pattern_128_v1.h5`; inner-HPO score replaced NLL surrogate with TRUE IBS (`survival_at_times` + `sksurv.metrics.integrated_brier_score` on common-support grid from interior bin edges); config-driven batch/grad_accum/max_epochs/patience/amp when h5 cache present (synthetic path preserved for tests).
>   - `scripts/run_method_y_image_training_v3.py` run mode: full loop amyloid→tau × {scalar_tabular, scalar_tabular_plus_pattern_image} × outer_0..4; per-unit resume skips: winner JSON with matching master contract → skip HPO; `development/<branch>/<variant>/outer_<k>/final/locked_checkpoint.pt` exists → skip final fit; `OuterFoldController` state OUTER_EVAL_COMMITTED/FINALIZED → skip outer (exactly-once preserved).
> - `c514df4` fix: smoke resume `TypeError: RNG state must be a torch.ByteTensor`
>   - `src/method_y_revival/image_training_runtime_v3.py` `_restore_rng`: checkpoint saved with `map_location=cuda` put CPU ByteTensor rng states on cuda:0; now `.cpu()` applied to torch_cpu and every torch_cuda state before `set_rng_state`/`set_rng_state_all`. Verified: second smoke run resumes and passes (`{"smoke":"PASSED","device":"cuda","peak_vram":18384896}` at `results/method_y_revival_v3_image_v1/.smoke_test/locked_checkpoint.pt`).
> - `97dec14` fix: `RUN_CONTRACT_MISMATCH:orchestrator_master_state` on first real run
>   - Cause: placeholder-era `orchestration/<branch>/orchestrator_state.json` written with dummy `0*64` digests (commit c514... era) vs real membership/cache/split/readiness digests.
>   - Fix in `scripts/run_method_y_image_training_v3.py`: on that specific mismatch only, delete stale `orchestrator_state.json` + stale `winner_*.json`, re-instantiate orchestrator with real digests. Genuine config/cache/split drift still fails closed (Plan 009 rule).
> - `eaf6d57` perf: terminal throttle + GPU utilization
>   - `image_training_runtime_v3.py` `fit_survival_model`: `EPOCH_COMMITTED` printed only when epoch==first, ==target, early-terminal, or ≥40 s since last print (`last_print` gate) — terminal spam removed; test `test_launcher_emits_epoch_progress_with_eta` still green.
>   - `seed_everything`: TF32 enabled (`set_float32_matmul_precision("high")`, `allow_tf32=True`) always; under env `IMAGE_V3_PERF=1` → `cudnn.benchmark=True`, `deterministic=False`; without it deterministic/benchmark=False kept for bitwise-resume tests.
>   - `_loader`: `num_workers=2` auto for `H5PatternSurvivalDataset` when perf flag set (H5 handle reopened per worker via `__getstate__`); `persistent_workers=False`.
>   - `configs/method_y_image_training_v3_1.yaml` model: `batch_size 2→4`, `gradient_accumulation_steps 4→2` — effective batch stays 8; optimizer-step count unchanged. `scripts/run_method_y_image_training_v3.ps1` sets `IMAGE_V3_PERF='1'`.
> - `41d1c7f` research+perf: 10× feasibility + channels_last_3d
>   - hyperresearch full-tier run `research/runs/gpu-3d-cnn-10x-9a1e13/` (query.md, scaffold.md, prompt-decomposition.json, temp/search-plan.md 38 queries, shims rendered). Web search lane blocked (Exa/SemanticScholar 403; Playwright CDN geo-blocked 403) → fallback synthesis from PyTorch docs + local benchmarks: `research/notes/final_report_gpu-3d-cnn-10x-9a1e13.md`. Ship gate intentionally left `blocked(verify)` (length/citation-density below full-tier floor) — recorded as honest partial, NOT presented as shipped full-tier report.
>   - Conclusion: single-GPU 10× not reachable without breaking locked science; measured combined technical headroom ≈3.8× (TF32 1.3 × batch4×2 1.8 × channels_last 1.2 × workers2 1.2 × cudnn.benchmark 1.15); effective wall-time vs naive-90h ≈9× because early stopping averages ~12 epochs of max 40. REJECTED as science-changing: DDP on one GPU, patch 64³ (breaks 128³ reference contract), pruning/compression (changes locked GROUPNORM_RESIDUAL_TINY_3D capacity), gradient checkpointing (adds compute), torch.compile on Windows (triton unavailable — verified RuntimeError, reverted).
>   - `image_training_runtime_v3.py`: `model.to(memory_format=torch.channels_last_3d)` and input `.to(memory_format=torch.channels_last_3d)` in `_forward`, both gated on `IMAGE_V3_PERF=1`+cuda; try/except no-op fallback.
>
> **Live operational actions this session (no code change):** removed stale `.image_training.writer.lock` after orphaned python runs (pids 16540/21624/18532/2436 — verified dead before unlink, never killed by name alone); deleted placeholder-era `orchestration/*/orchestrator_state.json`; earlier tau cache rebuild under new contract already covered by 13:30Z append.
>
> **Measured timings on RTX 4060 Laptop (real 128³ cache, development subjects):** 100 subj bs2/ga4 perf-off = 15.2 s/epoch; 100 subj bs4/ga2 perf-on = 9.0 s/epoch; 560 subj = ~40 s/epoch either way (I/O-bound tail). Estimated full programme ≈20 h wall (amyloid ≈12 h, tau ≈8 h) assuming mean ~12 epochs/run via early stopping; resumable at every epoch/checkpoint/outer commit.
> **Verification:** `pytest tests\method_y_revival -q` = 435 passed (rerun after each change batch); preflight READY; readiness hashes unchanged (19baa31d..., memberships c02a93.../b511ff..., source artifacts 784fa.../41f59...). No v1/v2/raw mutation; all outputs/temp on H.

> **Append 2026-08-25T13:30Z — Method Y image v3 hard audit (Plans 009-013).** This session is image-v3-only.
> - Branch `method-y-post-revival-sanitized-20260812` HEAD `5405716` + `4ef3495` → new image v3 commit (42 files, 7062 insertions). No `git reset/clean/checkout --`, no branch switch, no push of leaky refs.
> - Canonical output `H:/New project/results/method_y_revival_v3_image_v1` READY (19baa31d...), 560/110 + 306/75, cache 895/306 on H, provenance-bound (strict HDF5 source_sha, readiness source_artifact, true IBS via `survival_at_times`).
> - Hard audit fixed: strict `HDF5_GROUP_SOURCE_SHA_MISMATCH`, readiness `source_artifact_sha256` binding (amyloid 784fa..., tau 41f59...), `image_training_orchestrator` true IBS (not NLL), `LogisticRegression` sklearn compat, `run_method_y_image_training_v3.py` smoke (cuda 26 MiB).
> - Tests: 435 passed in `tests/method_y_revival` (was 363), focused 16+19+15 all green, preflight `READY_TO_RUN_FINAL_INTERNAL_EXPLORATORY_IMAGE_TRAINING`.
> - Reports: `advisor-plans/agent-reports/method-y-image-v3-plan-010..013-*.md` + `...hard-audit-20260825T133000Z.md`. Launcher: `powershell -NoProfile -ExecutionPolicy Bypass -File 'H:\New project\scripts\run_method_y_image_training_v3.ps1'` (resumable).
> - No v1/v2/raw mutation, temp/cache on H only, claim gates still closed. See `docs/PROJECT_MEMORY_INDEX.md` 2026-08-25 image v3.

Read this before continuing Method Y or assuming the earlier closeout was valid.
Machine twin: `results/method_y_revival_v2/journals/session_ledger_20260815.json`.
Method X twin (untouched): `results/method_x_registry/SESSION_NOTE_METHOD_Y_DEFECT_REPAIR_20260815.json`.

This file does **not** start GPU training, Method Y HPO, or a git push.

## 2026-08-16 four-gate execution (this session)

Machine twin: `results/method_y_revival_v2/journals/four_gate_closeout_20260816.json`.

Terminal status: `TASK4_COMMITTED + PREFLIGHT_ONLY_TRAINING_NOT_STARTED + SMOKE_DISPATCHED + SCIENCE_GATES_STILL_CLOSED`.

- Live HEAD `54057162efe69556539d4c3e092b9e4871c4f161` (Method-Y-only AMYLCENT
  honesty gate). History still `NOT_PUBLISHABLE`. Do not push leaky refs.
- Task 4 is proof-bearing `COMMITTED` + `AGENT_MULTI_PASS_QC_PASS` + QUALITY
  `PASS` under the sanitized-SOT rubric amendment. Six locked science hashes
  unchanged.
- `--preflight-only` is `PREFLIGHT_ONLY_TRAINING_NOT_STARTED`.
- `PRETRAINING_READINESS_V2.json` keeps `training_allowed=false` and is not the
  Task-6 receipt. Synthetic smoke used
  `TASK6_SOURCE_REGIONAL_SYNTHETIC_SMOKE_RECEIPT_V2.json` with `test_only=true`.
- Label `SYNTHETIC_SMOKE_ONLY`. Not A4 CoxNet/RSF/GBSA. Image / outer / claim /
  external remain false. Tau 0/1314 `BLOCKED` / `NO_CLAIM`. CTX unproven. DUA
  `PENDING`.
- Method X files were not staged. This session did not start MX GPU training.

## 2026-08-16 isolation note (Method Y leftover closeout)

This Method Y session does **not** own the Method X battery and did not start
or stop GPU training. Isolation facts only:

- ResNet18 live resume is C staging optimizer step **2420** (H
  `fold_2_progress.pt` still stale at 1321). The 2026-08-15 row below that said
  **1534** is a historical snapshot from that day, not the live MX pointer.
- Method X remains the assumed GPU owner.
- Task 4 is still not `COMMITTED`. `training_allowed=false`.
- Rubric: `docs/METHOD_Y_TASK4_SPEC_QUALITY_V1.md`. Re-review journal:
  `results/method_y_revival_v2/journals/task4_adversarial_review_20260816.json`
  (SPEC PASS, QUALITY FAIL; `finalize_task4_agent_review` not called).
- 2026-08-16 P4/P5 execution journal:
  `results/method_y_revival_v2/journals/p4p5_execution_20260816.json`.
  Locked Task 4 bytes were not rebuilt. Tau sidecar fill and public
  dictionary hashes are append-only amendments.

## One-line status

Method X battery resume was **not edited**. Method Y Task 4 is proof-bearing `COMMITTED` with QUALITY `PASS` under sanitized-SOT isolation; `PRETRAINING_READINESS_V2.json` still has `training_allowed=false`; a `SYNTHETIC_SMOKE_ONLY` runtime was dispatched (not A4 HPO); live git history is **not publishable**.

## Git (do not push leaky refs)

| Ref | SHA | Meaning |
|---|---|---|
| Live HEAD (`method-y-post-revival-sanitized-20260812`) | `54057162efe69556539d4c3e092b9e4871c4f161` | Method-Y-only AMYLCENT honesty gate. Branch **name is false**: history still reaches the private parquet blob. |
| Defect-repair code | `b1b99df32f4ab0a2a7cc4d4cc887eb0e6638fcbb` | Proof-bearing review evidence; Task 4 not COMMITTED. |
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

### Pass 2 (defect repair — historical; superseded by four-gate section above)

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

Live gates (2026-08-16 four-gate; not A4 clearance):

| Artifact | Status |
|---|---|
| `results/method_y_revival_v2/readiness/PRETRAINING_READINESS_V2.json` | `TASK4_COMMITTED_PREFLIGHT_ONLY_TRAINING_NOT_STARTED_SMOKE_DISPATCHED_SCIENCE_GATES_STILL_CLOSED`, `training_allowed=false`, not a Task-6 receipt |
| `results/method_y_revival_v2/pretraining_contracts/pretraining_validation_state.json` | proof-bearing `COMMITTED` + `AGENT_MULTI_PASS_QC_PASS` |
| Adversarial layer | SPEC `PASS`, QUALITY `PASS` under sanitized-SOT amendment; live HEAD still `NOT_PUBLISHABLE` |
| `results/method_y_revival_v2/training_control/TRAINING_CONTROL_PREFLIGHT_V2.json` | `PREFLIGHT_ONLY_TRAINING_NOT_STARTED` |
| Synthetic smoke | `SMOKE_DISPATCHED` via `TASK6_SOURCE_REGIONAL_SYNTHETIC_SMOKE_RECEIPT_V2.json`; not A4 HPO |
| Tau public audit | still `BLOCKED` / `NO_CLAIM` (0/1314 identity; GE/Philips 9 events) |
| Source-regional `development_allowed` | true (amyloid / tau / paired) |
| Image / outer / claim / external | all false |

Targeted Method Y v2 suite on 2026-08-15: **150 passed** (junit SHA-256 `20f689611c88f2c050d82a14f12f40938e23227bc1de1e9be58307a506c0714f`).
Four-gate P1 (2026-08-16): **165 passed**, junit `e1f55917956f3e44a1997e6c2f04a6f491a5ee74221199af7d970f244f64d743`.
Full `pytest tests` (2026-08-15): **1547 passed / 36 failed / 1 skipped**. Failures are **out of scope** (Method X dirty APIs, legacy Method Y external/image ImportErrors, missing thesis reports). Do not “fix” Method X to green that log. Record: `results/method_y_revival_v2/reports/FULL_PYTEST_STATUS_20260815.json`.

## Orchestration / memory map (what to trust)

| Kind | Path | Trust |
|---|---|---|
| This ledger | this file + JSON twin | **current Method Y session** |
| Four-gate closeout | `results/method_y_revival_v2/journals/four_gate_closeout_20260816.json` | **2026-08-16 P0–P4 machine twin** |
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

## Next agent — Method Y (CPU; no A4 HPO)

1. Do not start A4 HPO, image-CNN, outer evaluation, or external outcomes.
   `SYNTHETIC_SMOKE_ONLY` already ran 2026-08-16; do not relabel it as A4 training.
2. Do not rewrite live git history; do not push leaky refs.
3. Do not edit Method X trainer files. Leave them unstaged.
4. Task 4 is already proof-bearing `COMMITTED`. A4 HPO still needs a versioned
   training-control YAML that is not `synthetic_smoke_only`, plus science gates
   that remain closed (Tau identity, CTX, DUA).
5. `PRETRAINING_READINESS_V2.json` is not a Task-6 execute receipt.

Operator preflight only (does not start A4):

```powershell
.\.venv\Scripts\python.exe scripts\run_method_y_training_v2.py --preflight-only
```
