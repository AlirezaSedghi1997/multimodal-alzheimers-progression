# Method X MX-03 on Google Colab — CONDITIONAL plan (DESIGNED, NOT AUTHORIZED)

Status: `DESIGNED_NOT_AUTHORIZED`. No data moves, no contract edits, no Colab
execution is permitted by this document. Execution unlocks only when every box
in §1 is signed. Until then the sole authorized path stays the local RTX 4060
sequential run after Method Y (`MX03_FIVE_MODEL_PREP_20260904.json`).

Related receipts (H-only): `results/method_x_registry/MX03_AUDIT_AND_FIX_20260905.json`
(Colab verdict NOT-VIABLE for the *locked* command), `MX03_TRAINER_FLAGS_FIX_20260905.json`.

## 0. Why a conditional plan can exist at all

The locked MX-03 command is Windows/H:/C:-bound and its digests
(`command_sha256`, `contract_sha256`, `storage_policy_digest`) *define* MX-03
validity — running it verbatim on Colab Linux is impossible, and editing it
ad hoc voids the evidence. The only legitimate route is a **new versioned run**:
same science (seed/cache/splits/params/budget), new platform, re-locked
digests, append-only amendment binding old→new. Within one Colab machine the
9 arm deltas stay internally consistent; battery-vs-arm numbers become
cross-platform and must be reported as such (§5).

## 1. Conditions precedent (ALL must be signed; absence = BLOCK)

- [ ] **G1 — Governance sign-off:** explicit written approval for transferring
  A4/LEARN-derived artifacts to Google infrastructure, recorded in
  `results/method_x_registry/` with approver identity + date.
- [ ] **G2 — DUA clearance:** the A4/LEARN data-use agreement reviewed for
  third-party-cloud clauses; currently DUA is `PENDING`
  (`docs/plans/method_y_remaining_work_20260815.md:48,71`). If the DUA
  forbids redistribution, this plan is VOID — stop here.
- [ ] **G3 — Transfer manifest:** exactly these files, nothing else:
  `data/cached/t1_volume_cache_128.h5` (6,531,064,014 bytes, sha256
  `ccd0d312…`), `splits/` (~1 MB total), code snapshot at the pinned commit,
  pinned env (`torch==2.5.1+cu121`, `monai==1.5.2`, see preflight runtime
  block). No raw IDs (cache attrs already `contains_raw_identifiers=False`;
  re-verify post-transfer by hash).
- [ ] **G4 — Retention/deletion:** Drive copy deleted after the run with a
  deletion receipt (hash of emptied paths + date), unless the DUA requires
  otherwise.
- [ ] **G5 — Single-machine rule accepted** (§4): all 9 arms on ONE GPU model;
  any exception triggers §4 fallback, never silent mixing.

## 2. Mechanical adaptations (enumerated; each lands in the re-lock diff)

1. `PROJECT_ROOT`-relative paths: code already resolves via
   `Path(__file__).parents[1]` — portable. Backslash `out_dir`s in stored
   contracts (`mx03_task_contracts.json`) must be regenerated with POSIX
   separators (changes `command_sha256` — expected, re-locked).
2. Interpreter: locked `PROJECT_ROOT\.venv\Scripts\python.exe` → Colab
   `python3` with a pinned `requirements.txt` env rebuilt from the preflight
   runtime block (torch/cuda/cudnn/monai/h5py versions recorded, not
   negotiated).
3. Storage policy: the `H:\New project` + `C:\…\checkpoint_staging` allowlist
   (`configs/policy/method_x_storage_locality_v1.yaml`,
   `storage_policy_digest` in every checkpoint) must be amended to a Drive
   mount root (e.g. `/content/drive/MyDrive/AMUSE-X/<run_id>/…`). Drive free
   tier (15 GB) is INSUFFICIENT — working set ≈ 6 GB cache + ~6 GB/arm live
   checkpoints + ~50 GB cumulative writes; provision ≥100 GB Drive or
   ephemeral disk + session-end sync (then every recycle re-uploads ~6 GB).
4. MX-03 checkpoints are per-epoch ~970 MB atomic saves (no 32-step journal
   in this path) — resume granularity stays one epoch; a recycle loses the
   running epoch only.
5. Science untouched: seed 13, folds 5, epochs 100, patience 5, batch 1,
   grad-accum 8, AMP on, deterministic flags on, `num_workers=2/pin_memory/
   no-persistent_workers/prefetch_factor=2`, augmentation locked off.
   `--device cuda --require_gpu` unchanged (TPU explicitly out: zero XLA
   imports in the stack; hard-CUDA via `torch.cuda.amp`, rng-state calls,
   MONAI nets).

## 3. Re-lock procedure (on the Colab machine, before any training)

1. Fresh clone at the pinned commit + pinned env install; record
   `nvidia-smi` (GPU model, driver) + `torch.cuda.get_device_capability()`.
2. Verify transferred hashes (`cache_sha256`, splits digest) — mismatch = STOP.
3. Regenerate `private/mx03_task_contracts.json` with the Linux
   interpreter/paths (new `command_sha256`/`contract_sha256` per arm).
4. Write append-only amendment `mx03_colab_platform_relock_v1.json` binding:
   old digests → new digests, GPU identity, env versions, storage-map
   old→new, governance refs (G1–G5), and the §5 scientific framing.
5. Run `--contract_preflight_only` per arm; require PASS before `--mode run`.
6. The existing `--mode run --resamples 1000 --seed 13` command (Linux
   spelling) executes arms sequentially with the same supervised retry
   semantics (75 = pause/resume, non-zero = triage, never `--no_resume`).

## 4. Operations runbook (VS Code Colab extension)

- Connect extension → Colab backend; pin a GPU backend and RECORD its model.
  T4 (typical free tier) is slower than the local 4060 — budget ~1.5–2× the
  local 15–25 h estimate. A100 (paid) is faster but rarer; either is
  acceptable IF §G5 holds for the whole run.
- Sessions cap at ~12 h with ~90 min idle reclaim: keep the session open;
  the runner's 15 s supervisor heartbeat is process activity, not Colab
  input — plan 2–4 sessions, resuming the identical command each time
  (committed folds skip, running fold resumes from its last epoch checkpoint).
- Sync checkpoints to Drive continuously (per-epoch atomic saves make this
  clean); verify hash after each session restore.
- If preempted onto a DIFFERENT GPU model: STOP, record, and either return
  to the original model or restart the interrupted arm as a new versioned
  run with disclosure — never silently mix architectures inside one MX-03.

## 5. Scientific framing (what the numbers may and may not claim)

- VALID on Colab: all within-run deltas (sanity arms must fail to learn —
  huge effects, platform-robust; drop-arm ordering; bootstrap CIs computed
  on Colab OOF). Promotion thresholds stay numerically locked; interpret
  drop-deltas (~0.005 scale) with the platform caveat.
- WEAKENED: any absolute comparison of a Colab arm against H-battery OOF
  numbers (cross-platform systematic shift possible; bitwise identity
  explicitly not claimed — determinism flags are warn-only, AMP/scaler
  trajectories are arch-dependent, RNG state is not restored on resume).
- UNCHANGED: cohort-confound ceiling (INTERNAL-only), no-diagnosis /
  no-causal / no-external claims, multiplicity reporting (raw + family
  method), inner-only calibration discipline.
- The Colab MX-03 report must carry the platform amendment ID and the GPU
  identity; without them it is inadmissible the same way an undigested run is.

## 6. Cost/benefit vs the authorized local path

- Local: 0 transfer risk, 0 re-lock risk, ~15–25 h GPU after Method Y, full
  bitwise story. Colab: same science + transfer/ops risk + re-lock
  ceremony + weaker battery comparability, in exchange for calendar overlap
  with Method Y. Recommended ONLY if the local GPU calendar slips by more
  than the re-lock + transfer overhead (roughly: cache upload hours + env
  build + slower T4 epochs).
