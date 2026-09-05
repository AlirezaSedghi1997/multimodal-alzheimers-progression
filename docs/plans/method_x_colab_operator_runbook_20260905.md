# MX-03 Colab operator runbook — synthetic smoke (AUTHORIZED) + official path (LOCKED)

Companion to `method_x_colab_conditional_plan_20260905.md` (conditional official
design, DESIGNED_NOT_AUTHORIZED). This runbook has two halves with opposite
authorization: Part A is authorized NOW (zero human data); Part B stays locked
until G1–G5 sign-offs.

Rule zero for both halves: NEVER point an official battery/validity run at
synthetic data, and NEVER report synthetic numbers as evidence. Synthetic
artifacts live under `tmp/` (or Colab `/content/…`) and are never merged into
`results/method_x_registry` or the battery output root.

## Part A — Synthetic smoke on Colab GPU (AUTHORIZED NOW)

Goal: prove the full mechanics on a Colab GPU with random data —
parse → resnet34 build → train step → checkpoint → metrics → outer eval —
before spending local GPU hours. Catches exactly the failure family already
found locally (unknown flags/backbones/kwargs).

### A1. Pack the transfer set (on H, CPU-only, ~2 min)

A ready-made pack has been built and verified from branch
`method-x/mx03-audit-fix` (full intra-repo import closure of the smoke path,
machine-computed — 36 code/text files, 345,472 bytes, zero data files):

- Path: `C:\Users\USER\AppData\Local\Temp\opencode\mx03_colab_pack.zip`
- SHA-256: `dc1ea80c052b2f91747d3e9899a123f9cd0a95ec1b781af7aaad6a30e8c90f43`
  (byte-identical to branch blobs; verified per-file, zero drift, zero non-code)
- Verify before upload: recompute the hash; any mismatch = rebuild, never
  hand-edit the zip.

To rebuild it after any branch change (same closure + `requirements.txt`):

```powershell
Set-Location -LiteralPath 'H:\New project'
.\.venv\Scripts\python.exe C:\Users\USER\AppData\Local\Temp\opencode\mx03_pack_closure.py
# take the TRACKED/UNTRACKED list minus amuse_x_hybrid3d.py (lazy import, out of scope),
# then: git archive -o <pack.zip> method-x/mx03-audit-fix -- <list> requirements.txt
```

Upload that zip to Google Drive. Nothing human-derived is inside (345 KB
total — one MRI volume alone is 8 MB; plus every entry is `.py`/`.txt`).

### A2. On Colab (VS Code extension backend or notebook terminal)

```bash
# 0. Pin and record the machine (paste outputs back to the agent)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 -c "import torch,monai,h5py; print(torch.__version__, monai.__version__, h5py.__version__)"

# 1. Stage code (Drive mounted at /content/drive, or direct upload)
mkdir -p /content/mx03smoke && cd /content/mx03smoke
unzip -o /content/drive/MyDrive/mx03_colab_pack.zip
pip install -q h5py  # torch/monai ship with Colab; record versions from step 0

# 2. Generate the miniature (128^3 for fidelity, N=48; ~400 MB)
python3 scripts/mx03_build_synthetic_smoke.py --out-root /content/mx03smoke/synth --n 48 --volume 128 --seed 13

# 3. Fresh-dir contract preflight (must print CONTRACT_PREFLIGHT_PASS)
SYN=$(python3 -c "import json;print(json.load(open('/content/mx03smoke/synth/synth_manifest.json'))['cache_sha256'])")
python3 -u scripts/train_3d_mri.py --cache /content/mx03smoke/synth/synth_cache.h5 \
  --cache_sha256 $SYN --spm_snapshot_sha256 2d659c87d451da0ebc14e847570ea52ea3804ccc2b153f2764ab56cbb1098ebb \
  --splits /content/mx03smoke/synth/splits_synth --out_dir /content/mx03smoke/run \
  --phase dev --backbone resnet34_3d --fusion cross_attention --task_loss bce \
  --lr 0.0001 --scheduler cosine --dropout 0.2 --resolution 128 --folds 5 --epochs 2 \
  --seed 13 --batch_size 1 --grad_accum 2 --device cuda --require_gpu --restore_best_at_end \
  --amp --checkpoint --num_workers 2 --pin_memory --no-persistent_workers --prefetch_factor 2 \
  --sanity_random_labels --input_mode mri_plus_tabular --contract_preflight_only

# 4. Mechanics run (expect EXIT 0; metrics are RANDOM-DATA noise, not evidence)
#    (same command without --contract_preflight_only)
```

Success = exit 0 + `method_x_3d_fold_0_progress.pt` + `three_d_metrics.json`
with `aggregate.status=COMPLETED`. Send back: EXIT code, GPU record from
step 0, and the tail 20 lines. Any `unrecognized arguments` / `Unknown
backbone` / `unexpected keyword` = a real latent bug — report verbatim,
do NOT patch on Colab; the fix lands on H with tests.

### A3. Cleanup
Delete `/content/mx03smoke` and the Drive zip when done (nothing scientific
to keep; the PASS/FAIL verdict is recorded on H from your report-back).

## Part B — Official 9-arm run on Colab (LOCKED until G1–G5)

Unlocks only with: governance sign-off (G1), DUA clearance (G2), transfer
manifest (G3), deletion receipt plan (G4), single-GPU rule (G5). Then:

1. Transfer EXACTLY the G3 manifest; verify `cache_sha256` + splits digest
   on arrival — mismatch = STOP.
2. Rebuild the pinned env (torch 2.5.1+cu121 / monai 1.5.2 per the H
   preflight runtime block, not Colab defaults).
3. Regenerate `private/mx03_task_contracts.json` on the Colab machine
   (Linux interpreter/paths → new digests), then write
   `amendments/mx03_colab_platform_relock_v1.json` from the template below.
4. `--contract_preflight_only` per arm → all PASS → identical
   `--mode run --resamples 1000 --seed 13` command (Linux spelling),
   supervised by `scripts/mx03_execution_watchdog.py --mode watch`.
5. One GPU model for all 9 arms (§4 of the conditional plan on preemption).
6. Sync per-epoch checkpoints to Drive; restore-verify by hash each session.
7. After completion: copy back ONLY `mx03_runs/*/three_d_metrics.json`,
   `*_outer_eval.json`, `run_contract.json`, `private/*.json`, `reports/`;
   verify every sha256 against the re-lock amendment before the H
   orchestrator reads them. Then execute the G4 Drive deletion.

### Re-lock amendment TEMPLATE (fill on the Colab machine, append-only)

```json
{
  "amendment_id": "mx03_colab_platform_relock_v1",
  "append_only": true,
  "supersedes_command_digests": {"<arm>": "<H command_sha256> (from mx03_task_contracts.json)"},
  "new_command_digests": {"<arm>": "<Colab command_sha256>"},
  "platform": {"gpu_model": "<nvidia-smi>", "cuda_capability": "<torch>",
               "torch": "<v>", "monai": "<v>", "os": "linux"},
  "science_unchanged": ["seed 13", "folds 5", "epochs 100", "patience 5",
    "batch 1", "grad_accum 8", "AMP on", "deterministic flags on",
    "cache_sha256 ccd0d312…", "splits digest <v>", "loader knobs",
    "augmentation locked off"],
  "storage_map": {"H_project_root": "<drive mount>/AMUSE-X/<run_id>"},
  "governance_refs": {"G1_signoff": "<path>", "G2_dua_clearance": "<path>",
    "G3_manifest": "<path>", "ethics_code": "<id, e.g. faculty code>"},
  "framing": "within-Colab deltas VALID; absolute battery-vs-arm numbers CROSS-PLATFORM (report with platform ID)"
}
```

## Watchdog note (both halves)

`scripts/mx03_execution_watchdog.py --mode once` is read-only and safe
anywhere (include its JSON in every report-back). `--mode watch` relaunches
only on safe-pause (exit 75) with a bounded count, refuses `--no_resume`,
and writes BLOCKED receipts otherwise. After ANY code fix, one manual
identical-command rerun is required first (BLOCKED entries re-execute;
COMPLETED entries are reused) — the watchdog never auto-clears a HARD class.
