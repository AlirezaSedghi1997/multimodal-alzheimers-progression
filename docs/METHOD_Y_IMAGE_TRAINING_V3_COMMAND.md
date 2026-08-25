# Method Y Image Training v3 — Command

Canonical production command (resumable, no timeout, no --no-resume):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File 'H:\New project\scripts\run_method_y_image_training_v3.ps1'
```

This PS1 sets all temp/cache roots to H:\New project\.runtime\method_y_image_v3 and delegates to:

```powershell
& 'H:\New project\.venv\Scripts\python.exe' 'H:\New project\scripts\run_method_y_image_training_v3.py' --config 'H:\New project\configs\method_y_image_training_v3_1.yaml' --resume
```

## Modes

- `--mode preflight` – read-only checks, prints READY_TO_RUN_FINAL_INTERNAL_EXPLORATORY_IMAGE_TRAINING
- `--mode smoke` – 4–8 scans, forward+backward, AMP, batch_size=2, gradient_accumulation=4, peak VRAM, finite loss
- `--mode run` (default) – resumable full pipeline: cache → GPU smoke → amyloid HPO → amyloid lock → amyloid outer → tau HPO → tau lock → tau outer → robustness → reports → privacy → FINALIZED
- `--mode status` – read-only state dump
- `--mode verify` – read-only hash verification

`--resume` is default true; `--no-resume` does not exist. No internal timeout. Interruption: rerun same PS1, it resumes from last committed epoch/stage/fold/candidate/inner-fold/outer commit.

## State Machine

```
INIT → PREFLIGHT_DONE → DUAL_GATE_CONTRACT_VERIFIED → READINESS_LOCKED → PATTERN_CACHES_COMPLETE → GPU_SMOKE_PASSED → AMYLOID_INNER_HPO → AMYLOID_MODELS_LOCKED → AMYLOID_OUTER_COMMITTED → TAU_INNER_HPO → TAU_MODELS_LOCKED → TAU_OUTER_COMMITTED → ROBUSTNESS_COMPLETE → REPORTS_COMPLETE → CONSISTENCY_PRIVACY_PASS → FINALIZED
```

Amyloid then tau order is locked for early runtime failure detection; it does not change selection or claim.

## Storage

All outputs and temps on H. No C temp. H free ≥20 GiB required.

## Claim Boundary

Training execution allowed; positive/external claims remain blocked until quantitative equivalence, scanner robustness, and locked external evaluation pass. Pattern-normalized images are not official Centiloid/CenTauR.
