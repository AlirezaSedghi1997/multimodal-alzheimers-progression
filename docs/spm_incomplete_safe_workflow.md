# Safe Work Before Full SPM Completion

This document defines what may run while SPM/MATLAB preprocessing is still incomplete.

## Source of Truth

The current SPM completion source of truth is:

- `results/logs/spm_mri_preprocess_v2_resume_wsl.json`
- `data/preprocessed_mri_96/*.nii.gz`
- expected config hash: `4324429c60083666`
- expected completed files: `1700`

If a secondary report disagrees with the manifest and file count, use the manifest plus file count.

## Blocked Until `SPM_COMPLETE`

These stages must stay blocked until `scripts/spm_completion_guard.py` returns `SPM_COMPLETE`:

- `mri-volume-cache`
- `train-3d`
- `optuna`
- `fusion`
- `external`
- `calibration`
- `final-analysis`

Do not create `data/cached/t1_volume_cache_96.h5`, run 3D training, refresh XAI, or refresh final analysis while the guard reports `SPM_INCOMPLETE`.

## Allowed Before Completion

Allowed work is limited to aggregate, public-safe, non-training tasks:

- guard and orchestration improvements
- docs/config/script/test changes
- privacy, leakage, public-bundle, and stage-protocol tests
- aggregate feasibility reports that do not depend on a full MRI cache
- smoke-only dependency checks that do not publish subject-level outputs


## Continue SPM To Completion

Use the script's built-in batch loop instead of wrapping single-batch runs in an external shell loop:

```bash
cd "/mnt/h/New project"

export DATA_ROOT="/mnt/h/A4 Learn"
export MATLAB_BIN="/mnt/c/Program Files/MATLAB/R2024a/bin/matlab.exe"
export SPM_DIR="/mnt/d/Matlab & Tolboxes/spm_25.01.02/spm"

conda run -n amyloid-tau python scripts/preprocess_mri_spm.py \
  --data_root "$DATA_ROOT" \
  --matlab "$MATLAB_BIN" \
  --spm_dir "$SPM_DIR" \
  --manifest results/logs/spm_mri_preprocess_v2_resume_wsl.json \
  --execute \
  --batch_size 10 \
  --auto_batches
```

Equivalent Make target:

```bash
make preprocess-full-mri-spm-auto \
  DATA_ROOT="/mnt/h/A4 Learn" \
  MATLAB_BIN="/mnt/c/Program Files/MATLAB/R2024a/bin/matlab.exe" \
  SPM_DIR="/mnt/d/Matlab & Tolboxes/spm_25.01.02/spm" \
  SPM_BATCH_SIZE=10
```

Without `--auto_batches`, `--batch_size 10` intentionally stops after one batch and reports `BATCH_COMPLETED` so a human or scheduler can resume later. With `--auto_batches`, it keeps taking the next pending batch until all catalog rows are done or a missing/failed item is encountered.
## MRI Volume Cache Smoke Check

`scripts/mri_volume_cache_smoke.py` is a smoke-only dependency and image-loading check. It reads only catalog rows whose `sample_hash` is marked `DONE` in the SPM manifest and has a matching preprocessed MRI file.

By default it:

- writes a temporary HDF5 file outside the project cache path
- validates the HDF5 structure
- deletes the temporary HDF5 at process exit
- writes only aggregate smoke summaries under `results/tables/` and `results/logs/`
- refuses to write under `data/cached/`
- refuses to overwrite official MRI cache summary or manifest files
- marks the output as `SMOKE_ONLY_NOT_FOR_TRAINING`

The smoke output is not a Method X training cache and must not be used for 3D training, fusion, XAI, final analysis, or thesis-level performance claims.

## GPU Policy

Use `--device auto` for smoke or post-SPM MRI volume conversion. With the existing loader, `auto` uses CUDA when `torch.cuda.is_available()` is true and otherwise falls back to CPU. Use `--device cuda` only when a fast failure is preferable if CUDA is unavailable. This policy changes processing speed, not the research target or model input contract.
