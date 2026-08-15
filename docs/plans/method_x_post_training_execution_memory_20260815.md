# Post-training execution memory — Method X / BrainIAC-CAFT

Store-only until the five effective CNN models are terminal.
Do not run these steps before battery completion. Do not start them in parallel with GPU CNN training.
Do not start Method Y PET training from this file.

Machine twin: `results/method_x_registry/post_training_execution_memory.json`.
Session isolation (Method Y repair, Method X untouched):
`results/method_x_registry/SESSION_NOTE_METHOD_Y_DEFECT_REPAIR_20260815.json`.

The July 2026 `post_training_roadmap.json` still says “all seven rows COMPLETED”. That wording is superseded: the live protocol is five effective models, and BrainIAC R1a–R1c sits after MX-03 and before MX-04.

## Resume commands (training, next session — not PRE)

ResNet18 first:

```powershell
Set-Location -LiteralPath 'H:\New project'
.\.venv\Scripts\python.exe scripts\run_method_x_claim_benchmark.py `
  --mode resume-backbone `
  --backbone resnet18_3d `
  --output-root 'H:\New project\results\final_results\method_x_model_battery_v3_depth_and_diversity_20260729'
```

After ResNet18 5/5 + metrics validation:

```powershell
Set-Location -LiteralPath 'H:\New project'
.\.venv\Scripts\python.exe scripts\run_method_x_claim_benchmark.py `
  --mode run `
  --output-root 'H:\New project\results\final_results\method_x_model_battery_v3_depth_and_diversity_20260729'
```

Never `--no_resume`. Never two GPU models at once. `BLOCKED_*` must not skip.

Expected automatic behaviour of `--mode run`: skip completed ResNet10/18; resume ResNet34 fold 3 step 668; exclude ResNet50 and EfficientNet; train DenseNet121 then MedNeXt-S.

Accepted checkpoint overlay: full exact payload every optimizer step on C; H mirror at epoch; replay window 0.

## After five models are terminal

### MX-03

Random-label, random-noise, ablation, calibration, cohort probe (audit-only), paired bootstrap ≥1000. Absence of scanner metadata = `NOT_MEASURABLE`, never PASS.

Command: `.\.venv\Scripts\python.exe scripts\run_method_x_validity_orchestrator.py --mode run --resamples 1000 --seed 13 --output-root 'H:\New project\results\final_results\method_x_model_battery_v3_depth_and_diversity_20260729'`

### R1a — BrainIAC gates (BrainIAC branch only)

- Install isolated venv from `research/brainiac_caft_20260808/brainiac_isolated_venv_requirements.txt` into `H:\New project\.venvs\brainiac`. Do not install those pins into the battery `.venv`.
- Pretraining overlap: prove subject-level disjointness or keep `UNKNOWN_PRETRAINING_OVERLAP`.
- Do not mutate 128³ cache. Private hash-indexed 128→96 adapter: single-step resize, nonzero channel-wise norm, no labels, no raw IDs/paths.
- Label-blind sensitivity n=64: direct-resize vs 1mm-resample→96.
- Mask-edge dilation/erosion/foreground perturbation (skull-strip shortcut).
- Load with `validate_vendor_equivalent_strict_load`. Naive MONAI 1.5.2 `strict=True` is a documented FAIL (84 unused `cross_attn` keys). Do not copy `BrainIAC.ckpt` into the vendored GitHub tree.
- If geometry/foreground/embedding stability fail: stop BrainIAC; CNN battery stays intact.

### R1b — frozen embeddings + probes

Extract frozen embeddings for all 6213 cache rows (1669 real + 4544 derivative). Validation/test remain real-only.

Baselines, inner-val C grid `{0.01, 0.1, 1, 10}` only:

- MRI-only L2 logistic on 768-d embedding
- BrainIAC + five allowed features (`age, sex, education_years, apoe4_carrier, mmse_base`)
- Locked ResNet34 + five allowed features

Report real-only training sensitivity separately.

### R1c — CAFT

Local encoder = prespecified locked `resnet34_3d` (not chosen from outer test).
Global = frozen BrainIAC 96³.
Both frozen in main CAFT. Fold-local LayerNorm + projection to 128-d.
Tabular projection independent.

```
final_logit = locked_resnet34_logit + gate(local, global, tabular) × foundation_residual
```

Gate regularized toward zero. Cohort/site/scanner/manufacturer/field strength never enter prediction or gate.

Outputs: calibrated probability; split-conformal sets `{Aβ−}`, `{Aβ+}`, `{Aβ−, Aβ+} = INDETERMINATE_RESEARCH_OUTPUT`.

Also: naïve concatenation ablation, linear CKA, residual predictive value.

Promotion requires: ΔROC-AUC ≥ 0.02; positive lower paired CI; Brier not worse by >0.01; ECE not worse by >0.02; confound probe not worse; conformal coverage at prespecified level; sanities pass.

All BrainIAC results: `POST_HOC_INTERNAL_TECHNICAL_EVIDENCE`.

### MX-04

Prespecified finetune for eligible CNNs. BrainIAC LoRA only if frozen CAFT promoted, separate branch:

- last two transformer blocks, q/v only
- rank=4, alpha=8, dropout=0.05, LR≈5e-5, LLRD=0.8
- no full fine-tune in v1
- LoRA active checkpoints may use C allowlist; pretrained weights, metrics, terminal ckpts stay on H

### MX-05

AMUSE-X-Hybrid3D. Do not merge with or replace BrainIAC except a future independent ablation.

### MX-06

Aggregate regional XAI for finalized models: token attribution / Integrated Gradients, occlusion, randomization sanity, region-additive sensitivity. No raw Grad-CAM on ViT. Allowed wording: «ناحیه‌های دارای attribution بالاتر در تصمیم مدل».

### R2 / R3 / MX-07

- R2 survival only after event/censoring and temporal origin are locked.
- R3 missingness-aware late fusion only after paired-modality and temporal-validity gates. FBP is circular for Aβ-status Method X. FTP/FDG/plasma are Method Y or future orthogonal branches.
- MX-07 freeze Method/Results/Discussion, representation registry, thesis packet.

## Locked scientific constraints

- PET never enters Method X (train or infer).
- Five official CNNs stay; BrainIAC is family 6 post-hoc.
- R0 scoping review is `COMPLETE`; novelty remains `NOVELTY_CANDIDATE`. Exact prior art → replication/extension.
- No Alzheimer diagnosis / external validation / causal region / diseased region claims.
