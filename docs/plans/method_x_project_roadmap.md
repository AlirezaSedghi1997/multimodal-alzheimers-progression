# Method X project roadmap — live 2026-08-21

Machine-readable twin: `results/method_x_registry/project_roadmap.json`.
Readiness authority:
`results/method_x_registry/READY_FOR_MAIN_BATTERY_TRAINING_20260821_FINAL.json`.

## Scientific boundary

Method X is PET-free in training and inference. Its predictive inputs are T1 MRI
plus exactly `age, sex, education_years, apoe4_carrier, mmse_base`. Cohort/site/
scanner fields are audit-only. The current output is an internal technical
amyloid-status benchmark, not Alzheimer diagnosis, causal evidence, external
validation, or a clinical tool.

External evaluation is optional and nonblocking because no external dataset is
currently approved. If an eligible dataset later becomes available, it may enter
only the locked no-refit branch. Otherwise the replacement package is labeled
`ROBUSTNESS_EVIDENCE_NOT_EXTERNAL_VALIDATION` and cannot widen claims.

## Live five-CNN battery

Root: `results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729`

| Model | Live status | Action |
|---|---|---|
| ResNet10 3D | `COMPLETED` | validate and skip; never retrain |
| ResNet18 3D | `COMPLETED` | validate and skip; never retrain |
| ResNet34 3D | `COMPLETED` | validate and skip; never retrain |
| DenseNet121 3D | `COMPLETED` | validate and skip; never retrain |
| MedNeXt-S | `COMPLETED` | validate and skip; terminal |
| ResNet50 3D | `PROTOCOL_EXCLUDED_NOT_STARTED` | remain excluded |
| EfficientNet3D-B0 | `PROTOCOL_EXCLUDED_NOT_STARTED` | remain excluded |

Battery is **TERMINAL** (2026-08-25). Manifest SHA `40a4a009b3a2453fb953e2b2e9b9ec789bdf896741dc2278a898a3b38fdb971d`, 5/5 effective `COMPLETED`, `.battery.lock` ABSENT. The scheduler validates all five rows and skips them. Replay distance is 0. Journal, mirror, reports, manifests and scientific artifacts remain authoritative on H.

## Automated gate status (2026-08-25)

- official preflight: `PASS` with manifest SHA `40a4a009b3a...` (terminal) and `verify_battery_terminal` `terminal:true`;
- battery: `TERMINAL` — `FIVE_CNN_BATTERY_NOT_TERMINAL` no longer sole blocker;
- BrainIAC `PREPARE_RAW_T1` gate: `allowed:true` (`build_live_readiness` PASS, `prepare_brainiac_raw_t1_v2.py --mode preflight` PASS);
- source contract: `9b0f90a5a14f8764397c6a15ff8016462f0872e2f827b2ec47fe54ab773871b3` (updated baseline);
- related regression suite: `791 passed` historical; BrainIAC focused `45 passed` (preprocessing v2) + `17 passed` (anchor);
- exact next action: `BRAINIAC_PREPROCESSING_P003_FULL` (1669 MRI, resumable, 45-75h);

## Exact next command — P003 BrainIAC preprocessing (battery is TERMINAL)

Battery command (now terminal, keep for reference):

```powershell
Set-Location -LiteralPath 'H:\New project'
.\.venv\Scripts\python.exe scripts\run_method_x_claim_benchmark.py --mode run --output-root 'H:\New project\results\final_results\method_x_model_battery_v3_depth_and_diversity_20260729'
# now returns TERMINAL — all 5 COMPLETED, no resume needed
```

**Next to run (resumable, 45-75h, Brainiac venv):**

```powershell
& 'H:\New project\.venvs\brainiac\Scripts\python.exe' 'H:\New project\scripts\research\prepare_brainiac_raw_t1_v2.py' --mode full --device cuda
```

Rerunning the *same* command after any interruption (Ctrl+C, close, power loss, exit 75) resumes from the last committed receipt — committed samples are skipped, uncommitted scratch is quarantined. No `--no_resume` / `--force` flags exist. Preflight: `... --mode preflight` must show `PASS` first.

## After the five-CNN battery becomes terminal

1. MX-03: sanity tests, calibration, confound audit and paired bootstrap.
2. P08b: finish at least 18 additional auditable BrainIAC full-text reviews.
3. R1a: official BrainIAC load/forward and affine-aware preprocessing/geometry.
4. R1b: frozen real-subject probes with split/lineage locks.
5. R1c: equal-budget CAFT comparison; LoRA only in a later version if inner/OOF
   evidence passes every finite-metric and confound gate.
6. MX-04: prespecified fine-tuning without outer-test selection.
7. MX-05: independent AMUSE-X-Hybrid3D ablations.
8. MX-06: atlas/affine-locked aggregate anatomical attribution.
9. MX-07: immutable publication/thesis freeze and optional eligible external
   no-refit evaluation, or the explicitly non-external robustness package.

Canonical detailed plan: `tasks/method_x_completion_20260821_plan.md`.
Live checklist: `tasks/method_x_completion_20260821_todo.md`.
