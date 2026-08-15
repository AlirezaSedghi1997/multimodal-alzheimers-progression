# Method X remaining work — 2026-08-15 (pre-training gate closed)

Authoritative programme: H-only storage, resume, BrainIAC-CAFT corrective plan.
This file is the execution inventory. It does **not** start GPU training.

Training on this git branch (`method-y-post-revival-sanitized-20260812`) is
**Method X CNN battery resume (MX-02)**, not Method Y PET training. Method Y
remains `LOCAL_PIPELINE_NOT_CLAIM_READY` / `science_gate_status=BLOCKED`. Do
not start Method Y image training from these cards.

A later 2026-08-15 agent session repaired Method Y pretraining contracts on this
same leaky checkout and on `methody/pretraining-sanitized-v2`. That session did
**not** change Method X battery science, YAML, or checkpoints. Record:
`docs/plans/agent_session_ledger_20260815.md` and
`results/method_x_registry/SESSION_NOTE_METHOD_Y_DEFECT_REPAIR_20260815.json`.
Resume step of record is still C staging **1534** (`docs/plans/method_x_project_roadmap.md`),
not the stale H progress 1321.

## Live battery (do not reset)

Root: `H:\New project\results\final_results\method_x_model_battery_v3_depth_and_diversity_20260729`

| Model | Status | Resume point |
|---|---|---|
| ResNet10 | `COMPLETED` | validation repaired; do not retrain |
| ResNet18 | `PAUSED_RESUMABLE_2_OF_5` | fold 2; C staging optimizer step 1534 (H progress stale at 1321) |
| ResNet34 | `PAUSED_RESUMABLE_3_OF_5` | fold 3, optimizer step 668 |
| DenseNet121 | `SMOKE_PASS` / `NOT_STARTED` | after ResNet34 |
| MedNeXt-S | `SMOKE_PASS` / `NOT_STARTED` | after DenseNet121 |
| ResNet50 / EfficientNet-B0 | `PROTOCOL_EXCLUDED_NOT_STARTED` | keep smoke reports |

Effective order: ResNet10 → ResNet18 → ResNet34 → DenseNet121 → MedNeXt-S.

Branch dirty. No reset, no unrelated discard, no `--no_resume`. No two GPU models at once. `BLOCKED_*` must not skip to the next model.

First GPU resume still auto-applies **source-only contract migration** (trainer digest drift). Do not rewrite finalized `.pt` / outer-test bytes.

Accepted runtime overlay (A5 fallback, do not reopen): full exact checkpoint every optimizer step on C NVMe; H mirror at epoch; journal/replay window `0`. Cadence-32 journal candidate was not promoted.

## PRE-training — closed this session (do not redo)

| ID | Item | Evidence |
|---|---|---|
| H-00 | Storage policy + locality audit | `configs/policy/storage_policy.json`, `storage_locality_baseline.json`, latest post-gate audit |
| A | State reconciliation | live `reports/state_reconciliation.json`; ResNet10 repaired without retrain |
| B | Five-model amendment | battery `amendments/method_x_battery_v3_five_model_amendment.v2.json`; locked v3 YAML untouched |
| B-future | Future five-model YAML | `configs/experiments/method_x_claim_benchmark_v3_five_model_future.yaml` — **not** runner default |
| C | `resume-backbone`, no skip on `BLOCKED_*` | runner |
| §3–4 | Staging / A5 acceptance | `A5_ACCEPTANCE_REPORT.json` PASS; runtime amendment v3 |
| A6 | Ten-layer autosupervision | `READY_FOR_MAIN_BATTERY_TRAINING.json` |
| R0 | Scoping review | 29 sources, 13 peer-reviewed, 25 `FULL_TEXT_REVIEWED`, status `COMPLETE`, novelty `NOVELTY_CANDIDATE` |
| Vendor | Pinned commit on H | `vendor/BrainIAC/ba60f45bed832ee0e5683c678caeaeefe2072f0d` |
| Weights | Official encoder on H | `models/pretrained/brainiac/BrainIAC.ckpt` 362119240 bytes, SHA-256 `f22bdbcae26823a9d9e8aee883c6f24386ba4617339c12269848b6666cc62693` |
| CPU load | Vendor-equivalent strict | 137/137 backbone keys; naive MONAI 1.5.2 `strict=True` remains FAIL |
| Isolated venv | Created, packages not installed | `H:\New project\.venvs\brainiac`; requirements frozen for R1a |
| Adapter helper | MONAI 1.5.2 unused `cross_attn` | `validate_vendor_equivalent_strict_load` in `src/research/brainiac_adapter.py` |

Unread paywalls (do not stamp): s04 AnatCL, s16 JAMA plasma, s19 AJNR A8899, s23 JMRI. citeprobe corroboration stays `INCONCLUSIVE`. Overlap stays `UNKNOWN_PRETRAINING_OVERLAP`. Geometry FOV interpretation stays `UNRESOLVED_PENDING_VENDOR`.

A6 generator still hardcodes R0 `IN_PROGRESS` for historical receipts. Do **not** re-run `scripts/build_method_x_a6_readiness.py` write_bundle; it would revert the live roadmap. Live twins were amended 2026-08-15.

## Next session — Method X training (MX-02). Not PRE. Not Method Y.

### Card MX-02a — ResNet18 only

- Preconditions: no live Method X process; `.battery.lock` absent; C staging free ≥30 GB and quota ≤20 GB; storage audit PASS; never `--no_resume`.
- Command:

```powershell
Set-Location -LiteralPath 'H:\New project'
.\.venv\Scripts\python.exe scripts\run_method_x_claim_benchmark.py `
  --mode resume-backbone `
  --backbone resnet18_3d `
  --output-root 'H:\New project\results\final_results\method_x_model_battery_v3_depth_and_diversity_20260729'
```

- Expected: skip folds 0–1; resume fold 2 at optimizer step 1321; then folds 3–4; validate metrics; exit without starting ResNet34.
- Stop: `BLOCKED_*`, contract mismatch, VRAM >7.5 GB, scientific drift.
- Resume: rerun the same command. Return 75 = clean pause.

### Card MX-02b — remainder of five-model battery

Only after ResNet18 is 5/5 + metrics validation.

```powershell
Set-Location -LiteralPath 'H:\New project'
.\.venv\Scripts\python.exe scripts\run_method_x_claim_benchmark.py `
  --mode run `
  --output-root 'H:\New project\results\final_results\method_x_model_battery_v3_depth_and_diversity_20260729'
```

Expected: skip ResNet10/18; resume ResNet34 fold 3 step 668; exclude ResNet50/EfficientNet; train DenseNet121 then MedNeXt-S.

## POST-training — stored, do not run until five effective models are terminal

Authoritative machine file:
`results/method_x_registry/post_training_execution_memory.json`

Order (corrective programme §8; supersedes July 29 `post_training_roadmap.json` seven-row wording):

1. MX-03 validity
2. R1a BrainIAC geometry/overlap/mask-edge (may stop BrainIAC only)
3. R1b frozen embeddings + linear probes
4. R1c CAFT residual gated fusion
5. MX-04 CNN finetune; LoRA only if CAFT promoted
6. MX-05 Hybrid3D (not merged with BrainIAC)
7. MX-06 aggregate XAI
8. R2 survival after endpoint lock
9. R3 late fusion after paired/temporal gates
10. MX-07 freeze Method/Results/Discussion

Method Y PET is a separate blocked track. It is not a post-MX-02 automatic next step.

## Explicit non-goals now

- GPU CNN training / 6213-row embedding extraction
- Method Y PET training
- Editing locked v3 YAML
- Resetting git / cleaning unrelated dirty files
- Writing project artifacts on C
- Claiming novelty, Alzheimer diagnosis, external validation, or causal regions
- Re-running A6 write_bundle
- Downloading the 6.91 GB Dropbox folder zip
- Copying `BrainIAC.ckpt` into the vendored GitHub tree
