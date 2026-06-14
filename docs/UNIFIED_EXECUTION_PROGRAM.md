# Unified Execution Program

## Project: Amyloid-Tau Prediction (A4/LEARN)

## Core Principles

1. **Method X**: Non-invasive, T1 MRI + tabular only (PET forbidden)
2. **Method Y**: PET-assisted exploratory (tautology caveat required)
3. **RTX 4060 8GB**: VRAM hard stop at 7.5GB
4. **Patient-grouped CV**: No patient leakage between folds
5. **Privacy**: No raw identifiers, no PET in Method X cache

## Branch Architecture

| Branch | Inputs | Status |
|--------|--------|--------|
| Method X (main) | T1 MRI + tabular | ✅ Active |
| Method X-Cog | T1 MRI + tabular + cognitive | ⏳ Pending fusion |
| Method Y (PET) | PET + MRI + cognitive | ✅ Trained (AUC=0.819) |
| Method Z (blood) | Blood biomarkers | ✅ Trained (AUC=0.999) |

## Prompts Status

| # | Description | Status |
|---|-------------|--------|
| 0 | Phase 0 governance | ✅ DONE |
| 1 | Baseline training | ✅ DONE (AUC=0.731) |
| 2 | Feature engineering | ✅ DONE |
| 3 | Incremental value | ✅ DONE |
| 4 | 3D MRI training | ✅ DONE (AUC=0.698) |
| 5 | 3D ablation | ✅ DONE |
| 6 | XAI | ✅ DONE |
| 7 | Calibration | ✅ DONE (ECE=0.044) |
| 8 | Method X-Cog | ⏳ NOT EXECUTED |
| 9 | MRI+cognitive fusion | ⏳ NOT EXECUTED |
| 10 | Method Y PET | ✅ DONE (AUC=0.819) |
| 11 | External validation | ⏳ DEFERRED |
| 12 | Phase 4 manuscript | ⏳ NOT STARTED |
| 13 | Public bundle | ⏳ PARTIAL |

## Training Results

| Model | AUC | Status |
|-------|-----|--------|
| Tabular baseline | 0.731 | ✅ |
| 3D tiny backbone | 0.698 | ✅ |
| Blood biomarkers | 0.999 | ✅ (pTau217 labels) |
| PET-only (Method Y) | 0.819 | ✅ |
| Final analysis | 0.751 | ✅ (bias risk HIGH) |

## Quality Gates

| Gate | Check | Status |
|------|-------|--------|
| G0 | Phase 0 governance | ✅ PASS |
| G1 | Baseline AUC >= 0.65 | ✅ PASS |
| G2 | Optuna AUC >= 0.70 | ✅ PASS |
| G3 | Fusion improvement | ✅ PASS |
| G4 | Calibration ECE < 0.05 | ✅ PASS |
| G5 | TRIPOD 7/9 | ✅ PASS |
| G6 | PROBAST 6/6 | ✅ PASS |

## Mandatory Cycle

```
OBSERVE → DECIDE → PATCH → VERIFY → RECORD
```

## Token Efficiency

1. File Scope Lock
2. Minimal diffs
3. One task per cycle
4. Read only needed lines
