# amyloid-tau-prediction-a4learn-methodx

PET-free **Method X-Strict** pipeline for early prediction of amyloid biomarker status and preclinical risk on A4/LEARN-style data, aligned with execution program **v3.9** (thesis-grade validity controls).

**Repository:** [github.com/AlirezaSedghi1997/amyloid-tau-prediction-a4learn-methodx](http<LOCAL_PATH_REDACTED>)

## Scientific object (protected thesis anchor)

**Method X-Strict** predicts amyloid **biomarker status** (Aβ+ vs Aβ−) — early prediction of amyloid biomarker status and preclinical risk — not Alzheimer disease diagnosis.

| Allowed inputs | Forbidden inputs |
|----------------|------------------|
| T1 structural MRI (96³ cache) | PET image / SUVR / centiloid as predictor |
| `age`, `sex`, `education_years`, `apoe4_carrier`, `mmse_base` | Blood biomarkers in Method X |
| | Cognitive features beyond fixed tabular set |
| | Raw subject IDs / raw paths in public outputs |

- **Target/label only:** PET-derived centiloid (frozen cutoff **33.2705**), VA/visual-read sensitivity label.
- **Validation term:** locked **pseudo-external** evaluation — not true external validation.
- **Cohort caveat:** A4/LEARN enrolled sample has strong cohort–label coupling (Cramér's V ≈ 1 within enrolled-only analyses).

## Branch architecture

| Branch | Role | Status (local) |
|--------|------|----------------|
| **Method X-Strict** | Protected thesis anchor (T1 + fixed tabular [+ 3D CNN]) | Phase 2 MVP **complete** |
| **Method X-Cog/SF** | Confounding-oriented cognitive extension + SF internal decoupling | Stress tests **done** |
| **MRI ROI/WMH/reads** | Interpretable MRI branch (no leaky surrogates) | Phase 2 MVP **complete** |
| **Method Y** | Exploratory PET-assisted multimodal | **COMPLETED** (all incremental modalities trained) |
| **Bio-Cog / Method Z** | Minimally invasive blood benchmark | **FEASIBILITY_ONLY** (Phase 3) |
| **Clinical endpoint** | Exploratory progression proxy | **FEASIBILITY_ONLY** |

## Execution progress (v3.9)

| Phase | Gate / deliverable | Status |
|-------|-------------------|--------|
| **0** | Branch catalogs, SF governance, leakage review, label freeze | **PASS** |
| **1** | Interpretable baselines (tabular, cognitive, ROI, WMH, reads) | **DONE** |
| **0.6** | Cohort-probe + within-cohort centiloid regression | **DONE** |
| **Literature** | SynIB loss, MFSCA texture, worst-dimension audit (24-paper registry) | **DONE** |
| **2** | Minimum MVP battery (3 headline models) | **MVP_COMPLETE** |
| **3** | Method Z + clinical endpoint + Method Y training | **Partial / not started** |

Key validity findings (aggregate, local runs):

- **3D MRI latent cohort-probe:** linear AUC ≈ **0.50** → cohort not encoded in CNN embeddings.
- **ROI/WMH nonlinear cohort-probe:** AUC ≈ **0.99** → cohort artifact; report as limitation.
- **SF stress tests:** train-SF→enrolled cognitive transfer collapses (~0.51 AUC); SF is internal decoupling only.

## Data (local, read-only)

Default data root (not in git):

- Windows: `<LOCAL_PATH_REDACTED> Learn`
- WSL: `<LOCAL_PATH_REDACTED> Learn`

Large artifacts (`data/cached/*.h5`, `results/final_results/`, raw CSVs) are **gitignored**. Reproduce via staged pipeline below.

## Quick start

```bash
# Environment
conda activate amyloid-tau   # or see Makefile ENV_NAME

# Phase 0
make phase0-complete DATA_ROOT="<LOCAL_PATH_REDACTED> Learn"

# Phase 1 + validity probes (CPU)
make phase1-baselines cohort-probe centiloid-regression

# Phase 2 MVP audit (CPU)
make phase2-mvp-audit gap-matrix

# GPU stages (when SPM + 96³ cache ready)
make gpu-preflight three-d-ablation

# Method X-Cog/SF + Phase 3 feasibility (CPU)
make method-x-cog-sf-stress-tests bio-cog-method-z-readiness thesis-grade-report

# Aggregate reports
make final-analysis public-bundle
```

Orchestrator: `python scripts/run_project_stage.py --next`

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/final_execution_program.md](docs/final_execution_program.md) | Master execution program |
| [governance/sf_internal_decoupling_policy.md](governance/sf_internal_decoupling_policy.md) | SF stress-test policy |
| [docs/method_y_status_summary.md](docs/method_y_status_summary.md) | Method Y blockers |
| [docs/clinical_endpoint_readiness.md](docs/clinical_endpoint_readiness.md) | Clinical endpoint feasibility |
| [public_bundle/](public_bundle/) | Aggregate reproducibility bundle (no raw IDs/images) |

## Tests

```bash
python -m pytest tests/ -q
```

Current local gate: **203 passed**, 1 skipped (CPU contract tests use `--backbone tiny` for 16³ smoke).

## Literature-backed innovations (2026)

| Module | Role | Artifact |
|--------|------|----------|
| SynIB synergy loss | Multimodal MRI+tabular training regularizer | `src/models/losses/synib_synergy_loss.py` |
| MFSCA texture proxies | Lightweight MRI texture features | `results/tables/mfsca_texture_features.json` |
| Worst-dimension audit | Chebyshev fusion robustness check | `results/tables/worst_dimension_fusion_audit/` |
| Extended modality inventory | FLAIR/DWI/fMRI/T2 mapping (non–Method X inputs) | `results/tables/extended_modality_inventory.json` |

Registry: `docs/literature_backbone_evidence_2023_2026.yaml` (24 papers).

## Privacy & compliance

- No raw subject IDs, raw paths, or subject-level predictions in public bundle.
- PET is **never** a Method X model input.
- SF must **never** be described as external validation.
- Method Y is PET-assisted exploratory with PET-label tautology caveat; results are not for clinical claims.
- Synthetic MRI-to-PET is supplementary, fold-local, train-only, and never counted as real subjects.
- Augmentation is train-only exposure multiplication, not real-N expansion.

## Citation / contact

Master's thesis project — A4/LEARN early prediction of amyloid biomarker status and preclinical risk under cohort-label confounding.  
Maintainer: [AlirezaSedghi1997](http<LOCAL_PATH_REDACTED>)
