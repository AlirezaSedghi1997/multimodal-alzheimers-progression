# Method X/Y v3.8 Literature-Aligned Comprehensive Patch

This package upgrades the prior Method X/Y v3.6 + Method Y PET Phase 2 package using the literature-review outputs. It does **not** change the Method X non-invasive claim. It makes Method Y PET preprocessing, evaluation, and reporting stricter before any PET training is allowed.

## What this version adds

### Temporal rigor
- Adds `gold_standard_30` tier for high-validity sensitivity.
- Keeps `primary_strict <=90 days` as feasible Method Y training tier.
- Keeps `secondary_window <=120 days` as sensitivity.
- Excludes `>90 days` from primary Method Y training, but retains report-only summaries.

### PET quantification rigor
- Adds tracer-specific FBP/FBB SUVR thresholds and Centiloid equations.
- Keeps whole cerebellum as primary amyloid reference.
- Adds cerebellar gray as sensitivity, not primary replacement.
- Keeps inferior cerebellar gray for FTP/tau optional subset.
- Forbids SUVR/Centiloid scalar inputs in Method X and forbids them as primary Method Y features.

### Spatial rigor
- Requires PET->T1->MNI two-step spatial preprocessing for training-ready PET outputs.
- Keeps PETPrep/petBrain/ADprep-style precomputed outputs as acceptable if QC-pass.
- Adds private visual QC overlay hooks and numeric QC fields.

### Modeling rigor
- Requires PET-only and MRI-only baselines before fusion claims.
- Downgrades naive early channel concatenation to ablation-only.
- Recommends late/gated feature fusion as practical primary fusion after baselines.
- Keeps cross-attention as advanced optional variant if sample size and VRAM allow.

### Evaluation rigor
- Adds cohort-label, tracer-label, and gap-label audits.
- Requires patient-level split and one-PET-per-patient selection.
- Requires calibration plan: Brier, ECE, reliability diagram.
- Requires bootstrap CI as default uncertainty strategy.
- Keeps DeLong restricted.
- Requires pseudo-external wording and PET-label tautology caveat.

## First tests

```bash
python -m pytest \
  tests/test_v3_8_literature_policy_contract.py \
  tests/test_v3_8_pet_manifest_and_gap.py \
  tests/test_v3_8_pet_qc_and_readiness.py \
  -q
```

## Execution order

### Method X validity path

```bash
python scripts/run_validity_audit.py
python scripts/run_temporal_window_sensitivity.py
python scripts/run_external_eval.py --final_locked_config
python scripts/build_thesis_grade_report.py
```

### Method Y PET literature-aligned path

```bash
python scripts/method_y_gap_filter.py --catalog data/processed/master_catalog.csv
python scripts/build_pet_private_manifest.py --data_root "$DATA_ROOT" --tracer fbp --catalog data/processed/master_catalog.csv
python scripts/preprocess_pet_spatial.py --data_root "$DATA_ROOT" --tracer fbp --dry_run --limit 10
```

## Scientific boundaries

- Method X never receives PET image, SUVR, Centiloid, FBP/FBB/FTP, or PET-derived features.
- Method Y PET-containing models are PET-assisted exploratory analyses only.
- PET-only performance is interpreted as biomarker reproduction / reader-surrogate context, not clinical diagnosis.
- A-beta-positive vs A-beta-negative is biomarker status, not Alzheimer disease diagnosis.
