# Method X v3.6-validity-fixes Patch Pack

This patch pack implements the executable repair layer requested by the v3.6 final issues document.

It does **not** replace the Method Y PET patch. It complements it:

- PET patch = Method Y / PET-assisted exploratory, separate from Method X.
- v3.6 patch = Method X validity, temporal sensitivity, nested HPO, pseudo-external reporting, final thesis reporting.

## Files included

### Critical scripts

- `scripts/run_validity_audit.py`
- `scripts/run_temporal_window_sensitivity.py`
- `scripts/run_optuna_nested.py`
- `scripts/run_external_eval.py`
- `scripts/run_cognitive_family_ablation.py`
- `scripts/train_3d_mri.py`
- `scripts/build_thesis_grade_report.py`

### Model/evaluation support

- `src/evaluation/bootstrap_ci.py`
- `src/models/backbones/lightweight_3d.py`
- `src/models/backbones/backbone_factory.py`

### Policies and docs

- `configs/model/v3_6_validity_policy.yaml`
- `configs/model/cognitive_feature_family_ablation.yaml`
- `governance/v3_6_validity_fixes_policy.md`
- `docs/thesis_defense_talking_points.md`
- `patches/MAKEFILE_v3_6_targets.mk`

### Tests

- `tests/test_v3_6_policy_contract.py`
- `tests/test_validity_audit_outputs.py`
- `tests/test_temporal_window_variants.py`
- `tests/test_no_raw_ids_in_exports.py`
- `tests/test_full_pipeline_smoke_contracts.py`

## Apply

```bash
unzip method_x_v36_validity_fixes_patch_pack.zip
cd method_x_v36_validity_fixes_patch
bash apply_v3_6_patch.sh /path/to/amyloid-tau-prediction-a4learn
cd /path/to/amyloid-tau-prediction-a4learn
```

Append the contents of:

```text
patches/MAKEFILE_v3_6_targets.mk
```

to your existing `Makefile`.

## Run order

```bash
# 0) Snapshot first
git checkout method-x-cog
git tag v3.5-baseline-before-validity-fixes
git checkout -b v3.6-validity-fixes

mkdir -p results/baseline_snapshots
cp results/tables/final_metrics_table.md results/baseline_snapshots/final_metrics_table_v3_5.md || true
cp results/tables/cross_cohort_bias_report.md results/baseline_snapshots/cross_cohort_bias_report_v3_5.md || true
cp results/final_results/three_d/three_d_metrics.md results/baseline_snapshots/three_d_metrics_v3_5.md || true

# 1) Contract tests
python -m pytest tests/test_v3_6_policy_contract.py -q

# 2) Validity audit
python scripts/run_validity_audit.py

# 3) Temporal PET–T1 sensitivity
python scripts/run_temporal_window_sensitivity.py

# 4) Nested HPO
python scripts/run_optuna_nested.py --mode tabular_plus_mri_features

# 5) External-like final evaluation
python scripts/run_external_eval.py --final_locked_config

# 6) Method X-Cog family contribution
python scripts/run_cognitive_family_ablation.py

# 7) 3D final contract scaffold
python scripts/train_3d_mri.py --phase final --resolution 96 --folds 5 --input_mode mri_plus_tabular --fusion cross_attention

# 8) Thesis-grade report
python scripts/build_thesis_grade_report.py

# 9) Smoke tests
python -m pytest tests/test_validity_audit_outputs.py tests/test_temporal_window_variants.py tests/test_full_pipeline_smoke_contracts.py -q
```

## How to use alongside the PET patch

### Track A: Method X validity fixes

Run this patch on branch:

```bash
v3.6-validity-fixes
```

This branch repairs the core thesis claim: validity audit, temporal sensitivity, nested HPO, external-like reporting, 3D final contract and thesis report.

### Track B: Method Y PET patch

Run the previous PET patch on a separate branch:

```bash
method-y-pet-phase2
```

That patch adds:

- `scripts/preprocess_pet.py`
- `scripts/method_y_gap_filter.py`
- `docs/method_y_preregistration.md`
- `configs/model/method_y_gap_cohort_eval_policy.yaml`
- PET-specific gate policy and tests

### Safe combined workflow

1. Complete Method X v3.6 validity audit first.
2. In parallel, run PET dry-run / limited preprocessing in Method Y branch only.
3. Do not merge PET preprocessing outputs into Method X cache.
4. Merge only shared governance/docs/tests if they do not add PET as Method X input.
5. Report Method X and Method Y in separate result sections.

## Important warnings

- Do not overwrite `data/processed/master_catalog.csv` with gap-filtered catalogs.
- Do not use PET gap as a model feature.
- Do not run DeLong on pseudo-external holdout.
- Do not tune on external-like data.
- Do not call the current output Alzheimer diagnosis.
- Do not move PET images, SUVR or centiloid into Method X / Method X-Cog features.
