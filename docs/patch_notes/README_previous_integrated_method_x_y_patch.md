# Integrated Patch Pack — Method X v3.6 Validity Fixes + Method Y PET Phase-2

This package merges the two previous patch packs into one unified change set:

1. **Method X v3.6 validity fixes**
2. **Method Y PET Phase-2 preparation**

It contains all executable files, configs, governance docs, tests, Makefile targets, and patch notes from both packages, plus an integrated policy and integrated contract test.

## What this integrated patch does

### Method X / v3.6

Adds:

- `scripts/run_validity_audit.py`
- `scripts/run_temporal_window_sensitivity.py`
- `scripts/run_optuna_nested.py`
- `scripts/run_external_eval.py`
- `scripts/run_cognitive_family_ablation.py`
- `scripts/train_3d_mri.py`
- `scripts/build_thesis_grade_report.py`
- `configs/model/v3_6_validity_policy.yaml`
- `configs/model/cognitive_feature_family_ablation.yaml`
- `governance/v3_6_validity_fixes_policy.md`
- `docs/thesis_defense_talking_points.md`
- `src/evaluation/bootstrap_ci.py`
- `src/models/backbones/lightweight_3d.py`
- `src/models/backbones/backbone_factory.py`
- v3.6 tests and Makefile targets

### Method Y / PET

Adds:

- `scripts/preprocess_pet.py`
- `scripts/method_y_gap_filter.py`
- `docs/method_y_preregistration.md`
- `configs/model/method_y_gap_cohort_eval_policy.yaml`
- `governance/method_y_pet_phase2_gate_policy.md`
- `tests/test_pet_preprocess_contract.py`

### New integrated layer

Adds:

- `configs/model/integrated_method_x_y_patch_policy.yaml`
- `governance/integrated_method_x_y_patch_policy.md`
- `tests/test_integrated_patch_contract.py`
- `patches/MAKEFILE_INTEGRATED_METHOD_XY_TARGETS.mk`
- `apply_integrated_patch.sh`

## Scientific boundary

This package is unified for convenience, but the scientific lanes remain separate:

| Lane | Name | PET role | Reporting |
|---|---|---|---|
| A | Method X | label/target only, never model input | main thesis path |
| B | Method Y PET/MM | PET image input allowed | exploratory PET-assisted branch |

Do **not** report PET-containing results as Method X.

## Apply to the project

```bash
unzip method_x_y_integrated_v36_pet_patch_pack.zip
cd method_x_y_integrated_v36_pet_patch
bash apply_integrated_patch.sh /path/to/amyloid-tau-prediction-a4learn
cd /path/to/amyloid-tau-prediction-a4learn
```

Optionally append the integrated Makefile targets:

```bash
cat patches/MAKEFILE_INTEGRATED_METHOD_XY_TARGETS.mk >> Makefile
```

## Recommended branch workflow

Safer two-branch workflow:

```bash
git checkout method-x-cog
git checkout -b v3.6-integrated-method-x-validity

# apply this package
# run Method X audit stages
```

For PET-heavy work, either use a separate branch or keep PET stages clearly labeled Method Y:

```bash
git checkout method-x-cog
git checkout -b method-y-pet-phase2

# apply this package, but run only Method Y PET stages
```

If applying to one combined branch, keep reporting and outputs separate.

## First tests after applying

```bash
python -m pytest   tests/test_integrated_patch_contract.py   tests/test_v3_6_policy_contract.py   tests/test_pet_preprocess_contract.py   -q
```

## Recommended execution order

### Stage 0 — snapshot

```bash
git tag v3.5-baseline-before-integrated-v36-pet
mkdir -p results/baseline_snapshots
cp results/tables/final_metrics_table.md results/baseline_snapshots/final_metrics_table_v3_5.md || true
cp results/tables/cross_cohort_bias_report.md results/baseline_snapshots/cross_cohort_bias_report_v3_5.md || true
cp results/final_results/three_d/three_d_metrics.md results/baseline_snapshots/three_d_metrics_v3_5.md || true
```

### Stage 1 — Method X validity

```bash
python scripts/run_validity_audit.py
python scripts/run_temporal_window_sensitivity.py
python scripts/run_external_eval.py --final_locked_config
python scripts/run_cognitive_family_ablation.py
python scripts/build_thesis_grade_report.py
```

### Stage 2 — Method Y PET pre-training preparation

```bash
python scripts/method_y_gap_filter.py --catalog data/processed/master_catalog.csv
python scripts/preprocess_pet.py --data_root "$DATA_ROOT" --tracer fbp --dry_run
python scripts/preprocess_pet.py --data_root "$DATA_ROOT" --tracer fbp --execute --limit 10
```

### Stage 3 — optional model refresh

Run only after Method X validity audit is reviewed:

```bash
python scripts/run_optuna_nested.py --mode tabular_plus_mri_features
python scripts/train_3d_mri.py --phase final --resolution 96 --folds 5 --input_mode mri_plus_tabular --fusion cross_attention
```

## Warnings

- Do not use `git add .`.
- Do not overwrite `master_catalog.csv` with gap-filtered catalogs.
- Do not use PET–T1 gap as a Method X feature.
- Do not run DeLong on pseudo-external holdout.
- Do not tune on external-like data.
- Do not call the output Alzheimer diagnosis.
- Do not move PET images/SUVR/centiloid into Method X or Method X-Cog features.
- Do not place PET preprocessing outputs in Method X cache.

## Patch notes preserved

Original patch READMEs are preserved under:

- `docs/patch_notes/README_method_x_v36_validity_fixes_patch.md`
- `docs/patch_notes/README_method_y_pet_patch.md`
