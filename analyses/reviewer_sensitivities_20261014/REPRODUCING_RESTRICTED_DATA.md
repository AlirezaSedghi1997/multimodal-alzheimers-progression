# Reproduction using authorized restricted inputs

## Frozen inputs, not substituted cohorts

The revision scripts accept `--repo` (the caller's authorized local research root) and `--output` (a new destination). The relative input names below identify the exact frozen research artifacts. Their private execution manifest contains SHA256 digests locally, but is excluded from distribution because it also records restricted paths. Replacing a source release, endpoint, split or image cache changes the experiment; report this as a new run, not reproduction.

Required for the selected analyses:

```text
results/xyz_fusion_v2/private/contract.json
results/xyz_fusion_v2/private/jobs/mri_ensemble_cells.json
results/xyz_fusion_v2/private/jobs/pet_ensemble_cells.json
results/xyz_fusion_v2/private/jobs/tau_cox_ridge_o{0..4}_evaluation.json
results/xyz_fusion_v2/private/jobs/tau_rsf_o{0..4}_evaluation.json
results/xyz_fusion_postall_v2/private/units/tau_fold{0..4}_evaluation.json
results/xyz_fusion_postall_v2/private/pacc_longitudinal.parquet
thesis-workspace/supervisor_revision_20261011/private_analysis_frame.parquet
thesis-workspace/supervisor_revision_20261011/private_tau_frame.parquet
thesis-workspace/supervisor_revision_20261011/private_comparison_cells.json
thesis-workspace/supervisor_revision_20261011/performance_supplement.json
```

The scripts are the authoritative input contract for additional saved split/endpoint files: every JSON/Parquet read is recorded in a **private** execution manifest. Individual-cell JSON includes participant identity keys, event/censoring times, fold training/test membership, curves, and training censoring samples. Do not replace it with aggregate means. Missing inputs fail explicitly.

Example commands, replacing `AUTHORIZED_ROOT`, `RAW_CLINICAL` and `NEW_OUTPUT` with authorized local directories:

```text
python revision_analyses/selected_analyses.py --repo AUTHORIZED_ROOT --output NEW_OUTPUT/statistics --task all
python revision_analyses/tabular_sensitivities.py --repo AUTHORIZED_ROOT --output NEW_OUTPUT/statistics
python revision_analyses/cohort_and_censoring.py --repo AUTHORIZED_ROOT --raw-clinical RAW_CLINICAL --output NEW_OUTPUT/statistics --task all
```

`RAW_CLINICAL` contains `Derived Data/SUBJINFO.csv`, `Derived Data/PACC.csv`, `Raw Data/cdr.csv`, `External Data/imaging_SUVR_amyloid.csv`. The upstream endpoint reconstruction also uses `results/method_y_revival_v2/endpoints/amyloid_endpoint_summary.json` and the matching `amyloid_progression_endpoint.parquet`. The cohort check requires exact equality with the original 1,106-person upstream membership, not approximate counts. For censoring alone, `--task censoring` does not open raw clinical tables.

## Prepared frame and split contract

The frame is a one-row-per-participant Parquet table. Required fields are `patient_hash`, binary `event`, strictly positive `time_days`, and features below. `patient_hash` is a restricted linkage key, never a model input. Splits are five saved dictionaries with `train` and `test` lists; each participant is held out exactly once. Identical membership must be used for paired comparisons.

| Track | Prepared predictors |
|---|---|
| Clinical/amyloid | age, sex_binary, education_years, apoe4_carrier, baseline_pacc_raw, baseline_mmscore, treatment_active, amyloid_centiloid |
| Scalar/MRI regional and plasma branches | hippocampus_icv, ventricle_icv, entorhinal_icv, blood_ptau217, blood_log_ptau217; plasma missingness determines the eligible subset |
| Tau | clinical/amyloid fields plus tau_temporal: arithmetic mean of the exact left/right entorhinal SUVR rows, crus reference |
| MRI | scalar fields plus cache_row aligned to a float image cache of shape N×1×128×128×128 |
| PET | MRI inputs plus image_path_private, h5_group and source_image_sha256 in the authorized prepared PET frame |

Training-only median imputation and scaling are implemented in `source_snapshot/scripts/xyz_fusion_v2/models.py`; do not prefit them on the entire study. `run_models.py` rejects invalid split coverage and outcomes, and validates without fitting by default:

```text
python run_models.py --track tau --frame TAU_FRAME.parquet --splits SAVED_SPLITS.json --output NEW_OUTPUT/tau --mode validate
python run_models.py --track tau --frame TAU_FRAME.parquet --splits SAVED_SPLITS.json --output NEW_OUTPUT/tau --mode fit --families cox_ridge
python run_models.py --track scalar --frame MRI_FRAME.parquet --splits SAVED_SPLITS.json --output NEW_OUTPUT/scalar --mode fit
python run_models.py --track mri --frame MRI_FRAME.parquet --splits SAVED_SPLITS.json --mri-cache MRI_CACHE.npy --output NEW_OUTPUT/mri --mode fit
python run_models.py --track pet --frame PET_FRAME.parquet --splits SAVED_SPLITS.json --mri-cache MRI_CACHE.npy --output NEW_OUTPUT/pet --mode fit
```

Omitting `--families` uses both original tabular families. Image fitting uses the configuration's three seeds and three paired components in each of five outer folds. It is computationally substantial. The expert and its corresponding fusion head remain paired; only final survival curves are averaged. Historical development used the same study, so these are internal exploratory fits even if the membership guards pass. Outputs under `private/` must remain local. Identical hashes are required to resume an existing output.

## Raw-data reconstruction and source order

The original route starts with endpoint linkage (`src/method_y_revival/endpoint_builder.py`), MRI/source preprocessing (`src/data/a4_schema.py` and imported preparation modules), accepted image-QC/cache construction, then the versioned scripts under `scripts/xyz_mri_baseline_fast_v1`, `xyz_fusion_v2` and `xyz_fusion_postall_v2`. The selected local import closure is included. Historical orchestration reads earlier parent receipts; it is not a new one-command public download/training workflow. `A4_ROOT` supplies the local data root and `A4_CLINICAL_ROOT` the clinical directory when using the historical audit helper. Do not run historical smoke/orchestration commands from an unrelated working directory.

N4 correction, ANTs nonlinear registration and the MNI152NLin2009cAsym template/mask must match the saved preprocessing contract. The network used a template mask and no independent native skull-stripping stage; In that historical parent analysis, Jacobians served QC only. The later compact-CNN/Jacobian pilot is a separate development experiment described in the root pilot guide; it did not advance to full-cohort execution. Source regional measurements came from study-supplied NeuroQuant, not the registered network grid. Missing licensed data, template assets, cached scan lineage or review decisions cannot be reconstructed from aggregate results. Trained weights and participant-level predictions are excluded from this archive.

## Statistical interpretation and executed checks

The revision scripts resample people within fixed outer folds. A 40,000-draw tau run yields 40,000 valid IBS draws and 39,361 valid concordance draws; undefined draws have no weighted comparable pairs in at least one fold. The 2,000-draw prefix is exactly reconciled. Pooled calibration is descriptive and does not replace fold estimates or recalibrate test predictions. Covariate-dependent censoring sensitivity adjusts measured predictors only. Three new tabular contrasts constitute a separate post-hoc family; no new prespecification or clinical threshold is claimed.

The local release checks exercise synthetic numerics, aggregate display replay, all new statistical analyses against authorized inputs, and an original tau Cox fitting route from prepared inputs. The numerical receipts distinguish these from full CNN retraining, raw-data portability validation, prospective validation and independent-cohort evaluation, none of which was performed for this revision.


### Additional selection and horizon analyses

`revision_analyses/reviewer2_analyses.py` reads the same prepared data and fixed curves as the main statistical scripts. Descriptive selection reconstruction also needs `data/preprocessed_pet/_private/manifests/fbp_private_manifest.csv`, `results/method_y_revival_v1/completion_v1/image_branch/fbp_preprocessing_membership.parquet`, and the 45 `mri_s*_o*_c*_lock.json` receipts. These input files remain private. PET list absence must not be relabeled as lack of a parent-study scan. The three-year Brier calculation is checked against the installed survival-library Brier implementation and original integrated loss. Calibration bands are conditional and pointwise, not simultaneous.


## Later analysis inputs

Absolute-concordance precision additionally requires `results/xyz_oof_stacking_v1/private/units/*_cells.json`. Training-history export reads original model receipts and exports only model-level epoch summaries. Use the bundled OOF guide to reconstruct missing OOF artifacts in an isolated authorized tree. The compact CNN/Jacobian pilot is not an input to these completed statistics.
