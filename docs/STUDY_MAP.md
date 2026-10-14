# Active study map

Selection follows endpoints, contracts and amendments, not modification date or the highest score.

| Route | Scientific role | Entry points |
|---|---|---|
| mri_preprocessing | CURRENT_PREPROCESSING | `scripts/preprocess_mri_spm.py`<br>`scripts/preprocess_mri_full.py` |
| method_x | TECHNICAL_INTERNAL | `scripts/mx_fasttrack_v2_4_4/entry_v244.py`<br>`scripts/mx_fasttrack_v2_2/thesis_report.py` |
| method_x_sensitivity | POST_HOC_SENSITIVITY | `scripts/mx_science_extension_v1/supervise_extension.py`<br>`scripts/mx_quick_validation_v1/supervise.py` |
| pet_transforms | CURRENT_PREPROCESSING_NOT_QUANTITATIVE_EQUIVALENCE | `scripts/run_method_y_transforms_manual.ps1` |
| method_y_regional | SEPARATE_INTERNAL_ANALYSIS | `scripts/run_method_y_training_v2.ps1` |
| method_y_image | FACTORIAL_DESCRIPTIVE_WITH_CORRECTED_INNER_SELECTION | `scripts/run_method_y_image_factorial_v4.ps1`<br>`scripts/run_method_y_posttraining_correction_v4_2.py`<br>`scripts/run_method_y_posttraining_selection_flow_v4_2.py`<br>`scripts/run_method_y_posttraining_joint_pairs_v4_2.py`<br>`scripts/run_method_y_posttraining_bootstrap_v4_2.py`<br>`scripts/run_method_y_posttraining_pairwise_v4_2.py`<br>`scripts/run_method_y_posttraining_scanner_v4_2.py`<br>`scripts/run_method_y_posttraining_science_status_v4_2.py`<br>`scripts/run_method_y_local_validation_v5.ps1`<br>`scripts/run_method_y_amyloid_lmo_a4_v2.ps1`<br>`scripts/analyze_method_y_amyloid_lmo_a4_v2.py`<br>`scripts/analyze_method_y_amyloid_lmo_calibration_v1.py`<br>`scripts/review_method_y_after_v5.py`<br>`scripts/audit_method_y_exact_sidecar_sites_v1.py`<br>`scripts/analyze_method_y_remaining_subgroups_v1.py`<br>`scripts/build_method_y_reporting_crosswalk_v1.py` |
| fusion_preprocessing | REUSED_CACHE_NOT_REUSED_V1_PREDICTIVE_RESULT | `scripts/run_xyz_mri_baseline_fast_v1.ps1` |
| fusion | CURRENT_INTERNAL_PROGNOSTIC_ANALYSIS | `scripts/run_xyz_fusion_v2.ps1` |
| postfusion | CORRECTED_COMPUTED_NOT_FINAL_SCIENTIFIC_PROMOTION | `scripts/run_xyz_postfusion_final_v1.ps1` |
| fp32_sensitivity | REVIEWED_INTERNAL_PRECISION_SENSITIVITY | `scripts/run_xyz_fp32_spatial_v1.ps1` |
| xy_spatial | DIAGNOSTIC_NOT_DISEASE_LOCALIZATION | `scripts/run_xy_spatial_validation_v1.ps1` |

## Shared dependencies

Older helpers required by current imports are retained and marked `REQUIRED_COMPATIBILITY_DEPENDENCY` in source_inventory.json. Their inclusion does not restore obsolete results.
Method X weights/predictions are not automatically reused as prognostic fusion scores. The prognostic MRI experts were cross-fitted for the future endpoint.
The old blood classifier is not the current clinical/cognitive/blood contribution to fusion. Invalid target-derived baselines are excluded.
FBP belongs to amyloid and FTP to tau. Pattern-normalized images do not establish official Centiloid/CenTauR equivalence.

## Result selection

The regional paired tau result and the new fusion tau comparison answer different questions in different matched populations. They must not be pooled or substituted for one another.
Post-fusion computation is complete in the recorded parent dossier. The completed FP32 companion has been reviewed for sensitivity and numerical diagnostic reporting; it does not establish positive scientific claims or fully validated attribution. See [FP32 review](FP32_SENSITIVITY.md), active_routes.json and exclusions.json.


## Research update, 14 October 2026

See [completed statistical and OOF work](RESEARCH_UPDATE_20261014.md) and the [CNN/Jacobian pilot decision](CNN_JACOBIAN_PILOT.md). Original results remain historical evidence; the new guide identifies superseded intervals and limits.
