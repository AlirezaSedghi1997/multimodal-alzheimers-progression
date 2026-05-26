# Full MRI Preprocessing Toolchain Report

- status: `BATCH_COMPLETED`
- backend: `spm`
- preprocessing_version: `spm_v2_mni_brain_robust_intensity`
- quality_steps: `['spm_unified_segmentation_bias_correction', 'spm_tissue_probability_brain_mask', 'spm_forward_deformation_mni_normalization', 'brain_masked_robust_percentile_clipping_0_5_99_5', 'brain_masked_zscore_intensity_normalization', 'hashed_numeric_qc_brain_volume_tissue_ratios_cnr_intensity']`
- missing: `[]`
- summary: `{'ok': 10, 'failed': 0, 'missing': 0, 'skipped_done': 723, 'done_total': 733, 'deferred_pending': 967, 'attempted_this_run': 10, 'batches_completed': 1, 'total': 1700, 'limit': 0, 'batch_size': 10, 'is_smoke_test': False, 'is_batch_run': True, 'auto_batches': False}`
- contains_pet_input: `False`
- contains_raw_identifiers: `False`
- output_granularity: `hashed_sample_level`
