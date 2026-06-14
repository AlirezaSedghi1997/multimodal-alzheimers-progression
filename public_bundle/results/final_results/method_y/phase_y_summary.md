# Method Y Phase 2 Summary

## results/tables/method_y_gap_filter_report.json

- Status/verdict: `REPORT_READY_NOT_TRAINING_PERMISSION`

## results/tables/method_y_pet_manifest_summary.json

- Status/verdict: `method_y_pet_manifest`
- n_candidates: `5661`
- tracer: `fbp`

## results/tables/method_y_pet_preprocess_summary.json

- Status/verdict: `PASS_WITH_QC_OUTPUTS_NOT_YET_TRAINING_READY_UNTIL_READINESS_GATE`
- selected_work_items: `266`
- candidate_count: `5661`
- processed_items: `10`
- tracer: `fbp`

## results/tables/method_y_pet_cache_summary.json

- Status/verdict: `method_y_pet_cache`
- tracer: `fbp`

## results/tables/method_y_training_readiness.json

- Status/verdict: `READY_FOR_EXPLORATORY_METHOD_Y_TRAINING`
- tracer: `fbp`
- blockers: `[]`

## results/tables/method_y_incremental_value_table.json

- Status/verdict: `COMPLETED`
- rows: `[{'brier': 0.24387926381116148, 'caveat': 'Method Y paired PET-cache hash intersection only; PET-containing rows have PET-label tautology caveat and small-N/class-imbalance limitations.', 'comparison_id': 'Y2_mri_paired', 'delta_auc_valid': True, 'delta_auc_vs_pet_only': -0.062008978675645365, 'effective_n': 184, 'headline_allowed': False, 'label': 'MRI-only on PET-cache hash intersection', 'metrics_status': 'AVAILABLE', 'pr_auc': 0.9123869253016514, 'required_common_hash_intersection_n': 184, 'roc_auc': 0.5712682379349046, 'threshold_metrics': {'accuracy': 0.8811646514935989, 'f1': 0.9364043057157515, 'sensitivity': 1.0, 'specificity': 0.0}}, {'brier': 0.2416590530920701, 'caveat': 'Method Y paired PET-cache hash intersection only; PET-containing rows have PET-label tautology caveat and small-N/class-imbalance limitations.', 'comparison_id': 'Y3_late_fusion', 'delta_auc_valid': True, 'delta_auc_vs_pet_only': -0.10058922558922556, 'effective_n': 184, 'headline_allowed': False, 'label': 'Late fusion MRI+PET', 'metrics_status': 'AVAILABLE', 'pr_auc': 0.8984981220450114, 'required_common_hash_intersection_n': 184, 'roc_auc': 0.5326879910213244, 'threshold_metrics': {'accuracy': 0.8811646514935989, 'f1': 0.9364043057157515, 'sensitivity': 1.0, 'specificity': 0.0}}, {'brier': 0.24505878659035413, 'caveat': 'Method Y paired PET-cache hash intersection only; PET-containing rows have PET-label tautology caveat and small-N/class-imbalance limitations.', 'comparison_id': 'Y4_gated_fusion', 'delta_auc_valid': True, 'delta_auc_vs_pet_only': -0.16329966329966328, 'effective_n': 184, 'headline_allowed': False, 'label': 'Gated fusion MRI+PET', 'metrics_status': 'AVAILABLE', 'pr_auc': 0.8862022558782474, 'required_common_hash_intersection_n': 184, 'roc_auc': 0.46997755331088664, 'threshold_metrics': {'accuracy': 0.8864278093883357, 'f1': 0.9366941607882154, 'sensitivity': 0.9741935483870968, 'specificity': 0.14285714285714285}}, {'brier': 0.24351412083628135, 'caveat': 'Method Y paired PET-cache hash intersection only; PET-containing rows have PET-label tautology caveat and small-N/class-imbalance limitations.', 'comparison_id': 'Y5_mri_pet_cognitive', 'delta_auc_valid': True, 'delta_auc_vs_pet_only': 0.0, 'effective_n': 184, 'headline_allowed': False, 'label': 'MRI+PET+cognitive summaries', 'metrics_status': 'AVAILABLE', 'pr_auc': 0.927189121860636, 'required_common_hash_intersection_n': 184, 'roc_auc': 0.6332772166105499, 'threshold_metrics': {'accuracy': 0.8706383357041252, 'f1': 0.9290018530959075, 'sensitivity': 0.9806451612903226, 'specificity': 0.02857142857142857}}]`

## results/tables/synthetic_mri_pet_supplement.json

- Status/verdict: `DESIGN_READY_SUPPLEMENTARY_ONLY`

## results/tables/proposal_alignment_report.json

- Status/verdict: `COMPLETED`

## results/final_results/method_y/mri_pet_cognitive_training/method_y_mri_pet_cognitive_metrics.json

- Status/verdict: `COMPLETED`

