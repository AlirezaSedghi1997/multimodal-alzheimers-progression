# Phase 1 Incremental Value Table

- status: `COMPLETED`
- ROI nonlinear cohort-probe AUC: `0.9920219533026382`
- best **descriptive** pooled AUC: `fixed_tabular_plus_roi_wmh_reads`
- validity-safe headline step: `fixed_tabular_plus_cognitive`

| step | roc_auc | pr_auc | validity_flag |
|---|---:|---:|---|
| fixed_tabular | 0.728884377271474 | 0.8496227848172967 | HEADLINE_DESCRIPTIVE_WITH_CAVEAT |
| cognitive_only | 0.7150537634408602 | 0.8342719816788144 | HEADLINE_DESCRIPTIVE_WITH_CAVEAT |
| fixed_tabular_plus_cognitive | 0.7931623931623931 | 0.8842543025389877 | HEADLINE_DESCRIPTIVE_WITH_CAVEAT |
| roi_icv_norm_only | 0.7534289969205916 | 0.9141601479089289 | DESCRIPTIVE_ONLY_COHORT_ARTIFACT |
| wmh_only | 0.49818264425261244 | 0.805985350503306 | SINGLE_OFFICIAL_WMH_FEATURE |
| mri_reads_only | 0.5027727295673684 | 0.785239159258601 | WEAK_STANDALONE_SIGNAL |
| roi_wmh_reads | 0.7515939724367712 | 0.9134611481357562 | DESCRIPTIVE_ONLY_COHORT_ARTIFACT |
| fixed_tabular_plus_roi_wmh_reads | 0.8207254278358322 | 0.9239612486800821 | DESCRIPTIVE_ONLY_COHORT_ARTIFACT |
| method_x_strict_3d_cnn | 0.7221570544151189 | 0.8421626306129226 | VALIDITY_SAFE_LOW_COHORT_PROBE |

## Caveat

Do not present highest pooled ROC-AUC (often tabular+ROI) as best biological model when ROI/WMH nonlinear cohort-probe ≈0.99. Prefer fixed tabular or 3D (cohort-probe ≈0.50) for validity narrative.
