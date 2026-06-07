# Phase 1 Incremental Value Table

- status: `COMPLETED`

| step | roc_auc | pr_auc | brier | n_features |
|---|---:|---:|---:|---|
| fixed_tabular | 0.728884377271474 | 0.8496227848172967 | 0.21052292814535506 | 5 |
| cognitive_only | 0.7150537634408602 | 0.8342719816788144 | 0.21687964769314055 | 25 |
| fixed_tabular_plus_cognitive | 0.7931623931623931 | 0.8842543025389877 | 0.18685001029197396 | 30 |
| roi_icv_norm_only | 0.7534289969205916 | 0.9141601479089289 | 0.19924757372101745 | 53 |
| wmh_only | 0.49818264425261244 | 0.805985350503306 | 0.24032962063322277 | 1 |
| mri_reads_only | 0.5027727295673684 | 0.785239159258601 | 0.24983483112742966 | 4 |
| roi_wmh_reads | 0.7515939724367712 | 0.9134611481357562 | 0.20048358571103386 | 58 |
| fixed_tabular_plus_roi_wmh_reads | 0.8207254278358322 | 0.9239612486800821 | 0.17609371903837726 | 63 |
| method_x_strict_3d_cnn | 0.7221570544151189 | 0.8421626306129226 | 0.20841256008246753 | 3D_MRI_embedding_plus_tabular |

## Caveat

A4/LEARN enrolled catalog has strong cohort-label coupling; interpret increments as descriptive only.
