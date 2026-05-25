# Implementation Gap Matrix

| requirement | status | artifact | note |
|---|---|---|---|
| PET is never a model input | DONE | `final_analysis_summary.json` | PET remains label/target only. |
| Allowed tabular schema is fixed | DONE | `data/processed/master_catalog.csv` | Only aggregate column-level check; no raw rows exported. |
| Raw IDs/PII excluded from artifacts | DONE | `public_bundle/PUBLIC_BUNDLE_MANIFEST.json` | Public bundle manifest reports no raw identifiers. |
| Patient-level split locking | DONE | `splits/split_summary.json` | External holdout remains locked pseudo-external. |
| Baseline tabular and MRI-feature models | DONE | `results/final_results/` | Fast-track baselines generated. |
| Fusion/ablation variants | PARTIAL | `results/final_results/three_d_ablation/three_d_ablation_summary.json` | Feature/surrogate fusion exists; DONE requires real 3D concat/cross-attention/alignment ablation. |
| Real 3D MRI training | PARTIAL | `results/final_results/three_d/three_d_metrics.json` | CUDA 64/3-fold exists; final 96/5-fold remains pending. |
| Deep MRI preprocessing | MISSING | `results/tables/full_mri_preprocess_toolchain_report.json` | Specialized toolchain-gated stage is required before final 96 training. |
| Power analysis with actual N | DONE | `results/tables/power_analysis.json` | Must use catalog/split N, class balance, CIs, and caveats. |
| Site/scanner drift sensitivity | SKIP_WITH_JUSTIFICATION | `results/tables/site_scanner_drift_sensitivity_report.json` | No site/scanner metadata columns are present in the locked master catalog. |
| Subgroup performance audit | DONE | `results/final_results/subgroup_performance_audit.json` | Required for sex, age bins, APOE4, education, and MMSE with CIs. |
| XAI / interpretability | DONE | `results/final_results/xai/xai_summary.json` | Grad-CAM MRI and SHAP tabular must be aggregate-only in public outputs. |
| Tau exploratory branch | DONE | `results/final_results/tau/tau_regression_metrics.json` | Tau regression executed with grouped CV; current result is exploratory and negative versus the grouped mean baseline. |
| Public reproducibility bundle | DONE | `public_bundle/PUBLIC_BUNDLE_MANIFEST.json` | Aggregate text only; no raw data/HDF5/checkpoints/images. |
| Thesis-grade results report | MISSING | `results/final_results/thesis/` | Write after validation refresh, not from current fast-track-only package. |
