# Cohort-Probe Summary (Phase 0.6 Protocol)

| input | linear_auc | nonlinear_auc | gap | interpretation |
|---|---:|---:|---:|---|
| method_x_3d_mri_latent_oof | 0.5031 | 0.5102 | 0.0070 | Cohort encoding in the input representation is limited or ambiguous. |
| method_x_strict_tabular | 0.7313 | 0.7050 | -0.0263 | Moderate linear cohort encoding; under A4/LEARN coupling (Cramér's V≈1.0) this likely reflects substudy structure — do not call ambiguous. |
| mri_roi_wmh_reads | 0.7621 | 0.9920 | 0.2299 | Moderate linear cohort encoding; under A4/LEARN coupling (Cramér's V≈1.0) this likely reflects substudy structure — do not call ambiguous. |
