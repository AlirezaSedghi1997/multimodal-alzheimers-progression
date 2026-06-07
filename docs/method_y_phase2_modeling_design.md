# Method Y Phase 2 Modeling Design — v3.8 Literature-Aligned

## Status
Design-frozen before PET training. This document does not authorize training until `scripts/check_method_y_training_readiness.py` returns READY.

## Scientific framing
Method Y is PET-assisted exploratory modeling. PET-containing models are not non-invasive and are not Alzheimer disease diagnosis models. PET-only and MRI+PET models are evaluated against PET-derived biomarker labels; this creates a PET-label tautology risk. Report them as biomarker reproduction / reader-assist / fusion-behavior analyses.

## Required baselines before any fusion claim
1. **PET-only baseline**: reader-surrogate / biomarker reproduction ceiling.
2. **MRI-only baseline on the exact Method Y split**: structural baseline.
3. **Late/logit fusion baseline**: robust, low-parameter, missing-modality friendly.
4. **Gated feature fusion**: practical primary fusion variant if baselines pass.
5. **Early channel concatenation**: ablation only, not primary.
6. **Cross-attention**: advanced optional variant if sample size and VRAM allow.
7. **MRI+PET+cognitive**: optional extension after MRI-only, PET-only, and MRI+PET baselines are complete.

## Fusion hierarchy
- Primary low-risk baseline: late fusion / learned logit combiner.
- Practical primary fusion: separate encoders + gated feature fusion.
- Advanced: cross-attention with modality dropout.
- Future robustness: sparse MoE or modality dropout for missing PET.

## Metrics required for all training scripts
- ROC-AUC, PR-AUC, sensitivity, specificity, balanced accuracy.
- Brier score, ECE, reliability diagram.
- Bootstrap 95% CI, default 1000 iterations where feasible.
- Calibration fitted on validation only, never on test/pseudo-external.

## Ablation table required
| Variant | Required | Interpretation |
|---|---:|---|
| MRI-only | yes | Method Y structural baseline |
| PET-only | yes | PET reader-surrogate ceiling |
| MRI+PET late fusion | yes | robust fusion baseline |
| MRI+PET gated feature fusion | yes | practical primary fusion |
| MRI+PET early channel concat | optional | ablation only |
| MRI+PET cross-attention | optional | advanced, sample-size dependent |
| MRI+PET+cognitive | optional | exploratory extension |

## Sample-size caution
If primary strict tier has fewer than 100 subjects in either class, PET training is blocked. If fewer than 150 per class, training is allowed only with warnings and reduced claim strength.
