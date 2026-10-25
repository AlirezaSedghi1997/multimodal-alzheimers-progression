# Research update — 24 October 2026

The [new analysis bundle](../analyses/deepreview_revision_20261024/README.md)
adds executed post-hoc review analyses and preserves the original primary
estimates. All findings are internal and development-conditioned.

| Analysis | Increment and adjusted conditional interval | Interpretation |
|---|---|---|
| Standardized pooled OOF MRI | ΔIBS +0.000040 (-0.000223, +0.000317), K=2 | No established improvement |
| Standardized pooled OOF PET | ΔIBS -0.000440 (-0.001023, +0.000117), K=2 | No established improvement |
| Nested frozen-component head tuning, MRI | ΔIBS +0.000019 (-0.000306, +0.000357), K=4 | No established improvement |
| Common sample, tau minus plasma | ΔIBS +0.000118 (-0.002849, +0.003202), K=4 | No superiority or equivalence established in 220 people / 26 three-year events |
| Four fixed deformation summaries added to clinical + amyloid | ΔIBS +0.000531 (+0.000074, +0.001022), K=2 | This control increased error |

Lower IBS favors augmentation. These rounded values are for navigation; exact
outputs, family definitions, resampling counts, selected penalties and EDF are
in the aggregate records. All new paired families use 40000 draws. C intervals
are conditional on all-fold estimability, especially fragile in the common
sample (25085 retained draws). They do not cover image training/model selection.

The original 639 omitted tau C draws were reconstructed. Same-draw IBS and
alternative policies retained the original K=3 Cox ranking interpretation.
Under a broader K=40 adjustment, however, the tabular comparable-pair-weighted
C interval includes zero. That method changes the aggregation target and sums
within-fold pairs only. Universal robustness is not claimed.

New paired raw-OOF tau C increments are +0.0730 (0.0217, 0.1308) and
+0.0746 (0.0210, 0.1348), K=3; their IBS intervals remain uncertain.
Actual original fitted-head EDF was computed on 360 hash-verified heads.
The source/PACC audit reconciled ten one-score exclusions, and tau timing was
0–77 days before the landmark. A bounded PET audit did not establish source
voxel calibration or a valid background-only recovery mask.

The actual supplied 128-cubed image trainer ran on synthetic inputs through
training, validation, checkpoint/reload and inference. This is execution
evidence, not end-to-end A4 training reproduction. Aggregate assertions cover
90 original component partitions; membership digests stay private.

The full-cohort four-summary deformation control is separate from the existing
compact CNN/Jacobian pilot. The pilot's historical no-go record is preserved.
No full Jacobian-channel, native-space or pretrained encoder experiment and no
external cohort validation are claimed. No manuscript text, private records,
scans, predictions or weights are included in this update.
