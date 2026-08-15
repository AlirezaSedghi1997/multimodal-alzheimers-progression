# Tau quantitative validation v2

- Official masks: CenTauR Mesial, Meta, TP and Frontal; reference: `voi_CerebGry_tau_2mm`.
- Comparisons computed: Pearson, Spearman, bias, MAE and Bland–Altman limits.
- Canonical aggregate quantitative metrics (not claim-bearing):
  - Mesial_CenTauR: pairs `379`, Pearson `0.9299`, Spearman `0.9030`, bias `0.00235`, MAE `0.0461`, Bland–Altman mean bias `0.00235`, SD `0.06074`, 95% limits [`-0.11671`, `0.12141`].
  - Meta_CenTauR: pairs `379`, Pearson `0.9264`, Spearman `0.8769`, bias `0.02005`, MAE `0.0413`, Bland–Altman mean bias `0.02005`, SD `0.04963`, 95% limits [`-0.07723`, `0.11733`].
  - TP_CenTauR: pairs `379`, Pearson `0.8557`, Spearman `0.7791`, bias `0.07569`, MAE `0.0793`, Bland–Altman mean bias `0.07569`, SD `0.06166`, 95% limits [`-0.04516`, `0.19653`].
  - Frontal_CenTauR: pairs `379`, Pearson `0.7068`, Spearman `0.6620`, bias `0.15586`, MAE `0.1571`, Bland–Altman mean bias `0.15586`, SD `0.07813`, 95% limits [`0.00272`, `0.30900`].
- Metric sources: `scan_level_tau_quantitative_validation.json` SHA-256 `05c2a4a016393e1079e330044210aad03e48a6be08744b10f6c9e493d3381da1`, `tau_source_region_crosswalk_audit_v2.json` SHA-256 `b5bc619fff4ba82d99cbaf3e279b568a217b335fcb51084d2ebd49b727caa55a`.
- Stanford uses the non-PVC gray-matter cerebellum contract (`bi_Cerebellum.Cortex`), while the official CenTauR mask uses an inferior-cerebellar reference; equivalence is not proven.
<!-- scan-provenance audit: begin -->
## Scan-provenance audit (refresh-generated)

- Scan count: `1314`.
- Locked input manifest: `private_ftp_t1_transform_eligible.parquet`; rows `1314`; SHA-256 `73556b25cbf3c9a50b498f0c166aa8b9b8b6f95164f1571dff6aca52179f2917`.
- Sidecar identity status counts: `{"NO_EXACT_ID_FIELDS": 1314}`.
- NIfTI header/extension identity status counts: `{"NO_EXACT_ID_FIELDS": 1314}`.
- Aggregate exact identity scans (sidecar or header): `0`; fields: `none`.
- A4 linear-SUVR BID+VISCODE subject/visit associations: `1279`; statuses: `{"NO_A4_LINEAR_SUVR_SOURCE_KEY": 35, "SUBJECT_VISIT_ASSOCIATION_A4_LINEAR_SUVR_SOURCE": 1279}`.
- Stanford BID+VISCODE subject/visit associations (including numeric visit convention): `{"NO_STANFORD_SOURCE_KEY": 935, "NUMERICAL_VISIT_ASSOCIATION_TO_STANFORD_VISCODE_2": 379}`.
- BID+VISCODE associations are not exact acquisition identity; no UID/accession bridge or reference-pipeline equivalence is inferred.
- Science gate: **BLOCKED**; claim status: **NO_CLAIM**; claim-bearing work remains disallowed.
<!-- scan-provenance audit: end -->

The quantitative science gate remains blocked until exact scan identity, reference-pipeline equivalence, and a prespecified calibration path are proven.

Canonical aggregate metric artifacts are identified above by fixed name and SHA-256; no row-level artifact path is published.

Rows without exact source-visit linkage remain exploratory even when image metrics are finite; no transform rerun, training, HPO, outer evaluation, or image-CNN work is authorized by this audit.
