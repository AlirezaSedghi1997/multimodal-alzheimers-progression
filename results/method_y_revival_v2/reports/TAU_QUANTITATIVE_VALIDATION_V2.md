# Tau quantitative validation v2

- Official masks: CenTauR Mesial, Meta, TP and Frontal; reference: `voi_CerebGry_tau_2mm`.
- Comparisons computed: Pearson, Spearman, bias, MAE and Bland–Altman limits.
- Official-source metrics remain on the existing 379-row numerical-visit association and are not claim-bearing: Mesial Pearson `0.9299`, Spearman `0.9030`, bias `0.00235`, MAE `0.0461`; Meta Pearson `0.9264`, Spearman `0.8769`, bias `0.0201`, MAE `0.0413`; TP Pearson `0.8557`, Spearman `0.7791`, bias `0.0757`, MAE `0.0793`; Frontal Pearson `0.7068`, Spearman `0.6620`, bias `0.1559`, MAE `0.1571`.
- Stanford uses the non-PVC gray-matter cerebellum contract (`bi_Cerebellum.Cortex`), while the official CenTauR mask uses an inferior-cerebellar reference; equivalence is not proven.
<!-- scan-provenance audit: begin -->
## Scan-provenance audit (refresh-generated)

- Scan count: `1314`.
- Sidecar identity status counts: `{"NO_EXACT_ID_FIELDS": 1314}`.
- NIfTI header/extension identity status counts: `{"NO_EXACT_ID_FIELDS": 1314}`.
- Aggregate exact identity scans (sidecar or header): `0`; fields: `none`.
- A4 linear-SUVR BID+VISCODE subject/visit associations: `1279`; statuses: `{"NO_A4_LINEAR_SUVR_SOURCE_KEY": 35, "SUBJECT_VISIT_ASSOCIATION_A4_LINEAR_SUVR_SOURCE": 1279}`.
- Stanford BID+VISCODE subject/visit associations (including numeric visit convention): `{"NO_STANFORD_SOURCE_KEY": 935, "NUMERICAL_VISIT_ASSOCIATION_TO_STANFORD_VISCODE_2": 379}`.
- BID+VISCODE associations are not exact acquisition identity; no UID/accession bridge or reference-pipeline equivalence is inferred.
- Science gate: **BLOCKED**; claim status: **NO_CLAIM**; claim-bearing work remains disallowed.
<!-- scan-provenance audit: end -->

The quantitative science gate remains blocked until exact scan identity, reference-pipeline equivalence, and a prespecified calibration path are proven.

Source artifacts: `provenance/scan_level_tau_quantitative_validation.parquet`, `provenance/scan_level_tau_quantitative_validation.json`, `provenance/tau_source_family_registry_v2.json`, `provenance/tau_source_region_crosswalk_audit_v2.json`, `provenance/tau_tracer_consistency_audit_v2.json`, and `provenance/tau_scan_source_identity_audit_v2.json`.

Rows without exact source-visit linkage remain exploratory even when image metrics are finite; no transform rerun, training, HPO, outer evaluation, or image-CNN work is authorized by this audit.
