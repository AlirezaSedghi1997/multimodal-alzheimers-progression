# Tau quantitative validation v2

- Scan-level coverage: `1314`; official Stanford source-linked rows: `379`; unmatched/failed rows: `935`.
- Official masks: CenTauR Mesial, Meta, TP and Frontal; reference: `voi_CerebGry_tau_2mm`.
- Comparisons computed: Pearson, Spearman, bias, MAE and Bland–Altman limits.
- Official-source metrics on the 379 bridged rows are: Mesial Pearson `0.9299`,
  Spearman `0.9030`, bias `0.00235`, MAE `0.0461`; Meta Pearson `0.9264`,
  Spearman `0.8769`, bias `0.0201`, MAE `0.0413`; TP Pearson `0.8557`,
  Spearman `0.7791`, bias `0.0757`, MAE `0.0793`; Frontal Pearson `0.7068`,
  Spearman `0.6620`, bias `0.1559`, MAE `0.1571`.
- The source table uses the Stanford non-PVC gray-cerebellum contract
  (`bi_Cerebellum.Cortex`), while the official CenTauR mask uses an inferior-
  cerebellar reference. Their quantitative equivalence has not been proven.
- PET-to-**Stanford** identity currently uses a BID plus numerical visit bridge;
  a shared Series/Study UID or accession is unavailable. A separate audit found
  an exact BID+VISCODE bridge for `1279/1314` rows to the A4 linear-SUVR source
  table (`imaging_SUVR_tau.csv`); this is metadata provenance only and does not
  make that whole-cerebellum pipeline equivalent to Stanford or CenTauR. The
  remaining `35` rows have no A4 source key.
- The sidecar/header audit covered all `1314` linked FTP scans: `0` contained
  non-empty SeriesInstanceUID, StudyInstanceUID or AccessionNumber fields in
  either JSON sidecars or NIfTI extensions, and neither source CSV exposes those
  fields. Machine-readable details are in
  `provenance/tau_scan_source_identity_audit_v2.json` and its hashed row table.
- The source method documents and source-family contract are locked in
  `provenance/tau_source_family_registry_v2.json`. The A4 linear-SUVR table,
  Stanford FreeSurfer non-PVC table, and PETSurfer PVC/GTM table are separate
  quantitative contracts and are never pooled.

These numerical results do **not** pass the claim-bearing gate. The anatomical
region basis is now documented, but exact Stanford scan identity, reference-
pipeline equivalence and a prespecified quantitative calibration path remain
unproven. The local transform scale has not been calibrated to the Stanford
quantitative pipeline. The image-CNN tau branch remains forbidden.

Source artifacts:

- `provenance/scan_level_tau_quantitative_validation.parquet`
- `provenance/scan_level_tau_quantitative_validation.json`
- `provenance/tau_source_family_registry_v2.json`
- `provenance/tau_source_region_crosswalk_audit_v2.json`
- `provenance/tau_tracer_consistency_audit_v2.json`
- `provenance/tau_scan_source_identity_audit_v2.json`
- `provenance/tau_scan_source_identity_audit_v2.parquet`

Tracer audit excludes 143 selected scans with explicit non-FTP sidecar evidence
and 29 without an exact Flortaucipir source row from the claim-bearing path.
Generic fluorine-only text was not treated as a tracer identity. Rows without
exact source-visit linkage remain exploratory even when their image metrics are
finite.
