# Task 1 implementation report

Status: `DONE_WITH_CONCERNS`

## Scope and safety

- Worked only in `H:\New project` on branch `methody/pet-branch-implementation`.
- Treated `H:\A4 Learn` as read-only evidence.
- No PET transform, transform validation rerun, HPO, outer evaluation, image-CNN, or claim-bearing training was started.
- The pre-existing dirty worktree was preserved; only the scoped files listed below were staged.

## Evidence audit

The audit covered the 1,314 rows in
`results/method_y_revival_v2/manifests/private_ftp_t1_transform_eligible.parquet`,
the existing transform/linkage naming, all corresponding FTP JSON sidecars and
NIfTI headers/extensions, and the A4 linear-SUVR and Stanford FreeSurfer source
CSVs.

Evidence found:

- `1,279/1,314` linked FTP rows have an exact BID plus canonical VISCODE key in
  `H:\A4 Learn\Clinical\External Data\imaging_SUVR_tau.csv` (192 regional
  rows per key). This is a direct bridge to the separate A4 linear-SUVR source
  family only.
- `379` rows have the existing numerical BID plus visit-2 bridge to the
  Stanford table; this is not exact acquisition identity.
- The Stanford method document records a non-PVC FreeSurfer gray-matter
  cerebellum reference (`bi_Cerebellum.Cortex`, normalized to 1). The A4 SUVR
  method records whole-cerebellum `blcere_all`.
- The CenTauR supplement documents the FreeSurfer-derived regional basis used
  by the anatomical crosswalk.

Evidence not found:

- No non-empty `SeriesInstanceUID`, `StudyInstanceUID`, or `AccessionNumber`
  appeared in any of the 1,314 sidecars or NIfTI header extensions.
- Neither source CSV contains DICOM UID/accession columns.
- No direct Stanford source-table-to-transform UID/accession bridge exists.
- No local documented equivalence between `voi_CerebGry_tau_2mm`, Stanford
  `bi_Cerebellum.Cortex`, and the A4 source pipeline was found.
- No prespecified local calibration path or common reference dataset was found.

## Implementation

- Added `src/method_y_revival/tau_scan_provenance_audit.py`, which performs an
  atomic, row-level sidecar/NIfTI identity audit; canonical BID/VISCODE matching
  to A4 linear-SUVR; separate Stanford visit-bridge classification; source
  schema/hash capture; and fail-closed science-gate status.
- Added `scripts/audit_method_y_tau_scan_provenance_v2.py` to refresh the audit
  artifacts without reading or rewriting transform voxels.
- Added focused tests in
  `tests/method_y_revival/test_tau_scan_provenance_audit_v2.py` covering exact
  identity parsing, nested sidecar identity fields, direct A4 bridge detection,
  and preservation of the science-gate block.
- Refreshed `results/method_y_revival_v2/provenance/tau_scan_source_identity_audit_v2.json`,
  `.../tau_scan_source_identity_audit_v2.parquet`, and
  `reports/TAU_QUANTITATIVE_VALIDATION_V2.md` with the direct-bridge distinction
  and the all-scan identity negative result.

## Exact commands and results

1. Raw/process inspection: `Get-CimInstance Win32_Process` filtered for the
   Method Y/PET/tau/training command patterns; no matching active writer was
   present.
2. Metadata search: `rg -n -i`
   `SeriesInstanceUID|StudyInstanceUID|AccessionNumber` over the raw/source
   metadata tree; no exact identity fields were found.
3. Audit refresh:
   `& .venv\\Scripts\\python.exe scripts/audit_method_y_tau_scan_provenance_v2.py --output-root results/method_y_revival_v2`
   exited `0`; report status is `COMPLETED_WITH_SCIENCE_GATE_BLOCKED` with
   `scan_count=1314`, exact sidecar identity `0`, exact NIfTI-header identity
   `0`, A4 direct bridge `1279`, no A4 key `35`, and Stanford numerical bridge
   `379`.
4. Focused tests:
   `& .venv\\Scripts\\python.exe -m pytest -q tests/method_y_revival/test_tau_scan_provenance_audit_v2.py tests/method_y_revival/test_tau_official_crosswalk.py tests/method_y_revival/test_tau_source_region_crosswalk_audit_v2.py tests/method_y_revival/test_tau_source_registry_v2.py tests/method_y_revival/test_quantitative_scan_gates_v2.py`
   result: `16 passed`.
5. Syntax check:
   `& .venv\\Scripts\\python.exe -m py_compile src/method_y_revival/tau_scan_provenance_audit.py scripts/audit_method_y_tau_scan_provenance_v2.py`
   exited `0`.

## Artifact status and concerns

- The Tau quantitative science gate remains `BLOCKED`; claim status remains
  `NO_CLAIM`; image training remains disallowed.
- The A4 direct bridge is not a Stanford/CenTauR equivalence and must not be
  used to clear the reference or calibration blockers.
- The remaining 35 rows have no exact A4 source key; 935 rows remain outside
  the Stanford numerical bridge in the existing scan-level table.
- The generated artifacts are intentionally private provenance outputs and
  include hashes/paths needed for reproducibility; no public bundle was changed.
