# Method Y PET Phase 2 Spatial+QC Changelog

## Added

- Mandatory spatial preprocessing policy.
- SUVR/reference-region policy with scalar feature ban.
- Manifest-first private PET linking.
- One-PET-per-patient selection.
- Resumable preprocessing manifest.
- Controlled parallel execution.
- QC failure gates.
- Training readiness script.
- Agent playbook and compact policy index.

## Explicitly not added

- PET image input to Method X.
- SUVR/Centiloid scalar model inputs.
- Automatic primary tier switching.
- DICOM conversion.
- DeepSUVR dependency.
- Direct streaming to Method Y HDF5 before QC.
