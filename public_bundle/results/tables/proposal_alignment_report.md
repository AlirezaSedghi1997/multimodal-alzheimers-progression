# Proposal Alignment Report

- status: `COMPLETED`
- title framing: `Early prediction of amyloid biomarker status and preclinical risk`
- clinical Alzheimer diagnosis requires independent label: `True`
- Method X-Strict PET-free: `True`

| claim | status | current mapping | caveat |
|---|---|---|---|
| P01 | SCIENTIFIC_ETHICAL_LIMIT | Reframed as early prediction of amyloid biomarker status and preclinical risk unless an independent clinical label is used. | AB-positive vs AB-negative is not clinical Alzheimer diagnosis. |
| P02 | BETTER_THAN_PROPOSAL | Method X-Strict protects PET-free inputs: T1 MRI plus age/sex/education/APOE4/MMSE. | PET is label source only, never Method X input. |
| P03 | ADDED_CONTROLLED_EXTENSION | Method Y includes PET-only, MRI-only paired, MRI+PET fusion, and MRI+PET+cognitive controlled extension. | Method Y is paired subset only and PET-label tautology must be reported. |
| P04 | ADDED_CONTROLLED_EXTENSION | MONAI ResNet/EfficientNet controlled ablations exist; true pretrained 3D fine-tuning is not claimed because no validated compatible pretrained 3D weights are locked in this repo. | Report as controlled ablation, not confirmed pretrained transfer learning. |
| P05 | ADDED_CONTROLLED_EXTENSION | Train-fold-only MONAI augmentation reports 4x augmented training exposures while real N remains unchanged. | 6000 is augmented exposure count, not 6000 real subjects. |
| P06 | ADDED_CONTROLLED_EXTENSION | Supplementary fold-local policy: generator fit only inside train fold; synthetic PET added only to same train fold. | Validation/test/holdout stay real; Method X never receives synthetic PET. |
| P07 | ADDED_CONTROLLED_EXTENSION | Threshold metrics are computed using thresholds selected from fold-internal validation data only. | Test labels are not used for threshold selection. |
| P08 | DONE | Clinical endpoint branch uses CDR worsening and PACC decline as progression proxies. | Progression proxy, not clinical AD diagnosis. |
| P09 | BETTER_THAN_PROPOSAL | Public bundle is aggregate/hash-level only with raw data, HDF5, images, raw identifiers, raw paths, and subject-level predictions excluded. | Subject-level prediction vectors remain private. |
| P10 | BETTER_THAN_PROPOSAL | Current project preserves SynIB, MFSCA, worst-dimension audit, FLAIR sensitivity, KD, cohort-probe, calibration, DCA, privacy, and claim-control workflows. | These are retained as current upgrades, not reverted to the initial proposal. |
