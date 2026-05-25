# Full MRI Preprocessing Toolchain Report

- status: `BLOCKED`
- missing: `['brain_extraction', 'template']`
- required: `{'bias_correction': ['N4BiasFieldCorrection'], 'registration': ['antsRegistrationSyNQuick.sh'], 'brain_extraction': ['mri_synthstrip', 'hd-bet', 'bet']}`
- template: `/home/aiuser/fsl/data/standard/MNI152_T1_1mm.nii.gz`
- contains_pet_input: `False`
- contains_raw_identifiers: `False`
- output_granularity: `hashed_sample_level`
