# Full MRI Preprocessing Toolchain Report

- status: `BLOCKED`
- missing: `['bias_correction', 'brain_extraction', 'registration', 'template']`
- required: `{'bias_correction': ['N4BiasFieldCorrection'], 'registration': ['antsRegistrationSyNQuick.sh'], 'brain_extraction': ['mri_synthstrip', 'hd-bet', 'bet']}`
- template: `None`
- contains_pet_input: `False`
- contains_raw_identifiers: `False`
- output_granularity: `hashed_sample_level`
