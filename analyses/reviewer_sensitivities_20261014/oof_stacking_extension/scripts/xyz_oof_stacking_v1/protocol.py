"""Fixed model/contrast definitions, decided before this experiment's evaluation."""
from scripts.xyz_fusion_v2.protocol import CLINICAL, MRI, PET


def arms(track):
    base = CLINICAL + ['amyloid_centiloid']
    values = {'clinical_amyloid': base, 'clinical_mri_amyloid': base + MRI}
    if track == 'mri':
        return values
    if track == 'pet':
        return dict(values, clinical_mri_amyloid_pet=base + MRI + PET)
    if track == 'blood':
        return {'clinical_amyloid': base,
                'clinical_amyloid_blood': base + ['blood_log_ptau217'],
                'clinical_mri_amyloid_blood': base + MRI + ['blood_log_ptau217']}
    if track == 'tau':
        return dict(values, clinical_amyloid_tau=base + ['tau_temporal'],
                    clinical_mri_amyloid_tau=base + MRI + ['tau_temporal'])
    raise ValueError('UNKNOWN_OOF_TRACK')


def contrasts():
    # Families are separate questions, never evidence of global multiplicity control.
    return [
        ('mri', 'clinical_amyloid', 'clinical_mri_amyloid', 'imaging', 2),
        ('pet', 'clinical_mri_amyloid', 'clinical_mri_amyloid_pet', 'imaging', 2),
        ('blood', 'clinical_amyloid_blood', 'clinical_mri_amyloid_blood', 'biomarker', 3),
        ('tau', 'clinical_amyloid', 'clinical_amyloid_tau', 'biomarker', 3),
        ('tau', 'clinical_mri_amyloid', 'clinical_mri_amyloid_tau', 'biomarker', 3),
        ('mri', 'split_clinical_mri_amyloid', 'clinical_mri_amyloid', 'pooling', 2),
        ('pet', 'split_clinical_mri_amyloid_pet', 'clinical_mri_amyloid_pet', 'pooling', 2),
    ]


def validate(options):
    expected = {
        'protocol': 'POOLED_CROSSFIT_SURVIVAL_STACKING_V1',
        'claim_status': 'POST_HOC_INTERNAL_EXPLORATORY',
        'tracks': ['mri', 'pet', 'blood', 'tau'],
        'ridge_alpha': 10.0,
        'precision': 'FP32_CNN_SCORES_FLOAT64_COX',
        'allow_new_cnn_training': False,
        'allow_outer_model_selection': False,
        'tune_head_on_pooled_oof': False,
        'aggregation': 'MEAN_COMPONENT_SURVIVAL_THEN_MEAN_SEED_SURVIVAL',
    }
    for key, value in expected.items():
        if options.get(key) != value:
            raise ValueError('PROTOCOL_CHANGE_REQUIRES_NEW_VERSION:' + key)
    if options['bootstrap_resamples'] < 40000 or not 1 <= options['cpu_threads'] <= 8:
        raise ValueError('INVALID_COMPUTE_BUDGET')
