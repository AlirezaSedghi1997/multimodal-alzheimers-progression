"""Immutable analysis specifications; no choices may depend on outer performance."""
from __future__ import annotations

CLINICAL = ['age', 'sex_binary', 'education_years', 'apoe4_carrier',
            'baseline_pacc_raw', 'baseline_mmscore', 'treatment_active']
VOLUMES = ['hippocampus_icv', 'ventricle_icv', 'entorhinal_icv']
MRI = ['mri_risk_1', 'mri_risk_2', 'mri_risk_3']
PET = ['pet_risk_1', 'pet_risk_2', 'pet_risk_3']
CLAIM = 'POST_HOC_INTERNAL_EXPLORATORY'


def arms(track):
    c = CLINICAL
    base = {'clinical': c, 'clinical_amyloid': c + ['amyloid_centiloid'],
            'clinical_volumes': c + VOLUMES,
            'clinical_volumes_amyloid': c + VOLUMES + ['amyloid_centiloid']}
    if track == 'scalar':
        return dict(base, blood_clinical=c,
                    blood_clinical_amyloid=c+['amyloid_centiloid'],
                    blood_clinical_blood=c+['blood_log_ptau217'],
                    blood_clinical_amyloid_blood=c+['amyloid_centiloid', 'blood_log_ptau217'])
    if track == 'tau':
        return {'clinical_amyloid': c+['amyloid_centiloid'],
                'clinical_amyloid_tau': c+['amyloid_centiloid', 'tau_temporal']}
    if track not in ('mri', 'pet'):
        raise ValueError('UNKNOWN_FUSION_TRACK')
    result = dict(base, mri_only=MRI, clinical_mri=c+MRI,
                  clinical_mri_amyloid=c+MRI+['amyloid_centiloid'],
                  clinical_volumes_mri_amyloid=c+VOLUMES+MRI+['amyloid_centiloid'])
    if track == 'pet':
        result['clinical_mri_amyloid_pet'] = c+MRI+['amyloid_centiloid']+PET
    else:
        result.update(blood_clinical_amyloid=c+['amyloid_centiloid'],
                      blood_clinical_amyloid_blood=c+['amyloid_centiloid', 'blood_log_ptau217'],
                      blood_clinical_mri_amyloid=c+MRI+['amyloid_centiloid'],
                      blood_clinical_mri_amyloid_blood=c+MRI+['amyloid_centiloid', 'blood_log_ptau217'])
    return result


def contrasts(track):
    if track == 'tau':
        return [('clinical_amyloid', 'clinical_amyloid_tau')]
    pairs = [('clinical', 'clinical_amyloid'), ('clinical', 'clinical_volumes'),
             ('clinical_amyloid', 'clinical_volumes_amyloid')]
    if track == 'scalar':
        return pairs + [('blood_clinical', 'blood_clinical_blood'),
                        ('blood_clinical_amyloid', 'blood_clinical_amyloid_blood')]
    pairs += [('clinical', 'clinical_mri'), ('clinical_amyloid', 'clinical_mri_amyloid'),
              ('clinical_volumes_amyloid', 'clinical_volumes_mri_amyloid')]
    if track == 'pet':
        return pairs + [('clinical_mri_amyloid', 'clinical_mri_amyloid_pet')]
    return pairs + [('blood_clinical_amyloid', 'blood_clinical_amyloid_blood'),
                    ('blood_clinical_amyloid_blood', 'blood_clinical_mri_amyloid_blood')]


def eligible(frame, arm):
    return frame.loc[frame.blood_ptau217.notna()].copy() if arm.startswith('blood_') else frame.copy()


def validate_config(c):
    if c['training']['backbone']!='amuse_x_hybrid3d' or c['training']['hybrid_variant']!='baseline':
        raise ValueError('UNIMPLEMENTED_ARCHITECTURE_CONTRACT')
    if c['training']['tf32'] or not c['training']['deterministic']:
        raise ValueError('EXACT_RESUME_MATH_POLICY_REQUIRED')
    if c['outer_folds'] != 5 or c['components'] != 3 or len(set(c['training_seeds'])) < 2:
        raise ValueError('FIVE_OUTER_THREE_COMPONENTS_AND_REPEATED_SEEDS_REQUIRED')
    if c['claim_status'] != CLAIM or c['training']['selection_metric'] != 'IPCW_IBS':
        raise ValueError('PROTOCOL_CLAIM_OR_SELECTION_DRIFT')
    if c['endpoint']['primary'] != 'confirmation_time':
        raise ValueError('VERSIONED_ENDPOINT_CONTRACT_REQUIRED')
    if c['training']['epochs'] < c['training']['minimum_epochs']:
        raise ValueError('INVALID_TRAINING_BUDGET')
    if c['training']['augmentation'] != 'SMALL_TRANSLATION_NOISE_MRI_ONLY':
        raise ValueError('AUGMENTATION_CONTRACT')
    for track in ('scalar', 'mri', 'pet', 'tau'):
        for a, b in contrasts(track):
            if a not in arms(track) or b not in arms(track):
                raise ValueError('CONTRAST_ARM_MISSING')
