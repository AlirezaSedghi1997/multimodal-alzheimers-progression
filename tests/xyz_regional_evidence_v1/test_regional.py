from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from scripts.xyz_regional_evidence_v1.statistics import holm_family, cox_association


def test_holm_keeps_failed_regions_in_family():
    values = {'a': {'p': .02}, 'b': {'p': None}, 'c': {'p': .04}}
    result = holm_family(values)
    assert result['a']['p_holm'] == pytest.approx(.06)
    assert result['b']['p_holm'] is None
    assert not any(x['multiplicity_supported'] for x in result.values())


def test_holm_is_not_unadjusted_significance():
    result = holm_family({'a': {'p': .001}, 'b': {'p': .02}, 'c': {'p': .9}})
    assert result['a']['p_holm'] == pytest.approx(.003)
    assert result['b']['p_holm'] == pytest.approx(.04)


def frame(n=500, seed=29):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    death = rng.exponential(600 / np.exp(.6*x))
    censor = rng.uniform(400, 1640, n)
    result = pd.DataFrame({'patient_hash': [str(i) for i in range(n)], 'time_days': np.minimum(death, censor),
                          'event': (death <= censor).astype(int), 'region_z': x,
                          'age_z': rng.normal(size=n), 'mri_manufacturer': ['A','B']*(n//2)})
    return result


OPTIONS = {'minimum_subjects': 100, 'minimum_events': 60, 'minimum_events_per_parameter': 5, 'ph_step_days':730.5}


def test_cox_recovers_known_direction_without_training_a_predictor():
    value = cox_association(frame(), ['region_z','age_z'], OPTIONS)
    assert value['status'] == 'ESTIMATED'
    assert .3 < value['log_hr'] < .9
    assert value['hazard_ratio'] > 1
    assert value['p'] < .001
    assert value['pacc_or_saliency_was_not_target']
    assert value['ph']['status']=='TESTED_TWO_YEAR_ROI_TIME_INTERACTION'
    assert value['linearity']['status']=='TESTED_QUADRATIC_DEPARTURE'


def test_no_events_does_not_become_pass_or_null_effect():
    data = frame(); data['event'] = 0
    value = cox_association(data, ['region_z','age_z'], OPTIONS)
    assert value['status'].startswith('NOT_ESTIMABLE') and value.get('p') is None


def test_subject_duplication_is_rejected_not_pseudoreplicated():
    data = frame(); data.loc[1,'patient_hash'] = data.loc[0,'patient_hash']
    with pytest.raises(ValueError, match='DUPLICATE_SUBJECT'):
        cox_association(data, ['region_z','age_z'], OPTIONS)


def test_nonfinite_region_is_not_automatically_imputed():
    data = frame(); data.loc[0,'region_z'] = np.inf
    with pytest.raises(ValueError, match='NONFINITE'):
        cox_association(data, ['region_z','age_z'], OPTIONS)


def test_missing_region_family_cannot_be_finalized():
    from scripts.xyz_regional_evidence_v1.reporting import require_complete
    with pytest.raises(ValueError, match='REGIONAL_COVERAGE'):
        require_complete({'a': {}}, {'a':{}, 'b':{}})


def test_image_region_name_requires_exact_atlas_identity():
    from scripts.xyz_regional_evidence_v1.reporting import anatomical_mapping
    mapping = anatomical_mapping({'109':'Left Hippocampus','119':'Right Hippocampus',
                                  '31':'Parahippocampal Gyrus, anterior division'})
    assert mapping['hippocampus']['labels'] == [109,119]
    assert mapping['entorhinal']['status'] == 'NO_EXACT_LABEL_NO_IMAGE_MAPPING'


def test_pacc_repeated_measures_are_estimable_and_direction_correct():
    from scripts.xyz_regional_evidence_v1.statistics import pacc_association
    rng=np.random.default_rng(291);rows=[]
    for i in range(120):
        region,age=rng.normal(size=2);intercept,slope=rng.normal(size=2)*[1,.4]
        for years in [.5,1,2,3,4]:
            rows.append({'patient_hash':str(i),'years':years,'region_z':region,'age_z':age,
                'mri_manufacturer':'A' if i%2 else 'B',
                'pacc':intercept+.2*age+years*(.5*region+slope)+rng.normal(scale=.5)})
    value=pacc_association(pd.DataFrame(rows),['age_z'])
    assert value['status']=='ESTIMATED'
    assert value['n_subjects']==120 and value['n_measurements']==600
    assert .25<value['estimate']<.75
    assert value['p']<.001
