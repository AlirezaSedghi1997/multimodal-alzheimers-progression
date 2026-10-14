"""Leakage, alignment, immutable receipts and paired inference regression tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scripts.xyz_oof_stacking_v1.inputs import require_keys, pool_oof, FrozenUnits, Inputs
from scripts.xyz_oof_stacking_v1.reporting import paired_ibs
from scripts.xyz_oof_stacking_v1.protocol import validate, arms, contrasts
from scripts.xyz_oof_stacking_v1.runner import output_path, verify
from scripts.xyz_oof_stacking_v1.pipeline import fit_heads, predict_head, head_name
from scripts.xyz_fusion_postall_v2.core import Units, atomic_json, digest, file_sha
from scripts.xyz_fusion_v2.models import fit_model, curves, assert_pair
from scripts.xyz_fusion_v2.pipeline import average_predictions


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def options():
    return json.loads((ROOT / 'configs/xyz_oof_stacking_v1.json').read_text())


def test_pooled_oof_requires_exactly_once_and_aligns_subjects():
    a = pd.DataFrame({'patient_hash': ['b', 'a'], 'mri_risk_1': [20., 10.]})
    b = pd.DataFrame({'patient_hash': ['c'], 'mri_risk_1': [30.]})
    result = pool_oof([b, a], {'a', 'b', 'c'})
    assert list(result.mri_risk_1) == [10., 20., 30.]
    with pytest.raises(ValueError, match='EXACT_UNIQUE'):
        pool_oof([a, a], {'a', 'b'})
    with pytest.raises(ValueError, match='EXACT_UNIQUE'):
        pool_oof([a], {'a', 'b', 'c'})


@pytest.mark.parametrize('selection', [False, True])
def test_training_and_selection_ancestry_cannot_include_meta_or_test(selection):
    receipt = {'train_subjects': ['train'], 'selection_subjects': ['stop']}
    receipt['selection_subjects' if selection else 'train_subjects'].append('held')
    with pytest.raises(ValueError, match='ANCESTRY_OVERLAP'):
        assert_pair(receipt, ['meta'], ['held'])


def test_outer_order_must_match_before_averaging():
    a = {'arm': {'subjects': ['a', 'b'], 'survival': [[.9, .8], [.8, .7]]}}
    b = copy.deepcopy(a); b['arm']['subjects'].reverse()
    with pytest.raises(ValueError, match='MEMBERSHIP_MISMATCH'):
        average_predictions([a, b])


def test_average_survival_not_average_transformed_features():
    a = {'arm': {'subjects': ['a'], 'survival': [[.8, .4]]}}
    b = {'arm': {'subjects': ['a'], 'survival': [[.2, .1]]}}
    got = average_predictions([a, b])['arm']['survival']
    assert np.allclose(got, [[.5, .25]])
    assert not np.allclose(got, np.sqrt(np.array(a['arm']['survival']) * b['arm']['survival']))


def test_held_values_do_not_change_training_transform():
    rng = np.random.default_rng(41)
    train = pd.DataFrame({'x': rng.normal(size=160), 'event': rng.random(160) < .6,
                          'time_days': rng.uniform(10, 1800, 160)})
    train.loc[0, 'x'] = np.nan
    model = fit_model(train, ['x'], 'cox_ridge', 10, {}, 13)
    med = model[0].statistics_.copy(); mean = model[1].mean_.copy()
    held = pd.DataFrame({'x': [1e5, np.nan, -1e5]})
    model[:2].transform(held)
    assert np.array_equal(model[0].statistics_, med)
    assert np.array_equal(model[1].mean_, mean)
    assert np.allclose(med, np.nanmedian(train[['x']], axis=0))


def test_paired_resampling_preserves_equal_predictions_and_fold_weights():
    result = paired_ibs([np.zeros(3), np.zeros(8)], 150, 123, 2)
    assert result['delta_ibs'] == 0 and result['ci95'] == [0, 0]
    result = paired_ibs([np.ones(3), np.full(8, -1)], 150, 123, 2)
    assert result['delta_ibs'] == pytest.approx(-5/11)
    assert result['ci95'] == pytest.approx([-5/11, -5/11])


def test_paired_bootstrap_matches_explicit_within_fold_reference():
    diffs = [np.array([-.1, .2, .3]), np.array([-.2, -.1, .2, .8])]
    b = 31; rng = np.random.default_rng(7)
    expected = np.zeros(b)
    for d in diffs:
        expected += d[rng.integers(0, len(d), (b, len(d)))].sum(axis=1) / 7
    result = paired_ibs(diffs, b, 7, 3)
    assert np.allclose(result['ci95'], np.quantile(expected, [.025, .975]))
    assert np.allclose(result['family_adjusted_ci'], np.quantile(expected, [.05/6, 1-.05/6]))


@pytest.mark.parametrize('field,value', [('allow_new_cnn_training', True), ('ridge_alpha', .1),
                                       ('tune_head_on_pooled_oof', True), ('allow_outer_model_selection', True)])
def test_unapproved_method_change_fails(options, field, value):
    options[field] = value
    with pytest.raises(ValueError, match='PROTOCOL_CHANGE'):
        validate(options)


def test_output_cannot_overlap_parent(options, tmp_path):
    options['output_root'] = options['parent_root']
    with pytest.raises(ValueError):
        output_path(tmp_path, options, False)
    options['output_root'] = '../outside'
    with pytest.raises(ValueError):
        output_path(tmp_path, options, False)


def test_receipt_and_model_tampering_fail(tmp_path):
    units = Units(tmp_path, {'test': 1})
    model = tmp_path / 'private/model.bin'; model.write_bytes(b'original')
    units.job('head', lambda: {'path': 'private/model.bin', 'sha256': file_sha(model)})
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='MODEL_DRIFT'):
        units.read('head')


def test_changed_protocol_cannot_resume_existing_run(tmp_path):
    Units(tmp_path, {'version': 1})
    with pytest.raises(ValueError, match='CONTRACT_MISMATCH'):
        Units(tmp_path, {'version': 2})


def test_precision_mixing_rejected(tmp_path):
    parent = SimpleNamespace(binding='base', execution_binding='executed')
    doc = {'parent_contract': 'base', 'parent_execution_contract': 'executed',
           'math': {'cnn_precision': 'FP32', 'amp': True, 'tf32': False}, 'cnn_weights_frozen': True}
    atomic_json(tmp_path / 'private/contract.json', doc)
    with pytest.raises(ValueError, match='MIXED_PRECISION'):
        FrozenUnits(tmp_path, parent)


class SyntheticInputs:
    def __init__(self, options):
        self.options = options
        self.config = {'training_seeds': [13], 'fusion': {'ibs_grid_days': [365., 730., 1095.]}}
        rng = np.random.default_rng(19)
        all_columns = set(c for cols in arms('mri').values() for c in cols)
        frame = pd.DataFrame({col: rng.normal(size=180) for col in sorted(all_columns)})
        frame['patient_hash'] = [f'SYNTHETIC_{j}' for j in range(180)]
        frame['event'] = rng.random(180) < .5
        frame['time_days'] = rng.uniform(100, 1800, 180)
        self.train = frame.iloc[:150].copy().sort_values('patient_hash').reset_index(drop=True)
        self.test = frame.iloc[150:].copy().sort_values('patient_hash').reset_index(drop=True)
        self.train.loc[149, 'time_days'] = 1900
    def frames_for_head(self, track, seed, fold):
        return self.train.copy(), ['component0', 'component1', 'component2']
    def split_frame(self, track, fold, role):
        return (self.train if role == 'train' else self.test).copy()
    def validate_task(self, task):
        return {'task': task, 'meta_is_disjoint': True}
    def outer_for_head(self, track, task, fold):
        return self.test.drop(columns=['event', 'time_days'])


def test_real_head_fit_predict_resume_and_missing_head_detection(tmp_path, options):
    inputs = SyntheticInputs(options); units = Units(tmp_path, {'scope': 'synthetic'})
    first = fit_heads(inputs, units, 'mri', 13, 0)
    result = predict_head(inputs, units, 'mri', 13, 0)
    before = {k: r['sha256'] for k, r in first['heads'].items()}
    # Replaying must recover exact artifacts, never refit to changed held outcomes.
    inputs.test['event'] = ~inputs.test['event']; inputs.test['time_days'] += 50
    repeat = fit_heads(inputs, units, 'mri', 13, 0)
    again = predict_head(inputs, units, 'mri', 13, 0)
    assert before == {k: r['sha256'] for k, r in repeat['heads'].items()}
    assert result == again
    units.job('all_heads_locked', lambda: {'head_count': 2})
    units.job('smoke_complete', lambda: {'smoke': True})
    assert verify(units, inputs, smoke=True)['status'] == 'SMOKE_VERIFIED'
    name = head_name('mri', 13, 0) + '_clinical_amyloid'
    units.path(name).unlink()
    with pytest.raises(ValueError, match='REQUIRED_HEAD_RECEIPT'):
        verify(units, inputs, smoke=True)


def test_all_planned_contrasts_have_valid_arms():
    for track, base, aug, family, size in contrasts():
        assert base.removeprefix('split_') in arms(track)
        assert aug in arms(track)
        assert size == (3 if family == 'biomarker' else 2)


class FullSyntheticInputs:
    """All four routes and five outer folds, with no real records or CNN inference."""
    def __init__(self, options):
        self.options = options
        self.config = {'training_seeds': [13, 29, 47], 'fusion': {
            'ibs_grid_days': [365., 730., 1095.], 'horizons_days': [547.875, 1095.], 'min_at_risk': 20}}
        rng = np.random.default_rng(2026); n = 150
        features = sorted({c for tr in options['tracks'] for cs in arms(tr).values() for c in cs})
        f = pd.DataFrame({c: rng.normal(size=n) for c in features})
        f['patient_hash'] = [f'SYNTHETIC_COMPLETE_{j:03d}' for j in range(n)]
        f['event'] = rng.random(n) < .55
        f['time_days'] = rng.uniform(20, 1700, n)
        for fold in range(5):
            f.loc[fold * 30, 'time_days'] = 50
            f.loc[fold * 30 + 29, 'time_days'] = 1800
            f.loc[fold * 30 + 29, 'event'] = False
        self.frame = f
        self.frames = {tr: f.copy() for tr in options['tracks']}
        self.splits = [{'train': list(f.loc[(f.index // 30) != fold, 'patient_hash']),
                        'test': list(f.loc[(f.index // 30) == fold, 'patient_hash'])} for fold in range(5)]
        self.scores = SimpleNamespace(read=self.legacy)
    def split_frame(self, track, fold, role):
        keys = self.splits[fold]['train' if role == 'train' else 'test']
        return self.frame.loc[self.frame.patient_hash.isin(keys)].copy().reset_index(drop=True)
    def frames_for_head(self, track, seed, fold):
        return self.split_frame(track, fold, 'train'), [f'{track}_{fold}_{c}' for c in range(3)]
    def validate_task(self, task):
        return {'task': task, 'synthetic_only': True}
    def outer_for_head(self, track, task, fold):
        return self.split_frame(track, fold, 'outer').drop(columns=['event', 'time_days'])
    def legacy(self, name):
        track = name.split('_')[0]; fold = int(name.split('_o')[1].split('_')[0])
        test = self.split_frame(track, fold, 'outer'); keys = list(test.patient_hash)
        selected = ['clinical_amyloid', 'clinical_mri_amyloid'] if track == 'mri' else ['clinical_mri_amyloid', 'clinical_mri_amyloid_pet']
        return {'payload': {'fold': fold, 'test_subjects': keys,
                'arms': {a: {'subjects': keys, 'curves': np.tile([.94, .86, .76], (len(keys), 1)).tolist()} for a in selected}}}


def test_full_five_fold_four_track_execution_and_reporting(tmp_path, options):
    from scripts.xyz_oof_stacking_v1.pipeline import run_fit
    from scripts.xyz_oof_stacking_v1.reporting import report
    from threadpoolctl import threadpool_limits
    inputs = FullSyntheticInputs(options); units = Units(tmp_path, {'scope': 'ALL_SYNTHETIC_ONLY'})
    units.job('preflight', lambda: {'support': []})
    with threadpool_limits(2):
        result = run_fit(inputs, units)
        assert result['head_count'] == 180 and result['outer_cells'] == 20
        report(inputs, units)
    receipt = verify(units, inputs)
    assert receipt['status'] == 'OOF_EXECUTION_VERIFIED_NOT_SCIENTIFIC_VALIDATION'
    public = json.loads((tmp_path / 'public/aggregate_results.json').read_text())
    assert len(public['comparisons']) == 7
    assert set(public['arms']) == {'mri', 'pet', 'blood', 'tau'}
    assert 'SYNTHETIC_COMPLETE' not in (tmp_path / 'public/aggregate_results.json').read_text()
    for item in public['comparisons']:
        assert item['resamples'] == 40000
    # A partial set of folds cannot be promoted merely because a report file exists.
    units.path('tau_o4_evaluation').unlink()
    with pytest.raises(ValueError, match='REQUIRED_EVALUATION'):
        verify(units, inputs)
