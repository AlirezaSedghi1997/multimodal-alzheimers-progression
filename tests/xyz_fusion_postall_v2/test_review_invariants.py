"""Adversarial fixtures only. Never mirror, delete or mutate live results."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.xyz_fusion_postall_v2.core import Units, ReadOnlyParent, atomic_json, digest, file_sha, inside, parent_exclusion
from scripts.xyz_fusion_postall_v2.audit import incremental_verdict, metric_replay
from scripts.xyz_fusion_postall_v2.secondary import subject_standardize
from scripts.xyz_fusion_postall_v2.spatial_target import cox_risk, ensemble_mean
from scripts.xyz_fusion_postall_v2.release import assert_public, closure, public_report


def test_resume_does_not_recompute_committed_unit(tmp_path):
    store = Units(tmp_path, {'config': 'fixed'})
    counter = []
    assert store.job('a', lambda: counter.append(1) or {'status': 'DONE'})['status'] == 'DONE'
    assert store.job('a', lambda: pytest.fail('recomputed'))['status'] == 'DONE'
    assert len(counter) == 1


def test_failed_unit_can_resume_without_repeating_previous(tmp_path):
    store = Units(tmp_path, {'config': 1})
    store.job('previous', lambda: {'status': 'DONE'})
    def fail():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        store.job('interrupted', fail)
    assert store.read('interrupted') is None
    assert store.job('previous', lambda: pytest.fail('repeated')) == {'status': 'DONE'}
    assert store.job('interrupted', lambda: {'status': 'RECOVERED'})['status'] == 'RECOVERED'


@pytest.mark.parametrize('kind', ['source', 'runtime', 'policy'])
def test_contract_drift_never_deletes_old_outputs(tmp_path, kind):
    original = {kind: 'a'}
    store = Units(tmp_path, original)
    store.job('done', lambda: {'status': 'DONE'})
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='CONTRACT_MISMATCH'):
        Units(tmp_path, {kind: 'b'})
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_binary_and_model_drift_fail_closed(tmp_path):
    store = Units(tmp_path, {})
    p = tmp_path / 'model.pkl'
    p.write_bytes(b'abc')
    store.job('head', lambda: {'path': 'model.pkl', 'sha256': file_sha(p)})
    store.job('map', lambda: {'outputs': {'model.pkl': file_sha(p)}})
    p.write_bytes(b'abd')
    with pytest.raises(ValueError, match='MODEL_DRIFT'):
        store.read('head')
    with pytest.raises(ValueError, match='BINARY_DRIFT'):
        store.read('map')


def test_dependency_drift_is_not_resume(tmp_path):
    store = Units(tmp_path, {})
    store.job('a', lambda: {'value': 1}, {'parent': 'sha1'})
    with pytest.raises(ValueError, match='DEPENDENCY_DRIFT'):
        store.job('a', lambda: {'value': 2}, {'parent': 'sha2'})


def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match='ESCAPES_ROOT'):
        inside(tmp_path, '../outside')


def test_parent_adapter_never_invokes_recovery_writer(tmp_path, monkeypatch):
    from scripts.xyz_fusion_v2 import verification
    contract = {'protocol': {}}
    atomic_json(tmp_path / 'private/contract.json', contract)
    payload = {'status': 'READY'}
    atomic_json(tmp_path / 'private/jobs/unit.json', {'contract': digest(contract), 'payload': payload, 'payload_sha256': digest(payload)})
    journal = tmp_path / 'private/access_journal/unit.json'
    atomic_json(journal, {'state': 'COMMITTED', 'payload_sha256': digest(payload)})
    before = (journal.read_bytes(), journal.stat().st_mtime_ns)
    def fake_verify(store, config, *, final):
        return checked_job(store, 'unit')  # resolved in the cloned verifier namespace
    fake_verify.__module__ = verification.__name__
    monkeypatch.setattr(verification, 'verify', fake_verify)
    assert ReadOnlyParent(tmp_path).verify() == payload
    assert before == (journal.read_bytes(), journal.stat().st_mtime_ns)


def test_parent_write_callback_forbidden(tmp_path):
    atomic_json(tmp_path / 'private/contract.json', {})
    with pytest.raises(ValueError, match='COMPUTATION_FORBIDDEN'):
        ReadOnlyParent(tmp_path).job('new', lambda: {})


def test_parent_lock_not_created_or_modified(tmp_path):
    with pytest.raises(ValueError, match='EXISTING_PARENT'):
        with parent_exclusion(tmp_path):
            pass
    assert not (tmp_path / '.writer.lock').exists()
    p = tmp_path / '.writer.lock'
    p.write_bytes(b'0')
    before = (p.read_bytes(), p.stat().st_mtime_ns)
    with parent_exclusion(tmp_path):
        with pytest.raises((OSError, PermissionError)):
            with parent_exclusion(tmp_path):
                pass
    assert before == (p.read_bytes(), p.stat().st_mtime_ns)


@pytest.mark.parametrize('uno', [None, [-.1, .2], [-.001, 0]])
def test_positive_ibs_does_not_override_missing_or_adverse_uno_ci(uno):
    value = {'delta_ibs_bonferroni_ci': [-.02, -.01], 'delta_uno_bonferroni_ci': uno}
    assert incremental_verdict(value) != 'INTERNAL_EXPLORATORY_SUPPORT'


def test_positive_gate_requires_both_full_intervals():
    assert incremental_verdict({'delta_ibs_bonferroni_ci': [-.02, -.01], 'delta_uno_bonferroni_ci': [0, .02]}) == 'INTERNAL_EXPLORATORY_SUPPORT'


def test_standardization_is_not_visit_weighted():
    table = pd.DataFrame({'patient_hash': ['a'] * 10 + ['b'] * 2, 'score': [0.] * 10 + [2.] * 2})
    standardized, scale = subject_standardize(table, 'score')
    assert scale['mean'] == 1.
    assert np.isclose(standardized.iloc[0], -1 / np.sqrt(2))


def test_baseline_changing_within_subject_is_rejected():
    table = pd.DataFrame({'patient_hash': ['a', 'a'], 'score': [1, 2]})
    with pytest.raises(ValueError, match='CHANGES_WITHIN_SUBJECT'):
        subject_standardize(table, 'score')


def fake_head(coef, mean=(1., 2.), scale=(2., 4.), h0=.3):
    return SimpleNamespace(steps=[('imputer', SimpleNamespace(statistics_=np.array(mean), strategy='median', add_indicator=False)),
        ('scaler', SimpleNamespace(mean_=np.array(mean), scale_=np.array(scale))),
        ('cox', SimpleNamespace(coef_=np.array(coef), cum_baseline_hazard_=lambda _: h0))])


def test_full_cox_risk_and_gradient_match_closed_form():
    x = torch.tensor([[3., 6.]], dtype=torch.float64, requires_grad=True)
    head = fake_head([.2, -.3])
    result = cox_risk(head, x, 3)
    eta = -.1
    expected = 1 - np.exp(-.3 * np.exp(eta))
    assert np.isclose(result.item(), expected, atol=1e-14)
    result.sum().backward()
    assert np.allclose(x.grad.numpy(), .3 * np.exp(eta) * (1 - expected) * np.array([[.2/2, -.3/4]]), atol=1e-14)


def test_cox_imputation_exact_and_each_ensemble_head_is_used():
    x = torch.tensor([[float('nan'), 6.]], dtype=torch.float64)
    heads = [fake_head([.2, -.3], h0=.1), fake_head([.9, .7], h0=.8)]
    targets = [lambda value, h=h: cox_risk(h, value, 3).sum() for h in heads]
    expected = sum((1 - np.exp(-h.steps[-1][1].cum_baseline_hazard_(3) * np.exp(h.steps[-1][1].coef_[1]))) for h in heads) / 2
    assert np.isclose(ensemble_mean(targets, x).item(), expected)
    assert not np.isclose(ensemble_mean(targets, x).item(), targets[0](x).item())


def test_real_sklearn_cox_pipeline_prediction_parity():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 2)); x[3, 1] = np.nan
    y = Surv.from_arrays(rng.random(80) < .6, rng.uniform(1, 9, 80))
    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), CoxPHSurvivalAnalysis(alpha=10))
    model.fit(x, y)
    expected = np.array([1 - f(5.) for f in model.predict_survival_function(x)])
    actual = cox_risk(model, torch.tensor(x), 5.).numpy()
    assert np.allclose(expected, actual, atol=1e-12, rtol=0)


@pytest.mark.parametrize('value', [{'nested': {'censoring_reference': []}}, {'a': {'event': [1, 0]}},
                                  {'map': 'volume'}, {'x': r'H:\private\file'}, {'a': {'patient_hash': 'x'}},
                                  {'a': [list(range(2))] * 3}, {'arbitrary': list(range(61))}])
def test_recursive_public_privacy(value):
    with pytest.raises(ValueError, match='PUBLIC_'):
        assert_public(value)


def test_aggregate_allowed():
    assert_public({'fold_metrics': [{'events': 10, 'n': 100, 'ibs': .15}], 'ci95': [-.03, .01]})


def test_incomplete_tasks_do_not_finalize(tmp_path):
    units = Units(tmp_path, {})
    for task in ('f00', 'f01', 'f02', 'f03', 'f05', 'f06', 'f07'):
        units.job(task, lambda: {'status': 'COMPLETED'})
    assert closure(units)['missing_tasks'] == ['f04']
    with pytest.raises(ValueError, match='FINALIZATION_REQUIRES'):
        public_report(units)


def test_probe_support_uses_scanner_classes_not_disease_events():
    from scripts.xyz_fusion_postall_v2.robustness import probe_support, probe_arrays
    tr = np.array(['GE'] * 10 + ['SIEMENS'] * 10)
    te = np.array(['GE'] * 5 + ['SIEMENS'] * 5)
    assert probe_support(tr, te)[0] == ['GE', 'SIEMENS']
    rng = np.random.default_rng(4)
    result = probe_arrays(rng.normal(size=(20, 3)), tr, rng.normal(size=(10, 3)), te, 13)
    assert result['status'] == 'HELDOUT_SCANNER_PROBE_COMPLETED'
    assert 'balanced_accuracy' in result and 'macro_ovr_auc' in result


def test_synthetic_ipcw_metric_replay_and_tamper():
    from scripts.xyz_fusion_postall_v2.statistics import metrics
    rng = np.random.default_rng(1)
    tr = pd.DataFrame({'event': rng.integers(0, 2, 100), 'time_days': np.linspace(1, 20, 100)})
    te = pd.DataFrame({'event': rng.integers(0, 2, 40), 'time_days': np.linspace(2, 19, 40)})
    grid = [3., 5., 7.]
    curves = np.tile([.9, .8, .7], (40, 1))
    m = metrics(tr, te, curves, grid, [5.], 5)
    assert m['horizons']['5.0']['status'] == 'NOT_ESTIMABLE'
    assert m['horizons']['5.0']['dynamic_auc'] == .5
    arm = {'event': te.event.tolist(), 'time': te.time_days.tolist(), 'curves': curves.tolist(),
           'censor_train': {'event': tr.event.tolist(), 'time': tr.time_days.tolist()}, 'metrics': m}
    config = {'fusion': {'ibs_grid_days': grid, 'horizons_days': [5.], 'min_at_risk': 5}}
    assert metric_replay([{'fold': 0, 'arms': {'a': arm}}], config)['status'] == 'ALL_STORED_METRICS_REPRODUCED'
    arm['metrics']['ibs'] += .01
    with pytest.raises(ValueError, match='METRIC_REPLAY_MISMATCH'):
        metric_replay([{'fold': 0, 'arms': {'a': arm}}], config)


def test_roi_perturbation_preserves_random_control_volume():
    from scripts.xyz_fusion_postall_v2.roi_fidelity import region_controls
    shape = (8, 8, 8)
    labels = np.zeros(shape, dtype=int); labels[1:3, 1:3, 1:3] = 1
    labels[5:7, 5:7, 5:7] = 2
    image = torch.ones((1, 1, *shape))
    baseline = torch.zeros_like(image)
    weights = torch.tensor((labels == 1).astype(float)).reshape_as(image)
    seen = []
    def forward(x):
        seen.append(int((x == 0).sum()))
        return (x * weights).sum()
    a = weights.numpy().squeeze()
    value = region_controls(forward, image, baseline, a, np.ones(shape, bool), labels, seed=1, repeats=4)
    assert value['positive']['atlas_label'] == 1
    assert set(seen[1:]) == {8}
    assert value['positive']['directional_margin_vs_random_mean'] > 0


def test_no_new_cnn_training_callable_in_postall_package():
    import ast
    folder = Path(__file__).resolve().parents[2] / 'scripts/xyz_fusion_postall_v2'
    forbidden = {'train_expert', 'paired_crossfit', 'run_tabular'}
    for path in folder.glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (forbidden & {name.name for name in node.names}), path


def test_pet_scanner_support_uses_pet_not_mri_labels():
    from scripts.xyz_fusion_postall_v2.robustness import pet_scanner_splits
    frame = pd.DataFrame({'patient_hash': [str(i) for i in range(100)], 'event': [1] * 50 + [0] * 50,
                          'pet_manufacturer': ['GE'] * 25 + ['SIEMENS'] * 25 + ['GE'] * 25 + ['SIEMENS'] * 25,
                          'mri_manufacturer': ['WRONG'] * 100})
    config = {'secondary': {'min_group_subjects': 30, 'min_group_events': 10}}
    # Other-group events must reach 30; the global group rule must not be weakened.
    splits, excluded = pet_scanner_splits(frame, config)
    assert not splits and len(excluded) == 2
    frame['event'] = 1
    splits, excluded = pet_scanner_splits(frame, config)
    assert {s['group'] for s in splits} == {'GE', 'SIEMENS'}
    assert all(set(s['train']).isdisjoint(s['test']) for s in splits)


def test_unexpected_public_file_is_never_adopted(tmp_path):
    from scripts.xyz_fusion_postall_v2.release import require_public_file_whitelist
    (tmp_path / 'unreviewed.csv').write_text('id,value\nsecret,.5', encoding='utf-8')
    with pytest.raises(ValueError, match='UNAPPROVED_PUBLIC_FILE'):
        require_public_file_whitelist(tmp_path)


@pytest.mark.parametrize('out', ['results/xyz_fusion_v2/child', 'results',
                                'results/xyz_fusion_postall_v1/child'])
def test_no_child_or_ancestor_output_of_protected_results(monkeypatch, out):
    from scripts.xyz_fusion_postall_v2 import runner
    monkeypatch.setattr(runner, 'read_json', lambda _: {'allow_new_cnn_training': False,
                        'parent_root': 'results/xyz_fusion_v2', 'output_root': out})
    with pytest.raises(ValueError, match='SEPARATE_VERSIONED_OUTPUT_REQUIRED'):
        runner.execute(Path('H:/New project'), 'preflight', 'unused')


def test_saturated_cox_risk_has_finite_input_gradient():
    x = torch.tensor([[1e6, -1e6]], dtype=torch.float64, requires_grad=True)
    risk = cox_risk(fake_head([1., -1.]), x, 3.)
    risk.sum().backward()
    assert risk.item() == 1. and torch.isfinite(x.grad).all()


def test_fixed_parent_batch_keeps_other_images_constant():
    from scripts.xyz_fusion_postall_v2.spatial_target import FixedInferenceBatch
    class Model(torch.nn.Module):
        def forward(self, x):
            # Batch dependence is explicit in this adversarial fixture.
            return x.flatten(1).sum(1, keepdim=True) + len(x)
    context = torch.zeros((4, 1, 2, 2, 2))
    model = FixedInferenceBatch(Model(), context, 2)
    image = torch.ones((1, 1, 2, 2, 2), requires_grad=True)
    value = model(image)
    value.sum().backward()
    assert value.item() == 12. and torch.equal(image.grad, torch.ones_like(image))
    assert not context.requires_grad and context.sum() == 0


def test_scaled_ig_recovers_unscaled_linear_attributions():
    from scripts.xyz_fusion_postall_v2.integrated_gradients import integrated_gradients
    x = torch.ones((1, 1, 2, 2, 2))
    value, status = integrated_gradients(lambda a: a.sum() * .02, x, torch.zeros_like(x),
                                        steps=(4,), gradient_scale=1024.)
    assert np.allclose(value, .02) and status['numerical_status'] == 'PASS'
    assert status['completeness_in_original_risk_units']


def test_supersession_requires_actual_post_run_review(tmp_path, monkeypatch):
    from scripts.xyz_fusion_postall_v2 import supersession
    monkeypatch.setattr(supersession, 'candidate', lambda _: {'f08_digest': 'locked'})
    ctx = SimpleNamespace(out=tmp_path, units=SimpleNamespace(binding='contract'))
    with pytest.raises(ValueError, match='REVIEW_NOT_YET_RECORDED'):
        supersession.promote(ctx)


def test_small_matched_heads_serialize_and_resume(tmp_path):
    from scripts.xyz_fusion_v2.pipeline import locked_tabular
    rng = np.random.default_rng(18)
    frame = pd.DataFrame({'patient_hash': [str(i) for i in range(60)], 'event': [1, 0] * 30,
                          'time_days': np.arange(60) + 10., 'age': rng.normal(70, 5, 60)})
    units = Units(tmp_path, {'test': 'matched_heads'})
    model, receipt = locked_tabular(units, 'head', frame, ['age'], 'cox_ridge', 10., {}, 13)
    again, recovered = locked_tabular(units, 'head', frame, ['age'], 'cox_ridge', 10., {}, 13)
    assert receipt == recovered and np.array_equal(model.predict(frame[['age']]), again.predict(frame[['age']]))
    changed = frame.copy(); changed.loc[0, 'event'] = 0
    with pytest.raises(ValueError, match='TABULAR_LOCK_DRIFT'):
        locked_tabular(units, 'head', changed, ['age'], 'cox_ridge', 10., {}, 13)


def test_public_release_uses_only_whitelisted_aggregate_artifacts(tmp_path):
    from scripts.xyz_fusion_postall_v2.release import write_release, require_public_file_whitelist
    units = Units(tmp_path, {'fixture': True})
    for name, value in {
        'f00': {'status': 'PASS', 'locked_image_checkpoints': 410, 'prediction_journals': 133, 'parent_writes': 0},
        'f02': {'status': 'PASS', 'intersection_counts': {'mri': {'n': 100, 'events': 25}}},
        'f03': {'status': 'NOT_ESTIMABLE_SUPPORT'},
        'f04': {'status': 'DIAGNOSTIC', 'cases': [], 'requested_units': 30, 'eligible_units': 0},
        'f05': {'status': 'PROBES', 'probes': {'units': []}, 'lmo': {'groups': [], 'site_lmo_status': 'NOT_ESTIMABLE'}},
        'f06': {'status': 'NOT_ESTIMABLE_SUPPORT'},
        'f07': {'status': 'NOT_ESTIMABLE_SUPPORT'}}.items():
        units.job(name, lambda value=value: value)
    metrics = [{'fold': f, 'arm': arm, 'horizons': {'1095.75': {'calibration_bins': [
        {'predicted_risk': .1, 'ipcw_observed_fraction': .15, 'n': 20}]}}}
        for f in range(5) for arm in ('clinical', 'clinical_amyloid')]
    comparison = {'base': 'clinical', 'augmented': 'clinical_amyloid', 'delta_ibs_mean': -.001,
                  'delta_ibs_ci95': [-.003, .002], 'verdict': 'INCREMENTAL_VALUE_NOT_DEMONSTRATED'}
    analysis = {'metrics': {'fold_metrics': metrics}, 'arm_summary': {
        arm: {'n': 100, 'events': 25, 'ibs': .1, 'uno_c_index': .7, 'ibs_conditional_ci95': [.08, .12]}
        for arm in ('clinical', 'clinical_amyloid')}, 'paired_comparisons': [comparison]}
    units.job('f01', lambda: {'status': 'COMPLETED', 'analyses': {'mri': analysis, 'pet': analysis},
                              'subgroups': {}, 'sensitivity': {}})
    result = write_release(SimpleNamespace(out=tmp_path, units=units))
    require_public_file_whitelist(tmp_path / 'public', complete=True)
    assert result['privacy'] == 'PASS' and result['positive_or_external_claim'] is False


def test_pacc_unchanged_random_slope_model_recovers_known_association():
    from scripts.xyz_fusion_postall_v2.secondary import fit_pacc_models
    rng = np.random.default_rng(24)
    n, visits = 80, 5
    table = pd.DataFrame({'patient_hash': np.repeat(np.arange(n).astype(str), visits),
                          'years': np.tile(np.linspace(.3, 4.5, visits), n)})
    for col in ('age_z', 'education_years_z', 'baseline_pacc_raw_z', 'baseline_mmscore_z', 'oof_risk_3y_z'):
        table[col] = np.repeat(rng.normal(size=n), visits)
    for col in ('sex_binary', 'apoe4_carrier', 'treatment_active'):
        table[col] = np.repeat(rng.integers(0, 2, n), visits)
    table['pacc'] = (table.baseline_pacc_raw_z - .6 * table.years * table.oof_risk_3y_z
                     + np.repeat(rng.normal(size=n), visits)
                     + table.years * np.repeat(rng.normal(scale=.4, size=n), visits)
                     + rng.normal(scale=.3, size=n * visits))
    result = fit_pacc_models(table)
    assert all(v['status'] == 'ESTIMATED' for v in result.values())
    coef = result['augmented']['coefficients']['years:oof_risk_3y_z']
    assert -.8 < coef['estimate'] < -.4


def test_roi_occlusion_exact_risk_difference_not_additive_voxel_claim():
    from scripts.xyz_fusion_postall_v2.occlusion_maps import roi_effects
    labels = np.zeros((2, 2, 2), int); labels[0] = 1; labels[1] = 2
    x = torch.ones((1, 1, 2, 2, 2))
    targets = [lambda value: torch.sigmoid(value.sum() - 4)]
    image, rows = roi_effects(targets, x, torch.zeros_like(x), labels, np.ones_like(labels, bool))
    expected = float(torch.sigmoid(torch.tensor(4.))) - .5
    assert len(rows) == 2 and np.allclose(image, expected)
    assert all(np.isclose(r['signed_risk_difference'], expected) for r in rows)
    # The displayed ROI value is a finite model-risk change, not a sum of voxel causes.
    assert not np.isclose(image.sum(), float(targets[0](x) - targets[0](torch.zeros_like(x))))


def test_random_weight_control_does_not_keep_custom_learned_parameters(tmp_path, monkeypatch):
    from scripts.xyz_fusion_postall_v2.experts import load_expert
    from scripts.xyz_fusion_v2 import training
    class Net(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.scale_gates = torch.nn.Parameter(torch.zeros(4))
            self.layer = torch.nn.Linear(4, 1)
    monkeypatch.setattr(training, 'MRISurvival', Net)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    original = Net({})
    original.scale_gates.data.fill_(7.)
    task = 'mri_s13_o0_c0'
    path = tmp_path / 'private/models' / (task + '_mri_expert') / 'locked.pt'
    path.parent.mkdir(parents=True)
    torch.save({'model': original.state_dict()}, path)
    parent = SimpleNamespace(root=tmp_path, hashes=file_sha,
         job=lambda _: {'experts': {'mri': {'sha256': file_sha(path)}}})
    ctx = SimpleNamespace(parent=parent, config={'training': {}, 'seed': 13})
    randomized, _ = load_expert(ctx, task, 'mri', randomized=True)
    assert torch.equal(randomized.scale_gates, torch.zeros(4))
    assert not torch.equal(randomized.layer.weight, original.layer.weight)
    assert not any(p.requires_grad for p in randomized.parameters())


def test_analytic_survival_backward_matches_cumulative_product():
    from scripts.xyz_fusion_postall_v2.spatial_target import LockedSurvivalCurve
    logits = torch.tensor([[.8, -2., 4., -10.], [-1., .5, 3., 1.]], requires_grad=True)
    weights = torch.tensor([[1., 2., 3., 4.], [-2., 3., -1., .5]])
    reference = torch.cumprod(torch.sigmoid(-logits), dim=1)
    expected, = torch.autograd.grad((reference * weights).sum(), logits)
    actual = LockedSurvivalCurve.apply(logits)
    observed, = torch.autograd.grad((actual * weights).sum(), logits)
    assert np.array_equal(reference.detach().numpy().astype(float), actual.detach().numpy())
    assert torch.allclose(observed, expected, rtol=1e-6, atol=1e-8)
