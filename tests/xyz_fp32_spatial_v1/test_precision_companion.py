import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts.xyz_fp32_spatial_v1.numerics import quadrature, compare_maps, converged
from scripts.xyz_fp32_spatial_v1.pipeline import evaluate_committed, task_names, chosen_arms, ancestry
from scripts.xyz_fp32_spatial_v1.runner import make_contract, verify
from scripts.xyz_fusion_postall_v2.core import Units, atomic_json, read_json, digest

ROOT = Path(__file__).resolve().parents[2]
OPTIONS = json.loads((ROOT / 'configs/xyz_fp32_spatial_sensitivity_v1.json').read_text())


def test_fp32_completeness_and_two_level_stability():
    x = torch.linspace(0, 1, 64).reshape(1, 1, 4, 4, 4)
    b = torch.zeros_like(x)
    f = lambda y: torch.sigmoid((y * y).mean())
    a, n1 = quadrature(f, x, b, 8, OPTIONS)
    z, n2 = quadrature(f, x, b, 16, OPTIONS)
    assert n1['numerical_status'] == n2['numerical_status'] == 'PASS'
    assert not converged([n1], compare_maps(a, z), OPTIONS)
    assert converged([n1, n2], compare_maps(a, z), OPTIONS)
    assert n1['attribution_rescaled_to_force_completeness'] is False


def test_wrong_gradient_cannot_pass_by_increasing_steps():
    class Wrong(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x): return x.sum()
        @staticmethod
        def backward(ctx, g): return g.expand(1, 1, 2, 2, 2) * .5
    x = torch.ones(1, 1, 2, 2, 2); b = torch.zeros_like(x)
    a, n1 = quadrature(Wrong.apply, x, b, 8, OPTIONS)
    z, n2 = quadrature(Wrong.apply, x, b, 16, OPTIONS)
    assert n1['numerical_status'] == 'FAIL_COMPLETENESS'
    assert not converged([n1, n2], compare_maps(a, z), OPTIONS)


def test_zero_control_not_passing_sanity():
    x = torch.ones(1, 1, 2, 2, 2)
    _, n = quadrature(lambda z: z.sum() * 0, x, x * 0, 8, OPTIONS)
    assert n['numerical_status'] == 'NOT_ESTIMABLE_ZERO_MAP'


def test_nonfinite_rejected():
    x = torch.ones(1, 1, 2, 2, 2); x.flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='FINITE_FP32'):
        quadrature(lambda z: z.sum(), x, x * 0, 8, OPTIONS)


def test_contract_and_binary_drift_fail_closed(tmp_path):
    units = Units(tmp_path, {'precision': 'FP32'})
    units.job('complete', lambda: {'value': 2})
    assert units.job('complete', lambda: pytest.fail('must not rerun')) == {'value': 2}
    with pytest.raises(ValueError, match='CONTRACT_MISMATCH'):
        Units(tmp_path, {'precision': 'AMP'})
    p = tmp_path / 'private/artifact.bin'; p.write_bytes(b'a')
    from scripts.xyz_fusion_postall_v2.core import file_sha
    units.job('binary', lambda: {'outputs': {'private/artifact.bin': file_sha(p)}})
    p.write_bytes(b'b')
    with pytest.raises(ValueError, match='BINARY_DRIFT'):
        units.read('binary')


def test_evaluation_resume_after_result_before_journal(tmp_path):
    units = Units(tmp_path, {'protocol': 1}); dep = {'heads': 'fixed'}; calls = []
    result = evaluate_committed(units, 'eval', dep, lambda: calls.append(1) or {'ibs': .05})
    j = tmp_path / 'private/evaluation_journal/eval.json'
    record = read_json(j); record['state'] = 'IN_PROGRESS'; atomic_json(j, record)
    assert evaluate_committed(units, 'eval', dep, lambda: pytest.fail('second evaluation')) == result
    assert calls == [1] and read_json(j)['state'] == 'COMMITTED'
    with pytest.raises(ValueError):
        evaluate_committed(units, 'eval', {'heads': 'changed'}, lambda: {})


def test_interrupted_uncommitted_evaluation_replay_is_counted(tmp_path):
    units = Units(tmp_path, {'protocol': 1}); dep = {'heads': 'fixed'}
    def interrupt(): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): evaluate_committed(units, 'eval', dep, interrupt)
    evaluate_committed(units, 'eval', dep, lambda: {'ibs': .05})
    assert read_json(tmp_path / 'private/evaluation_journal/eval.json')['attempts'] == 2


def test_all_expected_component_and_head_counts():
    conf = {'training_seeds': [13, 29, 47], 'components': 3}
    assert len(task_names(conf, 'mri')) == 45
    assert len(task_names(conf, 'pet', 2)) == 9
    assert chosen_arms('pet') == ['clinical_mri_amyloid', 'clinical_mri_amyloid_pet']


def test_meta_outer_overlap_rejected_before_file_loading():
    lock = {'expert_subjects': ['a'], 'meta_subjects': ['b'], 'outer_subjects': ['b'], 'experts': {}}
    ctx = SimpleNamespace(parent=SimpleNamespace(job=lambda name: lock))
    with pytest.raises(ValueError, match='ANCESTRY_OVERLAP'):
        ancestry(ctx, 'fixture')


def test_code_byte_changes_contract(tmp_path):
    for p in ['scripts/run_xyz_fp32_spatial_v1.py', 'scripts/run_xyz_fp32_spatial_v1.ps1',
              'src/evaluation/signed_spatial_validation_v1.py']:
        f = tmp_path / p; f.parent.mkdir(parents=True, exist_ok=True); f.write_text('a')
    parent = SimpleNamespace(binding='parent', execution_binding='execution')
    first = make_contract(tmp_path, parent, OPTIONS)
    (tmp_path / 'scripts/run_xyz_fp32_spatial_v1.py').write_text('b')
    second = make_contract(tmp_path, parent, OPTIONS)
    assert digest(first) != digest(second)
    assert first['math']['amp'] is False and first['same_model_as_amp_parent'] is False


def test_incomplete_outputs_cannot_finalize(tmp_path, monkeypatch):
    from scripts.xyz_fp32_spatial_v1 import runner
    units = Units(tmp_path, {'protocol': 1})
    monkeypatch.setattr(runner, 'make_contract', lambda *args: {'protocol': 1})
    ctx = SimpleNamespace(units=units, out=tmp_path, root=tmp_path, parent=None, options={}, frame=None, pet=None,
        config={'outer_folds': 5, 'components': 3, 'training_seeds': [13, 29, 47]})
    result = verify(ctx)
    assert result['status'] == 'INCOMPLETE' and len(result['missing']) == 102


def test_aggregate_privacy_rejects_private_outcomes(tmp_path):
    from scripts.xyz_fusion_v2.privacy import scan_public_bundle
    atomic_json(tmp_path / 'results.json', {'patient_hash': 'private', 'event': [1], 'time_days': [100]})
    assert scan_public_bundle(tmp_path)


def test_entire_spatial_cell_outputs_are_readable_and_resumable(tmp_path, monkeypatch):
    import pandas as pd
    import nibabel as nib
    from scripts.xyz_fp32_spatial_v1 import spatial
    from scripts.xyz_fusion_postall_v2.core import file_sha
    units = Units(tmp_path / 'child', {'precision': 'FP32'})
    x = torch.linspace(.1, 1., 64).reshape(1, 1, 4, 4, 4)
    forward = lambda z: torch.sigmoid((z * z).mean())
    value = float(forward(x))
    task = 'mri_s13_o0_c0'; arm = 'clinical_mri_amyloid'
    units.job(task + '_child_lock', lambda: {'test': True})
    units.job(task + '_child_predictions', lambda: {'arms': {arm: {'subjects': ['fixture'],
                                                                 'survival': [[1 - value, 1 - value]]}}})
    units.job('mri_child_cells', lambda: {'cells': [{'arms': {arm: {'subjects': ['fixture'],
                                                                'curves': [[1 - value, 1 - value]]}}}]})
    atlas = np.ones((4, 4, 4), dtype='int16'); atlas[2:] = 2
    p = tmp_path / 'atlas.nii'; nib.save(nib.Nifti1Image(atlas, np.eye(4)), p)
    atomic_json(tmp_path / 'results/xyz_fusion_v2_assets/source_registry.json', {'labels': {'1': 'first', '2': 'second'}})
    parent = SimpleNamespace(root=tmp_path, job=lambda name: {
        'path': 'atlas.nii', 'status': 'APPROXIMATE_ANATOMICAL_LABELING_ONLY'})
    config = {'training_seeds': [13], 'components': 1, 'seed': 13, 'fusion': {'ibs_grid_days': [0, 1640]},
              'spatial': {'baseline_sigma': [1.]}}
    options = dict(OPTIONS, ig_steps=[8, 16], random_controls=3)
    bank = SimpleNamespace(batch=lambda rows, device: x.clone())
    ctx = SimpleNamespace(root=tmp_path, out=units.root, units=units, parent=parent, config=config,
        options=options, frame=pd.DataFrame([{'patient_hash': 'fixture', 'cache_row': 0}]), banks=lambda: {'mri': bank})
    monkeypatch.setattr(spatial, 'geometry', lambda *a: (nib.Nifti1Image(x.numpy().squeeze(), np.eye(4)), {'actual_subject': True}))
    monkeypatch.setattr(spatial, 'make_target', lambda *a, randomized=False: (lambda z: z.sum() * 0) if randomized else forward)
    result = spatial.case_maps(ctx, 0, 0, 'fixture', 'mri', 'mri')
    assert len(result['baselines']) == 2
    assert all(b['trained_ig_numeric'] == 'PASS' for b in result['baselines'].values())
    assert all(b['fully_validated_attribution'] is False for b in result['baselines'].values())
    for b in result['baselines'].values():
        assert nib.load(ctx.out / b['map']).shape == (4, 4, 4)
    original = {str(p): file_sha(p) for p in (ctx.out / 'private/spatial/components').glob('*.npz')}
    again = spatial.case_maps(ctx, 0, 0, 'fixture', 'mri', 'mri')
    assert original == {str(p): file_sha(p) for p in (ctx.out / 'private/spatial/components').glob('*.npz')}
    assert again['parity'] == result['parity']


def test_ram_storage_optimization_is_bitwise_equivalent(tmp_path, monkeypatch):
    import pandas as pd
    from scripts.xyz_fusion_v2 import training, inference
    monkeypatch.setattr(training, 'available_ram', lambda: 32 * 1024**3)
    monkeypatch.setattr(inference, 'available_ram', lambda: 32 * 1024**3)
    a = np.arange(3 * 8, dtype='f4').reshape(3, 1, 2, 2, 2)
    path = tmp_path / 'cache.npy'; np.save(path, a)
    slow, fast = training.ImageBank(path, 0), training.ImageBank(path, 12)
    assert slow.mode == 'MEMMAP' and fast.mode == 'RAM_ONCE'
    assert torch.equal(slow.batch([2, 0], torch.device('cpu')), fast.batch([2, 0], torch.device('cpu')))
    monkeypatch.setattr(inference.PETBank, '_read', lambda self, row: np.full((1, 128, 128, 128), row, dtype='f4'))
    frame = pd.DataFrame({'cache_row': [2, 9]})
    slow_pet, fast_pet = inference.PETBank(frame, 0), inference.PETBank(frame, 6)
    assert slow_pet.mode == 'HDF5_LAZY' and fast_pet.mode == 'RAM_ONCE'
    assert torch.equal(slow_pet.batch([9, 2], torch.device('cpu')), fast_pet.batch([9, 2], torch.device('cpu')))


def test_aggressive_ram_budget_keeps_reserved_memory(tmp_path, monkeypatch):
    import pandas as pd
    from scripts.xyz_fp32_spatial_v1 import resources
    free = 20 * 1024**3
    monkeypatch.setattr(resources, 'available_ram', lambda: free)
    assert resources.can_cache(12 * 1024**3, 24 * 1024**3, 4 * 1024**3)
    assert not resources.can_cache(17 * 1024**3, 24 * 1024**3, 4 * 1024**3)
    a = np.arange(24, dtype='f4').reshape(3, 1, 2, 2, 2)
    path = tmp_path / 'bank.npy'; np.save(path, a)
    ram = resources.FastMRI(path, 24 * 1024**3, 4 * 1024**3)
    mapped = resources.FastMRI(path, 0, 4 * 1024**3)
    assert ram.mode == 'RAM_ONCE' and mapped.mode == 'MEMMAP'
    assert torch.equal(ram.batch([1, 0], torch.device('cpu')), mapped.batch([1, 0], torch.device('cpu')))
    monkeypatch.setattr(resources.FastPET, '_read', lambda self, row: np.full((1, 128, 128, 128), row, dtype='f4'))
    frame = pd.DataFrame({'cache_row': [2, 9]})
    a = resources.FastPET(frame, 24 * 1024**3, 4 * 1024**3)
    b = resources.FastPET(frame, 0, 4 * 1024**3)
    assert a.mode == 'RAM_ONCE' and b.mode == 'HDF5_LAZY'
    assert torch.equal(a.batch([2, 9], torch.device('cpu')), b.batch([2, 9], torch.device('cpu')))
