"""Manual, atomic and versioned precision sensitivity; parent is read-only."""
from __future__ import annotations
import copy
import importlib.metadata
import json
import sys
from pathlib import Path
import numpy as np
from scripts.xyz_fusion_postall_v2.core import (
    ReadOnlyParent, Units, digest, file_sha, read_json, atomic_json, atomic_text,
    writer_lock, parent_exclusion)
from .resources import FastContext as Context
from .pipeline import task_names, chosen_arms, run_predictions, report_track


def make_contract(root, parent, options, smoke=False):
    import torch
    paths = set()
    for folder in ('scripts/xyz_fp32_spatial_v1', 'scripts/xyz_fusion_postall_v2',
                   'scripts/xyz_fusion_v1', 'scripts/xyz_fusion_v2', 'scripts/xyz_postprimary_v1', 'src/models'):
        paths.update((root / folder).rglob('*.py'))
    paths.update(root / p for p in ('scripts/run_xyz_fp32_spatial_v1.py', 'scripts/run_xyz_fp32_spatial_v1.ps1',
                                   'src/evaluation/signed_spatial_validation_v1.py'))
    return {'schema_version': 1, 'options': options, 'parent_contract': parent.binding,
        'parent_execution_contract': parent.execution_binding, 'scope': 'SMOKE' if smoke else 'FULL_SENSITIVITY',
        'source_sha256': {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(paths)},
        'runtime': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'pandas', 'scipy',
                     'scikit-learn', 'scikit-survival', 'statsmodels', 'nibabel', 'h5py')},
        'python': sys.version, 'cuda_runtime': torch.version.cuda,
        'math': {'cnn_precision': 'FP32', 'quadrature_accumulation': 'FP64',
            'cox_head_precision': 'FP64', 'tf32': False, 'amp': False, 'deterministic': True},
        'cnn_weights_frozen': True, 'same_model_as_amp_parent': False,
        'clinical_claim_allowed': False, 'external_claim_allowed': False, 'disease_localization_allowed': False}


def expected_tasks(config):
    return [t for track in ('mri', 'pet') for t in task_names(config, track)]


def verify(ctx):
    from scripts.xyz_fusion_v2.privacy import scan_public_bundle
    from .pipeline import ancestry
    locked = read_json(ctx.out / 'private/contract.json')
    if make_contract(ctx.root, ctx.parent, ctx.options) != locked:
        raise ValueError('SOURCE_OR_RUNTIME_CHANGED_DURING_RUN')
    missing = []; seen = {'mri': set(), 'pet': set()}
    if ctx.config['outer_folds'] != 5:
        raise ValueError('FIVE_FOLDS_REQUIRED')
    # Every receipt and binary is checked, not only the headline state.
    for path in (ctx.out / 'private/units').glob('*.json'):
        ctx.units.read(path.stem)
    for task in expected_tasks(ctx.config):
        child = ctx.units.read(task + '_child_lock')
        pred = ctx.units.read(task + '_child_predictions')
        if child is None or pred is None:
            missing.append(task); continue
        lock = ancestry(ctx, task)
        if child['parent_lock_digest'] != digest(lock) or pred['child_lock_sha256'] != digest(child):
            raise ValueError('CHILD_BINDING_DRIFT')
        for arm in chosen_arms(task.split('_')[0]):
            head = ctx.units.read(task + '_' + arm)
            if head != child['heads'][arm] or set(head['ancestry']) != set(lock['meta_subjects']):
                raise ValueError('HEAD_ANCESTRY_DRIFT')
            if set(pred['arms'][arm]['subjects']) != set(lock['outer_subjects']):
                raise ValueError('PREDICTION_MEMBERSHIP_DRIFT')
    for track in ('mri', 'pet'):
        for fold in range(5):
            name = f'{track}_o{fold}_evaluation'
            cell = ctx.units.read(name)
            if cell is None: missing.append(name); continue
            ids = set(cell['test_subjects'])
            if seen[track] & ids or ids & set(cell['train_subjects']):
                raise ValueError('FOLD_OVERLAP')
            seen[track].update(ids)
            journal = read_json(ctx.out / 'private/evaluation_journal' / (name + '.json'))
            if journal['state'] != 'COMMITTED' or journal['payload_sha256'] != digest(cell):
                raise ValueError('EVALUATION_JOURNAL_INCOMPLETE')
    for track, frame in [('mri', ctx.frame), ('pet', ctx.pet)]:
        if frame is not None and seen[track] and seen[track] != set(frame.patient_hash):
            missing.append(track + '_oof_subject_coverage')
    spatial = ctx.units.read('spatial_complete')
    if spatial is None: missing.append('spatial_complete')
    else:
        expected = ctx.config['spatial']['cases_per_fold'] * 5 * 3
        if spatial['requested_units'] != expected or len(spatial['cases']) != expected:
            raise ValueError('SPATIAL_PANEL_INCOMPLETE')
    release = ctx.units.read('release')
    if release is None: missing.append('release')
    elif scan_public_bundle(ctx.out / 'public'):
        raise ValueError('PUBLIC_PRIVACY_FAILURE')
    if missing:
        return {'status': 'INCOMPLETE', 'missing': missing, 'contract': ctx.units.binding}
    return {'status': 'PASS_FP32_SENSITIVITY_EXECUTION_NOT_POSITIVE_SCIENCE',
        'contract': ctx.units.binding, 'locked_components': len(expected_tasks(ctx.config)),
        'small_heads': 2 * len(expected_tasks(ctx.config)), 'outer_cells': 10,
        'claim_status': 'POST_HOC_INTERNAL_EXPLORATORY',
        'numerically_passing_spatial_baselines': spatial['numerically_passing_baselines'],
        'evaluated_spatial_baselines': spatial['evaluated_baselines'],
        'actual_image_and_statistical_review_required': True,
        'fully_validated_attribution': False, 'same_model_as_amp_parent': False}


def write_release(ctx, reports, spatial):
    from scripts.xyz_fusion_v2.privacy import scan_public_bundle
    # Explicit whitelist: no subject IDs, individual curves, paths or images.
    public = ctx.out / 'public'
    case_counts = {}
    for case in spatial['cases']:
        status = case['status']; case_counts[status] = case_counts.get(status, 0) + 1
    counts = {}
    for case in spatial['cases']:
        for baseline in case.get('baselines', {}).values():
            status = baseline['trained_ig_numeric']; counts[status] = counts.get(status, 0) + 1
    result = {'protocol': ctx.options['protocol'], 'analyses': reports,
        'claim_status': ctx.options['claim_status'], 'spatial_case_status_counts': case_counts,
        'spatial_numeric_counts': counts, 'requested_spatial_units': spatial['requested_units'],
        'eligible_spatial_units': spatial['eligible_units'], 'cnn_training_performed': False,
        'small_heads_refit_on_original_meta_only': True, 'same_model_as_amp_parent': False,
        'fully_validated_attribution': False, 'external_claim_allowed': False,
        'clinical_claim_allowed': False, 'disease_localization_allowed': False,
        'label_permutation_status': 'NOT_RUN_REQUIRES_NEW_CNN_TRAINING',
        'scope': 'SEPARATE_PRECISION_SENSITIVITY_NOT_A_REPLACEMENT_FOR_PARENT_RESULTS',
        'uncertainty': 'CONDITIONAL_ON_FITTED_MODELS_AND_TRAIN_CENSORING',
        'parent_artifacts_modified': False}
    atomic_json(public / 'results.json', result)
    atomic_text(public / 'START_HERE_FA.md',
        '# مسیر مکمل FP32\n\nاین تحلیل حساسیت مستقل است، نه توضیح دقیق مدل AMP والد. '
        'CNNها ثابت و headهای کوچک فقط روی افراد meta اولیه بازسازی شده‌اند. '
        'نتایج منفی و نقشه‌های نامعتبر حذف نشده‌اند.\n\n'
        'تصاویر فردی فقط در private/spatial هستند؛ زمینه تصویر واقعی همان فرد در فضای مدل است، نه native. '
        'قبولی کنترل عددی IG به معنی اثبات ناحیه آسیب آلزایمر یا قبولی تمام کنترل‌های علمی نیست. '
        'کنترل آموزش با label permutation اجرا نشده و نقشه‌ها diagnostic/exploratory باقی می‌مانند.\n\n'
        'پس از پایان، اعداد، حساسیت به baseline، fidelity و تصاویر باید بازبینی شوند. '
        'این فرمان بسته فصل‌های ۴ و ۵ را نهایی نمی‌کند و هیچ خروجی قبلی را جایگزین نمی‌کند.\n')
    if scan_public_bundle(public):
        raise ValueError('PUBLIC_PRIVACY_FAILURE')
    return {'status': 'AGGREGATE_RELEASE_READY_FOR_REVIEW',
            'outputs': {p.relative_to(ctx.out).as_posix(): file_sha(p) for p in public.iterdir() if p.is_file()}}


def smoke(ctx):
    import torch
    from scipy.ndimage import gaussian_filter
    from .pipeline import fit_heads, prediction
    from .spatial import make_target, adaptive_map
    results = {}
    for track, modality in [('mri', 'mri'), ('pet', 'pet')]:
        # Fixed before any child performance result; MRI is the originally failing component.
        task = f'{track}_s13_o0_c1'; frame = ctx.frame if track == 'mri' else ctx.pet
        fit_heads(ctx, task, frame)
        pred = prediction(ctx, task, frame); arm = chosen_arms(track)[-1]
        keys = set(ctx.splits[0]['test']) & set(frame.patient_hash)
        key = sorted(keys, key=lambda k: digest({'seed': ctx.config['seed'], 'key': k}))[0]
        row = frame.loc[frame.patient_hash.eq(key)]
        image = ctx.banks()[modality].batch([int(row.iloc[0].cache_row)], torch.device('cuda'))
        array = image.cpu().numpy().squeeze(); baseline = gaussian_filter(array, 2.); baseline[array == 0] = 0
        target = make_target(ctx, task, arm, modality, row)
        expected = pred['arms'][arm]; i = expected['subjects'].index(key)
        wanted = 1 - np.interp(1095.75, ctx.config['fusion']['ibs_grid_days'], expected['survival'][i])
        with torch.no_grad(): observed = float(target(image))
        if abs(observed - wanted) > ctx.options['spatial_parity_absolute_tolerance']:
            raise ValueError('SMOKE_PREDICTION_PARITY_FAILURE')
        _, numeric = adaptive_map(ctx, 'smoke_' + track + '_component_blur2', target, image,
            torch.as_tensor(baseline, device='cuda').reshape_as(image), {'child_lock': digest(ctx.units.read(task + '_child_lock'))})
        results[track] = {'numeric': numeric, 'prediction_parity_error': abs(observed - wanted)}
        del target
        torch.cuda.empty_cache()
    passed = all(r['numeric']['status'] == 'PASS' for r in results.values())
    value = {'status': 'GPU_SMOKE_NUMERIC_PASS' if passed else 'GPU_SMOKE_NUMERIC_NOT_VALIDATED',
        'modalities': results, 'all_panel_validated': False, 'new_cnn_training': False,
        'scope': 'TWO_FIXED_COMPONENTS_NO_OUTER_PERFORMANCE_SELECTION'}
    atomic_json(ctx.out / 'smoke.json', value)
    print(json.dumps(value), flush=True)
    return 0 if passed else 2


def execute(root, mode, config_path):
    from scripts.xyz_fusion_postall_v2.audit import verify_parent, freeze_once
    root = Path(root).resolve(); options = read_json(config_path)
    if str(root).replace('\\', '/').lower() != 'h:/new project':
        raise ValueError('AUTHORITATIVE_H_PROJECT_REQUIRED')
    if options['amp'] or options['tf32'] or options['allow_new_cnn_training'] or options['model_selection_from_outer_results']:
        raise ValueError('FP32_FROZEN_CNN_SENSITIVITY_POLICY_REQUIRED')
    out = (root / options['smoke_root' if mode == 'smoke' else 'output_root']).resolve()
    parent = ReadOnlyParent(root / options['parent_root'])
    if out.parent != root / 'results' or not out.name.startswith('xyz_fp32_spatial_sensitivity_'):
        raise ValueError('SEPARATE_CHILD_OUTPUT_REQUIRED')
    with writer_lock(out), parent_exclusion(parent.root):
        print(json.dumps({'event': 'READ_ONLY_PARENT_VERIFICATION'}), flush=True)
        freeze = verify_parent(root, parent)
        freeze_once(out, freeze)
        units = Units(out, make_contract(root, parent, options, mode == 'smoke'))
        ctx = Context(root, parent, units, options)
        ctx.config = copy.deepcopy(ctx.config)
        ctx.config['training']['amp'] = False
        ctx.config['training']['tf32'] = False
        ctx.config['training']['cpu_threads'] = options['cpu_threads']
        # Fixed per-image FP32 function for BOTH score extraction and attribution.
        # Never retain a four-image background batch whose other three gradients
        # are thrown away. This is part of the new child's math contract only.
        ctx.config['training']['microbatch_size'] = options['inference_microbatch']
        try:
            # Rehash raw/cache bytes on every invocation, never trust an old readiness marker.
            proof = ctx.load_inputs()
            units.job('input_reconstruction', lambda: proof)
            if mode == 'verify' or units.read('release') is not None:
                result = verify(ctx)
                atomic_json(out / 'orchestration_state.json', result)
                print(json.dumps(result), flush=True)
                return 0 if result['status'].startswith('PASS') else 2
            if mode == 'preflight':
                from .pipeline import ancestry
                for task in expected_tasks(ctx.config): ancestry(ctx, task)
                result = {'status': 'READY_FOR_MANUAL_FP32_SENSITIVITY', 'contract': units.binding,
                          'tasks': len(expected_tasks(ctx.config)), 'no_training_started': True}
                atomic_json(out / 'readiness.json', result); print(json.dumps(result), flush=True)
                return 0
            from scripts.xyz_fusion_v1.runtime import gpu_guard
            from scripts.xyz_fusion_v2.training import deterministic
            gpu_guard(); deterministic(ctx.config['seed'], options['cpu_threads'])
            if mode == 'smoke': return smoke(ctx)
            cells = run_predictions(ctx)
            reports = {t: units.job(t + '_report', lambda t=t: report_track(ctx, t, cells[t]), {'cells': digest(cells[t])})
                       for t in ('mri', 'pet')}
            from .spatial import run_spatial
            spatial = units.job('spatial_complete', lambda: run_spatial(ctx))
            units.job('release', lambda: write_release(ctx, reports, spatial),
                      {'reports': digest(reports), 'spatial': digest(spatial)})
            result = verify(ctx)
            atomic_json(out / 'orchestration_state.json', result)
            print(json.dumps(result), flush=True)
            return 0 if result['status'].startswith('PASS') else 2
        except BaseException as exc:
            atomic_json(out / 'orchestration_state.json', {'status': 'INTERRUPTED_RESUMABLE' if isinstance(exc, KeyboardInterrupt)
                        else 'BLOCKED', 'reason': type(exc).__name__ + ':' + str(exc), 'contract': units.binding})
            raise
        finally:
            ctx.close()
