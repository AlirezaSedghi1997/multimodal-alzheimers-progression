"""Bounded, resumable CLI; immutable inputs and separately versioned outputs."""
from pathlib import Path
import importlib.metadata
import json
import sys
import time
from contextlib import ExitStack
from scripts.xyz_fusion_postall_v2.core import (Units, digest, file_sha, atomic_json,
                                               read_json, writer_lock, parent_exclusion)
from .inputs import Inputs
from .protocol import validate
from .pipeline import run_fit, head_name
from .protocol import arms
from .reporting import report


def output_path(root, options, smoke):
    root = Path(root).resolve()
    target = (root / options['smoke_root' if smoke else 'output_root']).resolve()
    if not target.is_relative_to(root / 'results') or not target.name.startswith('xyz_oof_stacking_'):
        raise ValueError('NEW_OOF_OUTPUT_UNDER_PROJECT_RESULTS_REQUIRED')
    for field in ('parent_root', 'score_root'):
        parent = (root / options[field]).resolve()
        if target == parent or target.is_relative_to(parent) or parent.is_relative_to(target):
            raise ValueError('OUTPUT_OVERLAPS_READ_ONLY_PARENT')
    return target


def make_contract(root, options, inputs, smoke):
    paths = set()
    for folder in ('scripts/xyz_oof_stacking_v1', 'scripts/xyz_fusion_v1', 'scripts/xyz_fusion_v2',
                   'scripts/xyz_fusion_postall_v2', 'scripts/xyz_fp32_spatial_v1'):
        paths.update((root / folder).glob('*.py'))
    paths.add(root / 'scripts/run_xyz_oof_stacking_v1.py')
    return {'schema': 1, 'protocol': options, 'scope': 'SMOKE' if smoke else 'FULL_OOF_EXPERIMENT',
            'parent_contract': inputs.parent.binding, 'parent_execution_contract': inputs.parent.execution_binding,
            'frozen_score_contract': inputs.scores.binding,
            'frozen_score_receipts': dict(sorted(inputs.scores.receipts.items())),
            'parent_locks': {k: digest(v) for k, v in sorted(inputs.locks.items())},
            'input_sha256': inputs.input_hashes,
            'source_sha256': {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(paths)},
            'runtime': {name: importlib.metadata.version(name) for name in
                        ('numpy', 'pandas', 'scipy', 'scikit-learn', 'scikit-survival', 'statsmodels', 'pyarrow')},
            'python': sys.version, 'new_cnn_training': False, 'historical_outcome_exposure': True}


def verify(units, inputs, smoke=False):
    names = [p.stem for p in (units.root / 'private/units').glob('*.json')]
    for name in names:
        units.read(name)
    required = 'smoke_complete' if smoke else 'aggregate_report'
    value = units.read(required)
    if value is None:
        return {'status': 'INCOMPLETE', 'required': required, 'checked_units': len(names)}
    count = units.read('all_heads_locked')
    if count is None or count['head_count'] != (2 if smoke else 180):
        raise ValueError('WRONG_HEAD_COUNT')
    tasks = [('mri', inputs.config['training_seeds'][0], 0)] if smoke else [
        (track, seed, fold) for track in inputs.options['tracks']
        for seed in inputs.config['training_seeds'] for fold in range(5)]
    for track, seed, fold in tasks:
        prefix = head_name(track, seed, fold)
        lock = units.read(prefix + '_lock')
        pred = units.read(prefix + '_predictions')
        if lock is None or pred is None or pred['head_lock'] != digest(lock):
            raise ValueError('REQUIRED_HEAD_OR_PREDICTION_MISSING')
        if set(lock['heads']) != set(arms(track)) or set(pred['arms']) != set(arms(track)):
            raise ValueError('REQUIRED_ARM_MISSING')
        expected_train = set(inputs.split_frame(track, fold, 'train').patient_hash)
        expected_test = list(inputs.split_frame(track, fold, 'outer').patient_hash)
        from scripts.xyz_fusion_v1.statistics import check_curves
        for arm, receipt in lock['heads'].items():
            if units.read(prefix + '_' + arm) != receipt or set(receipt['ancestry']) != expected_train:
                raise ValueError('REQUIRED_HEAD_RECEIPT_MISSING_OR_CHANGED')
            value = pred['arms'][arm]
            if value['subjects'] != expected_test:
                raise ValueError('REQUIRED_PREDICTION_ORDER_MISMATCH')
            check_curves(value['survival'], len(expected_test), inputs.config['fusion']['ibs_grid_days'])
    if not smoke:
        for track in inputs.options['tracks']:
            bundle = units.read(track + '_cells')
            if bundle is None or len(bundle['cells']) != 5:
                raise ValueError('REQUIRED_EVALUATION_BUNDLE_MISSING')
            for fold in range(5):
                name = f'{track}_o{fold}_evaluation'
                cell = units.read(name)
                if cell is None or cell != bundle['cells'][fold]:
                    raise ValueError('REQUIRED_EVALUATION_MISSING_OR_CHANGED')
                journal = read_json(units.root / 'private/evaluation_journal' / (name + '.json'))
                if journal['state'] != 'COMMITTED' or journal['payload_sha256'] != digest(cell):
                    raise ValueError('EVALUATION_JOURNAL_INCOMPLETE')
    return {'status': 'SMOKE_VERIFIED' if smoke else 'OOF_EXECUTION_VERIFIED_NOT_SCIENTIFIC_VALIDATION',
            'checked_units': len(names), 'head_count': count['head_count'], 'new_cnn_training': False}


def execute(root, mode, config_path):
    root = Path(root).resolve(); options = read_json(config_path); validate(options)
    smoke = mode == 'smoke'; out = output_path(root, options, smoke)
    started = time.time()
    from threadpoolctl import threadpool_limits
    with ExitStack() as stack:
        stack.enter_context(threadpool_limits(options['cpu_threads']))
        # Existing parent lock bytes are held without rewriting their contents.
        for field in ('parent_root', 'score_root'):
            stack.enter_context(parent_exclusion(root / options[field]))
        inputs = Inputs(root, options)
        preflight = inputs.preflight()
        contract = make_contract(root, options, inputs, smoke)
        compact = {k: v for k, v in preflight.items() if k != 'support'}
        print(json.dumps(dict(compact, elapsed_seconds=round(time.time()-started, 2))), flush=True)
        if mode == 'preflight':
            check = root / 'results/xyz_oof_stacking_v1_checks/preflight.json'
            atomic_json(check, dict(preflight, contract_sha256=digest(contract), elapsed_seconds=time.time()-started))
            return 0
        if mode in ('report', 'verify') and not (out / 'private/contract.json').exists():
            raise ValueError('EXISTING_FIT_REQUIRED')
        stack.enter_context(writer_lock(out))
        units = Units(out, contract)
        if mode == 'verify':
            result = verify(units, inputs)
        else:
            units.job('preflight', lambda: preflight)
            if mode in ('all', 'fit', 'smoke'):
                result = run_fit(inputs, units, smoke=smoke)
                if smoke:
                    units.job('smoke_complete', lambda: result)
                    result = verify(units, inputs, smoke=True)
            if mode in ('all', 'report'):
                report(inputs, units)
                result = verify(units, inputs)
        atomic_json(out / 'execution_status.json', dict(result, elapsed_seconds=time.time()-started))
        print(json.dumps(dict(result, elapsed_seconds=round(time.time()-started, 2))), flush=True)
        return 0 if result['status'] != 'INCOMPLETE' else 2
