"""One pooled head per seed/fold; lock all heads before outer evaluation."""
import pickle
import numpy as np
from scripts.xyz_fusion_postall_v2.core import digest, inside
from scripts.xyz_fusion_v2.pipeline import locked_tabular, average_predictions
from scripts.xyz_fusion_v2.models import curves
from scripts.xyz_fusion_postall_v2.statistics import evaluate_cell
from scripts.xyz_fp32_spatial_v1.pipeline import evaluate_committed
from .protocol import arms


def head_name(track, seed, fold):
    return f'{track}_s{seed}_o{fold}_pooled'


def fit_heads(inputs, units, track, seed, fold):
    frame, tasks = inputs.frames_for_head(track, seed, fold)
    prefix = head_name(track, seed, fold)
    heads = {}
    for arm, columns in arms(track).items():
        _, heads[arm] = locked_tabular(units, prefix + '_' + arm, frame, columns,
            'cox_ridge', inputs.options['ridge_alpha'], inputs.config, seed)
    expected = set(inputs.split_frame(track, fold, 'train').patient_hash)
    for head in heads.values():
        if set(head['ancestry']) != expected:
            raise ValueError('HEAD_NOT_FITTED_ON_COMPLETE_ELIGIBLE_OOF')
    dependencies = {'parent_locks': digest([inputs.validate_task(t) for t in tasks]),
                    'head_receipts': digest(heads)}
    return units.job(prefix + '_lock', lambda: {
        'heads': heads, 'tasks': tasks, 'n': len(frame), 'events': int(frame.event.sum()),
        'pooled_subjects': list(frame.patient_hash), 'fixed_ridge_alpha': inputs.options['ridge_alpha'],
        'inner_selection_on_pooled_oof': False, 'outer_outcomes_used_for_fitting': False,
        'new_cnn_training': False}, dependencies)


def predict_head(inputs, units, track, seed, fold):
    prefix = head_name(track, seed, fold)
    lock = units.read(prefix + '_lock')
    if lock is None:
        raise ValueError('ALL_HEADS_MUST_BE_LOCKED_BEFORE_PREDICTION')
    dependencies = {'head_lock': digest(lock)}
    def compute():
        components = []
        for task in lock['tasks']:
            frame = inputs.outer_for_head(track, task, fold)
            if any(c in ('event', 'time_days') or c.endswith(('_event', '_time_days')) for c in frame):
                raise ValueError('OUTER_OUTCOMES_ENTERED_PREDICTION')
            predictions = {}
            for arm, receipt in lock['heads'].items():
                if units.read(prefix + '_' + arm) != receipt:
                    raise ValueError('HEAD_RECEIPT_CHANGED')
                # Only our own byte-verified, freshly produced model files are unpickled.
                with inside(units.root, receipt['path']).open('rb') as stream:
                    model = pickle.load(stream)
                predictions[arm] = {'subjects': list(frame.patient_hash),
                    'survival': curves(model, frame, receipt['features'], inputs.config['fusion']['ibs_grid_days']).tolist()}
            components.append(predictions)
        return {'arms': average_predictions(components), 'head_lock': digest(lock),
                'outcomes_passed_to_prediction': False,
                'aggregation': 'MEAN_OF_COMPONENT_SURVIVAL_NOT_MEAN_OF_FEATURES'}
    return units.job(prefix + '_predictions', compute, dependencies)


def matched_legacy(inputs, track, fold, subjects):
    cell = inputs.scores.read(f'{track}_o{fold}_evaluation')['payload']
    if cell['fold'] != fold or set(cell['test_subjects']) != set(subjects):
        raise ValueError('LEGACY_EVALUATION_SUBSET_MISMATCH')
    result = {}
    for arm, value in cell['arms'].items():
        if len(value['subjects']) != len(set(value['subjects'])) or set(value['subjects']) != set(subjects):
            raise ValueError('LEGACY_ARM_SUBSET_MISMATCH')
        order = {key: i for i, key in enumerate(value['subjects'])}
        result['split_' + arm] = {'subjects': list(subjects),
            'survival': np.asarray(value['curves'])[[order[k] for k in subjects]].tolist()}
    return result


def run_fit(inputs, units, smoke=False):
    tasks = [('mri', inputs.config['training_seeds'][0], 0)] if smoke else [
        (track, seed, fold) for track in inputs.options['tracks']
        for seed in inputs.config['training_seeds'] for fold in range(5)]
    locks = [fit_heads(inputs, units, *task) for task in tasks]
    units.job('all_heads_locked', lambda: {'head_count': sum(len(x['heads']) for x in locks),
        'pool_count': len(locks), 'smoke_only': smoke}, {'head_locks': digest(locks)})
    for task in tasks:
        predict_head(inputs, units, *task)
    if smoke:
        return {'status': 'SMOKE_FIT_PREDICT_PASSED_NO_PERFORMANCE_EVALUATION',
                'head_count': 2, 'folds': 1, 'seeds': 1, 'new_cnn_training': False,
                'scientific_improvement_established': False}
    cells = {}
    for track in inputs.options['tracks']:
        cells[track] = []
        for fold in range(5):
            train = inputs.split_frame(track, fold, 'train')
            test = inputs.split_frame(track, fold, 'outer')
            seeds = [units.read(head_name(track, seed, fold) + '_predictions')['arms']
                     for seed in inputs.config['training_seeds']]
            values = average_predictions(seeds)
            if track in ('mri', 'pet'):
                values.update(matched_legacy(inputs, track, fold, list(test.patient_hash)))
            dependencies = {'predictions': digest(values), 'split': digest(inputs.splits[fold]),
                            'eligible_train': digest(list(train.patient_hash))}
            cells[track].append(evaluate_committed(units, f'{track}_o{fold}_evaluation', dependencies,
                lambda: evaluate_cell(train, test, values, inputs.config, fold)))
        units.job(track + '_cells', lambda: {'cells': cells[track]}, {'cells': digest(cells[track])})
    return {'status': 'OOF_FIT_AND_EVALUATION_COMPLETE_REPORT_PENDING',
            'head_count': sum(len(x['heads']) for x in locks), 'outer_cells': 20,
            'new_cnn_training': False}
