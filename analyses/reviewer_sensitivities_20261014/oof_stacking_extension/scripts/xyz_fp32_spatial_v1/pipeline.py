"""Same expert/selection ancestry; precision-specific meta scores and small heads."""
import json
import pickle
import numpy as np
from scripts.xyz_fusion_postall_v2.core import digest, file_sha, inside, atomic_json, read_json
from scripts.xyz_fusion_v2.protocol import arms


def task_names(config, track, fold=None):
    return [f'{track}_s{s}_o{f}_c{c}' for s in config['training_seeds']
            for f in (range(5) if fold is None else [fold]) for c in range(config['components'])]


def chosen_arms(track):
    return (['clinical_amyloid', 'clinical_mri_amyloid'] if track == 'mri' else
            ['clinical_mri_amyloid', 'clinical_mri_amyloid_pet'])


def ancestry(ctx, task):
    from scripts.xyz_fusion_v2.models import assert_pair
    lock = ctx.parent.job(task + '_lock')
    pool, meta, held = [set(lock[k]) for k in ('expert_subjects', 'meta_subjects', 'outer_subjects')]
    if not all((pool, meta, held)) or pool & meta or pool & held or meta & held:
        raise ValueError('ANCESTRY_OVERLAP_OR_EMPTY')
    for mod, expert in lock['experts'].items():
        assert_pair(expert, meta, held)
        winner = ctx.parent.job(task + '_' + mod + '_winner')
        if set(winner['selection_subjects']) - pool:
            raise ValueError('SELECTION_OUTSIDE_EXPERT_POOL')
        path = ctx.parent.root / 'private/models' / f'{task}_{mod}_expert/locked.pt'
        if ctx.parent.hashes(path) != expert['sha256']:
            raise ValueError('FROZEN_CNN_DRIFT')
    return lock


def scores(ctx, task, role, source, lock):
    from scripts.xyz_fusion_postall_v2.experts import load_expert
    from scripts.xyz_fusion_v2.training import predict
    from scripts.xyz_fusion_v2.models import score_features
    import torch
    keys = lock[role + '_subjects']
    frame = source.loc[source.patient_hash.isin(keys)].copy().reset_index(drop=True)
    if len(frame) != len(keys) or set(frame.patient_hash) != set(keys) or frame.patient_hash.duplicated().any():
        raise ValueError('EXACT_SUBJECT_COVERAGE_REQUIRED')
    for modality in lock['experts']:
        name = f'{task}_{modality}_{role}_fp32'
        def infer():
            model, policy = load_expert(ctx, task, modality)
            if policy['amp']:
                raise ValueError('FP32_REQUIRED')
            try:
                # Only identifiers for cache addressing; no outcomes, covariates or dates.
                values = predict(model, ctx.banks()[modality], frame[['patient_hash', 'cache_row']],
                                 policy, next(model.parameters()).device, ctx.config['fusion']['ibs_grid_days'])
                return {'subjects': list(frame.patient_hash), 'survival': values.tolist(),
                        'model_sha256': lock['experts'][modality]['sha256'],
                        'outcomes_passed_to_cnn': False, 'precision': 'FP32'}
            finally:
                del model
                torch.cuda.empty_cache()
        receipt = ctx.units.job(name, infer, {'parent_lock': digest(lock), 'subjects': digest(list(frame.patient_hash))})
        if receipt['subjects'] != list(frame.patient_hash):
            raise ValueError('SCORE_ORDER_DRIFT')
        frame = score_features(frame, receipt['survival'], ctx.config['fusion']['ibs_grid_days'],
                               ctx.config['fusion']['score_times_days'], modality)
    return frame


def fit_heads(ctx, task, source):
    from scripts.xyz_fusion_v2.pipeline import locked_tabular
    lock = ancestry(ctx, task)
    name = task + '_child_lock'
    existing = ctx.units.read(name)
    if existing is not None:
        if existing['parent_lock_digest'] != digest(lock):
            raise ValueError('PARENT_LOCK_DRIFT')
        for arm, head in existing['heads'].items():
            if ctx.units.read(task + '_' + arm) != head:
                raise ValueError('CHILD_HEAD_DRIFT')
        return existing
    meta = scores(ctx, task, 'meta', source, lock)
    selected = arms(task.split('_')[0]); heads = {}
    for arm in chosen_arms(task.split('_')[0]):
        _, heads[arm] = locked_tabular(ctx.units, task + '_' + arm, meta, selected[arm], 'cox_ridge',
            ctx.config['fusion']['ridge_alpha'], ctx.config, int(task.split('_s')[1].split('_')[0]))
    return ctx.units.job(name, lambda: {'experts': lock['experts'], 'heads': heads,
        'meta_subjects': list(meta.patient_hash), 'outer_subjects': lock['outer_subjects'],
        'expert_subjects': lock['expert_subjects'], 'parent_lock_digest': digest(lock),
        'new_cnn_training': False, 'precision': 'FP32', 'outer_model_selection': False})


def load_head(ctx, task, arm):
    receipt = ctx.units.read(task + '_' + arm)
    if receipt is None:
        raise ValueError('CHILD_HEAD_NOT_LOCKED')
    path = inside(ctx.out, receipt['path'])
    if ctx.units.hashes(path) != receipt['sha256']:
        raise ValueError('CHILD_HEAD_HASH_DRIFT')
    with path.open('rb') as stream:
        return pickle.load(stream), receipt


def prediction(ctx, task, source):
    from scripts.xyz_fusion_v2.models import curves
    child = ctx.units.read(task + '_child_lock')
    if child is None:
        raise ValueError('CHILD_LOCK_REQUIRED_BEFORE_OUTER_INFERENCE')
    def infer():
        raw = source.drop(columns=[c for c in source if c in ('event', 'time_days') or c.endswith(('_event', '_time_days'))])
        value = scores(ctx, task, 'outer', raw, ancestry(ctx, task))
        output = {}
        for arm in chosen_arms(task.split('_')[0]):
            head, receipt = load_head(ctx, task, arm)
            output[arm] = {'subjects': list(value.patient_hash), 'survival': curves(head, value,
                receipt['features'], ctx.config['fusion']['ibs_grid_days']).tolist()}
        return {'arms': output, 'child_lock_sha256': digest(child), 'outcomes_passed': False}
    return ctx.units.job(task + '_child_predictions', infer, {'child_lock': digest(child)})


def evaluate_committed(units, name, dependency, compute):
    """One committed evaluation; deterministic interrupted replay is logged, not hidden."""
    journal = units.root / 'private/evaluation_journal' / (name + '.json')
    done = units.read(name, dependency)
    if done is not None:
        record = read_json(journal)
        if record['dependency'] != dependency:
            raise ValueError('EVALUATION_DEPENDENCY_CHANGED')
        atomic_json(journal, dict(record, state='COMMITTED', payload_sha256=digest(done)))
        return done
    record = read_json(journal) if journal.exists() else {'attempts': 0, 'dependency': dependency}
    if record['dependency'] != dependency:
        raise ValueError('EVALUATION_DEPENDENCY_CHANGED')
    record.update(attempts=record['attempts'] + 1, state='IN_PROGRESS',
                  no_selection_or_refit_during_replay=True)
    atomic_json(journal, record)
    value = units.job(name, compute, dependency)
    atomic_json(journal, dict(record, state='COMMITTED', payload_sha256=digest(value)))
    return value


def run_predictions(ctx):
    from scripts.xyz_fusion_v2.pipeline import average_predictions
    from scripts.xyz_fusion_postall_v2.statistics import evaluate_cell
    # Lock all 180 heads before calculating any child outer statistic.
    for track in ('mri', 'pet'):
        frame = ctx.frame if track == 'mri' else ctx.pet
        for task in task_names(ctx.config, track):
            fit_heads(ctx, task, frame)
    result = {}
    for track in ('mri', 'pet'):
        frame = ctx.frame if track == 'mri' else ctx.pet
        cells = []
        for fold, split in enumerate(ctx.splits):
            predictions = [prediction(ctx, task, frame)['arms'] for task in task_names(ctx.config, track, fold)]
            mean = average_predictions(predictions)
            train = frame.loc[frame.patient_hash.isin(split['train'])]
            test = frame.loc[frame.patient_hash.isin(split['test'])]
            deps = {'predictions': digest(mean), 'split': digest(split)}
            cells.append(evaluate_committed(ctx.units, f'{track}_o{fold}_evaluation', deps,
                lambda: evaluate_cell(train, test, mean, ctx.config, fold)))
        result[track] = ctx.units.job(track + '_child_cells', lambda: {'cells': cells}, {'cells': digest(cells)})['cells']
    return result


def report_track(ctx, track, cells):
    from scripts.xyz_fusion_v2.reporting import paired_ci
    from scripts.xyz_fusion_v1.report import cell_frame, individual_bs
    from scripts.xyz_fusion_postall_v2.audit import incremental_verdict
    arms_summary = {}
    for arm in chosen_arms(track):
        metrics = [c['arms'][arm]['metrics'] for c in cells]
        losses = []
        differences = []
        for c, old in zip(cells, ctx.parent.job(track + '_ensemble_cells')):
            val = c['arms'][arm]; tr, te = cell_frame(val)
            losses.extend(individual_bs(tr, te, np.asarray(val['curves']), ctx.config['fusion']['ibs_grid_days'])[0])
            old_arm = old['arms'][arm]
            if val['subjects'] != old_arm['subjects']:
                raise ValueError('AMP_FP32_MATCHED_ORDER_CHANGED')
            differences.extend((np.asarray(val['curves']) - np.asarray(old_arm['curves'])).ravel())
        losses = np.asarray(losses); diff = np.asarray(differences)
        rng = np.random.default_rng(ctx.config['seed'])
        draws = [float(losses[rng.integers(0, len(losses), len(losses))].mean()) for _ in range(ctx.options['bootstrap_resamples'])]
        cs = [m['uno_c_index'] for m in metrics]
        arms_summary[arm] = {'n': len(losses), 'events': sum(m['events'] for m in metrics),
            'ibs': float(losses.mean()), 'ibs_conditional_ci95': np.quantile(draws, [.025, .975]).tolist(),
            'uno_c': float(np.average(cs, weights=[m['n'] for m in metrics])) if all(c is not None for c in cs) else None,
            'fold_metrics': metrics, 'amp_parent_absolute_curve_change_mean': float(np.abs(diff).mean()),
            'amp_parent_absolute_curve_change_max': float(np.abs(diff).max()),
            'precision_comparison_is_descriptive_not_model_selection': True}
    base, aug = chosen_arms(track)
    comparison = paired_ci(cells, base, aug, ctx.config['fusion']['ibs_grid_days'], ctx.config['seed'],
                           ctx.options['bootstrap_resamples'], 2)
    return {'arms': arms_summary, 'paired_comparison': dict(comparison, verdict=incremental_verdict(comparison),
             family_size=2), 'selection_from_outer_results': False, 'same_as_parent_model': False}
