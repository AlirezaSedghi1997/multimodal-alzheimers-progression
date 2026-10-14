"""Repeated split-stacking: never move MRI scores between independently fitted experts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scripts.xyz_fusion_v1.pipeline import partition, stop_partition, assert_disjoint, save_pickle
from scripts.xyz_fusion_v1.statistics import metrics
from scripts.xyz_fusion_v1.store import atomic_json, digest, file_sha
from .training import train_expert, predict
from .models import fit_model, curves, score_features, assert_pair, score_diagnostic
from .protocol import arms, eligible, MRI, PET
from .store import checked_job


def locked_tabular(store, task, frame, features, family, parameter, config, seed):
    import pickle
    path = store.root/'private/models'/f'{task}.pkl'
    binding = digest({'parent': store.binding, 'family': family, 'parameter': parameter,
                      'features': features, 'seed': seed,
                      'rows': json.loads(frame[['patient_hash', 'event', 'time_days', *features]].to_json(orient='records'))})
    def compute():
        model = fit_model(frame, features, family, parameter, config, seed)
        save_pickle(path, model)
        return {'binding': binding, 'sha256': file_sha(path), 'features': features,
                'ancestry': sorted(frame.patient_hash), 'family': family, 'parameter': parameter,
                'path': str(path.relative_to(store.root))}
    r = store.job(task, compute)
    if r['binding'] != binding or r['sha256'] != file_sha(path):
        raise ValueError('V2_TABULAR_LOCK_DRIFT')
    with path.open('rb') as f: model = pickle.load(f)  # trusted, byte-verified local artifact
    return model, r


def prediction_unit(store, name, lock, compute):
    """One committed result; replay only an incomplete deterministic prediction unit."""
    journal = store.root/'private/access_journal'/f'{name}.json'
    path = store.root/'private/jobs'/f'{name}.json'
    if not path.exists():
        old = json.loads(journal.read_text()) if journal.exists() else {'attempts': 0}
        if old.get('lock', lock) != lock: raise ValueError('OUTER_MODEL_CHANGED_DURING_REPLAY')
        atomic_json(journal, {'lock': lock, 'attempts': old['attempts']+1,
                             'state': 'IN_PROGRESS', 'outcomes_passed_to_prediction': False})
    result = store.job(name, compute)
    if not journal.exists(): raise ValueError('PREDICTION_JOURNAL_MISSING')
    r = json.loads(journal.read_text())
    if r['lock'] != lock: raise ValueError('PREDICTION_LOCK_MISMATCH')
    r.update(state='COMMITTED', payload_sha256=digest(result))
    atomic_json(journal, r)
    return result


def evaluate_cell(train, held, arm_predictions, config, fold):
    result = {}
    for arm in arm_predictions:
        p = arm_predictions[arm]
        tr = eligible(train, arm)
        te = held.set_index('patient_hash').loc[p['subjects']].reset_index()
        result[arm] = {'subjects': p['subjects'], 'curves': p['survival'],
                       'event': te.event.astype(int).tolist(), 'time': te.time_days.tolist(),
                       'censor_train': {'event': tr.event.astype(int).tolist(), 'time': tr.time_days.tolist()},
                       'metrics': metrics(tr, te, p['survival'], config['fusion']['ibs_grid_days'],
                                          config['fusion']['horizons_days'], config['fusion']['min_at_risk'])}
    return {'fold': fold, 'status': 'OUTER_EVAL_COMMITTED', 'arms': result,
            'train_subjects': sorted(train.patient_hash), 'test_subjects': sorted(held.patient_hash)}


def paired_crossfit(frame, banks, config, store, track, outer_splits):
    import torch
    output = []; grid = config['fusion']['ibs_grid_days']
    columns = arms(track)
    for seed in config['training_seeds']:
        seed_cells = []
        for fold, split in enumerate(outer_splits):
            prefix = f'{track}_s{seed}_o{fold}'
            if (store.root/'private/jobs'/f'{prefix}_evaluation.json').exists():
                seed_cells.append(checked_job(store, prefix+'_evaluation')); continue
            tr = frame.loc[frame.patient_hash.isin(split['train'])].copy().reset_index(drop=True)
            te = frame.loc[frame.patient_hash.isin(split['test'])].copy().reset_index(drop=True)
            assert_disjoint(tr.patient_hash, te.patient_hash)
            components = []
            for inner, (u, v) in enumerate(partition(tr, config['components'], config['seed']+fold)):
                task = f'{prefix}_c{inner}'
                if (store.root/'private/jobs'/f'{task}_predictions.json').exists():
                    components.append(checked_job(store, task+'_predictions')); continue
                pool, meta = tr.iloc[u].copy(), tr.iloc[v].copy()
                fit, stop = stop_partition(pool, config['seed']+100*fold+inner)
                assert_disjoint(fit.patient_hash, stop.patient_hash, meta.patient_hash, te.patient_hash)
                models = {}; receipts = {}; meta_aug = meta.copy()
                for modality, bank in banks.items():
                    policy = dict(config['training'], modality=modality)
                    candidates = []
                    for j, lr in enumerate(policy['learning_rates']):
                        model, r = train_expert(bank, fit, stop, policy, grid,
                            store.root/'private/models'/f'{task}_{modality}_tune{j}',
                            store.binding, lr, seed+fold*1000+inner*100+j)
                        candidates.append({'lr': lr, 'metric': r['inner_ibs'], 'epochs': r['best_epoch'],
                                           'sha256': r['sha256']})
                        del model
                    winner = min(candidates, key=lambda x: (x['metric'], x['lr']))
                    store.job(f'{task}_{modality}_winner', lambda: {'winner': winner, 'candidates': candidates,
                              'selection_subjects': sorted(stop.patient_hash),
                              'never_seen': sorted(set(meta.patient_hash)|set(te.patient_hash))})
                    model, receipt = train_expert(bank, pool, None, policy, grid,
                        store.root/'private/models'/f'{task}_{modality}_expert', store.binding,
                        winner['lr'], seed+50000+fold*100+inner, fixed_epochs=winner['epochs'])
                    assert_pair(receipt, meta.patient_hash, te.patient_hash)
                    def meta_prediction():
                        value = predict(model, bank, meta.drop(columns=['event', 'time_days']), policy,
                                        next(model.parameters()).device, grid)
                        return {'subjects': list(meta.patient_hash), 'survival': value.tolist(),
                                'model_sha256': receipt['sha256']}
                    value = store.job(f'{task}_{modality}_meta', meta_prediction)
                    if value['model_sha256'] != receipt['sha256'] or value['subjects'] != list(meta.patient_hash):
                        raise ValueError('META_SCORE_MODEL_OR_MEMBERSHIP_DRIFT')
                    meta_aug = score_features(meta_aug, value['survival'], grid,
                                              config['fusion']['score_times_days'], modality)
                    models[modality] = model; receipts[modality] = receipt
                heads = {}; locks = {}
                for arm, features in columns.items():
                    subset = eligible(meta_aug, arm)
                    heads[arm], locks[arm] = locked_tabular(store, f'{task}_{arm}', subset, features,
                        'cox_ridge', config['fusion']['ridge_alpha'], config, seed)
                lock = store.job(task+'_lock', lambda: {'experts': receipts, 'heads': locks,
                    'expert_subjects': sorted(pool.patient_hash), 'meta_subjects': sorted(meta.patient_hash),
                    'outer_subjects': sorted(te.patient_hash), 'no_outer_refitted_expert': True})
                def infer():
                    values = te.drop(columns=[c for c in te if c == 'event' or c == 'time_days' or c.endswith(('_event', '_time_days'))]).copy()
                    diagnostic = {}
                    for modality, model in models.items():
                        p = predict(model, banks[modality], values, dict(config['training'], modality=modality),
                                    next(model.parameters()).device, grid)
                        values = score_features(values, p, grid, config['fusion']['score_times_days'], modality)
                        diagnostic[modality] = score_diagnostic(meta_aug, values, MRI if modality=='mri' else PET,
                                                                receipts[modality]['sha256'])
                    predictions = {}
                    for arm, features in columns.items():
                        subset = eligible(values, arm)
                        predictions[arm] = {'subjects': list(subset.patient_hash),
                                            'survival': curves(heads[arm], subset, features, grid).tolist()}
                    return {'arms': predictions, 'diagnostics': diagnostic, 'lock_sha256': digest(lock)}
                components.append(prediction_unit(store, task+'_predictions', digest(lock), infer))
                del models, heads
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            combined = average_predictions([x['arms'] for x in components])
            cell = store.job(prefix+'_evaluation', lambda: evaluate_cell(tr, te, combined, config, fold))
            seed_cells.append(cell)
            print(json.dumps({'event': 'V2_OUTER_COMMITTED', 'track': track, 'seed': seed, 'fold': fold}), flush=True)
        output.append({'seed': seed, 'cells': seed_cells})
    return output


def average_predictions(predictions):
    if not predictions: raise ValueError('NO_PREDICTION_COMPONENTS')
    result = {}
    for arm, cell in predictions[0].items():
        if any(set(p) != set(predictions[0]) or p[arm]['subjects'] != cell['subjects'] for p in predictions):
            raise ValueError('ENSEMBLE_MEMBERSHIP_MISMATCH')
        result[arm] = {'subjects': cell['subjects'],
                      'survival': np.mean([p[arm]['survival'] for p in predictions], axis=0).tolist()}
    return result


def ensemble_seeds(runs, frame, config):
    cells = []
    for fold in range(5):
        source = runs[0]['cells'][fold]
        train = frame.loc[frame.patient_hash.isin(source['train_subjects'])]
        held = frame.loc[frame.patient_hash.isin(source['test_subjects'])]
        predictions = []
        for run in runs:
            c = run['cells'][fold]
            if c['train_subjects'] != source['train_subjects'] or c['test_subjects'] != source['test_subjects']:
                raise ValueError('SEED_MEMBERSHIP_MISMATCH')
            predictions.append({a: {'subjects': x['subjects'], 'survival': x['curves']} for a, x in c['arms'].items()})
        cells.append(evaluate_cell(train, held, average_predictions(predictions), config, fold))
    return cells
