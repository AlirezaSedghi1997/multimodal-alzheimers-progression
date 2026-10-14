"""Equal-budget nested tabular comparators and source-matched sensitivity experiments."""
from __future__ import annotations
import numpy as np
from scripts.xyz_fusion_v1.pipeline import partition, assert_disjoint
from scripts.xyz_fusion_v1.statistics import ibs_score
from scripts.xyz_fusion_v1.store import digest
from .protocol import arms, eligible
from .models import curves
from .pipeline import locked_tabular, prediction_unit, evaluate_cell
from .store import checked_job


def run_tabular(frame, config, store, label, track, splits, families=None):
    families = families or config['secondary']['families']
    output = {}
    for family in families:
        cells = []
        for fold, split in enumerate(splits):
            prefix = f'{label}_{family}_o{fold}'
            if (store.root/'private/jobs'/f'{prefix}_evaluation.json').exists():
                cells.append(checked_job(store, prefix+'_evaluation')); continue
            train = frame.loc[frame.patient_hash.isin(split['train'])].copy().reset_index(drop=True)
            test = frame.loc[frame.patient_hash.isin(split['test'])].copy().reset_index(drop=True)
            assert_disjoint(train.patient_hash, test.patient_hash)
            heads = {}; locks = {}
            for arm, features in arms(track).items():
                subset = eligible(train, arm).reset_index(drop=True)
                grid = config['secondary']['cox_alpha_grid' if family=='cox_ridge' else 'rsf_leaf_grid']
                ranks = []
                for j, parameter in enumerate(grid):
                    scores = []
                    for inner, (a, b) in enumerate(partition(subset, 3, config['seed']+fold)):
                        fit, val = subset.iloc[a], subset.iloc[b]
                        name = f'{prefix}_{arm}_p{j}_i{inner}'
                        def score():
                            model, receipt = locked_tabular(store, name+'_model', fit, features, family,
                                                            parameter, config, config['seed']+fold)
                            p = curves(model, val, features, config['fusion']['ibs_grid_days'])
                            return {'ibs': ibs_score(fit, val, p, config['fusion']['ibs_grid_days']),
                                    'model_sha256': receipt['sha256'], 'validation': sorted(val.patient_hash)}
                        scores.append(store.job(name+'_score', score)['ibs'])
                    ranks.append({'parameter': parameter, 'mean_ibs': float(np.mean(scores)),
                                  'standard_error': float(np.std(scores, ddof=1)/np.sqrt(len(scores)))})
                best = min(ranks, key=lambda x: x['mean_ibs'])
                candidates = [r for r in ranks if r['mean_ibs'] <= best['mean_ibs']+best['standard_error']]
                # Larger ridge/leaf is the prespecified simpler model.
                winner = max(candidates, key=lambda x: x['parameter'])
                store.job(f'{prefix}_{arm}_selection', lambda: {'candidates': ranks, 'winner': winner,
                    'metric': 'IPCW_IBS', 'rule': 'ONE_STANDARD_ERROR_LARGER_PENALTY_OR_LEAF'})
                heads[arm], locks[arm] = locked_tabular(store, f'{prefix}_{arm}_final', subset, features,
                                      family, winner['parameter'], config, config['seed']+fold)
            lock = store.job(prefix+'_lock', lambda: locks)
            def infer():
                predictions = {}
                x = test.drop(columns=[c for c in test if c in ('event', 'time_days') or c.endswith(('_event', '_time_days'))])
                for arm, features in arms(track).items():
                    subset = eligible(x, arm)
                    predictions[arm] = {'subjects': list(subset.patient_hash),
                        'survival': curves(heads[arm], subset, features, config['fusion']['ibs_grid_days']).tolist()}
                return predictions
            p = prediction_unit(store, prefix+'_predictions', digest(lock), infer)
            cells.append(store.job(prefix+'_evaluation', lambda: evaluate_cell(train, test, p, config, fold)))
        output[family] = cells
    return output
