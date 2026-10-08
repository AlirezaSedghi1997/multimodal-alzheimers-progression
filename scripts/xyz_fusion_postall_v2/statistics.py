"""Post-fusion metric repair; keep the parent's prediction contract immutable."""
from __future__ import annotations

import numpy as np
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.nonparametric import CensoringDistributionEstimator
from scripts.xyz_fusion_v1.statistics import check_curves, support, ibs_score


def metrics(train, test, curves, grid, horizons, min_at_risk=20):
    p = check_curves(curves, len(test), grid)
    tr, te, cutoff = support(train, test, grid)
    result = {'ibs': ibs_score(train, test, p, grid), 'metric_name': 'IPCW_INTEGRATED_BRIER_SCORE',
              'grid_days': list(grid), 'administrative_cutoff_days': float(cutoff),
              'ipcw_ibs_ties': 'SCIKIT_SURVIVAL_REVERSE_KM_LIBRARY_CONVENTION',
              'calibration_ties': 'EVENT_LEFT_LIMIT_HORIZON_RIGHT_LIMIT',
              'n': len(test), 'events': int(te['event'].sum()), 'horizons': {}}
    try:
        uno = float(concordance_index_ipcw(tr, te, 1 - p[:, -1], tau=grid[-1])[0])
        result['uno_c_index'] = uno if np.isfinite(uno) else None
    except ValueError:
        result['uno_c_index'] = None
    censor = CensoringDistributionEstimator().fit(tr)
    for h in horizons:
        if not grid[0] <= h <= grid[-1]:
            raise ValueError('HORIZON_OUTSIDE_LOCKED_PREDICTION_GRID')
        risk = 1 - np.array([np.interp(h, grid, row) for row in p])
        cases, controls = te['event'] & (te['time'] <= h), te['time'] > h
        at_risk = int((te['time'] >= h).sum())
        record = {'n_at_risk': at_risk, 'events_by_horizon': int(cases.sum()), 'status': 'NOT_ESTIMABLE'}
        if at_risk >= min_at_risk and cases.sum() >= 2 and controls.sum() >= 2:
            try:
                auc = float(cumulative_dynamic_auc(tr, te, risk, np.array([h]))[0][0])
                if not np.isfinite(auc):
                    raise ValueError('NONFINITE_DYNAMIC_AUC')
                record['dynamic_auc'] = auc
                used = cases | controls
                query = np.where(cases[used], np.nextafter(te['time'][used], -np.inf), h)
                g = censor.predict_proba(query)
                if np.any(g <= 0):
                    raise ValueError('ZERO_CENSORING_SUPPORT')
                r = np.clip(risk[used], 1e-6, 1 - 1e-6)
                logit = np.log(r / (1 - r))
                # A constant risk has an estimable intercept-only calibration,
                # but NOT an identifiable intercept AND slope. Do not index a
                # one-column statsmodels fit or pretend its slope is zero.
                design = np.column_stack([np.ones(len(logit)), logit])
                if np.linalg.matrix_rank(design) < 2:
                    raise ValueError('CONSTANT_RISK_CALIBRATION_SLOPE_NOT_IDENTIFIABLE')
                import statsmodels.api as sm
                fit = sm.GLM(cases[used].astype(float), design, family=sm.families.Binomial(), freq_weights=1 / g).fit()
                if not fit.converged or not np.isfinite(fit.params).all():
                    raise ValueError('CALIBRATION_NOT_ESTIMABLE')
                record.update(status='ESTIMABLE', calibration_intercept=float(fit.params[0]),
                              calibration_slope=float(fit.params[1]), calibration_is_descriptive_not_refit=True)
                bins = []
                for ids in np.array_split(np.argsort(r), 3):
                    if len(ids) < 10:
                        continue
                    weights = 1 / g[ids]
                    bins.append({'n': len(ids), 'predicted_risk': float(np.average(r[ids], weights=weights)),
                                 'ipcw_observed_fraction': float(np.average(cases[used][ids], weights=weights))})
                record['calibration_bins'] = bins
            except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                pass
        result['horizons'][str(h)] = record
    return result


def evaluate_cell(train, held, predictions, config, fold):
    values = {}
    for arm, pred in predictions.items():
        test = held.set_index('patient_hash').loc[pred['subjects']].reset_index()
        values[arm] = {'subjects': pred['subjects'], 'curves': pred['survival'],
                       'event': test.event.astype(int).tolist(), 'time': test.time_days.tolist(),
                       'censor_train': {'event': train.event.astype(int).tolist(), 'time': train.time_days.tolist()},
                       'metrics': metrics(train, test, pred['survival'], config['fusion']['ibs_grid_days'],
                                          config['fusion']['horizons_days'], config['fusion']['min_at_risk'])}
    return {'fold': fold, 'status': 'OUTER_EVAL_COMMITTED', 'arms': values,
            'train_subjects': sorted(train.patient_hash), 'test_subjects': sorted(held.patient_hash)}
