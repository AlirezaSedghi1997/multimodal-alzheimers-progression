"""Matched longitudinal association; no claim of independent validation."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from .context import parquet_atomic
from .core import file_sha


def subject_standardize(table, column):
    unique = table[['patient_hash', column]].drop_duplicates()
    if unique.patient_hash.duplicated().any():
        raise ValueError('BASELINE_VALUE_CHANGES_WITHIN_SUBJECT')
    sd = float(unique[column].std(ddof=1))
    if not np.isfinite(sd) or sd <= 1e-8:
        raise ValueError('NON_ESTIMABLE_BASELINE_VARIANCE:' + column)
    mean = float(unique[column].mean())
    return (table[column] - mean) / sd, {'mean': mean, 'sd': sd, 'unit': 'UNIQUE_SUBJECT'}


def prepare_pacc(ctx):
    from scripts.xyz_fusion_v1.audit import _table, RAW
    from scripts.xyz_fusion_v2.protocol import CLINICAL
    ctx.ensure_inputs()
    cells = ctx.parent.job('mri_ensemble_cells')
    scores = []
    for cell in cells:
        a = cell['arms']['clinical_mri_amyloid']
        grid = a['metrics']['grid_days']
        for key, curve in zip(a['subjects'], a['curves']):
            scores.append({'patient_hash': key, 'fold': cell['fold'],
                           'oof_risk_3y': 1 - float(np.interp(1095.75, grid, curve))})
    score = pd.DataFrame(scores)
    if score.patient_hash.duplicated().any():
        raise ValueError('DUPLICATED_OOF_SCORE')
    base = ctx.frame[['patient_hash', 'landmark_days_t0', *CLINICAL, 'event']].merge(score, validate='one_to_one')
    pacc = _table(RAW / 'Derived Data/PACC.csv')
    visits = _table(RAW / 'Derived Data/SV.csv')
    visits = visits.loc[visits.patient_hash.isin(base.patient_hash)].copy()
    pacc = pacc.loc[pacc.patient_hash.isin(base.patient_hash)].copy()
    visits['day'] = pd.to_numeric(visits.SVSTDTC_DAYS_T0, errors='coerce')
    dates = visits.groupby(['patient_hash', 'VISITCD']).day.agg(['min', 'max'])
    exact = dates.loc[dates['min'].eq(dates['max']), 'min'].rename('day').reset_index().rename(columns={'VISITCD': 'VISCODE'})
    table = pacc.merge(exact, on=['patient_hash', 'VISCODE'], validate='many_to_one')
    table = table.merge(base, on='patient_hash', validate='many_to_one')
    table['years'] = (table.day - table.landmark_days_t0) / 365.25
    table['pacc'] = pd.to_numeric(table['PACC.raw'], errors='coerce')
    counts = {'parent_subjects': len(base), 'linked_measurements': len(table),
              'ambiguous_visit_dates_excluded': int((dates['min'] != dates['max']).sum())}
    table = table.loc[(table.years > 0) & (table.years <= 4.5)].copy()
    counts['post_landmark_window_measurements'] = len(table)
    table = table.dropna(subset=['pacc', 'years', 'oof_risk_3y', *CLINICAL]).copy()
    # Same-day repeats are one observation; contradictory same-day scores fail.
    conflicts = table.groupby(['patient_hash', 'day']).pacc.nunique()
    if (conflicts > 1).any():
        raise ValueError('CONFLICTING_SAME_DAY_PACC')
    table = table.drop_duplicates(['patient_hash', 'day'])
    enough = table.groupby('patient_hash').day.nunique()
    table = table.loc[table.patient_hash.isin(enough[enough >= 2].index)].copy()
    counts.update(complete_measurements=len(table), eligible_subjects=int(table.patient_hash.nunique()),
                  excluded_subjects=int(len(base) - table.patient_hash.nunique()))
    attrition = []
    for label, t in [('included', base.loc[base.patient_hash.isin(table.patient_hash)]),
                     ('excluded', base.loc[~base.patient_hash.isin(table.patient_hash)])]:
        attrition.append({'group': label, 'subjects': len(t), 'events': int(t.event.sum()),
                          'mean_age': float(t.age.mean()) if len(t) else None,
                          'mean_baseline_pacc': float(t.baseline_pacc_raw.mean()) if len(t) else None,
                          'mean_oof_risk': float(t.oof_risk_3y.mean()) if len(t) else None})
    scaling = {}
    if len(table):
        for column in ['oof_risk_3y', 'age', 'education_years', 'baseline_pacc_raw', 'baseline_mmscore']:
            table[column + '_z'], scaling[column] = subject_standardize(table, column)
    counts['subjects_observed_at_or_after_4y'] = int(table.loc[table.years >= 4].patient_hash.nunique())
    path = ctx.out / 'private/pacc_longitudinal.parquet'
    parquet_atomic(path, table[['patient_hash', 'fold', 'day', 'years', 'pacc', 'oof_risk_3y', *CLINICAL,
                               *[c + '_z' for c in scaling]]])
    return table, {'counts': counts, 'attrition': attrition, 'scaling': scaling,
                   'outputs': {path.relative_to(ctx.out).as_posix(): file_sha(path)}}


def fit_pacc_models(table):
    """Optimize the SAME ML model with an equivalent random-effect basis.

    Centering the time basis changes no fitted model when its 2x2 random-effect
    covariance remains unstructured. Optimizers target the same likelihood;
    neither p-values nor the direction of the risk association select a fit.
    API: https://www.statsmodels.org/v0.14.6/generated/statsmodels.regression.mixed_linear_model.MixedLM.fit.html
    """
    import statsmodels.formula.api as smf
    table = table.copy()
    table['random_time_centered'] = table.years - 2.25
    covariates = ['age_z', 'education_years_z', 'baseline_pacc_raw_z', 'baseline_mmscore_z',
                  'sex_binary', 'apoe4_carrier', 'treatment_active']
    models = {}
    for name, extra in [('base', []), ('augmented', ['oof_risk_3y_z'])]:
        formula = 'pacc ~ years * (' + ' + '.join(covariates + extra) + ')'
        model = smf.mixedlm(formula, table, groups=table.patient_hash,
                            re_formula='~random_time_centered', missing='raise')
        attempts, candidates = [], []
        for optimizer in ('lbfgs', 'bfgs', 'powell'):
          try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fit = model.fit(reml=False, method=optimizer, maxiter=1000)
                if optimizer == 'powell':
                    fit = model.fit(reml=False, method='lbfgs', start_params=fit.params_object, maxiter=1000)
            covariance = np.asarray(fit.cov_params())
            finite = np.isfinite(fit.params).all() and np.isfinite(covariance).all()
            variances_valid = bool(finite and np.linalg.eigvalsh(covariance).min() > 0 and
                                   np.linalg.eigvalsh(fit.cov_re).min() > 0)
            gradient_per_observation = float(np.max(np.abs(model.score(fit.params_object))) / len(table))
            warnings_seen = sorted(set(str(w.message) for w in caught))
            bad_hessian = any('not positive definite' in w.lower() or 'singular' in w.lower() for w in warnings_seen)
            valid = bool(fit.converged and finite and variances_valid and not bad_hessian and
                         np.isfinite(gradient_per_observation) and gradient_per_observation < 1e-4)
            attempts.append({'optimizer': optimizer, 'converged': bool(fit.converged),
                             'log_likelihood': float(fit.llf) if np.isfinite(fit.llf) else None,
                             'gradient_per_observation': gradient_per_observation,
                             'valid': valid, 'warnings': warnings_seen})
            if valid:
                candidates.append(fit)
            if len(candidates) >= 2 and abs(candidates[-1].llf - candidates[-2].llf) < .001:
                break
          except (ValueError, np.linalg.LinAlgError) as exc:
            attempts.append({'optimizer': optimizer, 'valid': False,
                             'reason': type(exc).__name__ + ':' + str(exc)})
        if candidates:
            fit = max(candidates, key=lambda value: value.llf)
            fixed_names = list(fit.fe_params.index)
            interval = fit.conf_int()
            models[name] = {'status': 'ESTIMATED', 'formula': formula,
                            'converged': bool(fit.converged), 'optimizer_attempts': attempts,
                            'random_effect_covariance_eigenvalues': np.linalg.eigvalsh(fit.cov_re).tolist(),
                            'coefficients': {k: {'estimate': float(fit.fe_params[k]), 'ci95': interval.loc[k].tolist()}
                                             for k in fixed_names},
                            'aic': float(fit.aic) if np.isfinite(fit.aic) else None,
                            'n': int(fit.nobs), 'random_effects': 'INTERCEPT_AND_TIME_SLOPE',
                            'equivalent_random_time_basis_center_years': 2.25,
                            'selected_by': 'MAXIMUM_VALID_LIKELIHOOD_NOT_SIGNIFICANCE',
                            'unregularized_ML_same_rows': True}
        else:
            models[name] = {'status': 'NOT_ESTIMABLE_NUMERICS', 'optimizer_attempts': attempts,
                            'formula': formula, 'coefficients': {}}
    return models


def pacc_analysis(ctx):
    table, report = prepare_pacc(ctx)
    report.update(status='NOT_ESTIMABLE_SUPPORT', missing_at_random_not_proven=True,
                  association_not_validation=True, cnn_refit=False,
                  uncertainty='CONDITIONAL_ON_LOCKED_GENERATED_OOF_PREDICTORS_NOT_JOINT_TRAINING_UNCERTAINTY')
    if table.patient_hash.nunique() < 30:
        return report
    models = report['models'] = fit_pacc_models(table)
    report['status'] = ('COMPLETED_SECONDARY_ASSOCIATION' if all(m['status'] == 'ESTIMATED' for m in models.values())
                        else 'NOT_ESTIMABLE_MIXED_MODEL_NUMERICS')
    return report
