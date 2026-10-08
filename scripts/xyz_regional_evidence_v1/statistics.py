"""Censored regional association with explicit support and diagnostic gates."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def holm_family(values, alpha=.05):
    names = list(values)
    p = [values[n].get('p') for n in names]
    if any(v is not None and (not np.isfinite(v) or not 0 <= v <= 1) for v in p):
        raise ValueError('INVALID_P_VALUE')
    adjusted = multipletests([1. if v is None else v for v in p], alpha=alpha, method='holm')[1]
    return {n: dict(values[n], p_holm=float(adjusted[i]) if p[i] is not None else None,
                    multiplicity_supported=bool(p[i] is not None and adjusted[i] < alpha),
                    correction='HOLM_FIXED_FAMILY_INCLUDING_NOT_ESTIMABLE_TESTS', family_size=len(names))
            for i, n in enumerate(names)}


def standardized(frame, names):
    result, parameters = frame.copy(), {}
    if frame.patient_hash.duplicated().any():
        raise ValueError('DUPLICATE_SUBJECT')
    for col in names:
        values = pd.to_numeric(result[col], errors='raise').to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError('NONFINITE_ASSOCIATION_INPUT')
        mean, sd = values.mean(), values.std(ddof=1)
        if not np.isfinite(sd) or sd < 1e-10:
            raise ValueError('NON_ESTIMABLE_COVARIATE_VARIANCE:' + col)
        result[col + '_z'] = (values - mean) / sd
        parameters[col] = {'mean': float(mean), 'sd': float(sd), 'unit': 'UNIQUE_SUBJECT'}
    return result, parameters


def _fit(times, x, event, strata, groups, entry=None):
    """Remove only rows absent from every event risk set, not outcome outliers.

    Their partial-likelihood score is identically zero; statsmodels otherwise
    marks their score residual NaN, making its sandwich covariance undefined.
    The mask is proven from the risk sets and recorded, not nan_to_num applied
    to a fitted covariance. It also respects delayed-entry split records.
    """
    times, x, event = np.asarray(times, float), np.asarray(x, float), np.asarray(event, int)
    strata, groups = np.asarray(strata, str), np.asarray(groups, str)
    entry = np.zeros(len(times)) if entry is None else np.asarray(entry, float)
    active = np.zeros(len(times), bool)
    for level in sorted(set(strata)):
        ids = np.flatnonzero(strata == level)
        failures = np.unique(times[ids][event[ids] == 1])
        if len(failures):
            active[ids] = ((entry[ids, None] <= failures) & (times[ids, None] >= failures)).any(axis=1)
    if np.any(event[~active]):
        raise ValueError('EVENT_LOST_FROM_RISK_SET')
    if np.linalg.matrix_rank(x[active]) != x.shape[1] or np.linalg.cond(x[active]) > 1e8:
        raise ValueError('ASSOCIATION_DESIGN_RANK_OR_CONDITION')
    model = PHReg(times[active], x[active], status=event[active], entry=entry[active],
                  strata=strata[active], ties='breslow', missing='raise')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        fit = model.fit(groups=groups[active], disp=0, maxiter=300)
    covariance = fit.cov_params()
    if (not np.isfinite(fit.params).all() or not np.isfinite(covariance).all()
            or np.linalg.eigvalsh(covariance).min() <= 0
            or np.max(np.abs(model.score(fit.params))) > 1e-3):
        raise ValueError('ASSOCIATION_FIT_OR_COVARIANCE_FAILED')
    return fit, {'risk_set_noncontributing_rows': int((~active).sum()),
                  'informative_subjects': len(set(groups[active])), 'robust_covariance': 'SUBJECT_SANDWICH'}


def cox_association(data, features, options, diagnostics=True):
    if data.patient_hash.duplicated().any():
        raise ValueError('DUPLICATE_SUBJECT')
    x = data[features].to_numpy(float)
    times, event = data.time_days.to_numpy(float), data.event.to_numpy(int)
    if not np.isfinite(x).all() or not np.isfinite(times).all() or (times <= 0).any():
        raise ValueError('NONFINITE_OR_INVALID_ASSOCIATION_INPUT')
    if not set(event) <= {0,1} or 'region_z' not in features:
        raise ValueError('ASSOCIATION_SCHEMA')
    counts = {'n':len(data), 'events':int(event.sum()), 'parameters':len(features)}
    if (len(data) < options['minimum_subjects'] or
            event.sum() < max(options['minimum_events'], options['minimum_events_per_parameter'] * len(features))):
        return dict(counts, status='NOT_ESTIMABLE_SUPPORT', p=None)
    strata, groups = data.mri_manufacturer.to_numpy(str), data.patient_hash.to_numpy(str)
    idx = features.index('region_z')
    try:
        fit, evidence = _fit(times, x, event, strata, groups)
        beta, se = float(fit.params[idx]), float(fit.bse[idx])
        if abs(beta) + 2 * se > 700:
            raise ValueError('UNSTABLE_LOG_HAZARD')
        result = dict(counts, status='ESTIMATED', features=features, log_hr=beta,
            standard_error=se, hazard_ratio=float(np.exp(beta)),
            ci95=np.exp([beta-1.96*se, beta+1.96*se]).tolist(),
            p=float(2*norm.sf(abs(beta/se))), standardization='ONE_SD_LOG_SOURCE_ROI_VOLUME',
            manufacturer_stratified=True, pacc_or_saliency_was_not_target=True,
            observational_association_not_prediction_validation=True, **evidence)
        if diagnostics:
            # The primary test is fixed; diagnostics never choose a nicer p-value.
            quadratic = np.column_stack([x, x[:,idx]**2])
            try:
                nonlinear, _ = _fit(times, quadratic, event, strata, groups)
                statistic = 2*(nonlinear.model.loglike(nonlinear.params)-fit.model.loglike(fit.params))
                if statistic < -1e-6:
                    raise ValueError('NONLINEAR_NESTED_LIKELIHOOD')
                result['linearity'] = {'status':'TESTED_QUADRATIC_DEPARTURE',
                    'statistic':max(0.,float(statistic)), 'p':float(chi2.sf(max(0.,statistic),1)),
                    'nonsignificance_does_not_prove_linearity':True}
            except (ValueError, np.linalg.LinAlgError, ConvergenceWarning) as exc:
                result['linearity'] = {'status':'NOT_ESTIMABLE_NUMERICS','reason':str(exc)}
            cut = options['ph_step_days']
            early = int(event[times <= cut].sum()); late = int(event[times > cut].sum())
            result['ph'] = {'status':'NOT_ESTIMABLE_PERIOD_SUPPORT','early_events':early,'late_events':late}
            if min(early,late) >= 10:
                a, b, start, stop, outcome, labels, ids = [], [], [], [], [], [], []
                for i in range(len(times)):
                    periods = [(0., times[i], event[i], 0.)] if times[i] <= cut else [
                        (0., cut, 0, 0.), (np.nextafter(cut, np.inf), times[i], event[i], 1.)]
                    for left,right,status,post in periods:
                        a.append(x[i]); b.append(np.r_[x[i], x[i,idx]*post])
                        start.append(left); stop.append(right); outcome.append(status)
                        labels.append(strata[i]); ids.append(groups[i])
                try:
                    null,_ = _fit(stop,a,outcome,labels,ids,start)
                    alt,_ = _fit(stop,b,outcome,labels,ids,start)
                    if not np.isclose(null.model.loglike(null.params),fit.model.loglike(fit.params),atol=1e-6,rtol=0):
                        raise ValueError('SPLITTING_CHANGED_RISK_SET_LIKELIHOOD')
                    lr=2*(alt.model.loglike(alt.params)-null.model.loglike(null.params))
                    if lr < -1e-6:raise ValueError('PH_NESTED_LIKELIHOOD')
                    result['ph']={'status':'TESTED_TWO_YEAR_ROI_TIME_INTERACTION',
                        'statistic':max(0.,float(lr)), 'p':float(chi2.sf(max(0.,lr),1)),
                        'time_cut_days':cut,'early_events':early,'late_events':late,
                        'test_scope':'ROI_STEP_ONLY_NOT_ALL_POSSIBLE_NONPH',
                        'nonsignificance_does_not_prove_ph':True}
                except (ValueError,np.linalg.LinAlgError,ConvergenceWarning) as exc:
                    result['ph']={'status':'NOT_ESTIMABLE_NUMERICS','reason':str(exc)}
        return result
    except (ValueError,np.linalg.LinAlgError,ConvergenceWarning) as exc:
        return dict(counts,status='NOT_ESTIMABLE_NUMERICS',p=None,reason=type(exc).__name__+':'+str(exc))


def pacc_association(table, covariates):
    """One baseline ROI, repeated measurements nested in subject; same fit family."""
    import statsmodels.formula.api as smf
    formula='pacc ~ years * ('+' + '.join(['region_z',*covariates])+') + C(mri_manufacturer)'
    table=table.copy();table['random_time_centered']=table.years-2.25
    model=smf.mixedlm(formula,table,groups=table.patient_hash,re_formula='~random_time_centered',missing='raise')
    attempts=[];valid=[]
    for method in ('lbfgs','bfgs'):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fit=model.fit(reml=False,method=method,maxiter=1000)
            good=(fit.converged and np.isfinite(np.asarray(fit.params)).all() and
                  np.isfinite(np.asarray(fit.cov_params())).all() and np.linalg.eigvalsh(fit.cov_re).min()>0 and
                  np.linalg.eigvalsh(fit.cov_params()).min()>0 and
                  np.max(np.abs(model.score(fit.params_object)))/len(table)<1e-4 and
                  not any('singular' in str(w.message).lower() or 'not positive definite' in str(w.message).lower() for w in caught))
            attempts.append({'optimizer':method,'valid':bool(good),
                'log_likelihood':float(fit.llf) if np.isfinite(fit.llf) else None})
            if good:valid.append(fit)
        except (ValueError,np.linalg.LinAlgError) as exc:
            attempts.append({'optimizer':method,'valid':False,'reason':str(exc)})
    if not valid:return {'status':'NOT_ESTIMABLE_MIXED_MODEL_NUMERICS','p':None,'attempts':attempts}
    fit=max(valid,key=lambda f:f.llf);term='years:region_z'
    beta,se=float(fit.fe_params[term]),float(fit.bse[term])
    return {'status':'ESTIMATED','n_subjects':int(table.patient_hash.nunique()),'n_measurements':len(table),
        'estimate':beta,'standard_error':se,'ci95':fit.conf_int().loc[term].tolist(),
        'p':float(2*norm.sf(abs(beta/se))),'effect_units':'PACC_POINTS_PER_YEAR_PER_SD_LOG_ROI_VOLUME',
        'formula':formula,'attempts':attempts,'random_effects':'INTERCEPT_AND_TIME_SLOPE',
        'missing_at_random_not_proven':True,'independent_replication':False}
