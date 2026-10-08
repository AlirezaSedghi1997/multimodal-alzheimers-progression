"""Censoring-aware fixed-grid evaluation; no outer-outcome model selection."""
from __future__ import annotations
import numpy as np
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.nonparametric import CensoringDistributionEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis


def outcome(frame):
    t=np.asarray(frame.time_days,dtype=float); e=np.asarray(frame.event,dtype=bool)
    if not np.isfinite(t).all() or np.any(t<=0) or not e.any():
        raise ValueError('SURVIVAL_OUTCOME_INVALID_OR_NO_EVENTS')
    return Surv.from_arrays(e,t)


def support(train,test,grid):
    tr=outcome(train); te=outcome(test)
    cutoff=np.nextafter(float(tr['time'].max()),0)
    te['event'] &= te['time']<cutoff
    te['time']=np.minimum(te['time'],cutoff)
    grid=np.asarray(grid,dtype=float)
    if grid[0]<te['time'].min() or grid[-1]>=te['time'].max() or grid[-1]>=cutoff:
        raise ValueError('PRESPECIFIED_SURVIVAL_GRID_NOT_ESTIMABLE')
    return tr,te,cutoff


def check_curves(curves, n, grid):
    p=np.asarray(curves,dtype=float)
    if p.shape!=(n,len(grid)) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or (np.diff(p,axis=1)>1e-6).any():
        raise ValueError('INVALID_SURVIVAL_CURVES')
    return p


def ibs_score(train,test,curves,grid):
    p=check_curves(curves,len(test),grid); tr,te,_=support(train,test,grid)
    value=float(integrated_brier_score(tr,te,p,np.asarray(grid)))
    if not np.isfinite(value) or not 0<=value<=1:
        raise ValueError('IPCW_IBS_NOT_ESTIMABLE')
    return value


def metrics(train,test,curves,grid,horizons,min_at_risk=20):
    p=check_curves(curves,len(test),grid); tr,te,cutoff=support(train,test,grid)
    result={'ibs':ibs_score(train,test,p,grid),'metric_name':'IPCW_INTEGRATED_BRIER_SCORE',
            'grid_days':list(grid),'administrative_cutoff_days':float(cutoff),
            'ipcw_ibs_ties':'SCIKIT_SURVIVAL_REVERSE_KM_LIBRARY_CONVENTION',
            'calibration_ties':'EVENT_LEFT_LIMIT_HORIZON_RIGHT_LIMIT',
            'n':len(test),'events':int(te['event'].sum()),'horizons':{}}
    try:
        result['uno_c_index']=float(concordance_index_ipcw(tr,te,1-p[:,-1],tau=grid[-1])[0])
    except ValueError:
        result['uno_c_index']=None
    censor=CensoringDistributionEstimator().fit(tr)
    for h in horizons:
        risk=1-np.array([np.interp(h,grid,row) for row in p])
        at_risk=int((te['time']>=h).sum()); cases=te['event']&(te['time']<=h); controls=te['time']>h
        record={'n_at_risk':at_risk,'events_by_horizon':int(cases.sum()),'status':'NOT_ESTIMABLE'}
        if at_risk>=min_at_risk and cases.sum()>=2 and controls.sum()>=2:
            try:
                auc=float(cumulative_dynamic_auc(tr,te,risk,np.array([h]))[0][0])
                record['dynamic_auc']=auc
                used=cases|controls
                query=np.where(cases[used],np.nextafter(te['time'][used],-np.inf),h)
                g=censor.predict_proba(query)
                if np.any(g<=0):
                    raise ValueError('ZERO_CENSORING_SUPPORT')
                import statsmodels.api as sm
                r=np.clip(risk[used],1e-6,1-1e-6); logit=np.log(r/(1-r))
                fit=sm.GLM(cases[used].astype(float),sm.add_constant(logit),family=sm.families.Binomial(),freq_weights=1/g).fit()
                if not fit.converged or not np.isfinite(fit.params).all():
                    raise ValueError('CALIBRATION_NOT_ESTIMABLE')
                record.update({'status':'ESTIMABLE','dynamic_auc':auc,
                               'calibration_intercept':float(fit.params[0]),'calibration_slope':float(fit.params[1]),
                               'calibration_is_descriptive_not_refit':True})
                # Risk-ordered aggregate bins: no outcome-defined thresholds or refit.
                bins=[]
                for ids in np.array_split(np.argsort(r),3):
                    if len(ids)<10:continue
                    weights=1/g[ids]
                    bins.append({'n':len(ids),'predicted_risk':float(np.average(r[ids],weights=weights)),
                                 'ipcw_observed_fraction':float(np.average(cases[used][ids],weights=weights))})
                record['calibration_bins']=bins
            except (ValueError,ZeroDivisionError,np.linalg.LinAlgError):
                pass
        result['horizons'][str(h)]=record
    return result


def fit_cox(frame,features,alpha):
    if any(x not in frame for x in features) or any(x in features for x in ('event','time_days','manufacturer','cohort','site')):
        raise ValueError('INVALID_FUSION_FEATURE_SCHEMA')
    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
                        CoxPHSurvivalAnalysis(alpha=float(alpha),n_iter=200,ties='breslow'))
    model.fit(frame[features],outcome(frame))
    return model


def cox_curves(model,frame,features,grid):
    funcs=model.predict_survival_function(frame[features])
    return np.asarray([f(np.asarray(grid)) for f in funcs])
