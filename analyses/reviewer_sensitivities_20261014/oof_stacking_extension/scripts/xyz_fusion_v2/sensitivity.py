"""Prespecified sensitivity analyses; unidentifiable assumptions remain explicit."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scripts.xyz_fusion_v1.report import individual_bs, cell_frame
from scripts.xyz_fusion_v1.audit import _table, RAW
from .protocol import CLINICAL, VOLUMES


def missingness(frame, cells, config):
    """Outer-train propensity; paired IPCW-loss reweighting, conditional on models."""
    feature = CLINICAL+['amyloid_centiloid']+VOLUMES
    samples = []; support = []
    for fold in cells:
        train = frame.loc[frame.patient_hash.isin(fold['train_subjects'])]
        test = frame.loc[frame.patient_hash.isin(fold['test_subjects'])]
        available = train.blood_ptau217.notna().astype(int)
        if available.nunique()!=2:
            return {'status': 'NOT_ESTIMABLE_PROPENSITY_CLASSES'}
        model = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(),
                              LogisticRegression(C=1., max_iter=2000, random_state=config['seed']))
        model.fit(train[feature], available)
        matched = test.loc[test.blood_ptau217.notna()]
        p = model.predict_proba(matched[feature])[:,1]
        clip = config['secondary']['ipw_clip']; clipped = np.clip(p, *clip)
        scale = train.blood_log_ptau217.std()
        if not np.isfinite(scale) or scale<1e-8: return {'status':'NOT_ESTIMABLE_BLOOD_VARIANCE'}
        z = (matched.blood_log_ptau217-train.blood_log_ptau217.mean())/scale
        weight = pd.Series(available.mean()/clipped, index=matched.patient_hash)
        z = pd.Series(z.to_numpy(), index=matched.patient_hash)
        support.append({'fold': fold['fold'], 'min_probability': float(p.min()),
                        'clipped_count': int((p!=clipped).sum()), 'n':len(p)})
        # The scalar comparator exists in both scalar and MRI tracks.
        a='blood_clinical_amyloid'; b='blood_clinical_amyloid_blood'
        if a not in fold['arms'] or b not in fold['arms']: return {'status':'NOT_ESTIMABLE_REQUIRED_MATCHED_ARMS'}
        ca, cb = fold['arms'][a], fold['arms'][b]
        if ca['subjects'] != cb['subjects']: raise ValueError('IPW_UNMATCHED_SUBJECTS')
        tr, te = cell_frame(ca)
        la, _, _ = individual_bs(tr, te, np.asarray(ca['curves']), config['fusion']['ibs_grid_days'])
        lb, _, _ = individual_bs(tr, te, np.asarray(cb['curves']), config['fusion']['ibs_grid_days'])
        samples.append((lb-la, weight.loc[ca['subjects']].to_numpy(), z.loc[ca['subjects']].to_numpy()))
    delta=np.concatenate([s[0] for s in samples]);w=np.concatenate([s[1] for s in samples]);z=np.concatenate([s[2] for s in samples])
    rng=np.random.default_rng(config['seed']); scenarios=[]
    for shift in config['secondary']['mnar_log_odds_delta']:
        weights=w*np.exp(np.clip(shift*z,-5,5));estimates=[]
        for _ in range(config['fusion']['bootstrap_resamples']):
            ids=rng.integers(0,len(delta),len(delta))
            estimates.append(float(np.average(delta[ids],weights=weights[ids])))
        scenarios.append({'log_selection_tilt':shift,'delta_ibs':float(np.average(delta,weights=weights)),
                          'conditional_ci95':np.quantile(estimates,[.025,.975]).tolist(),
                          'effective_sample_size':float(weights.sum()**2/(weights**2).sum())})
    return {'status':'COMPLETED_ASSUMPTION_SENSITIVITY','support':support,'scenarios':scenarios,
            'zero_tilt_assumption':'OBSERVED_COVARIATE_MAR_AND_POSITIVITY',
            'nonzero_tilt':'MNAR_SELECTION_ODDS_SENSITIVITY_ON_STANDARDIZED_OBSERVED_BLOOD',
            'mnar_identified_or_eliminated':False, 'retraining_uncertainty_in_ci':False}


def ph_diagnostic(frame, features):
    """Train-only nested likelihood-ratio test of a prespecified 2-year PH departure.

    An unpenalized reference model supplies the standard LR test, not an invented
    exact p-value for the penalized prediction model. RSF is the prediction sensitivity.
    """
    from statsmodels.duration.hazard_regression import PHReg
    from scipy.stats import chi2
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    x=StandardScaler().fit_transform(SimpleImputer(keep_empty_features=True).fit_transform(frame[features]))
    keep=np.std(x,axis=0)>1e-8
    x=x[:,keep];names=np.array(features)[keep]
    # A prespecified step in coefficients at 2 years; delayed-entry split records
    # preserve risk sets. Repeated intervals are not independent subjects.
    xx=[];zz=[];entry=[];stop=[];event=[]
    for i,row in enumerate(frame.itertuples()):
        if row.time_days>730.5:
            xx.append(x[i]);zz.append(np.zeros_like(x[i]));entry.append(0.);stop.append(730.5);event.append(0)
            xx.append(x[i]);zz.append(x[i]);entry.append(730.5);stop.append(row.time_days);event.append(row.event)
        else:
            xx.append(x[i]);zz.append(np.zeros_like(x[i]));entry.append(0.);stop.append(row.time_days);event.append(row.event)
    a=np.asarray(xx);b=np.column_stack([xx,zz])
    # PHReg inspects auxiliary .ndim before coercion; keep interval values unchanged.
    stop=np.asarray(stop,dtype=np.float64)
    entry=np.asarray(entry,dtype=np.float64)
    event=np.asarray(event,dtype=np.int64)
    if int(frame.event.sum())<5*b.shape[1]:
        return {'status':'NOT_ESTIMABLE_PH_REFERENCE_MODEL_EVENT_SUPPORT','events':int(frame.event.sum())}
    try:
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            null=PHReg(stop,a,status=event,entry=entry,ties='breslow').fit(disp=0,maxiter=200)
            alt=PHReg(stop,b,status=event,entry=entry,ties='breslow').fit(disp=0,maxiter=200)
        if not all(np.isfinite(f.params).all() and np.max(np.abs(f.model.score(f.params)))<1e-3 for f in (null,alt)):
            raise ValueError('PH_REFERENCE_MODEL_NOT_CONVERGED')
        value=2*(float(alt.model.loglike(alt.params))-float(null.model.loglike(null.params)))
        if not np.isfinite(value) or value < -1e-6:raise ValueError('NONFINITE_PH_LR')
        p=float(chi2.sf(max(0,value),a.shape[1]))
        return {'status':'TRAIN_ONLY_PH_STEP_INTERACTION_LR_REFERENCE_TEST','statistic':max(0,value),
                'p_approx':p,'degrees_of_freedom':a.shape[1],'features':names.tolist(),'time_cut_days':730.5,
                'formal_unpenalized_reference_test':True,'exact_test_of_penalized_predictor':False,
                'prediction_model_refit_on_test':False,
                'absence_of_significance_proves_ph':False}
    except (ValueError,np.linalg.LinAlgError,ConvergenceWarning) as exc:
        return {'status':'NOT_ESTIMABLE_PH_NUMERICS','reason':type(exc).__name__}


def scanner_splits(frame, column, config):
    if column not in ('mri_manufacturer','mri_site'): raise ValueError('SCANNER_SCOPE_REQUIRED')
    eligible=[]; excluded=[]
    for name,t in frame.groupby(column):
        n=len(t); e=int(t.event.sum()); other=frame.loc[frame[column]!=name]
        if name!='UNKNOWN' and n>=config['secondary']['min_group_subjects'] and e>=config['secondary']['min_group_events'] and int(other.event.sum())>=30:
            eligible.append({'group':str(name),'train':sorted(other.patient_hash),'test':sorted(t.patient_hash)})
        else: excluded.append({'group':str(name),'n':n,'events':e,'status':'NOT_ESTIMABLE_SUPPORT'})
    return eligible, excluded


def power_simulation(n, event_fraction, config):
    """Scenario power for ONE added standardized tau effect, not observed post-hoc power."""
    from statsmodels.duration.hazard_regression import PHReg
    rng=np.random.default_rng(config['seed']); rows=[]
    for beta in config['secondary']['power_effect_log_hazard']:
        successes=0; valid=0; event_counts=[]
        for _ in range(config['secondary']['power_repeats']):
            base=rng.normal(size=n); tau=.4*base+np.sqrt(.84)*rng.normal(size=n)
            event_time=rng.exponential(size=n)/np.exp(.4*base+beta*tau)
            # Exponential scenario; calibrate censoring approximately, not from observed associations.
            censor=rng.exponential(scale=event_fraction/max(1-event_fraction,1e-3),size=n)
            event=event_time<=censor;time=np.minimum(event_time,censor)
            try:
                fit=PHReg(time,np.column_stack([base,tau]),status=event).fit(disp=0)
                p=float(fit.pvalues[1])
                if np.isfinite(p):valid+=1;successes+=p<.05;event_counts.append(int(event.sum()))
            except (ValueError,np.linalg.LinAlgError):pass
        rows.append({'log_hazard_ratio_per_sd':beta,'valid_simulations':valid,
                     'scenario_power':successes/valid if valid else None,
                     'mean_simulated_events':float(np.mean(event_counts)) if event_counts else None})
    return {'status':'SCENARIO_SIMULATION_NOT_OBSERVED_POWER','n':n,'scenarios':rows,
            'assumed_tau_baseline_correlation':.4,'estimated_actual_effect':False,
            'simulation_does_not_select_favorable_split_or_gate_threshold':True}


def longitudinal_pacc(frame):
    import statsmodels.formula.api as smf
    pacc=_table(RAW/'Derived Data/PACC.csv');visits=_table(RAW/'Derived Data/SV.csv')
    visits['day']=pd.to_numeric(visits.SVSTDTC_DAYS_T0,errors='coerce')
    g=visits.groupby(['patient_hash','VISITCD']).day.agg(['min','max'])
    exact=g.loc[g['min']==g['max'],'min'].rename('day').reset_index().rename(columns={'VISITCD':'VISCODE'})
    t=pacc.merge(exact,on=['patient_hash','VISCODE'],validate='many_to_one')
    t=t.merge(frame[['patient_hash','landmark_days_t0',*CLINICAL,'amyloid_centiloid']],on='patient_hash',validate='many_to_one')
    t['years']=(t.day-t.landmark_days_t0)/365.25;t['pacc']=pd.to_numeric(t['PACC.raw'],errors='coerce')
    t=t.loc[t.years.between(0,4.5)].dropna(subset=['pacc','age','sex_binary','education_years','amyloid_centiloid'])
    enough=t.groupby('patient_hash').years.nunique();t=t.loc[t.patient_hash.isin(enough[enough>=2].index)]
    near=t.groupby('patient_hash').years.min()
    t=t.loc[t.patient_hash.isin(near[near<=183/365.25].index)]
    if t.patient_hash.nunique()<30:return {'status':'NOT_ESTIMABLE_LONGITUDINAL_SUPPORT'}
    unique=t.drop_duplicates('patient_hash').amyloid_centiloid
    if not np.isfinite(unique.std()) or unique.std()<1e-8:return {'status':'NOT_ESTIMABLE_PACC_AMYLOID_VARIANCE'}
    t['amyloid_sd']=(t.amyloid_centiloid-unique.mean())/unique.std()
    try:
        fit=smf.mixedlm('pacc ~ years*amyloid_sd + age + sex_binary + education_years + years*treatment_active',
                        t,groups=t.patient_hash,re_formula='~years').fit(reml=False,method='lbfgs',maxiter=1000)
        name='years:amyloid_sd';ci=fit.conf_int().loc[name].tolist()
        return {'status':'COMPLETED_ASSOCIATIONAL_SECONDARY' if fit.converged else 'NOT_ESTIMABLE_NOT_CONVERGED',
                'subjects':int(t.patient_hash.nunique()),'measurements':len(t),
                'time_by_amyloid_coefficient':float(fit.params[name]),'ci95':ci,
                'predictive_external_validation':False,'future_pacc_used_as_baseline_predictor':False}
    except (ValueError,np.linalg.LinAlgError): return {'status':'NOT_ESTIMABLE_MIXED_MODEL_NUMERICS'}
