"""Patient-paired analysis; seeds and perturbations never inflate sample size."""
import numpy as np
from scripts.mx_science_extension_v1.analysis import calibration
from scripts.mx_science_extension_v1.baselines import _number, _sex
from .common import PANEL, read, require, progress

METRICS = ('auroc','auprc','brier')


def metrics(y, p, w=None):
    y,p = np.asarray(y),np.asarray(p)
    w = np.ones(len(y)) if w is None else np.asarray(w)
    if set(y)!={0,1}:
        return np.array([np.nan,np.nan,np.average((p-y)**2,weights=w)])
    # Weighted rank sums with half credit for exact ties; AP at tied-score groups.
    order = np.argsort(p,kind='stable')
    pp,yy,ww = p[order],y[order],w[order]
    starts = np.r_[0,np.flatnonzero(np.diff(pp)!=0)+1]
    pos = np.add.reduceat(ww*yy,starts); neg = np.add.reduceat(ww*(1-yy),starts)
    auc = np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum())
    cp,cn = np.cumsum(pos[::-1]),np.cumsum(neg[::-1])
    ap = np.sum(pos[::-1]*cp/(cp+cn))/pos.sum()
    return np.array([auc,ap,np.average((p-y)**2,weights=w)],dtype=float)


def paired(y, first, second, *, resamples, seed, weights=None, strata=None, pairs=None):
    y = np.asarray(y,dtype=int); a,b = np.asarray(first,dtype=float),np.asarray(second,dtype=float)
    if a.ndim==1: a=a[:,None]
    if b.ndim==1: b=b[:,None]
    require(len(y)==len(a)==len(b) and a.shape==b.shape and
            np.isfinite(a).all() and np.isfinite(b).all() and
            ((a>=0)&(a<=1)).all() and ((b>=0)&(b<=1)).all(), 'PAIRED_INVALID_PANEL')
    counts = {'n':len(y),'positive':int(y.sum()),'negative':int(len(y)-y.sum())}
    if min(counts['positive'],counts['negative'])<2:
        return {**counts,'status':'NOT_ESTIMABLE_TWO_CLASSES_REQUIRED'}
    w = np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
    require(len(w)==len(y) and np.isfinite(w).all() and (w>0).all(), 'INVALID_SAMPLING_WEIGHTS')
    def average(p,idx):
        return np.mean([metrics(y[idx],p[idx,s],w[idx]) for s in range(p.shape[1])],axis=0)
    full = np.arange(len(y)); am,bm = average(a,full),average(b,full)
    rng = np.random.default_rng(seed)
    if strata is not None:
        require(len(strata)==len(y),'INVALID_STRATA')
        strata = np.asarray(strata)
        groups = [np.flatnonzero(strata==v) for v in np.unique(strata)]
    if pairs is not None:
        pairs = np.asarray(pairs,dtype=int)
        require(pairs.ndim==2 and pairs.shape[1]==2 and len(np.unique(pairs))==len(y)
                and set(pairs.ravel())==set(full),'MATCHING_REPEATED_OR_MISSING_PATIENT')
    draws = []
    for _ in range(resamples):
        if pairs is not None:
            idx = pairs[rng.integers(len(pairs),size=len(pairs))].ravel()
        elif strata is not None:
            idx = np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        else:
            idx = rng.integers(len(y),size=len(y))
        draws.append(average(a,idx)-average(b,idx))
    draws = np.asarray(draws)
    result = {**counts,'status':'DESCRIPTIVE_POST_HOC_INTERNAL','effects':{},
              'first_metrics':dict(zip(METRICS,am.tolist())), 'second_metrics':dict(zip(METRICS,bm.tolist())),
              'seed_count':a.shape[1],'seeds_are_independent_patients':False,
              'bootstrap_unit':'matched_pair' if pairs is not None else 'patient',
              'stratified_by_fold_class':strata is not None,
              'ci_scope':'conditional_on_fixed_fitted_models_not_training_variance',
              'calibration_ci':'NOT_ESTIMATED_IN_QUICK_ADDON', 'effects_direction':'first_minus_second'}
    if weights is None:
        result['calibration_by_seed'] = {'first':[calibration(y,a[:,s]) for s in range(a.shape[1])],
                                         'second':[calibration(y,b[:,s]) for s in range(b.shape[1])]}
    for j,name in enumerate(METRICS):
        valid = draws[:,j][np.isfinite(draws[:,j])]
        result['effects'][name] = {'delta':float(am[j]-bm[j]),'valid_resamples':len(valid),
                                  'ci95':np.quantile(valid,[.025,.975]).tolist() if len(valid)>=.9*resamples else None}
    return result


def match_pairs(rows, caliper):
    pairs = []
    for f in sorted({r['fold'] for r in rows}):
        for sex in (0.,1.):
            eligible = [i for i,r in enumerate(rows) if r['fold']==f and r['sex']==sex and np.isfinite(r['age'])]
            positive = sorted((i for i in eligible if rows[i]['label']==1),key=lambda i:(rows[i]['age'],rows[i]['patient_hash']))
            controls = {i for i in eligible if rows[i]['label']==0}
            for i in positive:
                options = [j for j in controls if abs(rows[j]['age']-rows[i]['age'])<=caliper]
                if options:
                    j = min(options,key=lambda j:(abs(rows[j]['age']-rows[i]['age']),rows[j]['patient_hash']))
                    pairs.append((i,j)); controls.remove(j)
    return pairs


def smd(rows):
    result = {}
    for name in ('age','sex','education_years'):
        values = [np.asarray([r[name] for r in rows if r['label']==c and np.isfinite(r[name])]) for c in (0,1)]
        a,b = values
        sd = np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2) if min(len(a),len(b))>=2 else 0
        result[name] = {'negative_n':len(a),'positive_n':len(b),'smd_positive_minus_negative':float((b.mean()-a.mean())/sd) if sd>0 else None,
                        'missing_n':len(rows)-len(a)-len(b)}
    return result


def run_statistics(store):
    from .audit import data_rows
    _,_,tests = data_rows(store.root)
    cov = {r['patient_hash']:r for fold in tests for r in fold}
    panel = read(store.root/PANEL)['rows']
    nonlinear = store.receipt('nonlinear_panel')
    rows = []
    for r in panel:
        c = cov[r['patient_hash']]
        rows.append({**r, 'age':_number(c['age']), 'sex':_sex(c['sex']),
                     'education_years':_number(c['education_years']),
                     'direction':np.sign(float(c['pet_day_t0'])-float(c['t1_day_t0']))})
    cfg = store.config
    subsets = {'all':list(range(len(rows)))}
    subsets.update({f'fold_{f}':[i for i,r in enumerate(rows) if r['fold']==f] for f in range(5)})
    subsets.update({f'leave_out_fold_{f}':[i for i,r in enumerate(rows) if r['fold']!=f] for f in range(5)})
    subsets.update({f'sex_{s}':[i for i,r in enumerate(rows) if r['sex']==s] for s in (0,1)})
    subsets['age_lt75'] = [i for i,r in enumerate(rows) if r['age']<cfg['age_cut']]
    subsets['age_ge75'] = [i for i,r in enumerate(rows) if r['age']>=cfg['age_cut']]
    subsets.update({k:[i for i,r in enumerate(rows) if r['direction']==d] for k,d in (('PET_before_MRI',-1),('PET_after_MRI',1),('same_day',0))})
    pairs = match_pairs(rows,cfg['matching_age_caliper_years'])
    subsets['age_sex_matched'] = [i for pair in pairs for i in pair]
    output = {}
    for name,indices in subsets.items():
        progress('paired_statistics',subset=name,status='RUNNING_OR_REUSING')
        subset = [rows[i] for i in indices]
        def compute(subset=subset,name=name):
            y = np.array([r['label'] for r in subset],dtype=int)
            counts = {'n':len(y),'positive':int(y.sum()),'negative':int(len(y)-y.sum())}
            if min(counts['positive'],counts['negative'])<cfg['subgroup_min_per_class']:
                return {**counts,'status':'NOT_ESTIMABLE_PRESPECIFIED_MINIMUM_20_PER_CLASS'}
            a = [[r['predictions']['hybrid_selected'][str(s)] for s in cfg['seeds']] for r in subset]
            b = [[nonlinear[r['patient_hash']][str(s)] for s in cfg['seeds']] for r in subset]
            matched = np.arange(len(y)).reshape(-1,2) if name=='age_sex_matched' else None
            result = paired(y,a,b,resamples=cfg['bootstrap_resamples'],seed=cfg['bootstrap_seed'],pairs=matched)
            result['per_seed'] = {str(s):{'hybrid':metrics(y,np.asarray(a)[:,j]).tolist(),
                                        'nonlinear':metrics(y,np.asarray(b)[:,j]).tolist()} for j,s in enumerate(cfg['seeds'])}
            result['demographic_balance'] = smd(subset)
            return result
        output[name] = store.receipt('statistics/'+name,compute,public=True)
    summary = {'status':'COMPLETED_INTERNAL_SENSITIVITY','subsets':output,
               'matched_pairs':len(pairs),'unmatched_patients':len(rows)-2*len(pairs),
               'missing_age':sum(not np.isfinite(r['age']) for r in rows),
               'missing_sex':sum(not np.isfinite(r['sex']) for r in rows),
               'leave_one_fold_out_is_new_CV':False,'matching_fixes_cohort_confounding':False}
    return store.receipt('sensitivity_summary',lambda:summary,public=True)
