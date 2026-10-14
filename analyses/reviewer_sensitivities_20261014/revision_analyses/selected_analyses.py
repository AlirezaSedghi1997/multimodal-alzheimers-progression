"""Post-hoc, aggregate-only analyses of frozen participant-level predictions.

Input data remain restricted. Call with --repo and --output; never upload the
private execution manifest or intermediate individual predictions.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, warnings
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.metrics import concordance_index_ipcw
from threadpoolctl import threadpool_limits

INPUTS={}
def load(path):
    path=Path(path); b=path.read_bytes()
    INPUTS[str(path)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    obj=json.loads(b.decode('utf-8-sig'))
    return obj.get('payload',obj) if isinstance(obj,dict) else obj

def parquet(path):
    path=Path(path);INPUTS[str(path)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return pd.read_parquet(path)

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf8')

def setup(repo):
    global ROOT,SUP,GRID,CONFIG,JOB,UNIT
    ROOT=Path(repo).resolve()
    bundled=Path(__file__).resolve().parents[1]/'source_snapshot'
    sys.path.insert(0,str(bundled if bundled.is_dir() else ROOT))
    SUP=ROOT/'thesis-workspace/supervisor_revision_20261011'
    JOB=ROOT/'results/xyz_fusion_v2/private/jobs'
    UNIT=ROOT/'results/xyz_fusion_postall_v2/private/units'
    CONFIG=load(ROOT/'results/xyz_fusion_v2/private/contract.json')['protocol']
    GRID=np.array(CONFIG['fusion']['ibs_grid_days'],float)

def fold_arrays(arm):
    from scripts.xyz_fusion_v1.report import cell_frame,individual_bs
    train,test=cell_frame(arm)
    loss,tr,te=individual_bs(train,test,np.asarray(arm['curves']),GRID)
    return loss,tr,te

def concordance_matrices(tr,te,risk,tau):
    """Exact weighted pair representation of sksurv IPCW C, including ties.

    Sampling participant multiplicities is equivalent to duplicating their
    records. No cross-fold comparison pairs or independence of seeds is assumed.
    """
    event,tm=te['event'],te['time'];active=tm<tau
    ipcw=np.zeros(len(te))
    if event[active].any():
        ipcw[active]=CensoringDistributionEstimator().fit(tr).predict_ipcw(te[active])
    comparable=event[:,None]&((tm[:,None]<tm[None,:])|((tm[:,None]==tm[None,:])&~event[None,:]))
    den=comparable*ipcw[:,None]**2
    diff=risk[:,None]-risk[None,:]
    reward=(diff>1e-8).astype(float)+0.5*(np.abs(diff)<=1e-8)
    return den*reward,den

def ratio_quadratic(counts,num,den):
    numerator=np.einsum('bi,bi->b',counts@num,counts)
    denominator=np.einsum('bi,bi->b',counts@den,counts)
    return np.divide(numerator,denominator,out=np.full(len(counts),np.nan),where=denominator>0)

def bootstrap_counts(lengths,b,seed):
    rng=np.random.default_rng(seed)
    out=[np.empty((b,n),float) for n in lengths]
    # Preserve the original interleaved fold/draw RNG sequence for reconciliation.
    for k in range(b):
        for f,n in enumerate(lengths):out[f][k]=np.bincount(rng.integers(0,n,n),minlength=n)
    return out

def interval(values,family=1):
    a=np.asarray(values);a=a[np.isfinite(a)]
    return np.quantile(a,[.05/(2*family),1-.05/(2*family)]).tolist() if len(a) else None

def validate_concordance():
    from sksurv.util import Surv
    rng=np.random.default_rng(81);errors=[];undefined=0
    for trial in range(10):
        tr=Surv.from_arrays(rng.random(180)<.45,rng.integers(1,80,180).astype(float))
        te=Surv.from_arrays(rng.random(45)<.35,rng.integers(1,60,45).astype(float))
        risk=np.round(rng.random(45),1);a,d=concordance_matrices(tr,te,risk,40)
        for k in range(10):
            idx=rng.integers(0,45,45);ct=np.bincount(idx,minlength=45)[None,:]
            accelerated=ratio_quadratic(ct,a,d)[0]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');reference=concordance_index_ipcw(tr,te[idx],risk[idx],tau=40)[0]
                if np.isfinite(reference):errors.append(abs(reference-accelerated))
                else:assert not np.isfinite(accelerated);undefined+=1
            except ValueError:assert not np.isfinite(accelerated);undefined+=1
    assert max(errors)<1e-12
    return {'synthetic_bootstrap_comparisons':len(errors),'max_absolute_error':max(errors),'undefined':undefined,'risk_and_time_ties_included':True}

def tau_bootstrap(out,b=40000):
    groups=[('cox',[load(JOB/f'tau_cox_ridge_o{i}_evaluation.json') for i in range(5)],'clinical_amyloid','clinical_amyloid_tau'),
            ('rsf',[load(JOB/f'tau_rsf_o{i}_evaluation.json') for i in range(5)],'clinical_amyloid','clinical_amyloid_tau'),
            ('mri',[load(UNIT/f'tau_fold{i}_evaluation.json') for i in range(5)],'base','plus_tau')]
    result={'status':'POST_HOC_FIXED_PREDICTION_SENSITIVITY','resamples':b,'seed':20261011,'accelerated_metric_check':validate_concordance(),'families':{'3':'Original three tau contrasts per metric','20':'All 20 specified main contrasts within each metric','40':'Those 20 contrasts across both IBS and concordance'},'covers_historical_model_selection':False,'models':{}}
    original=load(SUP/'tau_multiplicity.json')
    for name,cells,a,bname in groups:
        lengths=[len(c['arms'][a]['subjects']) for c in cells]
        counts=bootstrap_counts(lengths,b,20261011)
        ds=np.zeros(b);dc=np.zeros(b);valid=np.ones(b,bool);fails={};folds=[];pointc=[];pointbs=[]
        for f,c in enumerate(cells):
            av,bv=c['arms'][a],c['arms'][bname]
            for field in ['subjects','event','time','censor_train']:assert av[field]==bv[field]
            la,tr,te=fold_arrays(av);lb,_,_=fold_arrays(bv)
            ra=1-np.array(av['curves'])[:,-1];rb=1-np.array(bv['curves'])[:,-1]
            ma,de=concordance_matrices(tr,te,ra,GRID[-1]);mb,db=concordance_matrices(tr,te,rb,GRID[-1]);assert np.array_equal(de,db)
            ca=ratio_quadratic(counts[f],ma,de);cb=ratio_quadratic(counts[f],mb,de)
            bad=~(np.isfinite(ca)&np.isfinite(cb));valid&=~bad;fails[str(f+1)]=int(bad.sum())
            ds+=counts[f]@(lb-la)/sum(lengths);dc+=(cb-ca)*lengths[f]/sum(lengths)
            ones=np.ones((1,len(te)));pca=ratio_quadratic(ones,ma,de)[0];pcb=ratio_quadratic(ones,mb,de)[0]
            for risk,p in [(ra,pca),(rb,pcb)]:assert abs(p-concordance_index_ipcw(tr,te,risk,tau=GRID[-1])[0])<1e-12
            pointc.append(pcb-pca);pointbs.append(float((lb-la).mean()))
            folds.append({'fold':f+1,'n':len(te),'events_3y':int((te['event']&(te['time']<=GRID[-1])).sum()),'delta_ibs':pointbs[-1],'delta_uno':pointc[-1]})
        rec={'folds':folds,'delta_ibs':float(np.average(pointbs,weights=lengths)),'delta_uno':float(np.average(pointc,weights=lengths)),
             'valid_ibs_resamples':b,'valid_uno_resamples':int(valid.sum()),'undefined_uno_reason':'ZERO_IPCW_COMPARABLE_PAIR_WEIGHT_IN_AT_LEAST_ONE_FOLD',
             'undefined_by_fold':fails,'valid_first_2000':int(valid[:2000].sum()),'intervals':{}}
        for family in [1,3,20,40]:rec['intervals'][str(family)]={'ibs':interval(ds,family),'uno':interval(dc,family)}
        old=original[name]['difference'];rec['reconciliation']={'delta_ibs_error':rec['delta_ibs']-old['delta_ibs_mean'],'delta_uno_error':rec['delta_uno']-old['delta_uno_mean'],'first_2000_ibs_ci':interval(ds[:2000]),'first_2000_uno_ci':interval(dc[:2000])}
        assert abs(rec['reconciliation']['delta_ibs_error'])<1e-12 and abs(rec['reconciliation']['delta_uno_error'])<1e-12
        assert np.allclose(rec['reconciliation']['first_2000_ibs_ci'],old['delta_ibs_ci95'],atol=1e-12)
        assert np.allclose(rec['reconciliation']['first_2000_uno_ci'],old['delta_uno_ci95'],atol=1e-12)
        result['models'][name]=rec
        print('tau bootstrap',name,'valid C',int(valid.sum()),'/',b,flush=True)
    dump(out/'tau_bootstrap.json',result)

def logistic_calibration(logits,y,w,fixed_slope=False):
    """Newton solution, checked against statsmodels; no pseudo-row independence CI."""
    x=np.ones((len(y),1)) if fixed_slope else np.column_stack([np.ones(len(y)),logits])
    offset=logits if fixed_slope else np.zeros(len(y));beta=np.zeros(x.shape[1])
    if not fixed_slope:beta[1]=1
    if np.sum(w*y)<=0 or np.sum(w*(1-y))<=0:return None
    for k in range(60):
        mu=expit(x@beta+offset);v=w*mu*(1-mu)
        h=x.T@(v[:,None]*x);g=x.T@(w*(y-mu))
        try:step=np.linalg.solve(h,g)
        except np.linalg.LinAlgError:return None
        # Damping prevents separation-like explosions in sparse resamples.
        scale=max(1.,np.max(np.abs(step))/2);beta+=step/scale
        if not np.isfinite(beta).all() or np.max(np.abs(beta))>100:return None
        if np.max(np.abs(step))<1e-8:return beta.tolist()
    return None

def calibration_arrays(arm):
    _,tr,te=fold_arrays(arm);h=GRID[-1]
    cases=te['event']&(te['time']<=h);controls=te['time']>h;use=cases|controls
    weight=np.zeros(len(te));censor=CensoringDistributionEstimator().fit(tr)
    query=np.where(cases[use],np.nextafter(te['time'][use],-np.inf),h)
    g=censor.predict_proba(query);assert (g>0).all();weight[use]=1/g
    r=np.clip(1-np.array(arm['curves'])[:,-1],1e-6,1-1e-6)
    return r,cases.astype(float),weight,te

def calibration(out,b=2000):
    mri=load(JOB/'mri_ensemble_cells.json');pet=load(JOB/'pet_ensemble_cells.json');tau=[load(UNIT/f'tau_fold{i}_evaluation.json') for i in range(5)]
    groups=[('clinical_amyloid',mri,'clinical_amyloid'),('clinical_mri_amyloid',mri,'clinical_mri_amyloid'),('pet_augmented',pet,'clinical_mri_amyloid_pet'),('tau_base',tau,'base'),('tau_augmented',tau,'plus_tau')]
    result={'status':'DESCRIPTIVE_POOLED_OOF_CALIBRATION','resamples':b,'weights':'FOLD_TRAINING_REVERSE_KM_EVENT_LEFT_LIMIT_HORIZON_RIGHT_LIMIT','intervals':'CONDITIONAL_PARTICIPANT_BOOTSTRAP_WITHIN_FOLDS','models':{}}
    import statsmodels.api as sm
    for name,cells,arm in groups:
        aa=[calibration_arrays(c['arms'][arm]) for c in cells]
        r,y,w=[np.concatenate([a[i] for a in aa]) for i in range(3)];lg=logit(r)
        pars=logistic_calibration(lg,y,w);citl=logistic_calibration(lg,y,w,True)
        assert pars is not None and citl is not None
        glm=sm.GLM(y,np.column_stack([np.ones(len(y)),lg]),family=sm.families.Binomial(),freq_weights=w).fit()
        assert np.allclose(pars,glm.params,atol=1e-7)
        counts=bootstrap_counts([len(a[0]) for a in aa],b,20261013)
        counts=np.concatenate(counts,axis=1);draws=[];bins=[]
        # Bins fixed from all eligible predictions, not only observed cases/controls.
        binids=np.array_split(np.argsort(r,kind='stable'),3)
        for ids in binids:
            obs=float(np.average(y[ids],weights=w[ids]));pred=float(np.average(r[ids],weights=w[ids]))
            bc=counts[:,ids]*w[ids];den=bc.sum(axis=1);yy=np.divide(bc@y[ids],den,out=np.full(b,np.nan),where=den>0)
            bins.append({'n':len(ids),'events_3y':int(y[ids].sum()),'predicted_ipcw_mean':pred,'observed_ipcw_fraction':obs,'observed_ci95':interval(yy)})
        for row in counts:
            ww=w*row;p=logistic_calibration(lg,y,ww);ci=logistic_calibration(lg,y,ww,True)
            if p is not None and ci is not None:
                oe=float(np.dot(ww,y)/np.dot(ww,r));draws.append(p+ci+[oe])
        draws=np.asarray(draws);assert len(draws)>=.95*b
        hist,edges=np.histogram(r,bins=np.linspace(0,1,21))
        rec={'n':len(r),'events_3y':int(y.sum()),'usable_observation_status':int((w>0).sum()),'excluded_earlier_censoring':int((w==0).sum()),
             'intercept':pars[0],'slope':pars[1],'citl':citl[0],'oe_ratio_ipcw_hajek':float(np.dot(w,y)/np.dot(w,r)),
             'ci95':{k:interval(draws[:,j]) for j,k in enumerate(['intercept','slope','citl','oe_ratio_ipcw_hajek'])},'valid_resamples':len(draws),
             'weight_max':float(w.max()),'weight_effective_sample_size':float(w.sum()**2/(w@w)),
             'bins':bins,'risk_distribution':{'edges':edges.tolist(),'counts':hist.tolist(),'quantiles':np.quantile(r,[0,.25,.5,.75,1]).tolist()},
             'folds':[{'fold':i+1,'n':len(a[0]),'events_3y':int(a[1].sum())} for i,a in enumerate(aa)]}
        result['models'][name]=rec;print('calibration',name,'complete',flush=True)
    dump(out/'calibration.json',result)

def ipa(out,b=2000):
    cells=load(SUP/'private_comparison_cells.json');counts=bootstrap_counts([len(c['test_subjects']) for c in cells],b,20261013)
    draws={};point={}
    for arm in cells[0]['arms']:
        losses=[fold_arrays(c['arms'][arm])[0] for c in cells]
        draws[arm]=sum(ct@loss for ct,loss in zip(counts,losses))/sum(map(len,losses))
        point[arm]=float(np.concatenate(losses).mean())
    result={a:{'ibs':p,'ibs_ci95':interval(draws[a]),'ipa':1-p/point['null'],'ipa_ci95':interval(1-draws[a]/draws['null'])} for a,p in point.items()}
    dump(out/'ipa.json',{'conditional_fixed_predictions':True,'resamples':b,'models':result})
    print('IPA complete',flush=True)

def pacc(out):
    from scripts.xyz_fusion_postall_v2.secondary import subject_standardize,fit_pacc_models
    table=parquet(ROOT/'results/xyz_fusion_postall_v2/private/pacc_longitudinal.parquet')
    cells=load(JOB/'mri_ensemble_cells.json');result={'n_people':int(table.patient_hash.nunique()),'n_observations':len(table),'analyses':{}}
    for label,arm in [('clinical_amyloid','clinical_amyloid'),('mri_augmented','clinical_mri_amyloid')]:
        scores={sid:1-float(p[-1]) for c in cells for sid,p in zip(c['arms'][arm]['subjects'],c['arms'][arm]['curves'])}
        t=table.copy();t['oof_risk_3y']=t.patient_hash.map(scores);assert t.oof_risk_3y.notna().all()
        t['oof_risk_3y_z'],scale=subject_standardize(t,'oof_risk_3y')
        fits=fit_pacc_models(t);result['analyses'][label]={'risk_scale':scale,'models':fits}
        print('PACC',label,{k:v['status'] for k,v in fits.items()},flush=True)
    # Within-fold standardization changes the risk scale, not the underlying fits.
    # Fixed fold effects and fold-by-time interactions prevent fold-level shifts
    # from contributing to the estimated participant risk/time association.
    import statsmodels.formula.api as smf
    for label,arm in [('clinical_amyloid','clinical_amyloid'),('mri_augmented','clinical_mri_amyloid')]:
        t=table.copy();scores={sid:1-float(p[-1]) for c in cells for sid,p in zip(c['arms'][arm]['subjects'],c['arms'][arm]['curves'])};t['risk']=t.patient_hash.map(scores)
        u=t[['patient_hash','fold','risk']].drop_duplicates();sc=u.groupby('fold').risk.agg(['mean','std'])
        t['risk_z']=(t.risk-t.fold.map(sc['mean']))/t.fold.map(sc['std']);t['random_time_centered']=t.years-2.25
        cov=['age_z','education_years_z','baseline_pacc_raw_z','baseline_mmscore_z','sex_binary','apoe4_carrier','treatment_active']
        formula='pacc ~ years * ('+' + '.join(cov+['risk_z','C(fold)'])+')';model=smf.mixedlm(formula,t,groups=t.patient_hash,re_formula='~random_time_centered',missing='raise')
        candidates=[];attempts=[]
        for method in ['lbfgs','bfgs','powell']:
            with warnings.catch_warnings(record=True) as ww:
                warnings.simplefilter('always');fit=model.fit(reml=False,method=method,maxiter=1000)
                if method=='powell':fit=model.fit(reml=False,method='lbfgs',start_params=fit.params_object,maxiter=1000)
            grad=float(np.max(np.abs(model.score(fit.params_object)))/len(t));cv=np.asarray(fit.cov_params());valid=bool(fit.converged and np.isfinite(cv).all() and np.linalg.eigvalsh(cv).min()>0 and np.linalg.eigvalsh(fit.cov_re).min()>0 and grad<1e-4)
            attempts.append({'optimizer':method,'valid':valid,'log_likelihood':float(fit.llf),'gradient_per_observation':grad,'warnings':list({str(w.message) for w in ww})})
            if valid:candidates.append(fit)
            if len(candidates)>=2 and abs(candidates[-1].llf-candidates[-2].llf)<.001:break
        rec={'status':'NOT_ESTIMABLE_NUMERICS','attempts':attempts,'formula':formula,'risk_scale':'WITHIN_OUTER_FOLD_UNIQUE_SUBJECTS'}
        if candidates:
            f=max(candidates,key=lambda x:x.llf);term='years:risk_z';rec.update(status='ESTIMATED',estimate=float(f.fe_params[term]),ci95=f.conf_int().loc[term].tolist(),aic=float(f.aic))
        result['analyses'][label+'_fold_adjusted']=rec;print('PACC fold sensitivity',label,rec['status'],flush=True)
    dump(out/'pacc.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--task',choices=['tau','calibration','ipa','pacc','all'],default='all');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);setup(args.repo);start=time.time()
    tasks={'tau':tau_bootstrap,'calibration':calibration,'ipa':ipa,'pacc':pacc}
    with threadpool_limits(limits=2):
        for name,fn in tasks.items():
            if args.task in (name,'all'):fn(out)
    import sklearn,sksurv,scipy,statsmodels
    dump(out/f'{args.task}_execution.json',{'elapsed_seconds':time.time()-start,'date':'2026-10-13','python':sys.version,'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'sksurv':sksurv.__version__,'statsmodels':statsmodels.__version__},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    dump(out.parent/'private'/f'{args.task}_inputs.json',INPUTS)

if __name__=='__main__':main()
