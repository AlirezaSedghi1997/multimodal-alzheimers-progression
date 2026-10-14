"""Bounded post-hoc analyses of frozen predictions; never train image networks.

Only aggregate outputs are distributable. Input receipts remain in ../private.
The 3-year Brier family contains every 20 original contrast, not a selected subset.
"""
from pathlib import Path
import argparse, json, hashlib, sys, time
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sksurv.nonparametric import CensoringDistributionEstimator, kaplan_meier_estimator
from sksurv.metrics import brier_score
from threadpoolctl import threadpool_limits
import selected_analyses as s

def loss_grid(arm):
    old,tr,te=s.fold_arrays(arm)
    p=np.asarray(arm['curves'],float)
    censor=CensoringDistributionEstimator().fit(tr)
    ge=censor.predict_proba(te['time']);gt=censor.predict_proba(s.GRID)
    losses=np.zeros_like(p)
    for j,t in enumerate(s.GRID):
        case=te['event']&(te['time']<=t);control=te['time']>t
        assert np.all(ge[case]>0) and gt[j]>0
        losses[case,j]=p[case,j]**2/ge[case]
        losses[control,j]=(1-p[control,j])**2/gt[j]
    assert np.allclose(old,np.trapezoid(losses,x=s.GRID,axis=1)/(s.GRID[-1]-s.GRID[0]),atol=1e-12,rtol=0)
    _,check=brier_score(tr,te,p,s.GRID)
    assert np.allclose(check,losses.mean(axis=0),atol=1e-12,rtol=0)
    return losses,tr,te

def brier_horizon(out,b):
    base=Path(__file__).resolve().parents[1]
    prior=s.load(base/'figure_sources/aggregate_evidence.json')
    groups=[]
    for track in ['mri','pet']:
        cells=s.load(s.JOB/f'{track}_ensemble_cells.json')
        for c in prior['analysis'][track]['comparisons']:
            groups.append((track,cells,c['base'],c['augmented']))
    for track,fn,a,z in [('tau_cox',s.JOB/'tau_cox_ridge_o{}_evaluation.json','clinical_amyloid','clinical_amyloid_tau'),('tau_rsf',s.JOB/'tau_rsf_o{}_evaluation.json','clinical_amyloid','clinical_amyloid_tau'),('tau_mri',s.UNIT/'tau_fold{}_evaluation.json','base','plus_tau')]:
        groups.append((track,[s.load(str(fn).format(i)) for i in range(5)],a,z))
    cc=s.load(s.SUP/'private_comparison_cells.json')
    for a,z in [('full_clinical_amyloid','deep_mri_amyloid'),('full_clinical_volumes_amyloid','deep_volumes_mri_amyloid')]:groups.append(('stronger',cc,a,z))
    assert len(groups)==20
    results={'status':'POST_HOC_FIXED_PREDICTION_HORIZON_SENSITIVITY','horizon_days':float(s.GRID[-1]),'resamples':b,'seed':20261014,'family_size':20,'weights':'ORIGINAL_TRAINING_REVERSE_KM_RIGHT_CONTINUOUS_BRIER_CONVENTION','uncertainty':'CONDITIONAL_ON_FITTED_MODELS_AND_CENSORING_DISTRIBUTIONS','comparisons':[]}
    for track,cells,a,z in groups:
        la=[];lz=[];ns=[]
        for c in cells:
            aa,zz=c['arms'][a],c['arms'][z]
            for field in ['subjects','event','time','censor_train']:assert aa[field]==zz[field]
            la.append(loss_grid(aa)[0]);lz.append(loss_grid(zz)[0]);ns.append(len(aa['subjects']))
        counts=s.bootstrap_counts(ns,b,20261014)
        difference=[y[:,-1]-x[:,-1] for x,y in zip(la,lz)]
        draws=sum(ct@di for ct,di in zip(counts,difference))/sum(ns)
        ra=np.concatenate(la);rz=np.concatenate(lz)
        rec={'track':track,'reference':a,'augmented':z,'n':sum(ns),'bs_reference':float(ra[:,-1].mean()),'bs_augmented':float(rz[:,-1].mean()),'delta_bs':float(np.concatenate(difference).mean()),'nominal_ci95':s.interval(draws),'adjusted_20_ci':s.interval(draws,20),'grid_days':s.GRID.tolist(),'reference_brier_curve':ra.mean(axis=0).tolist(),'augmented_brier_curve':rz.mean(axis=0).tolist(),'all_resamples_finite':bool(np.isfinite(draws).all())}
        results['comparisons'].append(rec)
        print('Brier',track,a,z,rec['delta_bs'],rec['adjusted_20_ci'],flush=True)
    s.dump(out/'brier_3year.json',results)

def table2_precision(out,b=2000):
    result={'status':'DESCRIPTIVE_FIXED_PREDICTION_PRECISION','resamples':b,'intervals':'NOMINAL_95_PERCENT_NOT_SIMULTANEOUS','models':{}}
    for track in ['mri','pet']:
        cells=s.load(s.JOB/f'{track}_ensemble_cells.json')
        arms=list(cells[0]['arms'])
        for name in arms:
            losses=[];null=[]
            for c in cells:
                arm=c['arms'][name];x,tr,te=loss_grid(arm)
                losses.append(np.trapezoid(x,x=s.GRID,axis=1)/(s.GRID[-1]-s.GRID[0]))
                times,surv=kaplan_meier_estimator(tr['event'],tr['time'])
                idx=np.searchsorted(times,s.GRID,side='right')-1
                sp=np.where(idx>=0,surv[np.maximum(idx,0)],1.)
                na=dict(arm,curves=np.tile(sp,(len(te),1)).tolist())
                nx=loss_grid(na)[0];null.append(np.trapezoid(nx,x=s.GRID,axis=1)/(s.GRID[-1]-s.GRID[0]))
            ns=list(map(len,losses));ct=s.bootstrap_counts(ns,b,20261013)
            draws=sum(c@v for c,v in zip(ct,losses))/sum(ns)
            null_draws=sum(c@v for c,v in zip(ct,null))/sum(ns)
            point=float(np.concatenate(losses).mean());base=float(np.concatenate(null).mean())
            result['models'][track+':'+name]={'n':sum(ns),'ibs':point,'ibs_ci95':s.interval(draws),'null_ibs':base,'null_ci95':s.interval(null_draws),'ipa_percent':100*(1-point/base),'ipa_ci95_percent':s.interval(100*(1-draws/null_draws))}
    s.dump(out/'table2_precision.json',result)

def descriptive(out):
    frame=s.parquet(s.SUP/'private_analysis_frame.parquet');tau=s.parquet(s.SUP/'private_tau_frame.parquet')
    frame=frame.merge(tau[['patient_hash','tau_temporal']],on='patient_hash',how='left',validate='one_to_one')
    cols=['blood_ptau217','tau_temporal','hippocampus_icv','entorhinal_icv','ventricle_icv']
    results={'units':{'blood_ptau217':'U/mL (source assay scale)','tau_temporal':'SUVR, cerebellar crus reference','volumes':'100 x volume/ICV (%)'},'groups':{}}
    for name,mask in [('mri',np.ones(len(frame),bool)),('pet',frame.has_pet),('plasma',frame.has_blood),('tau',frame.has_tau)]:
        rec={}
        for col in cols:
            x=pd.to_numeric(frame.loc[mask,col],errors='coerce');v=x.dropna()
            if col.endswith('_icv'):v=100*v
            rec[col]={'available':len(v),'missing':int(x.isna().sum()),'median':float(v.median()),'q1':float(v.quantile(.25)),'q3':float(v.quantile(.75))}
        results['groups'][name]=rec
    s.dump(out/'baseline_biomarkers.json',results)
    qp=s.ROOT/'data/preprocessed_pet/_private/manifests/fbp_private_manifest.csv'
    s.INPUTS[str(qp)]={'bytes':qp.stat().st_size,'sha256':hashlib.sha256(qp.read_bytes()).hexdigest()}
    q=pd.read_csv(qp)
    source=s.ROOT/'results/method_y_revival_v1/completion_v1/image_branch/fbp_preprocessing_membership.parquet'
    b=s.parquet(source);link=q[['sample_hash','patient_hash']].drop_duplicates()
    bc=b.merge(link,left_on='subject_id_hash',right_on='sample_hash',how='left',validate='many_to_one')
    assert set(bc.loc[bc.patient_hash.isna(),'cohort'])=={'OTHER'}
    bc=bc.loc[bc.patient_hash.isin(frame.patient_hash)]
    z=frame[['patient_hash','sex_binary','has_pet']].merge(bc[['patient_hash','qc_status','exclusion_reason']],on='patient_hash',how='left',validate='one_to_one')
    z['reason']=np.where(z.qc_status.isna(),'NOT_IN_ASSEMBLED_IMAGE_LIST',np.where(z.qc_status.eq('PASS'),'INCLUDED','NONFINITE_SOURCE_ARRAY'))
    assert (z.reason.eq('INCLUDED')==z.has_pet).all()
    assert set(z.exclusion_reason.dropna())<= {'','FBP_SOURCE_NONFINITE'}
    outrows=[]
    for reason,t in z.groupby('reason'):
        outrows.append({'reason':reason,'n':len(t),'male_n':int(t.sex_binary.eq(1).sum()),'female_n':int(t.sex_binary.eq(0).sum())})
    s.dump(out/'pet_selection.json',{'n':len(frame),'counts':outrows,'scope':'Local input availability and processing; no claim about GRIP inventory or biological cause of sex imbalance','upstream_status_linked_to_exact_frozen_membership':True})
    expert=[];contract=s.load(s.ROOT/'results/xyz_fusion_v2/private/contract.json');m={v['patient_hash']:v for v in contract['manifest']}
    for f in s.JOB.glob('mri_s*_o*_c*_lock.json'):
        d=s.load(f);ks=d['expert_subjects'];expert.append({'n':len(ks),'events_full':sum(m[k]['event'] for k in ks),'events_3y':sum(m[k]['event'] and m[k]['time_days']<=1095.75 for k in ks)})
    assert len(expert)==45
    s.dump(out/'expert_support.json',{'fits':len(expert),'ranges':{c:[min(v[c] for v in expert),max(v[c] for v in expert)] for c in expert[0]}})

def calibration_curves(out,b=2000):
    original=s.load(out/'calibration.json')
    groups=[('clinical_mri_amyloid',s.load(s.JOB/'mri_ensemble_cells.json'),'clinical_mri_amyloid'),('pet_augmented',s.load(s.JOB/'pet_ensemble_cells.json'),'clinical_mri_amyloid_pet'),('tau_augmented',[s.load(s.UNIT/f'tau_fold{i}_evaluation.json') for i in range(5)],'plus_tau')]
    result={'status':'DESCRIPTIVE_WEIGHTED_LOGISTIC_CALIBRATION_CURVES','bands':'CONDITIONAL_POINTWISE_95_PERCENT_NOT_SIMULTANEOUS','resamples':b,'seed':20261013,'risk_plot_support':'2.5th to 97.5th percentiles of observed-status risk; no tail extrapolation','models':{}}
    for name,cells,arm in groups:
        aa=[s.calibration_arrays(c['arms'][arm]) for c in cells]
        r,y,w=[np.concatenate([a[i] for a in aa]) for i in range(3)];lg=logit(r)
        pars=s.logistic_calibration(lg,y,w)
        assert np.allclose(pars,[original['models'][name][k] for k in ['intercept','slope']],atol=1e-7)
        support=np.quantile(r[w>0],[.025,.975]);grid=np.linspace(*support,150)
        counts=np.concatenate(s.bootstrap_counts([len(a[0]) for a in aa],b,20261013),axis=1);pred=[]
        for ct in counts:
            p=s.logistic_calibration(lg,y,w*ct)
            if p is not None:pred.append(expit(p[0]+p[1]*logit(grid)))
        assert len(pred)>=.95*b
        result['models'][name]={'risk':grid.tolist(),'calibration':expit(pars[0]+pars[1]*logit(grid)).tolist(),'lower':np.quantile(pred,.025,axis=0).tolist(),'upper':np.quantile(pred,.975,axis=0).tolist(),'valid_resamples':len(pred),'parameters':pars,'n_observed_status':int((w>0).sum())}
    s.dump(out/'calibration_curves.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--task',choices=['all','brier','descriptive','precision','curves'],default='all');p.add_argument('--resamples',type=int,default=40000);a=p.parse_args()
    s.setup(a.repo);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    with threadpool_limits(limits=2):
        if a.task in ['all','descriptive']:descriptive(out)
        if a.task in ['all','brier']:brier_horizon(out,a.resamples)
        if a.task in ['all','precision']:table2_precision(out)
        if a.task in ['all','curves']:calibration_curves(out)
    s.dump(out/f'reviewer2_{a.task}_execution.json',{'date':'2026-10-14','seconds':time.time()-start,'task':a.task,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':sys.version,'new_cnn_training':False})
    s.dump(out.parent/'private'/f'reviewer2_{a.task}_inputs.json',s.INPUTS)

if __name__=='__main__':main()
