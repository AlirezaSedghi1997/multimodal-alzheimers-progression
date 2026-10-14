"""Aggregate selection audit and covariate-dependent censoring sensitivity."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import selected_analyses as s

def describe(x):
    x=pd.to_numeric(x,errors='coerce');v=x.dropna()
    return {'n':len(x),'missing':int(x.isna().sum()),'min':float(v.min()),'median':float(v.median()),'q1':float(v.quantile(.25)),'q3':float(v.quantile(.75)),'max':float(v.max())} if len(v) else {'n':len(x),'missing':len(x)}

def cohort(out,raw):
    from scripts.xyz_fusion_v1.audit import _table
    from src.method_y_revival.endpoint_builder import load_first_amyloid_scans,build_amyloid_endpoint_rows
    frame=s.parquet(s.SUP/'private_analysis_frame.parquet');subj=_table(raw/'Derived Data/SUBJINFO.csv')
    assert subj.patient_hash.is_unique
    joined=frame.merge(subj[['patient_hash','RNDMDT_DAYS_T0','RACE','ETHNIC']],validate='one_to_one')
    result={'landmark_minus_randomization_days':describe(joined.landmark_days_t0-joined.RNDMDT_DAYS_T0),'groups':{}}
    masks={'mri':np.ones(len(joined),bool),'pet':joined.has_pet,'plasma':joined.has_blood,'tau':joined.has_tau}
    for name,mask in masks.items():
        t=joined[mask];race=pd.to_numeric(t.RACE,errors='coerce');eth=pd.to_numeric(t.ETHNIC,errors='coerce')
        result['groups'][name]={'n':len(t),'events_full':int(t.event.sum()),'events_by_year':{str(y):int((t.event.astype(bool)&(t.time_days<=365.25*y)).sum()) for y in [1,2,3]},'at_risk_by_year':{str(y):int((t.time_days>=365.25*y).sum()) for y in [1,2,3]},'reported_white':int(race.eq(1).sum()),'race_unknown_or_missing':int((race.isna()|race.eq(97)).sum()),'reported_hispanic_latino':int(eth.eq(50).sum()),'ethnicity_unknown_or_missing':int((eth.isna()|eth.eq(97)).sum()),'mri_manufacturer':t.mri_manufacturer.fillna('UNKNOWN').value_counts().to_dict()}
    result['demographic_coding']={'source':'SUBJINFO and derived data dictionary','white':1,'hispanic_latino':50,'unknown_or_not_reported':97,'not_inferred_from_names_or_images':True}
    endpoint=s.load(s.ROOT/'results/method_y_revival_v2/endpoints/amyloid_endpoint_summary.json')['contract']['endpoint_definition']
    paths={'amyloid':raw/'External Data/imaging_SUVR_amyloid.csv','subj':raw/'Derived Data/SUBJINFO.csv','cdr':raw/'Raw Data/cdr.csv','pacc':raw/'Derived Data/PACC.csv'}
    rows={}
    for k,p in paths.items():
        s.INPUTS[str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
        with p.open(encoding='utf-8-sig',newline='') as f:rows[k]=list(csv.DictReader(f))
    first,_=load_first_amyloid_scans(rows['amyloid']);first_a4={k:v for k,v in first.items() if k[0].upper()=='A4'}
    built,audit=build_amyloid_endpoint_rows(first_a4,rows['subj'],rows['cdr'],rows['pacc'],{'endpoint':endpoint})
    existing=s.parquet(s.ROOT/'results/method_y_revival_v2/endpoints/amyloid_progression_endpoint.parquet')
    existing=existing[existing.cohort.eq('A4')]
    assert {v['patient_hash'] for v in built}==set(existing.patient_hash),'UPSTREAM_ENDPOINT_MEMBERSHIP_DRIFT'
    result['upstream_exclusions']={'source_a4':int(subj.SUBSTUDY.eq('A4').sum()),'first_eligible_amyloid_scan':len(first_a4),'rebuilt_endpoint_n':len(built),'endpoint_membership_matches_saved':True,'audit':audit}
    # Describe all 110 source exclusions together: avoid pretending they are one
    # mechanistic group, and do not create outcomes for participants lacking follow-up.
    inc=subj[subj.patient_hash.isin(frame.patient_hash)];exc=subj[subj.SUBSTUDY.eq('A4')&~subj.patient_hash.isin(frame.patient_hash)]
    smd={}
    for col in ['AGEYR','EDCCNTU','AMYLCENT','MMSETSV6','PACCV6']:
        a,b=pd.to_numeric(inc[col],errors='coerce').dropna(),pd.to_numeric(exc[col],errors='coerce').dropna()
        den=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        smd[col]={'included_n':len(a),'excluded_n':len(b),'included_mean':float(a.mean()),'excluded_mean':float(b.mean()),'smd_included_minus_excluded':float((a.mean()-b.mean())/den)}
    result['included_vs_all_source_exclusions']={'included':len(inc),'excluded':len(exc),'source_baseline_values_not_redefined_landmark_predictors':True,'continuous':smd}
    s.dump(out/'cohort_audit.json',result);print('cohort selection audit complete',flush=True)

def censoring(out):
    from scripts.xyz_fusion_v2.models import fit_model
    from scripts.xyz_fusion_v2.protocol import CLINICAL
    frame=s.parquet(s.SUP/'private_analysis_frame.parquet');tau=s.parquet(s.SUP/'private_tau_frame.parquet')
    groups=[('mri',s.load(s.JOB/'mri_ensemble_cells.json'),frame,'clinical_amyloid','clinical_mri_amyloid'),
            ('tau',[s.load(s.UNIT/f'tau_fold{i}_evaluation.json') for i in range(5)],tau,'base','plus_tau')]
    features=CLINICAL+['amyloid_centiloid'];result={'status':'POST_HOC_CENSORING_MODEL_SENSITIVITY','censoring_predictors':features,'ridge':10,'weights_clipped':False,'scope':'Fixed predictions; does not remove unmeasured informative censoring','models':{}}
    for name,cells,tab,a,b in groups:
        deltas=[];support=[];lossas=[];lossbs=[]
        for i,ce in enumerate(cells):
            av,bv=ce['arms'][a],ce['arms'][b];assert av['subjects']==bv['subjects']
            tr=tab.set_index('patient_hash').loc[ce['train_subjects']].reset_index();te=tab.set_index('patient_hash').loc[av['subjects']].reset_index()
            original_tr=tr.copy();tr['event']=~tr.event.astype(bool)
            assert len(tr)==len(av['censor_train']['time'])
            # Confirm that the linked training rows reproduce the censoring sample.
            assert sorted(zip(original_tr.event.astype(int),original_tr.time_days))==sorted(zip(av['censor_train']['event'],av['censor_train']['time']))
            model=fit_model(tr,features,'cox_ridge',10,s.CONFIG,20261013+i)
            sf=model.predict_survival_function(te[features]);_,_,yy=s.fold_arrays(av)
            eg=np.array([fn(t) for fn,t in zip(sf,yy['time'])]);hg=np.array([fn(s.GRID) for fn in sf])
            loss=[];usedweights=[]
            for arm in [av,bv]:
                pp=np.array(arm['curves']);ll=np.zeros_like(pp)
                for j,t in enumerate(s.GRID):
                    case=yy['event']&(yy['time']<=t);control=yy['time']>t
                    assert (eg[case]>0).all() and (hg[control,j]>0).all()
                    ll[case,j]=pp[case,j]**2/eg[case];ll[control,j]=(1-pp[control,j])**2/hg[control,j]
                    if len(loss)==0:usedweights.extend((1/eg[case]).tolist());usedweights.extend((1/hg[control,j]).tolist())
                loss.append(np.trapezoid(ll,x=s.GRID,axis=1)/(s.GRID[-1]-s.GRID[0]))
            lossas.append(loss[0]);lossbs.append(loss[1]);deltas.append(loss[1]-loss[0])
            ww=np.array(usedweights);support.append({'fold':i+1,'n':len(te),'minimum_used_G':float(1/ww.max()),'weight_max':float(ww.max()),'weight_q99':float(np.quantile(ww,.99))})
        counts=s.bootstrap_counts(list(map(len,deltas)),2000,20261013);draw=sum(c@d for c,d in zip(counts,deltas))/sum(map(len,deltas))
        result['models'][name]={'ibs_reference':float(np.concatenate(lossas).mean()),'ibs_augmented':float(np.concatenate(lossbs).mean()),'delta_ibs':float(np.concatenate(deltas).mean()),'conditional_nominal_ci95':s.interval(draw),'conditional_two_comparison_ci':s.interval(draw,2),'resamples':2000,'diagnostics':support}
        print('conditional censoring',name,'complete',flush=True)
    s.dump(out/'censoring_sensitivity.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--raw-clinical',required=True);p.add_argument('--output',required=True);p.add_argument('--task',choices=['all','cohort','censoring'],default='all');a=p.parse_args();s.setup(a.repo);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    with threadpool_limits(limits=2):
        if a.task in ['cohort','all']:cohort(out,Path(a.raw_clinical))
        if a.task in ['censoring','all']:censoring(out)
    s.dump(out/f'{a.task}_audit_execution.json',{'seconds':time.time()-start,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'date':'2026-10-13','python':sys.version})
    s.dump(out.parent/'private'/f'{a.task}_audit_inputs.json',s.INPUTS)
