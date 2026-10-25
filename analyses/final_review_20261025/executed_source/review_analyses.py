"""Append-only fixed-prediction checks for the user-supplied final review.

Private input records stay under the authorized H: workspace. Distributable
outputs contain aggregates only. Original models, predictions and results are
read-only and are never selected or overwritten by these checks.
"""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,hashlib,importlib.util,time,argparse
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
ROOT=Path(r'H:\New project');OUT=Path(__file__).parent;AGG=OUT/'aggregate';PRIV=OUT/'private'
AGG.mkdir(exist_ok=True);PRIV.mkdir(exist_ok=True);sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from sksurv.util import Surv
from sksurv.nonparametric import CensoringDistributionEstimator,kaplan_meier_estimator
from sksurv.metrics import integrated_brier_score
from scripts.xyz_fusion_postall_v2.core import digest
from scripts.xyz_fusion_v2.protocol import contrasts
from scripts.xyz_fusion_v1.report import cell_frame
INPUTS={};NUMERICAL=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def read(path):
    p=Path(path);raw=p.read_bytes();INPUTS[str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)};d=json.loads(raw.decode('utf-8-sig'))
    if isinstance(d,dict) and 'payload' in d:
        if 'payload_sha256' in d:assert digest(d['payload'])==d['payload_sha256'],'Frozen payload hash drift'
        return d['payload']
    return d
CONFIG=read(ROOT/'results/xyz_fusion_v2/private/contract.json')['protocol'];GRID=np.asarray(CONFIG['fusion']['ibs_grid_days'],float)
LATE=np.r_[730.,GRID[GRID>730.]];B=40000;SEED=20261025
assert LATE[-1]==1095.75 and GRID[0]==365
def interval(a,k=1):
    a=np.asarray(a,float);assert np.isfinite(a).all()
    return np.quantile(a,[.05/(2*k),1-.05/(2*k)]).tolist()
def loss(arm,grid=GRID,curves=None):
    tr=Surv.from_arrays(np.asarray(arm['censor_train']['event'],bool),np.asarray(arm['censor_train']['time'],float))
    te=Surv.from_arrays(np.asarray(arm['event'],bool),np.asarray(arm['time'],float))
    cutoff=np.nextafter(float(tr['time'].max()),0.);te['event'] &=te['time']<cutoff;te['time']=np.minimum(te['time'],cutoff)
    p=np.asarray(arm['curves'] if curves is None else curves,float)
    if curves is None and not np.array_equal(grid,GRID):p=np.asarray([np.interp(grid,GRID,row) for row in p])
    assert p.shape==(len(te),len(grid)) and np.isfinite(p).all() and grid[-1]<cutoff
    cen=CensoringDistributionEstimator().fit(tr);ge=cen.predict_proba(te['time']);gt=cen.predict_proba(grid);terms=np.zeros_like(p)
    for j,t in enumerate(grid):
        case=te['event']&(te['time']<=t);control=te['time']>t
        assert (ge[case]>0).all() and gt[j]>0
        terms[case,j]=p[case,j]**2/ge[case];terms[control,j]=(1-p[control,j])**2/gt[j]
    integrated=np.trapezoid(terms,x=grid,axis=1)/(grid[-1]-grid[0])
    if grid[0]>=te['time'].min() and grid[-1]<te['time'].max():
        check=float(integrated_brier_score(tr,te,p,grid));error=abs(check-float(integrated.mean()));assert error<1e-12;NUMERICAL.append(error)
    return integrated,tr,te
def counts(n,b=B,seed=SEED,fold=0):
    # Restarting this explicitly seeded stream keeps each matched contrast paired.
    return np.random.default_rng(np.random.SeedSequence([seed,fold,n])).multinomial(n,np.full(n,1./n),size=b).astype(np.int16)
def specifications():
    jobs=ROOT/'results/xyz_fusion_v2/private/jobs';units=ROOT/'results/xyz_fusion_postall_v2/private/units';cases=[]
    for track in ('mri','pet'):
        cells=read(jobs/f'{track}_ensemble_cells.json')
        for a,b in contrasts(track):cases.append((track,cells,a,b))
    for track,template,a,b in [('tau_cox',jobs/'tau_cox_ridge_o{}_evaluation.json','clinical_amyloid','clinical_amyloid_tau'),('tau_rsf',jobs/'tau_rsf_o{}_evaluation.json','clinical_amyloid','clinical_amyloid_tau'),('tau_mri',units/'tau_fold{}_evaluation.json','base','plus_tau')]:
        cases.append((track,[read(str(template).format(i)) for i in range(5)],a,b))
    cells=read(ROOT/'thesis-workspace/supervisor_revision_20261011/private_comparison_cells.json')
    for a,b in [('full_clinical_amyloid','deep_mri_amyloid'),('full_clinical_volumes_amyloid','deep_volumes_mri_amyloid')]:cases.append(('stronger',cells,a,b))
    assert len(cases)==20
    return cases
def late_window():
    cases=specifications();results=[]
    for track,cells,a,b in cases:
        lengths=[];diffs=[];original=[];la=[];lb=[]
        for c in cells:
            av,bv=c['arms'][a],c['arms'][b]
            for key in ('subjects','event','time','censor_train'):assert av[key]==bv[key]
            va,_,_=loss(av,LATE);vb,_,_=loss(bv,LATE);old_a,_,_=loss(av);old_b,_,_=loss(bv)
            la.append(va);lb.append(vb);lengths.append(len(va));diffs.append(vb-va);original.append(old_b-old_a)
        n=sum(lengths);draws=np.zeros(B)
        for f,(nn,d) in enumerate(zip(lengths,diffs)):
            c=counts(nn,fold=f);draws+=c@d/n
        r={'track':track,'reference':a,'augmented':b,'n':n,'late_ibs_reference':float(np.concatenate(la).mean()),'late_ibs_augmented':float(np.concatenate(lb).mean()),'late_delta_ibs':float(np.concatenate(diffs).mean()),'nominal_ci95':interval(draws),'adjusted_20_ci':interval(draws,20),'original_full_window_delta_ibs':float(np.concatenate(original).mean()),'resamples':B,'all_resamples_finite':bool(np.isfinite(draws).all())}
        results.append(r);dump(AGG/'A10_late_window_ibs_partial.json',results)
        print('LATE_WINDOW',track,a,b,r['late_delta_ibs'],r['adjusted_20_ci'],flush=True)
    dump(AGG/'A10_late_window_ibs.json',{'status':'EXECUTED_POST_HOC_FIXED_PREDICTIONS','original_grid_days':GRID.tolist(),'new_grid_days':LATE.tolist(),'new_grid_boundary_interpolation':'Survival at day 730 linearly interpolated only between frozen original prediction times.','resamples':B,'seed':SEED,'sampling':'Paired participant multinomial multiplicities within each original outer fold; stream SeedSequence([20261025, fold_index, fold_size]).','family_size':20,'uncertainty':'Conditional on original predictions, outer partitions and training censoring distributions; no refitting or selection uncertainty.','original_results_unchanged':True,'comparisons':results,'library_reconciliation_calls':len(NUMERICAL),'max_library_absolute_error':max(NUMERICAL)})
def subset_nulls():
    cases=specifications();groups={}
    for track,cells,a,b in cases:
        if track=='stronger':continue
        for name in (a,b):
            key=track+':'+name
            if key in groups:continue
            values=[];nulls=[];sizes=[];training=[]
            for c in cells:
                arm=c['arms'][name];v,tr,te=loss(arm)
                t,s=kaplan_meier_estimator(tr['event'],tr['time']);ix=np.searchsorted(t,GRID,side='right')-1
                curve=np.where(ix>=0,s[np.maximum(ix,0)],1.)
                nv,_,_=loss(arm,curves=np.tile(curve,(len(te),1)))
                values.append(v);nulls.append(nv);sizes.append(len(v));training.append({'n':len(tr),'events':int(tr['event'].sum())})
            model=float(np.concatenate(values).mean());null=float(np.concatenate(nulls).mean());n=sum(sizes)
            groups[key]={'n':n,'ibs_model':model,'ibs_training_km_null':null,'ipa_percent':100*(1-model/null),'fold_training_support':training,'fold_evaluation_sizes':sizes}
    prior_path=Path(r'C:\Users\USER\Desktop\draft\GeroScience_Submission_Package_v7_20261014\SUPPLEMENTARY_FOR_UPLOAD\ESM_3\aggregate_results\table2_precision.json')
    known=read(prior_path)['models'];errors=[]
    for k,v in groups.items():
        if k in known:
            errors.extend([abs(v['ibs_model']-known[k]['ibs']),abs(v['ibs_training_km_null']-known[k]['null_ibs']),abs(v['ipa_percent']-known[k]['ipa_percent'])]);assert max(errors)<1e-10
    nulls={}
    for track in ('mri','pet','plasma','tau'):
        if track=='mri':keys=[k for k in groups if k.startswith('mri:') and ':blood_' not in k]
        elif track=='plasma':keys=[k for k in groups if k.startswith('mri:blood_')]
        elif track=='pet':keys=[k for k in groups if k.startswith('pet:')]
        else:keys=[k for k in groups if k.startswith('tau_')]
        vals=[groups[k]['ibs_training_km_null'] for k in keys]
        assert max(vals)-min(vals)<1e-12,(track,vals)
        nulls[track]={'n':groups[keys[0]]['n'],'ibs':vals[0]}
    dump(AGG/'A11_subset_nulls_and_ipa.json',{'status':'EXECUTED_FROZEN_PREDICTION_RECONSTRUCTION','definition':'IPA percent = 100*(1 - participant-weighted IBS_model / participant-weighted IBS_null), calculated before display rounding.','nulls':nulls,'models':groups,'existing_table2_values_reconciled':True,'maximum_prior_absolute_error':max(errors),'tau_ipa_status':'Additional descriptive point estimates, not new significance tests or a comparison of modalities.'})
    for key,v in nulls.items():print('NULL',key,v,flush=True)
def subgroup_and_pair_support():
    prev=ROOT/'thesis-workspace/deepreview_revision_20261024/aggregate';cases=[]
    for track,contrast in [('mri','clinical_amyloid__clinical_mri_amyloid'),('pet','clinical_mri_amyloid__clinical_mri_amyloid_pet')]:
        d=read(ROOT/f'results/xyz_fusion_v2/private/jobs/{track}_subgroups.json')
        for group,label in [('sex_0','women'),('sex_1','men')]:
            g=d['groups'][group];assert (track,label,g['n']) in [('mri','women',631),('mri','men',428),('pet','women',385),('pet','men',159)]
            cases.append({'track':track,'sex':label,'n':g['n'],'all_followup_events':g['events'],'contrast':contrast,**g['paired_contrasts'][contrast]})
    dump(AGG/'reported_sex_diagnostics.json',{'status':'EXISTING_AGGREGATES_REPORTED_MORE_EXPLICITLY','intervals':'Nominal conditional 95% intervals; no new correction and no interaction/equivalence claim.','censoring':'Original outer-training censoring distribution; not a subgroup-specific refit or population-selection adjustment.','records':cases})
    spec=importlib.util.spec_from_file_location('review_original',Path(r'C:\Users\USER\Desktop\draft\GeroScience_Submission_Package_v7_20261014\SUPPLEMENTARY_FOR_UPLOAD\ESM_3\revision_analyses\selected_analyses.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.setup(ROOT)
    jobs=ROOT/'results/xyz_fusion_v2/private/jobs';units=ROOT/'results/xyz_fusion_postall_v2/private/units';reports=[]
    targets=[('tau_cox',[read(jobs/f'tau_cox_ridge_o{i}_evaluation.json') for i in range(5)],'clinical_amyloid',20261011),('tau_mri',[read(units/f'tau_fold{i}_evaluation.json') for i in range(5)],'base',20261011),('overlap',[read(ROOT/'thesis-workspace/deepreview_revision_20261024/private/overlap_cells.json')[i] for i in range(5)],'plasma',20261024)]
    for name,cells,arm,seed in targets:
        lengths=[len(c['arms'][arm]['subjects']) for c in cells];ct=mod.bootstrap_counts(lengths,B,seed);total=np.zeros(B);allvalid=np.ones(B,bool)
        for c,mult in zip(cells,ct):
            av=c['arms'][arm];_,tr,te=loss(av);_,den=mod.concordance_matrices(tr,te,1-np.asarray(av['curves'])[:,-1],GRID[-1]);d=np.einsum('bi,bi->b',mult@den,mult);total+=d;allvalid &=d>0
        assert int((total>0).sum())==B
        rec={'model':name,'resamples':B,'seed':seed,'all_fold_estimable':int(allvalid.sum()),'at_least_one_fold_estimable':int((total>0).sum()),'pair_weighted_valid_draws':int((total>0).sum())};reports.append(rec);print('PAIR_SUPPORT',rec,flush=True)
    tau=read(prev/'A01_tau_omission_robustness.json')['models'];over=read(prev/'A04_matched_biomarker_overlap.json')['results']['plasma__tau']
    values={'tau_cox':tau['cox'],'tau_mri':tau['mri'],'overlap':over}
    for r in reports:
        v=values[r['model']];assert r['all_fold_estimable']==v['valid_uno_resamples'];k='4' if r['model']=='overlap' else '3'
        r.update(fold_size_weighted_delta_uno=v['delta_uno'],conditional_interval=v['intervals'][k]['uno_original_estimable'],pair_weighted_delta_uno=v['pair_weighted_delta_uno'],pair_weighted_interval=v['intervals'][k]['uno_pair_weighted_within_fold'],family_size=int(k),pair_weighted_K40_interval=v['intervals']['40']['uno_pair_weighted_within_fold'])
    dump(AGG/'all_draw_concordance_support.json',{'status':'ORIGINAL_AGGREGATE_INTERVALS_RECONCILED_AND_DRAW_SUPPORT_RECHECKED','warning':'Pair weighting changes the aggregation estimand; all-draw estimability is not proof of unbiased inference or conservative coverage. No cross-fold pairs used.','models':reports})
def standardization_invariance():
    from scripts.xyz_oof_stacking_v1.inputs import Inputs
    from scripts.xyz_fusion_v2.protocol import CLINICAL,MRI,PET
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    inp=Inputs(ROOT,read(ROOT/'configs/xyz_oof_stacking_v1.json'));records=[]
    for source in ('mri','pet'):
        for fold in range(5):
            for seed in inp.config['training_seeds']:
                for component in range(3):
                    task=f'{source}_s{seed}_o{fold}_c{component}';meta=inp.feature_frame(task,'meta');held=inp.feature_frame(task,'outer');cols=MRI+(PET if source=='pet' else [])
                    means=meta[cols].mean();sd=meta[cols].std(ddof=0);assert (sd>1e-12).all()
                    zmeta=meta.copy();zheld=held.copy();zmeta.loc[:,cols]=(meta[cols]-means)/sd;zheld.loc[:,cols]=(held[cols]-means)/sd
                    features=CLINICAL+['amyloid_centiloid']+cols
                    pipe=lambda:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler())
                    a=pipe().fit(meta[features]);b=pipe().fit(zmeta[features]);trainerr=float(np.max(np.abs(a.transform(meta[features])-b.transform(zmeta[features]))));testerr=float(np.max(np.abs(a.transform(held[features])-b.transform(zheld[features]))))
                    assert max(trainerr,testerr)<1e-8,(task,trainerr,testerr)
                    records.append({'source':source,'fold':fold+1,'seed':seed,'component':component+1,'meta_n':len(meta),'held_n':len(held),'maxabs_training_standardized_difference':trainerr,'maxabs_test_standardized_difference':testerr})
            print('AFFINE_STANDARDIZATION_INVARIANCE',source,fold+1,flush=True)
    dump(AGG/'A12_component_standardization_invariance.json',{'status':'EXECUTED_DETERMINISTIC_PREPROCESSING_CHECK','component_count':len(records),'tolerance':1e-8,'new_model_fits':0,'outer_outcomes_used':False,'max_training_difference':max(x['maxabs_training_standardized_difference'] for x in records),'max_test_difference':max(x['maxabs_test_standardized_difference'] for x in records),'interpretation':'For a separate component head with its own training StandardScaler, a preceding fixed nondegenerate component-affine transform cancels in the second standardization, up to floating-point error. This does not apply to pooling rows from multiple experts into one head; jointly standardized and leakage-safe tuned pooled fusion was not evaluated.','records':records})
def main():
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['late','nulls','support','invariance']);args=parser.parse_args();started=datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=2):
        {'late':late_window,'nulls':subset_nulls,'support':subgroup_and_pair_support,'invariance':standardization_invariance}[args.task]()
    dump(PRIV/('input_receipt_'+args.task+'.json'),{'task':args.task,'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'plan_sha256':sha(OUT/'amendment_plan.json'),'inputs':INPUTS,'numpy':np.__version__,'pandas':pd.__version__})
if __name__=='__main__':main()
