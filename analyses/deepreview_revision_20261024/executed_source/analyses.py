"""Append-only, post-hoc review analyses. No controlled data leave this root."""
from pathlib import Path
import os,sys,json,hashlib,importlib.util,time,warnings,argparse
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
ROOT=Path(r'H:\New project');OUT=Path(__file__).parent;PUB=OUT/'aggregate';PRIV=OUT/'private'
PUB.mkdir(exist_ok=True);PRIV.mkdir(exist_ok=True);sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from scripts.xyz_fusion_postall_v2.core import digest
from scripts.xyz_oof_stacking_v1.inputs import Inputs,pool_oof
from scripts.xyz_oof_stacking_v1.protocol import arms,contrasts
from scripts.xyz_fusion_v2.protocol import CLINICAL,MRI,PET
from scripts.xyz_fusion_v2.models import fit_model,curves
from scripts.xyz_fusion_v1.pipeline import partition
from scripts.xyz_fusion_v1.statistics import ibs_score,outcome
from scripts.xyz_fusion_postall_v2.statistics import evaluate_cell
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
INPUTS={}
def read(path):
    path=Path(path);raw=path.read_bytes();INPUTS[str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    doc=json.loads(raw.decode('utf-8-sig'))
    if isinstance(doc,dict) and 'payload' in doc:
        if 'payload_sha256' in doc:assert digest(doc['payload'])==doc['payload_sha256'],'PAYLOAD_HASH_DRIFT'
        return doc['payload']
    return doc
def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)),'utf-8')
spec=importlib.util.spec_from_file_location('frozen_analyses',Path(r'C:\Users\USER\Desktop\draft\GeroScience_Submission_Package_v7_20261014\SUPPLEMENTARY_FOR_UPLOAD\ESM_3\revision_analyses\selected_analyses.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.setup(ROOT)
GRID=old.GRID;B=40000
def brier_losses(train,test,p,grid):
    """Same IPCW estimator; also defined before the first observed validation time.

    The library convenience API rejects t < min(validation follow-up), even
    when all subjects are known survivors at t. This exact formula retains
    the fixed grid rather than selecting a different horizon in small folds.
    """
    tr=outcome(train);te=Surv.from_arrays(test.event.to_numpy(bool),test.time_days.to_numpy(float))
    cutoff=np.nextafter(float(tr['time'].max()),0);te['event'] &= te['time']<cutoff;te['time']=np.minimum(te['time'],cutoff)
    assert grid[-1]<cutoff
    censor=CensoringDistributionEstimator().fit(tr);eg=censor.predict_proba(te['time']);tg=censor.predict_proba(np.asarray(grid));errors=[]
    for j,t in enumerate(grid):
        case=te['event']&(te['time']<=t);control=te['time']>t
        if np.any(eg[case]<=0) or tg[j]<=0:raise ValueError('NO_TRAINING_CENSORING_SUPPORT')
        loss=np.zeros(len(te));loss[case]=np.asarray(p)[case,j]**2/eg[case];loss[control]=(1-np.asarray(p)[control,j])**2/tg[j];errors.append(loss)
    return np.trapezoid(np.asarray(errors).T,x=grid,axis=1)/(grid[-1]-grid[0]),tr,te
def fold_arrays(arm):
    from scripts.xyz_fusion_v1.report import cell_frame
    train,test=cell_frame(arm)
    return brier_losses(train,test,np.asarray(arm['curves']),GRID)
old.fold_arrays=fold_arrays
def evaluate_extended(train,held,pred,config,fold):
    values={}
    for arm,v in pred.items():
        teframe=held.set_index('patient_hash').loc[v['subjects']].reset_index();loss,tr,te=brier_losses(train,teframe,np.asarray(v['survival']),GRID)
        try:c=float(concordance_index_ipcw(tr,te,1-np.asarray(v['survival'])[:,-1],tau=GRID[-1])[0])
        except ValueError:c=None
        values[arm]={'subjects':v['subjects'],'curves':v['survival'],'event':teframe.event.astype(int).tolist(),'time':teframe.time_days.tolist(),
            'censor_train':{'event':train.event.astype(int).tolist(),'time':train.time_days.tolist()},'metrics':{'ibs':float(loss.mean()),'uno_c_index':c}}
    return {'fold':fold,'arms':values,'train_subjects':sorted(train.patient_hash),'test_subjects':sorted(held.patient_hash)}
def ci(a,fam=1):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return np.quantile(a,[.05/(2*fam),1-.05/(2*fam)]).tolist() if len(a) else None

def paired(cells,a,b,family=3,seed=20261011):
    lengths=[len(c['arms'][a]['subjects']) for c in cells];n=sum(lengths)
    counts=old.bootstrap_counts(lengths,B,seed)
    ds=np.zeros(B);raw=np.zeros(B);zero=np.zeros(B);rnum=np.zeros(B);rden=np.zeros(B)
    pnuma=np.zeros(B);pnumb=np.zeros(B);pden=np.zeros(B);valid=np.ones(B,bool)
    folds=[];point_bs=[];point_c=[];point_pna=0.;point_pnb=0.;point_pd=0.
    absolute={a:[],b:[]}
    for f,c in enumerate(cells):
        av,bv=c['arms'][a],c['arms'][b]
        for key in ['subjects','event','time','censor_train']:assert av[key]==bv[key]
        la,tr,te=old.fold_arrays(av);lb,_,_=old.fold_arrays(bv)
        ma,de=old.concordance_matrices(tr,te,1-np.asarray(av['curves'])[:,-1],GRID[-1])
        mb,db=old.concordance_matrices(tr,te,1-np.asarray(bv['curves'])[:,-1],GRID[-1]);assert np.array_equal(de,db)
        ct=counts[f]
        na=np.einsum('bi,bi->b',ct@ma,ct);nb=np.einsum('bi,bi->b',ct@mb,ct);den=np.einsum('bi,bi->b',ct@de,ct)
        difference=np.divide(nb-na,den,out=np.full(B,np.nan),where=den>0)
        good=np.isfinite(difference);weight=lengths[f]/n;valid &= good
        ds+=ct@(lb-la)/n;raw+=difference*weight;zero+=np.where(good,difference,0)*weight
        rnum+=np.where(good,difference,0)*weight;rden+=good*weight
        pnuma+=na;pnumb+=nb;pden+=den
        pa=ma.sum()/de.sum();pb=mb.sum()/de.sum();point_c.append(pb-pa);point_bs.append(float((lb-la).mean()))
        point_pna+=ma.sum();point_pnb+=mb.sum();point_pd+=de.sum()
        absolute[a].append({'ibs':float(la.mean()),'uno':float(pa)});absolute[b].append({'ibs':float(lb.mean()),'uno':float(pb)})
        folds.append({'fold':f+1,'n':lengths[f],'events3y':int((te['event']&(te['time']<=GRID[-1])).sum()),'undefined_resamples':int((~good).sum()),'delta_ibs':point_bs[-1],'delta_uno':float(pb-pa)})
    renorm=np.divide(rnum,rden,out=np.full(B,np.nan),where=rden>0)
    pairdiff=np.divide(pnumb-pnuma,pden,out=np.full(B,np.nan),where=pden>0)
    rec={'n':n,'resamples':B,'seed':seed,'paired_reference':a,'paired_augmented':b,'delta_ibs':float(np.average(point_bs,weights=lengths)),
         'delta_uno':float(np.average(point_c,weights=lengths)),'valid_uno_resamples':int(valid.sum()),'undefined_uno_resamples':int((~valid).sum()),'folds':folds,
         'absolute':{k:{metric:float(np.average([v[metric] for v in vs],weights=lengths)) for metric in ['ibs','uno']} for k,vs in absolute.items()},'intervals':{},
         'pair_weighted_delta_uno':float((point_pnb-point_pna)/point_pd),
         'alternative_statistic_warning':'Zero-filled and renormalized differences change undefined-fold handling. Pair-weighted C changes the aggregation estimand; no cross-fold pairs are introduced. None is a proof of conservative coverage.'}
    for fam in sorted(set([1,family,40])):
        rec['intervals'][str(fam)]={'ibs_all':ci(ds,fam),'ibs_same_estimable_draws':ci(ds[valid],fam),'uno_original_estimable':ci(raw,fam),
            'uno_undefined_fold_delta_zero':ci(zero,fam),'uno_estimable_fold_renormalized':ci(renorm,fam),'uno_pair_weighted_within_fold':ci(pairdiff,fam)}
    return rec

def original_tau():
    jobs=ROOT/'results/xyz_fusion_v2/private/jobs';units=ROOT/'results/xyz_fusion_postall_v2/private/units'
    configs=[('cox',[read(jobs/f'tau_cox_ridge_o{i}_evaluation.json') for i in range(5)],'clinical_amyloid','clinical_amyloid_tau'),
             ('rsf',[read(jobs/f'tau_rsf_o{i}_evaluation.json') for i in range(5)],'clinical_amyloid','clinical_amyloid_tau'),
             ('mri',[read(units/f'tau_fold{i}_evaluation.json') for i in range(5)],'base','plus_tau')]
    old_record=read(Path(r'C:\Users\USER\Desktop\draft\GeroScience_Submission_Package_v7_20261014\SUPPLEMENTARY_FOR_UPLOAD\ESM_3\aggregate_results\tau_bootstrap.json'))
    results={}
    for name,cells,a,b in configs:
        v=paired(cells,a,b,3)
        known=old_record['models'][name]
        assert abs(v['delta_ibs']-known['delta_ibs'])<1e-12
        assert abs(v['delta_uno']-known['delta_uno'])<1e-12
        assert np.allclose(v['intervals']['3']['uno_original_estimable'],known['intervals']['3']['uno'],atol=1e-12)
        assert v['valid_uno_resamples']==known['valid_uno_resamples']
        results[name]=v;print('ORIGINAL_TAU_RECONCILED',name,v['valid_uno_resamples'],v['intervals']['40'],flush=True)
    dump(PUB/'A01_tau_omission_robustness.json',{'status':'EXECUTED_POST_HOC_FIXED_PREDICTIONS','models':results,'originals_reconciled':True})

def raw_pooled_tau():
    cells=[read(ROOT/f'results/xyz_oof_stacking_v1/private/units/tau_o{i}_evaluation.json') for i in range(5)]
    results={}
    for a,b in [('clinical_amyloid','clinical_amyloid_tau'),('clinical_mri_amyloid','clinical_mri_amyloid_tau')]:
        key=a+'__'+b;results[key]=paired(cells,a,b,3,20261014);print('RAW_OOF_PAIRED_C',key,results[key]['delta_uno'],flush=True)
    dump(PUB/'A02_raw_pooled_tau_paired.json',{'status':'EXECUTED_POST_HOC_FIXED_PREDICTIONS','family_per_metric':3,'models':results})

def effective_df(model,train,features,alpha):
    x=model[:-1].transform(train[features]);beta=model[-1].coef_;tm=train.time_days.to_numpy(float);ev=train.event.to_numpy(bool)
    eta=x@beta;eta-=eta.max();risk=np.exp(eta);info=np.zeros((x.shape[1],x.shape[1]))
    for t in np.unique(tm[ev]):
        at=tm>=t;w=risk[at];xx=x[at];d=int((ev&(tm==t)).sum());s0=w.sum();mu=(w[:,None]*xx).sum(0)/s0
        info+=d*((xx.T@(w[:,None]*xx))/s0-np.outer(mu,mu))
    return float(np.trace(np.linalg.solve(info+float(alpha)*np.eye(len(beta)),info)))

def select_alpha(train,features,config,seed):
    rows=[]
    splits=list(partition(train,3,seed))
    for alpha in [.1,1.,10.]:
        vals=[]
        for ai,bi in splits:
            fit,val=train.iloc[ai],train.iloc[bi]
            model=fit_model(fit,features,'cox_ridge',alpha,config,seed)
            vals.append(float(brier_losses(fit,val,curves(model,val,features,GRID),GRID)[0].mean()))
        rows.append({'alpha':alpha,'mean':float(np.mean(vals)),'se':float(np.std(vals,ddof=1)/np.sqrt(3))})
    best=min(rows,key=lambda r:r['mean']);winner=max(r['alpha'] for r in rows if r['mean']<=best['mean']+best['se'])
    return winner,rows

def overlap_analysis(inp):
    f=inp.frames['tau'].loc[inp.frames['tau'].blood_ptau217.notna()].copy().reset_index(drop=True)
    base=CLINICAL+['amyloid_centiloid'];defs={'base':base,'plasma':base+['blood_log_ptau217'],'tau':base+['tau_temporal'],'both':base+['blood_log_ptau217','tau_temporal']}
    cells=[];selection=[];models=PRIV/'overlap_models';models.mkdir(exist_ok=True)
    import pickle
    for fold,split in enumerate(inp.splits):
        train=f.loc[f.patient_hash.isin(split['train'])].copy().reset_index(drop=True);held=f.loc[f.patient_hash.isin(split['test'])].copy().sort_values('patient_hash').reset_index(drop=True)
        assert not set(train.patient_hash)&set(held.patient_hash);pred={}
        for name,features in defs.items():
            alpha,ranks=select_alpha(train,features,inp.config,inp.config['seed']+fold)
            model=fit_model(train,features,'cox_ridge',alpha,inp.config,inp.config['seed']+fold)
            with (models/f'{name}_o{fold}.pkl').open('wb') as stream:pickle.dump(model,stream)
            selection.append({'fold':fold+1,'arm':name,'alpha':alpha,'inner':ranks,'effective_df':effective_df(model,train,features,alpha)})
            x=held.drop(columns=[c for c in held if c in ('event','time_days') or c.endswith(('_event','_time_days'))])
            pred[name]={'subjects':list(x.patient_hash),'survival':curves(model,x,features,GRID).tolist()}
        cells.append(evaluate_extended(train,held,pred,inp.config,fold));print('OVERLAP_FOLD_LOCKED_EVALUATED',fold+1,len(train),len(held),flush=True)
    dump(PRIV/'overlap_cells.json',cells)
    results={a+'__'+b:paired(cells,a,b,4,20261024) for a,b in [('base','plasma'),('base','tau'),('plasma','tau'),('tau','both')]}
    dump(PUB/'A04_matched_biomarker_overlap.json',{'status':'EXECUTED_POST_HOC_NESTED_REFIT','n':len(f),'events':int(f.event.sum()),'events3y':int((f.event.astype(bool)&(f.time_days<=GRID[-1])).sum()),'family_per_metric':4,'results':results,'selection':selection})

def pooled_harmonized(inp):
    allcells={};diagnostics=[];edf=[]
    for track in ['mri','pet','blood','tau']:
        cells=[]
        for fold in range(5):
            cache=PRIV/f'harmonized_{track}_o{fold}.json'
            if cache.exists():cells.append(read(cache));print('RESUME_HARMONIZED',track,fold,flush=True);continue
            train=inp.split_frame(track,fold,'train');held=inp.split_frame(track,fold,'outer');seed_predictions=[]
            for seed in inp.config['training_seeds']:
                source='pet' if track=='pet' else 'mri';tasks=[f'{source}_s{seed}_o{fold}_c{c}' for c in range(3)]
                parts=[];tests=[]
                for task in tasks:
                    # Learn a component's unsupervised transform on ALL its disjoint meta rows, before biomarker subset filtering.
                    meta=inp.feature_frame(task,'meta');outer=inp.outer_for_head(track,task,fold)
                    cols=MRI+(PET if track=='pet' else [])
                    means=meta[cols].mean();sd=meta[cols].std(ddof=0)
                    if (sd<=1e-12).any():raise ValueError('NONIDENTIFIABLE_COMPONENT_SCORE_SCALE')
                    meta.loc[:,cols]=(meta[cols]-means)/sd;outer.loc[:,cols]=(outer[cols]-means)/sd
                    diagnostics.append({'track':track,'seed':seed,'fold':fold+1,'component':int(task[-1])+1,'transform_n':len(meta),'mean_after_maxabs':float(meta[cols].mean().abs().max()),'sd_after_maxerror':float((meta[cols].std(ddof=0)-1).abs().max()),'uses_outcomes':False})
                    parts.append(meta);tests.append(outer)
                pooled=pool_oof(parts,set(inp.split_frame(source,fold,'train').patient_hash))
                pooled=pooled.loc[pooled.patient_hash.isin(train.patient_hash)].copy()
                if track=='tau':pooled=pooled.merge(train[['patient_hash','tau_temporal']],validate='one_to_one')
                heads={}
                for arm,features in arms(track).items():
                    model=fit_model(pooled,features,'cox_ridge',10.,inp.config,seed);heads[arm]=model
                    edf.append({'workflow':'harmonized_pooled','track':track,'seed':seed,'fold':fold+1,'arm':arm,'n':len(pooled),'events':int(pooled.event.sum()),'p':len(features),'alpha':10.,'effective_df':effective_df(model,pooled,features,10.)})
                # All heads are fitted before any outer outcome is used.
                seed_predictions.append({arm:{'subjects':list(held.patient_hash),'survival':np.mean([curves(heads[arm],test,features,GRID) for test in tests],axis=0).tolist()} for arm,features in arms(track).items()})
            pred={arm:{'subjects':list(held.patient_hash),'survival':np.mean([p[arm]['survival'] for p in seed_predictions],axis=0).tolist()} for arm in arms(track)}
            cell=evaluate_cell(train,held,pred,inp.config,fold);dump(cache,cell);cells.append(cell)
            dump(PUB/'A03_transform_diagnostics.json',diagnostics);dump(PUB/'A05_pooled_edf.json',edf)
            print('HARMONIZED_FOLD',track,fold+1,flush=True)
        allcells[track]=cells
    result={}
    for track,a,b,family,k in contrasts():
        if a.startswith('split_'):continue
        result[track+'__'+a+'__'+b]=paired(allcells[track],a,b,k,20261024)
    # Direct raw vs harmonized augmented comparisons use the same outer test rows and censoring-training population.
    direct={}
    for track,arm in [('mri','clinical_mri_amyloid'),('pet','clinical_mri_amyloid_pet')]:
        cells=[]
        for fold,new in enumerate(allcells[track]):
            raw=read(ROOT/f'results/xyz_oof_stacking_v1/private/units/{track}_o{fold}_evaluation.json')['arms'][arm]
            nv=new['arms'][arm];order={key:i for i,key in enumerate(raw['subjects'])};idx=[order[key] for key in nv['subjects']]
            import copy
            raw=copy.deepcopy(raw)
            for key in ['subjects','event','time','curves']:raw[key]=[raw[key][i] for i in idx]
            assert raw['censor_train']==nv['censor_train']
            cells.append({'arms':{'raw':raw,'harmonized':nv}})
        direct[track]=paired(cells,'raw','harmonized',2,20261024)
    dump(PUB/'A03_harmonized_pooled_results.json',{'status':'EXECUTED_POST_HOC_FROZEN_CNN_HEAD_REFIT','new_CNN_training':False,'transform':'component mean/SD on disjoint meta rows; locked on outer predictions','results':result,'direct_raw_vs_harmonized':direct})

def component_tuned(inp):
    allcells={};edf=[];selection=[]
    for track in ['mri','tau']:
        cells=[]
        for fold in range(5):
            train=inp.split_frame(track,fold,'train');held=inp.split_frame(track,fold,'outer');components=[]
            for seed in inp.config['training_seeds']:
                for comp in range(3):
                    task=f'mri_s{seed}_o{fold}_c{comp}';meta=inp.feature_frame(task,'meta');meta=meta.loc[meta.patient_hash.isin(train.patient_hash)].copy().reset_index(drop=True)
                    if track=='tau':meta=meta.merge(train[['patient_hash','tau_temporal']],validate='one_to_one')
                    test=inp.outer_for_head(track,task,fold);pred={}
                    for arm,features in arms(track).items():
                        alpha,ranks=select_alpha(meta,features,inp.config,inp.config['seed']+fold)
                        fixed=fit_model(meta,features,'cox_ridge',10.,inp.config,seed)
                        tuned=fit_model(meta,features,'cox_ridge',alpha,inp.config,seed)
                        edf.append({'workflow':'component','track':track,'seed':seed,'fold':fold+1,'component':comp+1,'arm':arm,'n':len(meta),'events':int(meta.event.sum()),'p':len(features),'alpha':10.,'effective_df':effective_df(fixed,meta,features,10.)})
                        selection.append({'track':track,'seed':seed,'fold':fold+1,'component':comp+1,'arm':arm,'selected_alpha':alpha,'inner_ranks':ranks})
                        pred['fixed_'+arm]={'subjects':list(test.patient_hash),'survival':curves(fixed,test,features,GRID).tolist()}
                        pred['tuned_'+arm]={'subjects':list(test.patient_hash),'survival':curves(tuned,test,features,GRID).tolist()}
                    components.append(pred)
            pred={arm:{'subjects':list(held.patient_hash),'survival':np.mean([p[arm]['survival'] for p in components],axis=0).tolist()} for arm in components[0]}
            cell=evaluate_cell(train,held,pred,inp.config,fold);dump(PRIV/f'tuned_{track}_o{fold}.json',cell);cells.append(cell)
            dump(PUB/'A05_component_edf.json',edf);dump(PUB/'A05_component_tuning_choices.json',selection)
            print('COMPONENT_TUNED',track,fold+1,flush=True)
        allcells[track]=cells
    comparisons=[('mri','tuned_clinical_amyloid','tuned_clinical_mri_amyloid'),('tau','tuned_clinical_amyloid','tuned_clinical_amyloid_tau'),('tau','tuned_clinical_mri_amyloid','tuned_clinical_mri_amyloid_tau'),('mri','fixed_clinical_mri_amyloid','tuned_clinical_mri_amyloid')]
    results={track+'__'+a+'__'+b:paired(allcells[track],a,b,4,20261024) for track,a,b in comparisons}
    dump(PUB/'A05_component_tuning_results.json',{'status':'EXECUTED_POST_HOC_FROZEN_EXPERT_NESTED_HEAD_TUNING','family_per_metric':4,'inner_independence':'Each frozen expert saw none of its entire meta pool; inner folds are confined to that meta pool, not the pooled OOF matrix.','results':results})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['tau','raw','overlap','harmonized','tune']);args=ap.parse_args()
    started=time.time()
    with threadpool_limits(limits=2):
        if args.action=='tau':original_tau()
        elif args.action=='raw':raw_pooled_tau()
        else:
            inp=Inputs(ROOT,read(ROOT/'configs/xyz_oof_stacking_v1.json'))
            {'overlap':overlap_analysis,'harmonized':pooled_harmonized,'tune':component_tuned}[args.action](inp)
    INPUTS.update(old.INPUTS);dump(PRIV/f'input_hashes_{args.action}.json',INPUTS)
    print('ACTION_COMPLETE',args.action,round(time.time()-started,2),'seconds',flush=True)
if __name__=='__main__':main()
