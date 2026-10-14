"""Small, nested tabular sensitivity analyses; original image models untouched."""
import argparse,copy,hashlib,json,time,sys
from pathlib import Path
import selected_analyses as s
import numpy as np
from threadpoolctl import threadpool_limits

def run(repo,out):
    s.setup(repo);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    from scripts.xyz_fusion_v2.models import fit_model,curves
    from scripts.xyz_fusion_v2.protocol import CLINICAL
    from scripts.xyz_fusion_v1.pipeline import partition
    from scripts.xyz_fusion_v1.statistics import ibs_score
    from scripts.xyz_fusion_postall_v2.statistics import evaluate_cell
    from scripts.xyz_fusion_v2.reporting import paired_ci
    frame=s.parquet(s.SUP/'private_analysis_frame.parquet');tau=s.parquet(s.SUP/'private_tau_frame.parquet');splits=s.load(s.JOB/'splits.json')
    clinical=CLINICAL+['amyloid_centiloid']
    specs=[('mri',frame,{'clinical_amyloid':clinical,'without_mmse':[x for x in clinical if x!='baseline_mmscore'],'plus_hippocampus':clinical+['hippocampus_icv']},[('clinical_amyloid','without_mmse'),('clinical_amyloid','plus_hippocampus')]),
           ('compact_tau',tau,{'compact':['age','sex_binary','baseline_pacc_raw','amyloid_centiloid'],'compact_tau':['age','sex_binary','baseline_pacc_raw','amyloid_centiloid','tau_temporal']},[('compact','compact_tau')])]
    result={'status':'POST_HOC_NESTED_TABULAR_SENSITIVITY','same_original_outer_folds':True,'tuning':'3 inner event/treatment stratified folds; minimum IBS plus one-SE, choosing the largest ridge penalty','results':{}}
    for label,tab,arms,contrasts in specs:
        allcells=[];selection=[]
        for f,split in enumerate(splits):
            tr=tab[tab.patient_hash.isin(split['train'])].reset_index(drop=True);te=tab[tab.patient_hash.isin(split['test'])].reset_index(drop=True)
            assert not set(tr.patient_hash)&set(te.patient_hash)
            preds={};foldsel={}
            for arm,features in arms.items():
                ranks=[]
                for alpha in s.CONFIG['secondary']['cox_alpha_grid']:
                    losses=[]
                    for a,b in partition(tr,3,s.CONFIG['seed']+f):
                        model=fit_model(tr.iloc[a],features,'cox_ridge',alpha,s.CONFIG,s.CONFIG['seed']+f)
                        pp=curves(model,tr.iloc[b],features,s.GRID)
                        losses.append(ibs_score(tr.iloc[a],tr.iloc[b],pp,s.GRID))
                    ranks.append({'alpha':alpha,'mean_ibs':float(np.mean(losses)),'se':float(np.std(losses,ddof=1)/np.sqrt(3))})
                best=min(ranks,key=lambda x:x['mean_ibs']);winner=max((v for v in ranks if v['mean_ibs']<=best['mean_ibs']+best['se']),key=lambda x:x['alpha'])
                model=fit_model(tr,features,'cox_ridge',winner['alpha'],s.CONFIG,s.CONFIG['seed']+f)
                preds[arm]={'subjects':list(te.patient_hash),'survival':curves(model,te,features,s.GRID).tolist()}
                foldsel[arm]={'features':features,'winner':winner,'candidates':ranks,'train_n':len(tr),'train_events':int(tr.event.sum())}
            allcells.append(evaluate_cell(tr,te,preds,s.CONFIG,f));selection.append(foldsel)
            print('tabular',label,'fold',f+1,'complete',flush=True)
        absolute={}
        for arm in arms:
            metrics=[c['arms'][arm]['metrics'] for c in allcells];weights=[m['n'] for m in metrics]
            absolute[arm]={'n':sum(weights),'events':sum(m['events'] for m in metrics),'ibs':float(np.average([m['ibs'] for m in metrics],weights=weights)),'uno':float(np.average([m['uno_c_index'] for m in metrics],weights=weights)),'fold_metrics':metrics}
        cc=[paired_ci(allcells,a,b,s.GRID,20261013,2000,3) for a,b in contrasts]
        result['results'][label]={'models':absolute,'contrasts':cc,'inner_selections':selection}
        s.dump(out.parent/'private'/f'{label}_sensitivity_cells.json',allcells)
    old=s.load(s.SUP/'performance_supplement.json')['models']['full_clinical_amyloid']
    new=result['results']['mri']['models']['clinical_amyloid']
    result['original_full_training_comparator_replay']={'ibs_error':new['ibs']-old['ibs'],'uno_error':new['uno']-old['uno_c']}
    assert abs(new['ibs']-old['ibs'])<1e-10 and abs(new['uno']-old['uno_c'])<1e-10
    s.dump(out/'tabular_sensitivities.json',result)
    s.dump(out.parent/'private/tabular_inputs.json',s.INPUTS)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();start=time.time()
    with threadpool_limits(limits=2):run(a.repo,a.output)
    s.dump(Path(a.output)/'tabular_execution.json',{'seconds':time.time()-start,'date':'2026-10-13','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':sys.version})
