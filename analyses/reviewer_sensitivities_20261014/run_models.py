"""Portable entry to the original tabular and paired-expert fitting code.

The caller supplies authorized prepared inputs. This does not fabricate an
independent cohort or silently reconstruct unavailable participant data.
"""
from pathlib import Path
import argparse,json,os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'source_snapshot'))

def read(path):
    j=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    return j.get('payload',j) if isinstance(j,dict) else j

def execute(a):
    import numpy as np,pandas as pd
    from scripts.xyz_fusion_v1.store import Store,writer_lock,atomic_json,file_sha
    from scripts.xyz_fusion_v2.protocol import validate_config,CLINICAL,arms,MRI,PET
    from scripts.xyz_fusion_v2.tabular import run_tabular
    from scripts.xyz_fusion_v2.pipeline import paired_crossfit,ensemble_seeds
    from scripts.xyz_fusion_v2.training import ImageBank
    from scripts.xyz_fusion_postall_v2.statistics import metrics
    from scripts.xyz_fusion_v1.report import cell_frame
    frame=pd.read_parquet(a.frame);splits=read(a.splits);config=read(a.config);validate_config(config)
    predictors={col for cols in arms(a.track).values() for col in cols}-set(MRI+PET)
    required=['patient_hash','event','time_days',*predictors]
    if a.track in ['scalar','mri']:required+=['blood_ptau217']
    if a.track in ['mri','pet']:required+=['cache_row']
    missing=set(required)-set(frame)
    if missing:raise ValueError('MISSING_FRAME_COLUMNS:'+','.join(sorted(missing)))
    if frame.patient_hash.duplicated().any():raise ValueError('DUPLICATE_PARTICIPANT')
    if len(splits)!=5:raise ValueError('FIVE_SAVED_OUTER_SPLITS_REQUIRED')
    ids=set(frame.patient_hash);seen=set();counts=[]
    for f,sp in enumerate(splits):
        train,test=set(sp['train'])&ids,set(sp['test'])&ids
        if train&test or seen&test or train|test!=ids:raise ValueError('INVALID_PARTICIPANT_SPLIT')
        seen|=test;counts.append({'fold':f+1,'train':len(train),'test':len(test)})
    if seen!=ids:raise ValueError('INCOMPLETE_HELD_OUT_COVERAGE')
    for col in ['event','time_days']:
        if not np.isfinite(frame[col]).all():raise ValueError('INVALID_OUTCOME')
    if not frame.event.isin([0,1,False,True]).all() or (frame.time_days<=0).any():raise ValueError('INVALID_EVENT_OR_TIME')
    output=Path(a.output).resolve()
    if output==Path(a.frame).resolve().parent or output in Path(a.frame).resolve().parents:raise ValueError('OUTPUT_MUST_NOT_CONTAIN_INPUTS')
    print(json.dumps({'mode':a.mode,'track':a.track,'n':len(frame),'events':int(frame.event.sum()),'folds':counts,'training_started':False}),flush=True)
    if a.mode=='validate':return
    binding={'purpose':'AUTHORIZED_RECONSTRUCTION_FROM_PREPARED_INPUTS','config':config,'track':a.track,'frame_sha256':file_sha(a.frame),'splits_sha256':file_sha(a.splits),'source_manifest_sha256':file_sha(HERE/'source_snapshot_manifest.json')}
    if a.track in ['mri','pet']:
        if not a.mri_cache:raise ValueError('MRI_CACHE_REQUIRED')
        mm=np.load(a.mri_cache,mmap_mode='r',allow_pickle=False)
        if mm.ndim!=5 or tuple(mm.shape[1:])!=(1,128,128,128) or frame.cache_row.max()>=len(mm):raise ValueError('MRI_CACHE_GEOMETRY_OR_ROW_MISMATCH')
        binding['mri_cache_sha256']=file_sha(a.mri_cache)
    with writer_lock(output):
        store=Store(output,binding)
        if a.track in ['scalar','tau']:
            result=run_tabular(frame,config,store,'reconstruction',a.track,splits,families=a.families)
        else:
            banks={'mri':ImageBank(a.mri_cache,config['training']['ram_cache_max_gib'])}
            if a.track=='pet':
                from scripts.xyz_fusion_v2.inference import PETBank
                for col in ['image_path_private','h5_group','source_image_sha256']:
                    if col not in frame:raise ValueError('PET_PREPARED_FRAME_MISSING:'+col)
                banks['pet']=PETBank(frame)
            runs=paired_crossfit(frame,banks,config,store,a.track,splits);result={'image_ensemble':ensemble_seeds(runs,frame,config)}
            if 'pet' in banks:banks['pet'].close()
        # Use the corrected final metric implementation on reconstructed curves.
        for cells in result.values():
            for cell in cells:
                for arm in cell['arms'].values():
                    tr,te=cell_frame(arm);arm['metrics']=metrics(tr,te,arm['curves'],config['fusion']['ibs_grid_days'],config['fusion']['horizons_days'])
        atomic_json(output/'private/reconstructed_cells.json',result)
        aggregate={}
        for family,cells in result.items():
            aggregate[family]={}
            for arm in cells[0]['arms']:
                vals=[c['arms'][arm]['metrics'] for c in cells];weights=[v['n'] for v in vals]
                aggregate[family][arm]={'n':sum(weights),'ibs':float(np.average([v['ibs'] for v in vals],weights=weights)),'uno':float(np.average([v['uno_c_index'] for v in vals],weights=weights)) if all(v['uno_c_index'] is not None for v in vals) else None}
        atomic_json(output/'aggregate_reconstruction.json',aggregate)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True);p.add_argument('--splits',required=True);p.add_argument('--config',default=str(HERE/'config/xyz_fusion_v2.json'));p.add_argument('--track',choices=['scalar','tau','mri','pet'],required=True);p.add_argument('--mri-cache');p.add_argument('--output',required=True);p.add_argument('--mode',choices=['validate','fit'],default='validate');p.add_argument('--families',nargs='+',choices=['cox_ridge','rsf']);execute(p.parse_args())
