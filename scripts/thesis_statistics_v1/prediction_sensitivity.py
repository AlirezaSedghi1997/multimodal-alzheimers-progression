from pathlib import Path
import sys,json,copy,hashlib
import numpy as np,pandas as pd
from .runtime import configure
R, P = configure()
from scripts.xyz_fusion_v2.models import fit_model,curves,score_features,assert_pair
from scripts.xyz_fusion_v2.protocol import CLINICAL,MRI
from scripts.xyz_fusion_postall_v2.statistics import evaluate_cell,metrics
from scripts.xyz_fusion_v2.reporting import paired_ci
from scripts.xyz_fusion_v1.statistics import ibs_score
from scripts.xyz_fusion_v1.report import cell_frame,individual_bs
from scripts.xyz_fusion_postall_v2.secondary import subject_standardize,fit_pacc_models
from sksurv.nonparametric import kaplan_meier_estimator
def read(path):
 j=json.loads(Path(path).read_text(encoding='utf-8-sig'));return j.get('payload',j) if isinstance(j,dict) else j
def dump(name,x):
 (P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def job(name):return read(R/'results/xyz_fusion_v2/private/jobs'/f'{name}.json')
def unit(name):return read(R/'results/xyz_fusion_postall_v2/private/units'/f'{name}.json')
c=read(R/'results/xyz_fusion_v2/private/contract.json')['protocol'];grid=c['fusion']['ibs_grid_days']
frame=pd.read_parquet(P/'private_analysis_frame.parquet');tau=pd.read_parquet(P/'private_tau_frame.parquet');splits=job('splits')
report=read(R/'results/xyz_fusion_postall_v2/public/results.json')
def recalc(cells):
 cells=copy.deepcopy(cells)
 for cell in cells:
  for arm,v in cell['arms'].items():
   tr,te=cell_frame(v)
   v['metrics']=metrics(tr,te,v['curves'],grid,c['fusion']['horizons_days'])
 return cells
def aggregate(cells):
 result={}
 for arm in cells[0]['arms']:
  vals=[ce['arms'][arm] for ce in cells];weights=[len(v['subjects']) for v in vals]
  ms=[v['metrics'] for v in vals]
  result[arm]={'n':sum(weights),'events':sum(sum(v['event']) for v in vals),'ibs':float(np.average([m['ibs'] for m in ms],weights=weights)),'uno_c':float(np.average([m['uno_c_index'] for m in ms],weights=weights)) if all(m['uno_c_index'] is not None for m in ms) else None,'fold_metrics':ms}
 return result
if not (P/'performance_supplement.json').exists():
 mri=recalc(job('mri_ensemble_cells'));scalar=recalc([job(f'scalar_confirmation_time_cox_ridge_o{i}_evaluation') for i in range(5)])
 combined=[]
 for i,ce in enumerate(mri):
  names={'paired_clinical':'clinical','paired_clinical_amyloid':'clinical_amyloid','deep_mri_amyloid':'clinical_mri_amyloid','deep_volumes_mri_amyloid':'clinical_volumes_mri_amyloid'}
  arms={new:ce['arms'][old] for new,old in names.items()}
  for old in ['clinical','clinical_amyloid','clinical_volumes_amyloid']:arms['full_'+old]=scalar[i]['arms'][old]
  ref=arms['full_clinical'];tr,te=cell_frame(ref);tx,s=kaplan_meier_estimator(tr.event.to_numpy(bool),tr.time_days.to_numpy())
  null=np.empty((len(te),len(grid)),dtype=float)
  # KM is a right-continuous step function, not linear interpolation.
  null[:]=np.array([1 if np.searchsorted(tx,t,side='right')==0 else s[np.searchsorted(tx,t,side='right')-1] for t in grid])
  nv=copy.deepcopy(ref);nv['curves']=null.tolist();nv['metrics']=metrics(tr,te,null,grid,c['fusion']['horizons_days']);arms['null']=nv
  assert all(v['subjects']==ref['subjects'] for v in arms.values())
  combined.append(dict(ce,arms=arms))
 out=aggregate(combined);null=out['null']['ibs']
 for a,v in out.items():v['ipa']=1-v['ibs']/null
 comparisons=[]
 for a,b in [('full_clinical_amyloid','deep_mri_amyloid'),('full_clinical_volumes_amyloid','deep_volumes_mri_amyloid')]:
  comparisons.append(paired_ci(combined,a,b,grid,20261011,2000,2));print('baseline contrast',a,flush=True)
 dump('performance_supplement.json',{'models':out,'comparisons':comparisons,'null_fitted_on_outer_train':True})
 dump('private_comparison_cells.json',combined)
if not (P/'tau_multiplicity.json').exists():
 tauset=[('cox',[job(f'tau_cox_ridge_o{i}_evaluation') for i in range(5)],'clinical_amyloid','clinical_amyloid_tau'),('rsf',[job(f'tau_rsf_o{i}_evaluation') for i in range(5)],'clinical_amyloid','clinical_amyloid_tau'),('mri',[unit(f'tau_fold{i}_evaluation') for i in range(5)],'base','plus_tau')]
 result={}
 for name,cells,a,b in tauset:
  cells=recalc(cells);diff=paired_ci(cells,a,b,grid,20261011,2000,3)
  result[name]={'difference':diff,'models':aggregate(cells)};print('tau multiplicity',name,flush=True)
 dump('tau_multiplicity.json',result)
if not (P/'nocognitive_cells.json').exists():
 from .runtime import require_frozen_inference
 require_frozen_inference()
 import torch,time
 from scripts.xyz_fusion_v2.training import MRISurvival,predict,deterministic
 deterministic(20261011,2)
 class FoldBank:
  def __init__(self,held):
   mm=np.load(R/'results/xyz_mri_baseline_fast_v1/private/mri_cache.npy',mmap_mode='r')
   self.rows=list(held.cache_row.astype(int));self.loc={v:i for i,v in enumerate(self.rows)};self.data=np.array(mm[self.rows])
  def batch(self,rows,device):return torch.from_numpy(self.data[[self.loc[int(i)] for i in rows]]).to(device)
 def full_outer(task,lock,held,bank):
  path=P/(task+'_expert_outer.json')
  if path.exists():return read(path)
  cp=R/'results/xyz_fusion_v2/private/models'/f'{task}_mri_expert/locked.pt'
  assert hashlib.sha256(cp.read_bytes()).hexdigest()==lock['experts']['mri']['sha256']
  policy=dict(c['training'],modality='mri');model=MRISurvival(policy)
  model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model'],strict=True)
  device=torch.device('cuda');model.to(device).eval();model.requires_grad_(False)
  start=__import__('time').time();pred=predict(model,bank,held[['patient_hash','cache_row']],policy,device,grid)
  v={'subjects':list(held.patient_hash),'survival':pred.tolist(),'model_sha256':lock['experts']['mri']['sha256']}
  dump(path.name,v);del model;torch.cuda.empty_cache();print('frozen inference',task,round(__import__('time').time()-start,1),'s',flush=True);return v
 cells=[];features=[v for v in CLINICAL if v not in ['baseline_pacc_raw','baseline_mmscore']]+MRI+['amyloid_centiloid']
 for fold,split in enumerate(splits):
  held=frame.set_index('patient_hash').loc[split['test']].reset_index();components=[];bank=FoldBank(held)
  for seed in c['training_seeds']:
   for comp in range(3):
    task=f'mri_s{seed}_o{fold}_c{comp}';lock=job(task+'_lock');assert_pair(lock['experts']['mri'],lock['meta_subjects'],lock['outer_subjects'])
    meta_score=job(task+'_mri_meta');outer_score=full_outer(task,lock,held,bank)
    assert meta_score['model_sha256']==outer_score['model_sha256']==lock['experts']['mri']['sha256']
    meta=frame.set_index('patient_hash').loc[meta_score['subjects']].reset_index()
    test=frame.set_index('patient_hash').loc[outer_score['subjects']].reset_index()
    meta=score_features(meta,meta_score['survival'],grid,c['fusion']['score_times_days'],'mri')
    test=score_features(test,outer_score['survival'],grid,c['fusion']['score_times_days'],'mri')
    model=fit_model(meta,features,'cox_ridge',10,c,seed)
    pred=curves(model,test,features,grid)
    pred=pd.DataFrame(pred,index=test.patient_hash).loc[held.patient_hash].to_numpy();components.append(pred)
  pred=np.mean(components,axis=0);train=frame[frame.patient_hash.isin(split['train'])]
  cells.append(evaluate_cell(train,held,{'no_cognitive':{'subjects':list(held.patient_hash),'survival':pred.tolist()}},c,fold));print('noncognitive fold',fold,flush=True)
  del bank
 dump('nocognitive_cells.json',cells)
if not (P/'nocognitive_pacc.json').exists():
 cells=read(P/'nocognitive_cells.json');scores={k:1-v[-1] for ce in cells for k,v in zip(ce['arms']['no_cognitive']['subjects'],ce['arms']['no_cognitive']['curves'])}
 t=pd.read_parquet(R/'results/xyz_fusion_postall_v2/private/pacc_longitudinal.parquet')
 t['oof_risk_3y']=t.patient_hash.map(scores);assert t.oof_risk_3y.notna().all()
 t['oof_risk_3y_z'],scaling=subject_standardize(t,'oof_risk_3y')
 fits=fit_pacc_models(t)
 dump('nocognitive_pacc.json',{'models':fits,'risk_scale':scaling,'n_people':int(t.patient_hash.nunique()),'n_observations':len(t),'risk_features_exclude':['baseline_pacc_raw','baseline_mmscore'],'longitudinal_adjustment_includes_baseline_cognition':True,'prediction':aggregate(cells),'new_fits':45,'frozen_cnn':True})
 print('PACC sensitivity complete',flush=True)
