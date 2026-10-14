"""Small matched CNNs and an outcome-blind preparation path; audit after all fits."""
from pathlib import Path
import json,copy,time,os,random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_absolute_error
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import CensoringDistributionEstimator
from scripts.xyz_fusion_v1.statistics import outcome,support,ibs_score
from scripts.xyz_fusion_v1.loss import discrete_time_survival_nll,survival_from_logits
from .prepare import sha,dump,binding

CLINICAL=['age','sex_binary','education_years','apoe4_carrier','baseline_pacc_raw','baseline_mmscore','treatment_active','amyloid_centiloid']
EDGES=np.array([0,182.5,365,547.875,730.5,913.125,1095.75,1278.375,1461,1640],float)
GRID=np.linspace(365,1095.75,9)

class FixedSpatialPool(nn.Module):
 def forward(self,x):
  if x.shape[-3:]!=(8,8,8):raise ValueError('FIXED_64_CUBED_INPUT_REQUIRED')
  # Exact nonoverlapping4-cubed block mean, equivalent to adaptive output2 here.
  # Explicit reshape/mean has deterministic CUDA backward on the installed torch.
  return x.reshape(x.shape[0],x.shape[1],2,4,2,4,2,4).mean((3,5,7))

class Compact(nn.Module):
 def __init__(self,outputs):
  super().__init__();blocks=[];prev=2
  for n in (8,16,32):
   blocks.extend([nn.Conv3d(prev,n,3,stride=2,padding=1,bias=False),nn.GroupNorm(4,n),nn.SiLU(),nn.Conv3d(n,n,3,padding=1,bias=False),nn.GroupNorm(4,n),nn.SiLU()]);prev=n
  self.encoder=nn.Sequential(*blocks,FixedSpatialPool(),nn.Flatten(),nn.Linear(256,64),nn.SiLU())
  self.head=nn.Linear(64,outputs)
 def forward(self,x):return self.head(self.encoder(x))

def setup(seed,cfg):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.set_num_threads(cfg['cpu_threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)

def load_inputs(repo,cfg):
 out=Path(repo)/cfg['output_root'];contract=json.loads((out/'private/contract.json').read_text())
 if contract['config']!=cfg:raise ValueError('CONFIG_DRIFT')
 for rel,h in contract['sources'].items():
  if sha(Path(repo)/rel)!=h:raise ValueError('SOURCE_DRIFT:'+rel)
 if sha(Path(repo)/cfg['parent_root']/'private/contract.json')!=contract['parent_contract_sha256']:raise ValueError('PARENT_DRIFT')
 index=json.loads((out/'private/cache_index.json').read_text())
 if sha(out/'private/pilot_frame.parquet')!=index['frame_sha256']:raise ValueError('PILOT_FRAME_DRIFT')
 frame=pd.read_parquet(out/'private/pilot_frame.parquet');splits=contract['splits'];idx={r:np.flatnonzero(frame.patient_hash.isin(k)) for r,k in splits.items()}
 if set.union(*(set(v) for v in splits.values()))!=set(frame.patient_hash) or frame.patient_hash.duplicated().any():raise ValueError('PILOT_FRAME_MEMBERSHIP')
 a=[]
 for i,row in enumerate(frame.itertuples()):
  p=out/'private/scans'/row.mri_sample_hash;rec=json.loads((p/'receipt.json').read_text())
  if binding(rec)!=index['receipts'][i] or sha(p/'channels.npy')!=rec['output_sha256']:raise ValueError('CHANNEL_HASH_DRIFT')
  a.append(np.load(p/'channels.npy',allow_pickle=False))
 a=np.stack(a);mask=a[idx['fit'],1]!=0;vals=a[idx['fit'],1][mask]
 mean=float(vals.mean());sd=float(vals.std())
 if not np.isfinite(sd) or sd<1e-4:raise ValueError('NO_JACOBIAN_VARIATION')
 a[:,1]=np.where(a[:,1]!=0,(a[:,1]-mean)/sd,0)
 return out,frame,idx,torch.from_numpy(a),{'jac_train_mean':mean,'jac_train_sd':sd},contract

def fit_cox(x,y):
 m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),CoxPHSurvivalAnalysis(alpha=10.,n_iter=500,ties='breslow'))
 m.fit(x,outcome(y));return m
def co_curves(m,x,times):
 # Cox cumulative baseline hazard is flat after its last observed training time.
 # The final neural bin ends at1640 even when fitting follow-up ends earlier.
 # Evaluation at1–3years stays inside observed training support.
 return np.stack([f(np.clip(times,f.domain[0],f.domain[1])) for f in m.predict_survival_function(x)])
def make_offset(m,x):
 s=co_curves(m,x,EDGES[1:]);prev=np.c_[np.ones(len(s)),s[:,:-1]];haz=np.clip(1-s/np.clip(prev,1e-12,None),1e-5,1-1e-5)
 return torch.tensor(np.log(haz/(1-haz)),dtype=torch.float32)
def grid_curve(logits):
 s=survival_from_logits(torch.as_tensor(logits)).numpy();return np.array([np.interp(GRID,EDGES[1:],v) for v in s])
def batches(ids,size):
 for i in range(0,len(ids),size):yield ids[i:i+size]
def input_batch(x,ids,arm,device):
 z=x[ids].to(device).clone()
 if arm=='t1':z[:,1]=0
 return z
def predictions(model,x,ids,arm,device,offset=None):
 model.eval();values=[]
 with torch.no_grad():
  for ix in batches(ids,16):
   y=model(input_batch(x,ix,arm,device)).cpu()
   if offset is not None:y=y+offset[ix]
   values.append(y.numpy())
 return np.concatenate(values)

def fit_all(repo,cfg):
 out,frame,idx,x,norm,contract=load_inputs(repo,cfg);device='cuda' if torch.cuda.is_available() else 'cpu'
 if device!='cuda':raise ValueError('CUDA_REQUIRED_FOR_BOUNDED_PILOT')
 train=frame.iloc[idx['fit']];sel=frame.iloc[idx['selection']];start=time.time();run_start=start
 co=fit_cox(train[CLINICAL],train);offset=make_offset(co,frame[CLINICAL]);edges=torch.tensor(EDGES,device=device)
 targets=frame[cfg['positive_targets']].to_numpy(float);mean=np.nanmean(targets[idx['fit']],axis=0);sd=np.nanstd(targets[idx['fit']],axis=0)
 if (sd<=1e-8).any():raise ValueError('POSITIVE_TARGET_CONSTANT')
 y=torch.tensor(np.nan_to_num((targets-mean)/sd),dtype=torch.float32);valid=torch.tensor(np.isfinite(targets),dtype=torch.float32)
 e=torch.tensor(frame.event.to_numpy(),dtype=torch.float32);t=torch.tensor(frame.time_days.to_numpy(),dtype=torch.float64)
 output=out/'private/models';output.mkdir(parents=True,exist_ok=True);locks=[]
 tiny_path=out/'checks/tiny_real_overfit.json'
 if not tiny_path.exists():
  setup(13,cfg);ids=np.array([i for i in idx['fit'] if valid[i].all()])[:8]
  if len(ids)!=8:raise ValueError('POSITIVE_CONTROL_SUPPORT')
  tiny=Compact(2).to(device);o=torch.optim.Adam(tiny.parameters(),lr=.003);z=x[ids].to(device);target=y[ids].to(device)
  initial=float(((tiny(z)-target)**2).mean().detach())
  for step in range(160):
   o.zero_grad(set_to_none=True);loss=((tiny(z)-target)**2).mean();loss.backward();o.step()
  final=float(((tiny(z)-target)**2).mean().detach());passed=final<initial*.25
  dump(tiny_path,{'status':'PASS' if passed else 'FAIL','initial_mse':initial,'final_mse':final,'n':8,'steps':160,'no_audit_rows':True,'contract':binding(contract)})
  del tiny,z,target;torch.cuda.empty_cache()
 else:
  q=json.loads(tiny_path.read_text());passed=q['status']=='PASS' and q['contract']==binding(contract)
 if not passed:raise ValueError('TINY_REAL_OVERFIT_FAILED_DO_NOT_SCALE')
 for kind,seeds in [('positive',[13]),('survival',cfg['training_seeds'])]:
  for seed in seeds:
   for arm in cfg['arms']:
    name=f'{kind}_{arm}_s{seed}';path=output/(name+'.pt');receipt=output/(name+'.json')
    if receipt.exists():
     rec=json.loads(receipt.read_text())
     if rec['contract']!=binding(contract) or sha(path)!=rec['sha256']:raise ValueError('MODEL_RESUME_DRIFT')
     locks.append(rec);continue
    setup(seed,cfg);m=Compact(2 if kind=='positive' else 9).to(device)
    if kind=='survival':nn.init.zeros_(m.head.weight);nn.init.zeros_(m.head.bias)
    opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    best=float('inf');best_state=None;bad=0;history=[];rng=np.random.default_rng(seed)
    for epoch in range(cfg['max_epochs']):
     if time.time()-run_start>cfg['max_training_seconds']:raise TimeoutError('PILOT_TRAINING_BUDGET_EXHAUSTED_COMPLETE_MODELS_PRESERVED')
     m.train();losses=[]
     for ids in batches(rng.permutation(idx['fit']),cfg['batch_size']):
      opt.zero_grad(set_to_none=True);pred=m(input_batch(x,ids,arm,device))
      if kind=='positive':
       mask=valid[ids].to(device);loss=(((pred-y[ids].to(device))**2)*mask).sum()/mask.sum().clamp(min=1)
      else:loss=discrete_time_survival_nll(pred+offset[ids].to(device),e[ids].to(device),t[ids].to(device),edges)
      if not torch.isfinite(loss):raise ValueError('NONFINITE_PILOT_LOSS')
      loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step();losses.append(float(loss.detach()))
     pr=predictions(m,x,idx['selection'],arm,device,offset if kind=='survival' else None)
     if kind=='positive':criterion=float(np.nanmean((pr-(targets[idx['selection']]-mean)/sd)**2))
     else:criterion=ibs_score(train,sel,grid_curve(pr),GRID)
     history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),'selection_loss':criterion})
     if criterion<best-1e-7:best=criterion;best_state=copy.deepcopy(m.state_dict());best_epoch=epoch+1;bad=0
     else:bad+=1
     if (epoch+1)%4==0:print(json.dumps({'stage':'fit','model':name,**history[-1],'elapsed_seconds':round(time.time()-start)}),flush=True)
     if epoch+1>=cfg['minimum_epochs'] and bad>=cfg['patience']:break
    m.load_state_dict(best_state);torch.save({'state_dict':{k:v.cpu() for k,v in best_state.items()},'kind':kind,'normalization':norm,'positive_mean':mean.tolist(),'positive_sd':sd.tolist()},path)
    rec={'model':name,'kind':kind,'arm':arm,'seed':seed,'contract':binding(contract),'sha256':sha(path),'best_epoch':best_epoch,'selection_loss':best,'history':history,'parameters':sum(p.numel() for p in m.parameters()),'audit_outcomes_used':False,'seconds':time.time()-start}
    dump(receipt,rec);locks.append(rec)
    del m;torch.cuda.empty_cache()
 dump(out/'private/all_models_locked.json',{'contract':binding(contract),'models':locks,'count':len(locks),'audit_evaluated':False})
 print(json.dumps({'stage':'all_models_locked','count':len(locks),'seconds':time.time()-start}),flush=True)

def individual_ibs(train,test,curves):
 tr,te,_=support(train,test,GRID);c=CensoringDistributionEstimator().fit(tr)
 gt=c.predict_proba(te['time']);gg=c.predict_proba(GRID);gt=np.where(gt==0,np.inf,gt);gg=np.where(gg==0,np.inf,gg)
 losses=np.zeros_like(curves)
 for k,v in enumerate(GRID):losses[:,k]=curves[:,k]**2*((te['time']<=v)&te['event'])/gt+(1-curves[:,k])**2*(te['time']>v)/gg[k]
 values=np.trapezoid(losses,GRID,axis=1)/(GRID[-1]-GRID[0])
 if not np.isclose(values.mean(),ibs_score(train,test,curves,GRID),rtol=1e-10,atol=1e-12):raise ValueError('IPCW_LIBRARY_PARITY_FAILED')
 return values

def evaluate(repo,cfg):
 out,frame,idx,x,norm,contract=load_inputs(repo,cfg);lock=json.loads((out/'private/all_models_locked.json').read_text())
 if lock['contract']!=binding(contract) or lock['count']!=6:raise ValueError('ALL_SIX_MODELS_MUST_BE_LOCKED')
 train=frame.iloc[idx['fit']];audit=frame.iloc[idx['audit']];co=fit_cox(train[CLINICAL],train);offset=make_offset(co,frame[CLINICAL])
 co_pred=co_curves(co,audit[CLINICAL],GRID);preds={};positives={};curves_by_arm={};device='cuda'
 for rec in lock['models']:
  path=out/'private/models'/(rec['model']+'.pt')
  if sha(path)!=rec['sha256']:raise ValueError('MODEL_HASH_DRIFT')
  state=torch.load(path,map_location='cpu',weights_only=True);m=Compact(2 if rec['kind']=='positive' else 9).to(device);m.load_state_dict(state['state_dict'])
  p=predictions(m,x,idx['audit'],rec['arm'],device,offset if rec['kind']=='survival' else None)
  if rec['kind']=='positive':
   p=p*np.array(state['positive_sd'])+np.array(state['positive_mean']);metrics={}
   for k,col in enumerate(cfg['positive_targets']):
    obs=audit[col].to_numpy(float);use=np.isfinite(obs)
    metrics[col]={'n':int(use.sum()),'r2':float(r2_score(obs[use],p[use,k])),'mae':float(mean_absolute_error(obs[use],p[use,k]))}
   positives[rec['arm']]=metrics
  else:
   curves_by_arm.setdefault(rec['arm'],[]).append(grid_curve(p));preds[rec['model']]=grid_curve(p)
  del m
 from sksurv.metrics import concordance_index_ipcw
 def score(p):
  tr,te,_=support(train,audit,GRID)
  return {'ibs':ibs_score(train,audit,p,GRID),'c_index':float(concordance_index_ipcw(tr,te,1-p[:,-1],tau=GRID[-1])[0]),'risk_sd_3y':float(np.std(1-p[:,-1]))}
 aggregate={'clinical_cox':score(co_pred)}
 for arm,values in curves_by_arm.items():aggregate[arm]=score(np.mean(values,axis=0))
 # Simple anatomical comparator and a spatially pooled Jacobian baseline use identical fit rows.
 region=CLINICAL+['hippocampus_icv','entorhinal_icv','ventricle_icv'];reg=fit_cox(train[region],train);aggregate['clinical_regional_volumes']=score(co_curves(reg,audit[region],GRID))
 jac=x[:,1].numpy().reshape(len(x),4,16,4,16,4,16).mean((2,4,6)).reshape(len(x),64)
 jf=pd.DataFrame(jac,columns=[f'jac_{i}' for i in range(64)]);tab=pd.concat([frame[CLINICAL].reset_index(drop=True),jf],axis=1)
 jc=fit_cox(tab.iloc[idx['fit']],train);aggregate['clinical_pooled_jacobian_cox']=score(co_curves(jc,tab.iloc[idx['audit']],GRID))
 a=np.mean(curves_by_arm['t1_logjac'],axis=0);b=np.mean(curves_by_arm['t1'],axis=0)
 diff=individual_ibs(train,audit,a)-individual_ibs(train,audit,b);rng=np.random.default_rng(cfg['split_seed']);boot=np.array([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(cfg['gate']['bootstrap_resamples'])]);ci=np.quantile(boot,[.1,.9]).tolist()
 byseed=[score(preds[f'survival_t1_logjac_s{s}'])['ibs']-score(preds[f'survival_t1_s{s}'])['ibs'] for s in cfg['training_seeds']]
 criteria={'adequate_event_support':int(((audit.event==1)&(audit.time_days<=1095.75)).sum())>=cfg['gate']['minimum_audit_3y_events'],'adequate_followup':int((audit.time_days>1095.75).sum())>=cfg['gate']['minimum_audit_at_risk_3y'],
 'positive_controls':all(positives['t1_logjac'][k]['r2']>0 for k in cfg['positive_targets']),
 'both_seeds_improve':all(v<0 for v in byseed),'conditional_80pct_upper_below_zero':ci[1]<0,
 'absolute_ibs_gain':float(diff.mean())<=-cfg['gate']['minimum_absolute_ibs_gain'],'relative_ibs_gain':float(diff.mean())<=-cfg['gate']['minimum_relative_ibs_gain']*aggregate['t1']['ibs'],
 'improves_clinical_comparator':aggregate['t1_logjac']['ibs']<aggregate['clinical_cox']['ibs'],
 'ranking_not_materially_worse':aggregate['t1_logjac']['c_index']>=aggregate['t1']['c_index']-cfg['gate']['maximum_c_index_drop']}
 result={'status':'DEVELOPMENT_GATE_PASSED_FULL_PROTOCOL_REVIEW_REQUIRED' if all(criteria.values()) else 'DEVELOPMENT_GATE_NOT_PASSED_DO_NOT_SCALE','claim_status':'POST_HOC_PILOT_NOT_PROOF_OF_CLINICAL_BENEFIT','support':{'fit_n':len(train),'selection_n':len(idx['selection']),'audit_n':len(audit),'audit_events_3y':int(((audit.event==1)&(audit.time_days<=1095.75)).sum()),'audit_events_full':int(audit.event.sum())},'metrics':aggregate,'positive_controls':positives,'jacobian_minus_t1_ibs':float(diff.mean()),'paired_conditional_80pct_ci':ci,'seed_differences':byseed,'gate_criteria':criteria,'audit_used_for_training_or_early_stopping':False,'original_outer_test_used':False,'primary_results_overwritten':False,'full_dataset_started':False}
 dump(out/'public/pilot_results.json',result);dump(out/'private/audit_prediction_hashes.json',{k:binding(v.tolist()) for k,v in preds.items()});print(json.dumps(result),flush=True);return result
