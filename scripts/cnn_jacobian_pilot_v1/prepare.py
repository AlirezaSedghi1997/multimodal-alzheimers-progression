from pathlib import Path
import hashlib,json,os,time,shutil
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .geometry import checked_logjac,geometry_selftest

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8');os.replace(tmp,p)
def binding(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def split_manifest(frame,outer_train,outer_test,cfg):
 if set(outer_train)&set(outer_test):raise ValueError('OUTER_OVERLAP')
 f=frame.loc[frame.patient_hash.isin(outer_train)].sort_values('patient_hash').reset_index(drop=True)
 if len(f)!=len(set(outer_train)) or f.patient_hash.duplicated().any():raise ValueError('INVALID_OUTER_TRAIN')
 strata=np.where((f.event==1)&(f.time_days<=1095.75),2,np.where(f.event==1,1,0))
 pilot,_=train_test_split(np.arange(len(f)),train_size=cfg['pilot_n'],random_state=cfg['split_seed'],stratify=strata)
 dev,audit=train_test_split(pilot,test_size=cfg['audit_n'],random_state=cfg['split_seed']+1,stratify=strata[pilot])
 fit,sel=train_test_split(dev,test_size=cfg['selection_n'],random_state=cfg['split_seed']+2,stratify=strata[dev])
 out={role:f.iloc[ids].sort_values('patient_hash').patient_hash.tolist() for role,ids in [('fit',fit),('selection',sel),('audit',audit)]}
 sets=[set(x) for x in out.values()]
 if any(sets[i]&sets[j] for i in range(3) for j in range(i)) or set.union(*sets)&set(outer_test):raise ValueError('PILOT_LEAKAGE')
 return out

def one_scan(job):
 row,repo,cfg=job;repo=Path(repo);out=repo/cfg['output_root'];folder=repo/cfg['cache_root']/'private/mri_scans'/row['mri_sample_hash']
 target=out/'private/scans'/row['mri_sample_hash'];target.mkdir(parents=True,exist_ok=True)
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']:os.environ[k]='1'
 receipt=json.loads((folder/'receipt.json').read_text());v=receipt['payload']
 if v['status']!='PASS' or row['mri_sample_hash']!=v['sample_key']:raise ValueError('PARENT_SCAN_QC_FAILED')
 names=['tensor.npy',cfg['jacobian']['warp'],'registration_0GenericAffine.mat']
 for name in names:
  if sha(folder/name)!=v['outputs'][name]:raise ValueError('PARENT_SCAN_HASH_DRIFT')
 inputs={name:v['outputs'][name] for name in names}
 bind=binding({'inputs':inputs,'jacobian':cfg['jacobian'],'size':cfg['image_size']})
 if (target/'receipt.json').exists():
  old=json.loads((target/'receipt.json').read_text())
  if old['binding']!=bind or sha(target/'channels.npy')!=old['output_sha256']:raise ValueError('PILOT_CACHE_DRIFT')
  return old
 if cfg.get('reuse_prepared_scans_from'):
  previous=(repo/cfg['reuse_prepared_scans_from']).resolve()
  if not previous.is_relative_to(repo/'results') or not previous.name.startswith('cnn_jacobian_pilot_') or previous==out.resolve():raise ValueError('INVALID_REUSE_ROOT')
  prior=previous/'private/scans'/row['mri_sample_hash']
  if (prior/'receipt.json').exists():
   old=json.loads((prior/'receipt.json').read_text())
   if old['binding']!=bind or sha(prior/'channels.npy')!=old['output_sha256']:raise ValueError('PRIOR_PREPARED_CACHE_DRIFT')
   shutil.copy2(prior/'channels.npy',target/'channels.npy');dump(target/'receipt.json',old);return old
 import ants,nibabel as nib,nibabel.processing as nip
 refpath=repo/cfg['template_dir']/'reference_128_1p5mm.nii.gz';ref=nib.load(refpath);fixed=ants.image_read(str(refpath))
 mask=nip.resample_from_to(nib.load(repo/cfg['template_dir']/'tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz'),ref,order=0).get_fdata()>.5
 warp=ants.image_read(str(folder/cfg['jacobian']['warp']))
 if (warp.shape!=fixed.shape or not np.allclose(warp.spacing,fixed.spacing,atol=1e-5) or not np.allclose(warp.origin,fixed.origin,atol=1e-4) or not np.allclose(warp.direction,fixed.direction,atol=1e-5)):raise ValueError('WARP_REFERENCE_PHYSICAL_GEOMETRY')
 det=ants.create_jacobian_determinant_image(fixed,str(folder/cfg['jacobian']['warp']),do_log=False,geom=True).numpy()
 log,qc=checked_logjac(det,mask,cfg['jacobian']['log_clip'])
 t1=np.load(folder/'tensor.npy',allow_pickle=False)
 if t1.shape!=log.shape or not np.isfinite(t1).all():raise ValueError('T1_LOGJAC_ALIGNMENT')
 # Aligned 2x2x2 block averages preserve the fixed grid; no arbitrary array zoom.
 a=np.stack([t1,log]);a=a.reshape(2,64,2,64,2,64,2).mean((2,4,6)).astype(np.float32)
 mat=ants.read_transform(str(folder/'registration_0GenericAffine.mat')).parameters[:9].reshape(3,3)
 affine_det=float(np.linalg.det(mat))
 if not np.isfinite(affine_det) or affine_det<=0:raise ValueError('INVALID_PARENT_AFFINE')
 with (target/'channels.tmp').open('wb') as f:np.save(f,a,allow_pickle=False)
 os.replace(target/'channels.tmp',target/'channels.npy')
 d={'binding':bind,'input_sha256':inputs,'output_sha256':sha(target/'channels.npy'),'jacobian':qc,'affine_determinant_excluded_from_channel':affine_det,'shape':list(a.shape)}
 dump(target/'receipt.json',d);return d

def prepare(repo,cfg):
 repo=Path(repo);out=repo/cfg['output_root'];out.mkdir(parents=True,exist_ok=True)
 if shutil.disk_usage(out).free<cfg['min_free_disk_gib']*1024**3:raise ValueError('DISK_RESERVE')
 from scripts.xyz_fusion_v1.mri_cache import validate_reference
 reference=validate_reference(repo/cfg['template_dir'])
 source=repo/cfg['parent_root']/'private/contract.json';parent=json.loads(source.read_text())
 frame=pd.DataFrame(parent['manifest']);s=parent['parent_splits'][cfg['outer_fold_excluded']]
 splits=split_manifest(frame,s['train'],s['test'],cfg)
 code={p.relative_to(repo).as_posix():sha(p) for p in (repo/'scripts/cnn_jacobian_pilot_v1').glob('*.py')}
 code['scripts/run_cnn_jacobian_pilot_v1.py']=sha(repo/'scripts/run_cnn_jacobian_pilot_v1.py')
 contract={'config':cfg,'parent_contract_sha256':sha(source),'reference':reference,'sources':code,'splits':splits,'excluded_outer_test_digest':binding(sorted(s['test'])),'historical_outcome_exposure':True,'test_outcomes_used_for_selection':False}
 path=out/'private/contract.json'
 if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('PILOT_CONTRACT_DRIFT_REQUIRES_NEW_VERSION')
 if not path.exists():dump(path,contract)
 tests=geometry_selftest(out/'checks/synthetic_warps');dump(out/'checks/geometry.json',{'status':'PASS','cases':tests})
 keys=sum(splits.values(),[]);selected=frame.set_index('patient_hash').loc[keys].reset_index()
 receipts=[];started=time.time()
 with ProcessPoolExecutor(max_workers=cfg['preparation_workers']) as pool:
  for i,r in enumerate(pool.map(one_scan,[(r,str(repo),cfg) for r in selected.to_dict('records')]),1):
   receipts.append(r)
   if i%32==0:print(json.dumps({'stage':'jacobian_preparation','done':i,'n':len(selected),'seconds':round(time.time()-started)}),flush=True)
 selected.to_parquet(out/'private/pilot_frame.parquet',index=False)
 dump(out/'private/cache_index.json',{'patient_order':keys,'sample_keys':selected.mri_sample_hash.tolist(),'receipts':[binding(x) for x in receipts],'frame_sha256':sha(out/'private/pilot_frame.parquet')})
 counts={role:{'n':len(ids),'events_full':int(frame.loc[frame.patient_hash.isin(ids),'event'].sum()),'events_3y':int(((frame.patient_hash.isin(ids))&(frame.event==1)&(frame.time_days<=1095.75)).sum())} for role,ids in splits.items()}
 result={'status':'PILOT_PREPARATION_COMPLETE_NOT_SCIENTIFIC_VALIDATION','support':counts,'scans':len(receipts),'seconds':time.time()-started,'max_logjac_clip_fraction':max(x['jacobian']['clipped_fraction'] for x in receipts),'outer_test_n_excluded':len(s['test'])}
 dump(out/'public/preparation.json',result);print(json.dumps(result),flush=True);return result
