"""Execution of the supplied image-training implementation on synthetic data."""
from pathlib import Path
import sys,os,json,hashlib,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
OUT=Path(__file__).parent
SOURCE=Path(r'C:\Users\USER\Desktop\draft\GeroScience_Submission_Package_v7_20261014\SUPPLEMENTARY_FOR_UPLOAD\ESM_3\source_snapshot')
sys.path.insert(0,str(SOURCE))
import numpy as np,pandas as pd,torch
from scripts.xyz_fusion_v2.training import MRISurvival,train_expert,predict,deterministic
from scripts.xyz_fusion_v1.store import digest
cfg=json.loads(Path(r'H:\New project\configs\xyz_fusion_v2.json').read_text('utf-8-sig'))
policy=dict(cfg['training'],amp=False,tf32=False,modality='mri',epochs=1,minimum_epochs=1,batch_size=2,microbatch_size=1,cpu_threads=2,checkpoint_every_optimizer_steps=1)
deterministic(20261024,2);rng=np.random.default_rng(20261024)
class Bank:
    def __init__(self):self.a=rng.normal(0,1,(12,1,128,128,128)).astype('float32')
    def batch(self,rows,device):return torch.from_numpy(self.a[np.asarray(rows,int)]).to(device)
bank=Bank();train=pd.DataFrame({'patient_hash':[f'SYNTHETIC_{i}' for i in range(8)],'cache_row':range(8),'event':[1,0,1,0,1,0,1,0],'time_days':[300.,600.,800.,1000.,1100.,1300.,1500.,1640.]})
val=pd.DataFrame({'patient_hash':[f'SYNTHETIC_{i}' for i in range(8,12)],'cache_row':range(8,12),'event':[0,1,0,1],'time_days':[200.,800.,1200.,1600.]})
grid=np.asarray(cfg['fusion']['ibs_grid_days'],float);start=time.time()
model,receipt=train_expert(bank,train,val,policy,grid,OUT/'private/synthetic_image_training',digest({'purpose':'SYNTHETIC_CODE_EXECUTION_NOT_A4','source':str(SOURCE)}),1e-4,20261024)
device=next(model.parameters()).device;p=predict(model,bank,val,policy,device,grid)
checkpoint=OUT/'private/synthetic_image_training/locked.pt';loaded=MRISurvival(policy).to(device)
loaded.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['model'],strict=True)
q=predict(loaded,bank,val,policy,device,grid)
assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and (np.diff(p,axis=1)<=1e-6).all();assert np.max(np.abs(p-q))<1e-7
result={'status':'SYNTHETIC_TRAIN_CHECKPOINT_RELOAD_AND_INFERENCE_EXECUTED','training_subjects':8,'validation_subjects':4,'volume_geometry':[1,128,128,128],'epochs':1,'device':str(device),'torch':torch.__version__,'source_training_sha256':hashlib.sha256((SOURCE/'scripts/xyz_fusion_v2/training.py').read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'reload_max_abs_error':float(np.max(np.abs(p-q))),'seconds':time.time()-start,'history':receipt['history'],'scope':'Synthetic execution of the supplied 3D trainer, validation, saved weights and inference. Not real-data training reproduction, external validation, or a new scientific performance result.'}
(OUT/'aggregate/A08_synthetic_training.json').write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8');print(json.dumps(result,indent=2),flush=True)
