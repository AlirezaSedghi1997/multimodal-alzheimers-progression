from pathlib import Path
import os,sys,json,time,hashlib
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import numpy as np,pandas as pd,ants,nibabel as nib
from nibabel.processing import resample_from_to
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits
import analyses as a
ROOT=a.ROOT;OUT=a.OUT;source=ROOT/'results/xyz_mri_baseline_fast_v1/private/mri_scans';cache=OUT/'private/deformation_features';cache.mkdir(exist_ok=True)
reference=ROOT/'data/templates/method_y_mni152nlin2009casym_v1/reference_128_1p5mm.nii.gz';ref=nib.load(reference);target=ants.image_read(str(reference))
mask=resample_from_to(nib.load(ROOT/'data/templates/method_y_mni152nlin2009casym_v1/tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz'),ref,order=0).get_fdata()>.5
atlas=nib.load(ROOT/'results/xyz_fusion_v2/private/atlas_bridge/labels.nii.gz');assert atlas.shape==ref.shape and np.allclose(atlas.affine,ref.affine,atol=1e-4)
labels=atlas.get_fdata().astype(int);hip=np.isin(labels,[109,119])&mask;phg=(labels==34)&mask;assert hip.sum()>50 and phg.sum()>50
inp=a.Inputs(ROOT,a.read(ROOT/'configs/xyz_oof_stacking_v1.json'));cols=['def_hip_logj','def_antphg_logj','def_brain_logj_median','def_brain_logj_iqr']
def extract(row):
    cp=cache/(str(row.mri_sample_hash)+'.json')
    if cp.exists():return a.read(cp)
    folder=source/str(row.mri_sample_hash);rp=folder/'receipt.json';receipt=a.read(rp)
    assert receipt['status']=='PASS'
    name='registration_1Warp.nii.gz';path=folder/name;sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==receipt['outputs'][name]
    j=ants.create_jacobian_determinant_image(target,str(path),do_log=False).numpy();good=np.isfinite(j)&(j>0)
    fractions=[float(good[m].mean()) for m in [mask,hip,phg]]
    v={'patient_hash':row.patient_hash,'transform_sha256':sha,'positive_finite_fractions':fractions,'status':'ELIGIBLE' if min(fractions)>=.99 else 'EXCLUDED_NONPOSITIVE_DEFORMATION'}
    if v['status']=='ELIGIBLE':
        logj=np.zeros(j.shape,float);logj[good]=np.log(j[good]);b=logj[mask&good];q=np.quantile(b,[.25,.5,.75])
        v.update(dict(zip(cols,[float(logj[hip&good].mean()),float(logj[phg&good].mean()),float(q[1]),float(q[2]-q[0])])))
    a.dump(cp,v);return v
start=time.time();features=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    for i,row in enumerate(pool.map(extract,list(inp.frame.itertuples())),1):
        features.append(row)
        if i%25==0:print('DEFORMATION_EXTRACTED',i,'/',len(inp.frame),'seconds',round(time.time()-start,1),flush=True)
        if i%100==0:a.dump(OUT/'private/deformation_extraction_progress.json',{'done':i,'total':len(inp.frame),'seconds':time.time()-start})
f=inp.frame.merge(pd.DataFrame(features),on='patient_hash',validate='one_to_one');excluded=int((f.status!='ELIGIBLE').sum());f=f.loc[f.status=='ELIGIBLE'].copy().reset_index(drop=True)
f.to_parquet(OUT/'private/deformation_analysis_frame.parquet',index=False)
base=a.CLINICAL+['amyloid_centiloid'];vols=['hippocampus_icv','entorhinal_icv','ventricle_icv'];defs={'base':base,'deformation':base+cols,'volumes':base+vols,'volumes_deformation':base+vols+cols}
cells=[];choices=[]
with threadpool_limits(limits=2):
    for fold,split in enumerate(inp.splits):
        train=f.loc[f.patient_hash.isin(split['train'])].copy().reset_index(drop=True);held=f.loc[f.patient_hash.isin(split['test'])].copy().sort_values('patient_hash').reset_index(drop=True);pred={}
        for name,features in defs.items():
            alpha,ranks=a.select_alpha(train,features,inp.config,20261024+fold);model=a.fit_model(train,features,'cox_ridge',alpha,inp.config,20261024+fold)
            choices.append({'fold':fold+1,'arm':name,'alpha':alpha,'inner':ranks,'edf':a.effective_df(model,train,features,alpha)})
            x=held[features];pred[name]={'subjects':list(held.patient_hash),'survival':a.curves(model,x,features,a.GRID).tolist()}
        cells.append(a.evaluate_extended(train,held,pred,inp.config,fold));print('DEFORMATION_FOLD',fold+1,flush=True)
    results={x+'__'+y:a.paired(cells,x,y,2,20261024) for x,y in [('base','deformation'),('volumes','volumes_deformation')]}
a.dump(OUT/'private/deformation_cells.json',cells)
a.dump(OUT/'aggregate/A09_deformation_control.json',{'status':'EXECUTED_POST_HOC_DEFORMATION_AWARE_CONTROL','n':len(f),'excluded':excluded,'events':int(f.event.sum()),'features':cols,'family_per_metric':2,'results':results,'choices':choices,'extraction_seconds':time.time()-start,'atlas_sha256':hashlib.sha256((ROOT/'results/xyz_fusion_v2/private/atlas_bridge/labels.nii.gz').read_bytes()).hexdigest(),'limitation':'Fixed summary features are not a full Jacobian CNN channel or pretrained representation; no causal account of the original imaging result is established.'})
print('DEFORMATION_COMPLETE',len(f),round(time.time()-start,1),'seconds',flush=True)
