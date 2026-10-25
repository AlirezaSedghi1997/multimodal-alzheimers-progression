import analyses as a
from pathlib import Path
import numpy as np,pickle,hashlib
from threadpoolctl import threadpool_limits
from scripts.xyz_fusion_v2.models import score_features
inp=a.Inputs(a.ROOT,a.read(a.ROOT/'configs/xyz_oof_stacking_v1.json'));records=[];hashes={}
def load_model(root,receipt):
    p=(root/receipt['path']).resolve();assert p.is_relative_to(root.resolve())
    sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==receipt['sha256'];hashes[str(p)]=sha
    with p.open('rb') as stream:return pickle.load(stream)
with threadpool_limits(limits=2):
    for fold in range(5):
        for seed in [13,29,47]:
            for comp in range(3):
                task=f'mri_s{seed}_o{fold}_c{comp}';lock=a.read(a.ROOT/f'results/xyz_fusion_v2/private/jobs/{task}_lock.json')
                pred=a.read(a.ROOT/f'results/xyz_fusion_v2/private/jobs/{task}_mri_meta.json')
                assert pred['model_sha256']==lock['experts']['mri']['sha256']
                f=inp.frame.set_index('patient_hash').loc[pred['subjects']].reset_index();f=score_features(f,np.asarray(pred['survival']),a.GRID,inp.config['fusion']['score_times_days'],'mri')
                for arm in ['clinical_amyloid','clinical_mri_amyloid']:
                    receipt=lock['heads'][arm];model=load_model(a.ROOT/'results/xyz_fusion_v2',receipt)
                    rows=f.loc[f.patient_hash.isin(receipt['ancestry'])];assert len(rows)==len(receipt['ancestry'])
                    records.append({'workflow':'original_component','track':'mri','fold':fold+1,'seed':seed,'component':comp+1,'arm':arm,'n':len(rows),'p':len(receipt['features']),'alpha':receipt['parameter'],'effective_df':a.effective_df(model,rows,receipt['features'],receipt['parameter'])})
                ft=f.merge(inp.frames['tau'][['patient_hash','tau_temporal']],on='patient_hash',validate='one_to_one')
                for arm in ['base','plus_tau']:
                    receipt=a.read(a.ROOT/f'results/xyz_fusion_postall_v2/private/units/tau_{task}_{arm}.json');model=load_model(a.ROOT/'results/xyz_fusion_postall_v2',receipt)
                    assert set(ft.patient_hash)==set(receipt['ancestry'])
                    records.append({'workflow':'original_component','track':'tau','fold':fold+1,'seed':seed,'component':comp+1,'arm':arm,'n':len(ft),'p':len(receipt['features']),'alpha':receipt['parameter'],'effective_df':a.effective_df(model,ft,receipt['features'],receipt['parameter'])})
            for track in ['mri','pet','blood','tau']:
                frame,tasks=inp.frames_for_head(track,seed,fold)
                for arm in a.arms(track):
                    receipt=a.read(a.ROOT/f'results/xyz_oof_stacking_v1/private/units/{track}_s{seed}_o{fold}_pooled_{arm}.json');model=load_model(a.ROOT/'results/xyz_oof_stacking_v1',receipt)
                    assert set(frame.patient_hash)==set(receipt['ancestry'])
                    records.append({'workflow':'original_raw_pooled','track':track,'fold':fold+1,'seed':seed,'arm':arm,'n':len(frame),'p':len(receipt['features']),'alpha':receipt['parameter'],'effective_df':a.effective_df(model,frame,receipt['features'],receipt['parameter'])})
        print('ORIGINAL_EDF_FOLD',fold+1,flush=True)
a.dump(a.PUB/'A05_original_fitted_head_edf.json',{'status':'EXECUTED_ON_HASH_VERIFIED_ORIGINAL_FITTED_HEADS','head_count':len(records),'records':records})
a.dump(a.PRIV/'original_fitted_head_model_hashes.json',hashes)
print('ORIGINAL_EDF_COMPLETE',len(records),flush=True)
