"""Additional descriptive planning calculations; no source run is modified."""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd

from .runtime import configure
R, P = configure()
frame = pd.read_parquet(P / 'private_analysis_frame.parquet')
mean_followup = float(frame.time_days.mean() / 365.25)
rate = float(frame.event.sum() / (frame.time_days.sum() / 365.25))
from .planning import riley_scenarios
result = riley_scenarios(rate, mean_followup)
(P/'riley_full_scenarios.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))

# Additional analysis populations are described separately in the appendix.
pacc=pd.read_parquet(R/'results/xyz_fusion_postall_v2/private/pacc_longitudinal.parquet')
ratios=['hippocampus_icv','entorhinal_icv','ventricle_icv']
valid=np.isfinite(frame[ratios]).all(axis=1)&frame[ratios].gt(0).all(axis=1)&frame[ratios].lt(1).all(axis=1)
volume=frame.loc[valid]
assert len(volume)==1053 and int(volume.event.sum())==237
pacc_ids=set(pacc.patient_hash)
volume_ids=set(volume.patient_hash)
elig=pd.read_parquet(R/'results/xyz_fusion_postall_v2/private/eligibility.parquet')
four=set(elig.loc[elig.subset.eq('four_way'),'patient_hash'])
groups={'volumes':volume,'no_volumes':frame.loc[~valid],
        'pacc':frame.loc[frame.patient_hash.isin(pacc_ids)],
        'no_pacc':frame.loc[~frame.patient_hash.isin(pacc_ids)],
        'regional_pacc':frame.loc[frame.patient_hash.isin(pacc_ids & volume_ids)],
        'four_way':frame.loc[frame.patient_hash.isin(four)]}
assert len(groups['pacc'])==1049 and len(groups['regional_pacc'])==1043
out={}
for name,t in groups.items():
    a={'n':len(t),'events':int(t.event.sum()),'continuous':{},'binary':{}}
    for v in ['age','education_years','baseline_pacc_raw','baseline_mmscore','amyloid_centiloid','followup_years']:
        s=t[v].dropna();a['continuous'][v]={'n':len(s),'mean':s.mean(),'sd':s.std(ddof=1),'median':s.median(),'q1':s.quantile(.25),'q3':s.quantile(.75)}
    for v in ['sex_binary','apoe4_carrier','treatment_active']:
        s=t[v].dropna();a['binary'][v]={'n':len(s),'ones':int(s.eq(1).sum()),'proportion':s.eq(1).mean()}
    out[name]=a
(P/'extra_demographics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=lambda x:x.item()),encoding='utf8')
print('Additional groups:',{k:(v['n'],v['events']) for k,v in out.items()})

# Effective information in each of the 45 tau fusion-head fits.
tau=pd.read_parquet(P/'private_tau_frame.parquet')
tau_ids=set(tau.patient_hash)
counts=[]
contract=json.loads((R/'results/xyz_fusion_v2/private/contract.json').read_text(encoding='utf-8-sig'))
contract=contract.get('payload',contract)
for seed in contract['protocol']['training_seeds']:
    for fold in range(5):
        for component in range(3):
            path=R/'results/xyz_fusion_v2/private/jobs'/f'mri_s{seed}_o{fold}_c{component}_lock.json'
            j=json.loads(path.read_text(encoding='utf-8-sig'));j=j.get('payload',j)
            selected=set(j['meta_subjects']) & tau_ids
            t=tau[tau.patient_hash.isin(selected)]
            counts.append((len(t),int(t.event.sum())))
values=np.asarray(counts)
out={'n_fits':len(counts),'min_n':int(values[:,0].min()),'max_n':int(values[:,0].max()),
     'min_events':int(values[:,1].min()),'max_events':int(values[:,1].max()),
     'median':np.median(values,axis=0).tolist(),
     'range_events_per_12_parameters':[float(values[:,1].min()/12),float(values[:,1].max()/12)]}
(P/'tau_training_support.json').write_text(json.dumps(out),encoding='utf8')
