from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
from .runtime import configure
R, P = configure()
from scripts.xyz_fusion_v1.audit import _table,RAW
from scripts.xyz_fusion_v2.data import load_tau
from scripts.xyz_postprimary_v1.feasibility import audit_tau_landmark
def read(p):
 j=json.loads(Path(p).read_text(encoding='utf-8-sig'));return j.get('payload',j)
contract=read(R/'results/xyz_fusion_v2/private/contract.json');frame=pd.DataFrame(contract['manifest'])
config=contract['protocol'];tau,audit=load_tau(R,{'frame':frame},frame,config)
dated=pd.DataFrame(audit_tau_landmark({'frame':frame})['private_rows'])
dated=dated.loc[dated.patient_hash.isin(tau.patient_hash)].merge(frame[['patient_hash','landmark_days_t0']],validate='one_to_one')
gap=dated.landmark_days_t0-dated.ftp_day
elig=pd.read_parquet(R/'results/xyz_fusion_postall_v2/private/eligibility.parquet')
petids=set(elig.loc[(elig.track=='pet')&(elig.subset=='all'),'patient_hash'])
frame['has_tau']=frame.patient_hash.isin(tau.patient_hash);frame['has_pet']=frame.patient_hash.isin(petids)
frame['has_blood']=frame.blood_ptau217.notna();frame['followup_years']=frame.time_days/365.25
from sksurv.nonparametric import kaplan_meier_estimator
def km_quant(t,s,q):
 v=t[s<=1-q];return float(v[0]/365.25) if len(v) else None
groups={'all':frame,'event':frame[frame.event.eq(1)],'censored':frame[frame.event.eq(0)],'tau':frame[frame.has_tau],'no_tau':frame[~frame.has_tau],'pet':frame[frame.has_pet],'no_pet':frame[~frame.has_pet],'blood':frame[frame.has_blood],'no_blood':frame[~frame.has_blood]}
continuous=['age','education_years','baseline_pacc_raw','baseline_mmscore','amyloid_centiloid','followup_years']
binary=['sex_binary','apoe4_carrier','treatment_active']
summary={};curves={}
for name,t in groups.items():
 a={'n':len(t),'events':int(t.event.sum()),'continuous':{},'binary':{}}
 for v in continuous:
  s=t[v].dropna();a['continuous'][v]={'n':len(s),'mean':s.mean(),'sd':s.std(ddof=1),'median':s.median(),'q1':s.quantile(.25),'q3':s.quantile(.75)}
 for v in binary:
  s=t[v].dropna();a['binary'][v]={'n':len(s),'ones':int(s.eq(1).sum()),'proportion':s.eq(1).mean()}
 times,surv=kaplan_meier_estimator(t.event.to_numpy(bool),t.time_days.to_numpy(),reverse=True)
 a['reverse_km_median_followup_years']=km_quant(times,surv,.5)
 times,surv=kaplan_meier_estimator(t.event.to_numpy(bool),t.time_days.to_numpy())
 curves[name]={'days':times.tolist(),'survival':surv.tolist(),'at_risk':{str(y):int((t.time_days>=y*365.25).sum()) for y in range(5)}}
 a['km_event_risk_3y']=float(1-surv[max(0,np.searchsorted(times,3*365.25,side='right')-1)])
 summary[name]=a
smd={}
for mod in ['tau','pet','blood']:
 a,b=groups[mod],groups['no_'+mod];smd[mod]={}
 for v in continuous[:-1]+binary:
  x,y=a[v].dropna(),b[v].dropna();den=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
  smd[mod][v]=float((x.mean()-y.mean())/den) if den>0 else None
subj=_table(RAW/'Derived Data/SUBJINFO.csv');ds=_table(RAW/'Derived Data/DS.csv')
endpoint=pd.read_parquet(R/'results/method_y_revival_v2/endpoints/amyloid_progression_endpoint.parquet')
missing=subj.loc[subj.SUBSTUDY.eq('A4')&~subj.patient_hash.isin(endpoint.loc[endpoint.cohort.eq('A4'),'patient_hash'])]
old=read(R/'results/xyz_mri_baseline_fast_v1/private/contract.json')
intake=read(R/'results/xyz_mri_baseline_fast_v1/private/jobs/intake.json')['summary']
qc=[]
selected=set(frame.mri_sample_hash)
for row in old['manifest']:
 if row['mri_sample_hash'] not in selected:
  folder=R/'results/xyz_mri_baseline_fast_v1/private/mri_scans'/row['mri_sample_hash']
  files=list(folder.glob('*.json'))
  qc.append({'scan':row['mri_sample_hash'],'files':{f.name:read(f) for f in files}})
metadata={}
for file in ['Derived Data/SUBJINFO.csv','Derived Data/DS.csv','Derived Data/PACC.csv','External Data/imaging_MRI_reads.csv','Raw Data/psychwell.csv','Raw Data/cfi.csv','Raw Data/adlpq.csv']:
 t=_table(RAW/file);metadata[file]={'columns':list(t.columns),'n':len(t)}
 if 'DS.csv' in file:
  for c in t.columns:
   if c not in ['BID','patient_hash'] and t[c].nunique()<40:metadata[file][c]=t[c].value_counts(dropna=False).to_dict()
for dic in ['derived','external','clinical']:
 t=pd.read_csv(RAW/f'Documents/Data Dictionaries/{dic}_datadic.csv',low_memory=False)
 mask=t.astype(str).apply(lambda c:c.str.contains('PACC|FCTOTAL96|CDGLOBAL|APOE|EDCCNTU|SEX|pTau217|disposition|DEATH|GDS',case=False,regex=True)).any(axis=1)
 (P/f'{dic}_dictionary_selection.csv').write_text(t[mask].to_csv(index=False),encoding='utf-8-sig')
report={'demographics':summary,'smd':smd,'tau_chronology':{'n':len(gap),'after_origin':int((gap<0).sum()),'same_day':int(gap.eq(0).sum()),'min_days_before':gap.min(),'median_days_before':gap.median(),'max_days_before':gap.max(),'audit':audit},'attrition':{'a4_source':int(subj.SUBSTUDY.eq('A4').sum()),'outside_initial_endpoint':len(missing),'initial_endpoint':1106,'intake':intake,'qc_excluded':len(qc)},'raw_metadata':metadata,'km':curves}
def dump(name,obj): (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
dump('data_audit.json',report);dump('private_qc_exclusions.json',qc)
frame.to_parquet(P/'private_analysis_frame.parquet',index=False);tau.to_parquet(P/'private_tau_frame.parquet',index=False)
print(json.dumps({'tau_chronology':report['tau_chronology'],'attrition':report['attrition'],'groups':{k:{'n':v['n'],'events':v['events'],'reverse_km':v['reverse_km_median_followup_years']} for k,v in summary.items()}},ensure_ascii=False,default=str))
