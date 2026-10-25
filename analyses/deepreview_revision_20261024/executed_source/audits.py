from pathlib import Path
import sys,os,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
ROOT=Path(r'H:\New project');OUT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np
from scripts.xyz_fusion_v1.audit import _table,RAW
from scripts.xyz_oof_stacking_v1.inputs import Inputs
from scripts.xyz_fusion_v1.store import digest
def read(p):
    x=json.loads(Path(p).read_text('utf-8-sig'));return x.get('payload',x) if isinstance(x,dict) else x
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if hasattr(v,'item') else str(v)),'utf-8')
inp=Inputs(ROOT,read(ROOT/'configs/xyz_oof_stacking_v1.json'));base=inp.frame
result={'status':'EXECUTED_SOURCE_AND_TIMING_AUDIT','source_hashes':{}}
paths={'pacc':RAW/'Derived Data/PACC.csv','visits':RAW/'Derived Data/SV.csv','cdr':RAW/'Raw Data/cdr.csv','tau':RAW/'External Data/imaging_SUVR_tau.csv'}
for k,p in paths.items():result['source_hashes'][k]=hashlib.sha256(p.read_bytes()).hexdigest()
pacc=_table(paths['pacc']);visits=_table(paths['visits']);pacc=pacc.loc[pacc.patient_hash.isin(base.patient_hash)].copy();visits=visits.loc[visits.patient_hash.isin(base.patient_hash)].copy()
visits['day']=pd.to_numeric(visits.SVSTDTC_DAYS_T0,errors='coerce');dates=visits.groupby(['patient_hash','VISITCD']).day.agg(['min','max'])
exact=dates.loc[dates['min'].eq(dates['max']),'min'].rename('day').reset_index().rename(columns={'VISITCD':'VISCODE'})
t=pacc.merge(exact,on=['patient_hash','VISCODE'],validate='many_to_one').merge(base[['patient_hash','landmark_days_t0']],on='patient_hash',validate='many_to_one')
t['years']=(t.day-t.landmark_days_t0)/365.25;t['pacc']=pd.to_numeric(t['PACC.raw'],errors='coerce')
window=t.loc[t.years.gt(0)&t.years.le(4.5)].dropna(subset=['pacc','years']).drop_duplicates(['patient_hash','day'])
counts=window.groupby('patient_hash').day.nunique().reindex(base.patient_hash,fill_value=0)
saved=pd.read_parquet(ROOT/'results/xyz_fusion_postall_v2/private/pacc_longitudinal.parquet')
assert set(saved.patient_hash)==set(counts[counts>=2].index)
result['pacc']={'cohort_n':len(base),'participants_no_eligible_postlandmark_scores':int((counts==0).sum()),'participants_one_eligible_postlandmark_score':int((counts==1).sum()),'participants_at_least_two_scores':int((counts>=2).sum()),'observations':len(saved),'saved_analysis_membership_reconciled':True,'eligibility':'At least two nonconflicting dated post-landmark PACC observations in (0,4.5] years, with complete model variables.'}
cdr=_table(paths['cdr']);cdr=cdr.loc[cdr.patient_hash.isin(base.patient_hash)].copy();cdr['day']=pd.to_numeric(cdr.CDADTC_DAYS_T0,errors='coerce');cdr['score']=pd.to_numeric(cdr.CDGLOBAL,errors='coerce')
cdr=cdr.dropna(subset=['day','score']);conflicts=cdr.groupby(['patient_hash','day']).score.nunique();bad=set(conflicts[conflicts>1].index)
cdr=cdr.loc[[(r.patient_hash,r.day) not in bad for r in cdr.itertuples()]].drop_duplicates(['patient_hash','day']).merge(base[['patient_hash','landmark_days_t0']],on='patient_hash',validate='many_to_one')
cdr['relative_day']=cdr.day-cdr.landmark_days_t0;cdr=cdr.loc[cdr.relative_day.ge(0)&cdr.relative_day.le(1640)].sort_values(['patient_hash','day'])
intervals=cdr.groupby('patient_hash').day.diff().dropna();confirm_intervals=[]
for key,g in cdr.groupby('patient_hash'):
    a=g[['relative_day','score']].to_numpy()
    for j in range(1,len(a)):
        if a[j-1,1]>=.5 and a[j,1]>=.5:
            confirm_intervals.append(a[j,0]-a[j-1,0]);break
result['cdr_intervals']={'distinct_day_intervals':len(intervals),'median_days':float(intervals.median()),'q25_q75_days':intervals.quantile([.25,.75]).tolist(),'confirmation_pairs_within_window':len(confirm_intervals),'confirmation_pair_median_days':float(np.median(confirm_intervals)),'interpretation':'Observed visit-schedule lag, not a mathematically fixed delay for every subject; endpoint date remains the second qualifying assessment.'}
tau_frame=pd.read_parquet(ROOT/'thesis-workspace/supervisor_revision_20261011/private_tau_frame.parquet')
print('TAU_FRAME_COLUMNS',list(tau_frame.columns),flush=True)
tau=_table(paths['tau']);print('TAU_SOURCE_COLUMNS',list(tau.columns),flush=True)
result['tau_frame_metadata_columns']=[c for c in tau_frame if any(x in c.lower() for x in ['day','date','tau'])]
# Publish only aggregate cohort/pool assertions. Membership checksums remain private.
summary=[];private=[]
for source in ['mri','pet']:
    for seed in [13,29,47]:
        for fold in range(5):
            for comp in range(3):
                task=f'{source}_s{seed}_o{fold}_c{comp}';lock=read(ROOT/f'results/xyz_fusion_v2/private/jobs/{task}_lock.json')
                row={'modality':source,'seed':seed,'outer_fold':fold+1,'component':comp+1,'counts':{}}
                for role in ['expert','meta','outer']:
                    keys=lock[role+'_subjects'];f=base.loc[base.patient_hash.isin(keys)];assert len(keys)==len(f)
                    row['counts'][role]={'n':len(f),'events':int(f.event.sum()),'events3y':int((f.event.astype(bool)&f.time_days.le(1095.75)).sum())}
                    private.append({'task':task,'role':role,'membership_sha256':digest(sorted(keys))})
                summary.append(row)
result['partition_assertions']={'rows':len(summary),'no_public_membership_or_weights':True,'summary_file':'A08_partition_assertions.json','hash_policy':'Participant-membership digests remain private; data-use approval is needed before sharing even hashed membership.'}
dump(OUT/'aggregate/A08_partition_assertions.json',summary);dump(OUT/'private/partition_membership_hashes.json',private)
dump(OUT/'aggregate/A06_timing_and_denominators.json',result)
print('AUDIT_RESULT',result,flush=True)
