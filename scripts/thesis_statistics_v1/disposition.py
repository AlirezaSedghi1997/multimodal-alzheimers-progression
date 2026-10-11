from pathlib import Path
import sys,json,collections
import pandas as pd,numpy as np
from .runtime import configure
R, P = configure()
from scripts.xyz_fusion_v1.audit import RAW,_table
frame=pd.read_parquet(P/'private_analysis_frame.parquet');sub=_table(RAW/'Derived Data/SUBJINFO.csv');ds=_table(RAW/'Derived Data/DS.csv')
ds=ds.merge(frame[['patient_hash','landmark_days_t0','time_days','event']],validate='many_to_one')
ds['relative_day']=pd.to_numeric(ds.DSSTDTC_DAYS_T0,errors='coerce')-ds.landmark_days_t0
window=ds[(ds.relative_day>0)&(ds.relative_day<=1640)&ds.DSCAT.eq('DISPOSITION EVENT')]
reasons=window.drop_duplicates(['patient_hash','DSDECOD']).DSDECOD.value_counts().to_dict()
death=window[window.DSDECOD.eq('DEATH')].sort_values('relative_day').drop_duplicates('patient_hash')
# No recoding of death after last CDR into independently adjudicated event-free time.
death_summary={'source':'Derived Data/DS.csv','window_days':1640,'n_deaths':len(death),'before_observed_endpoint':int((death.relative_day<=death.time_days).sum()),'among_censored':int(death.event.eq(0).sum()),'after_last_cdr_among_censored':int((death.event.eq(0)&(death.relative_day>death.time_days)).sum()),'disposition_counts_nonexclusive':reasons,'competing_risk_analysis':'NOT_PERFORMED_EVENT_FREE_INTERVAL_BETWEEN_LAST_CDR_AND_DEATH_NOT_ADJUDICATED'}
ap=sub[sub.patient_hash.isin(frame.patient_hash)]
apoe={'raw_missing':int(ap.APOEGNPRSNFLG.isna().sum()),'analysis_flag_counts':ap.AAPOEGNPRSNFLG.value_counts(dropna=False).to_dict(),'encoding':'ANALYSIS_CARRIER_FLAG_NOT_ALLELE_COUNT'}
reads=_table(RAW/'External Data/imaging_MRI_reads.csv')
qc={'source_safety_reads_people':int(reads.loc[reads.patient_hash.isin(frame.patient_hash),'patient_hash'].nunique()),'does_not_validate_processed_registration':True}
scenarios=[]
for p in [7,8,12]:
 for r2 in [.05,.10,.15,.20]:
  shrink=.9;n=int(np.ceil(p/((shrink-1)*np.log(1-r2/shrink))))
  scenarios.append({'parameters':p,'anticipated_cox_snell_r2':r2,'target_shrinkage':shrink,'n_shrinkage_criterion_only':n})
out={'disposition':death_summary,'APOE':apoe,'source_mri_safety_reads':qc,'riley_shrinkage_scenarios':scenarios,'riley_scope':'RETROSPECTIVE_SENSITIVITY_ONE_CRITERION_NOT_FULL_MINIMUM_NOT_CNN_FORMULA'}
(P/'additional_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,default=str))
