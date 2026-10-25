from pathlib import Path
import sys,os,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import analyses as a
import pandas as pd,numpy as np,nibabel as nib,h5py
from sksurv.linear_model.coxph import CoxPHOptimizer
from scripts.xyz_fusion_v1.statistics import ibs_score
inp=a.Inputs(a.ROOT,a.read(a.ROOT/'configs/xyz_oof_stacking_v1.json'))
out=a.OUT
timing=a.read(out/'aggregate/A06_timing_and_denominators.json')
e=pd.read_parquet(a.ROOT/'results/xyz_fusion_postall_v2/private/eligibility.parquet')[['patient_hash','landmark_days_t0','ftp_day']].dropna().drop_duplicates('patient_hash')
lead=e.landmark_days_t0-e.ftp_day;assert len(e)==358 and lead.ge(0).all() and lead.le(183).all()
timing['tau_timing']={'n':len(e),'same_day':int(lead.eq(0).sum()),'earlier':int(lead.gt(0).sum()),'lead_days_median':float(lead.median()),'lead_days_min':float(lead.min()),'lead_days_max':float(lead.max()),'all_within_183_days':True,'source':'Previously locked eligibility records, reconciled with the dated source-selection code.'}
a.dump(out/'aggregate/A06_timing_and_denominators.json',timing)
membership=pd.read_parquet(a.ROOT/'results/method_y_revival_v3_image_v2/private/amyloid_image_membership.parquet')
mapping=pd.read_csv(a.ROOT/'data/preprocessed_pet/_private/manifests/fbp_private_manifest.csv')
kept=membership.loc[membership.patient_hash.isin(inp.frames['pet'].patient_hash)]
headers=[];cache_attrs=[]
for row in kept.head(8).itertuples():
    m=mapping.loc[mapping.sample_hash.eq(row.image_key)];assert len(m)==1
    source=Path(m.iloc[0].raw_path_private);im=nib.load(source)
    headers.append({'shape':list(im.shape),'axis_codes':list(nib.aff2axcodes(im.affine)),'xyzt_units':list(im.header.get_xyzt_units()),'descrip':bytes(im.header['descrip']).rstrip(b'\x00').decode('utf-8','replace'),'spatial_sform_code':int(im.header['sform_code']),'spatial_qform_code':int(im.header['qform_code'])})
    with h5py.File(str(row.image_path_private),'r') as handle:
        g=handle[str(row.h5_group)];cache_attrs.append({'root_attribute_names':sorted(handle.attrs),'group_attribute_names':sorted(g.attrs),'shape':list(g['image'].shape)})
maskkeys=[c for c in mapping if any(x in c.lower() for x in ['mask','reference_region','suvr_normalization','transform_chain','source_space'])]
report={'status':'EXECUTED_BOUNDED_PET_PROVENANCE_AUDIT','retained_subset_n':len(kept),'manifest_rows':len(mapping),'manifest_columns':list(mapping),'explicit_mask_reference_or_chain_columns':maskkeys,'header_sample_n':len(headers),'sampled_headers':headers,'sampled_cache_attribute_schemas':cache_attrs,
    'voxel_intensity_units':'NOT_ESTABLISHED_AS_CALIBRATED_SUVR_OR_RAW_ACTIVITY_BY_THESE_RECORDS','brain_mask_source_space':'NOT_ESTABLISHED_BY_THE_SUPPLIED_PET_MANIFEST_OR_CACHE_ATTRIBUTE_SCHEMA','scope':'Eight retained-scan headers were inspected, plus complete manifest/cache schema. This is not an exhaustive source-space investigation of 288 excluded scans.',
    'decision':'No automatic background-NaN replacement or source-brain-mask substitution. A template-space T1 mask is not itself proof of a PET-source-space mask. Retain exclusions and explicitly limit selected-subset and uncalibrated-pattern conclusions. Full recovery requires documented source PET-to-T1 geometry, brain coverage and quantitative normalization, followed by a new versioned image analysis.'}
a.dump(out/'aggregate/A07_pet_provenance_audit.json',report)
# Independent numerical checks on the new implementation against the actual library.
meta=inp.feature_frame('mri_s13_o0_c0','meta');features=a.CLINICAL+['amyloid_centiloid']+a.MRI
model=a.fit_model(meta,features,'cox_ridge',10.,inp.config,13);x=model[:-1].transform(meta[features]);opt=CoxPHOptimizer(x,meta.event.to_numpy(bool),meta.time_days.to_numpy(float),np.full(x.shape[1],10.),'breslow');opt.update(model[-1].coef_)
pen_h=opt.hessian*len(x);info=pen_h-10*np.eye(x.shape[1]);expected=float(np.trace(np.linalg.solve(pen_h,info)));actual=a.effective_df(model,meta,features,10.)
assert abs(expected-actual)<1e-12
train=inp.split_frame('mri',0,'train');held=inp.split_frame('mri',0,'outer');defs=a.CLINICAL+['amyloid_centiloid'];mod=a.fit_model(train,defs,'cox_ridge',10.,inp.config,13);p=a.curves(mod,held,defs,a.GRID)
explicit=float(a.brier_losses(train,held,p,a.GRID)[0].mean());library=float(ibs_score(train,held,p,a.GRID));assert abs(explicit-library)<1e-12
metric=a.old.validate_concordance();zero_event_probability=(66/70)**70
verification={'status':'NUMERICAL_IMPLEMENTATIONS_RECONCILED','edf_explicit':actual,'edf_library_hessian':expected,'edf_absolute_error':abs(expected-actual),'ibs_explicit':explicit,'ibs_library':library,'ibs_absolute_error':abs(explicit-library),'concordance_duplicate_resample_check':metric,
    'zero_event_probability_70_draws_4_events':zero_event_probability,'zero_event_probability_percent':100*zero_event_probability,'reviewer_probability_correction':'(66/70)^70 is about 1.63 percent, not 1.55 percent. Observed omission remains 639/40000=1.5975 percent.',
    'penalty_objective':'Library minimizes [-partial_loglik + alpha*||beta||^2/2]/n. Multiplying by n gives the unnormalized partial likelihood penalty alpha/2; alpha=10 does not keep effective shrinkage constant across sample sizes.',
    'installed_cox_source_sha256':hashlib.sha256((out/'coxph_installed_source.txt').read_bytes()).hexdigest()}
a.dump(out/'aggregate/NUMERICAL_VERIFICATION.json',verification)
print(json.dumps({'tau_timing':timing['tau_timing'],'PET':{k:report[k] for k in ['status','retained_subset_n','header_sample_n','voxel_intensity_units','brain_mask_source_space']},'verification':verification},ensure_ascii=False,indent=2),flush=True)
