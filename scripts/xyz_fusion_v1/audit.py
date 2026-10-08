"""Read historical sources; write only a separate, private fusion intake report."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.utils.privacy import stable_hash
from .intake import select_blood, sustained_endpoint

XROOT = 'results/final_results/method_x_science_extension_v1_20260927'
YROOT = 'results/method_y_revival_v4_image_factorial_v1'
RAW = Path('H:/A4 Learn/Clinical')


def _table(path):
    t = pd.read_csv(path, dtype={'BID':str, 'SUBSTUDY':str}, low_memory=False)
    t['patient_hash'] = [stable_hash(str(a).strip().upper()+'|'+str(b).strip(), prefix='pt')
                         for a,b in zip(t.SUBSTUDY,t.BID)]
    return t


def build_intake(root, *, mri_override=None):
    root = Path(root)
    source_paths = {
        'subject': RAW/'Derived Data/SUBJINFO.csv', 'cdr':RAW/'Raw Data/cdr.csv',
        'pacc':RAW/'Derived Data/PACC.csv', 'visits':RAW/'Derived Data/SV.csv',
        'blood':RAW/'External Data/biomarker_pTau217.csv',
        'fbp':RAW/'External Data/imaging_SUVR_amyloid.csv',
        'ftp':RAW/'External Data/imaging_SUVR_tau.csv',
        'volumetric_mri':RAW/'External Data/imaging_volumetric_mri.csv',
        'cache_index':root/'docs/fasttrack_private/cache_full_index.csv',
        'amyloid_endpoint':root/'results/method_y_revival_v2/endpoints/amyloid_progression_endpoint.parquet',
        'amyloid_membership':root/'results/method_y_revival_v3_image_v2/private/amyloid_image_membership.parquet',
        'tau_membership':root/'results/method_y_revival_v3_image_v2/private/tau_image_membership.parquet',
        **{f'x_test_{f}':root/f'splits/fold_{f}_test.csv' for f in range(5)},
        **{f'x_train_{f}':root/f'splits/fold_{f}_train.csv' for f in range(5)},
        **{f'{b}_splits':root/f'results/method_y_revival_v2/splits/{b}_nested_splits.json'
           for b in ('amyloid','tau')},
    }
    src = {k:_table(source_paths[k]) for k in ('subject','cdr','pacc','visits','blood','fbp','ftp')}
    subjects = src['subject'].set_index('patient_hash',verify_integrity=True)
    endpoint = pd.read_parquet(source_paths['amyloid_endpoint'])
    population = set(endpoint.loc[endpoint.cohort.eq('A4'),'patient_hash'])
    if mri_override is None:
        real = pd.read_csv(source_paths['cache_index'])
        real = real.loc[real.is_synthetic.astype(str).str.lower().isin(['0','false'])]
        if real.sample_hash.duplicated().any():
            raise ValueError('DUPLICATE_MRI_CACHE_SAMPLE')
        mri = pd.concat([pd.read_csv(source_paths[f'x_test_{f}']).assign(x_fold=f) for f in range(5)]).merge(
            real[['sample_hash','row']],on='sample_hash',how='inner',validate='many_to_one')
    else:
        mri=mri_override.copy()
        # The fresh baseline route must not depend on changing legacy X splits/cache.
        source_paths={k:v for k,v in source_paths.items()
                      if k!='cache_index' and not k.startswith(('x_test_','x_train_')) and not k.endswith('_splits')}
    if mri.sample_hash.duplicated().any():
        raise ValueError('MRI_SPLIT_SAMPLE_OVERLAP')
    blood = src['blood'].rename(columns={'COLLECTION_DATE_DAYS_T0':'day','ORRES':'value',
        'METHOD':'assay','ORRESU':'units','SPEC':'specimen','SAMPLE_ID':'sample'})
    for c in ('day','value'):
        blood[c]=pd.to_numeric(blood[c],errors='coerce')
    # No independent numerator/denominator fallback: only pTau217 is primary Z.
    blood = blood.loc[blood.TESTCD.eq('PTAU217')]
    fbp = src['fbp'].loc[src['fbp'].ligand.str.casefold().eq('florbetapir') &
                         src['fbp'].scan_analyzed.str.casefold().eq('yes')].copy()
    fbp['scan_date_DAYS_T0']=pd.to_numeric(fbp.scan_date_DAYS_T0,errors='coerce')
    fbp = fbp.dropna(subset=['scan_date_DAYS_T0']).drop_duplicates(
        ['patient_hash','VISCODE','scan_number','scan_date_DAYS_T0'])
    fbp = fbp.sort_values(['scan_date_DAYS_T0','VISCODE','scan_number']).groupby('patient_hash').first()
    visits = src['visits'].copy()
    visits['day']=pd.to_numeric(visits.SVSTDTC_DAYS_T0,errors='coerce')
    visits=visits.dropna(subset=['day'])
    vg=visits.groupby(['patient_hash','VISITCD']).day.agg(['min','max'])
    exact=vg.loc[vg['min']==vg['max'],'min'].rename('day').reset_index().rename(columns={'VISITCD':'VISCODE'})
    pacc=src['pacc'].merge(exact,on=['patient_hash','VISCODE'],validate='many_to_one')
    cdr=src['cdr'].copy()
    cdr['day']=pd.to_numeric(cdr.CDADTC_DAYS_T0,errors='coerce')
    cdr['cdr']=pd.to_numeric(cdr.CDGLOBAL,errors='coerce')
    groups = {name:{str(k):v for k,v in tab.groupby('patient_hash')}
              for name,tab in [('mri',mri),('blood',blood),('cdr',cdr),('pacc',pacc)]}
    ym={b:pd.read_parquet(source_paths[b+'_membership']).set_index('patient_hash',verify_integrity=True)
        for b in ('amyloid','tau')}
    exclusions=Counter(); optional_exclusions=Counter(); rows=[]; private_lineage=[]
    for key in sorted(population):
        try:
            if key not in fbp.index or key not in groups['mri']:
                raise ValueError('MISSING_EXACT_MRI_OR_FBP')
            scan=fbp.loc[key]; petday=float(scan.scan_date_DAYS_T0)
            candidates=groups['mri'][key].copy()
            candidates['gap']=(pd.to_numeric(candidates.t1_day_t0,errors='coerce')-petday).abs()
            candidates=candidates.loc[candidates.gap<=183].sort_values(['gap','t1_day_t0'])
            if candidates.empty:
                raise ValueError('MRI_PET_GAP_OVER_183')
            best=candidates.iloc[0]
            if len(candidates.loc[(candidates.gap==best.gap)&(candidates.t1_day_t0==best.t1_day_t0)])!=1:
                raise ValueError('AMBIGUOUS_MRI_SCAN')
            person=subjects.loc[key]
            random_day=float(person.RNDMDT_DAYS_T0)
            if not np.isfinite(random_day):
                raise ValueError('RANDOMIZATION_TIME_MISSING')
            anchor=max(petday,float(best.t1_day_t0),random_day)
            cs=groups['cdr'].get(key,pd.DataFrame(columns=['day','cdr'])).dropna(subset=['day','cdr'])
            near=cs.assign(gap=(cs.day-anchor).abs()).sort_values(['gap','day'])
            if near.empty or near.iloc[0].gap>183 or near.iloc[0].cdr!=0:
                raise ValueError('NO_CDR_ZERO_BASELINE_WITHIN_183')
            origin=max(anchor,float(near.iloc[0].day))
            if origin-min(petday,float(best.t1_day_t0))>183:
                raise ValueError('PREDICTOR_TOO_OLD_AT_LANDMARK')
            event,time=sustained_endpoint(list(zip(cs.day,cs.cdr)),origin,1640.)
            pc=groups['pacc'].get(key,pd.DataFrame(columns=['day']))
            pc=pc.loc[(pc.day<=origin)&(pc.day>=origin-183)].sort_values('day')
            if pc.empty:
                raise ValueError('NO_DATED_PRE_LANDMARK_COGNITION')
            pc=pc.loc[pc.day==pc.day.max()]
            if len(pc.drop_duplicates(['day','PACC.raw','MMSCORE']))!=1:
                raise ValueError('AMBIGUOUS_COGNITION')
            pc=pc.iloc[0]
            try:
                bl=select_blood(groups['blood'].get(key,pd.DataFrame()),origin,183)
            except ValueError as exc:
                optional_exclusions[str(exc)]+=1
                bl=None  # Z absence must not remove a valid MRI/PET comparator subject.
            sex=str(person.SEX).strip()
            sex_value=1. if sex in ('2','2.0','M','Male','MALE') else (0. if sex in ('1','1.0','F','Female','FEMALE') else None)
            tx=str(person.TX).upper()
            if 'SOLANEZUMAB' not in tx and 'PLACEBO' not in tx:
                raise ValueError('UNKNOWN_TREATMENT_NOT_IMPUTED')
            cent=float(person.AMYLCENT)
            if not np.isfinite(cent):
                raise ValueError('MISSING_BASELINE_CENTILOID')
            row={'patient_hash':key,'landmark_days_t0':origin,'mri_day':float(best.t1_day_t0),
                'fbp_day':petday,'cognition_day':float(pc.day),'event':event,'time_days':time,
                'mri_sample_hash':best.sample_hash,'mri_cache_row':int(best.row),'x_fold':int(best.x_fold),
                'age':float(person.AGEYR),'sex_binary':sex_value,
                'education_years':float(person.EDCCNTU),'apoe4_carrier':float(person.AAPOEGNPRSNFLG),
                'baseline_pacc_raw':float(pc['PACC.raw']),'baseline_mmscore':float(pc.MMSCORE),
                'treatment_active':float('SOLANEZUMAB' in tx),
                'amyloid_centiloid':cent,'blood_ptau217':None if bl is None else float(bl['value']),
                'blood_day':None if bl is None else float(bl['day']),
                'fbp_image_eligible':key in ym['amyloid'].index,
                'ftp_image_available_not_yet_chronology_validated':key in ym['tau'].index}
            if key in ym['amyloid'].index:
                row['fbp_image_key']=ym['amyloid'].loc[key,'image_key']
                row['manufacturer']=ym['amyloid'].loc[key,'manufacturer']
            rows.append(row)
            private_lineage.append({'patient_hash':key,'blood':bl,'fbp_visit':float(scan.VISCODE),
                'fbp_scan_number':str(scan.scan_number),'pacc_visit':float(pc.VISCODE)})
        except (ValueError,TypeError,KeyError) as exc:
            exclusions[str(exc)[:100]]+=1
    frame=pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('EMPTY_COMMON_LANDMARK_POPULATION:'+str(dict(exclusions)))
    summary={'source_a4_subjects':len(population),'common_landmark_subjects':len(frame),
        'events':int(frame.event.sum()),'blood_subjects':int(frame.blood_ptau217.notna().sum()),
        'blood_events':int(frame.loc[frame.blood_ptau217.notna(),'event'].sum()),
        'fbp_image_subjects':int(frame.fbp_image_eligible.sum()),'exclusions':dict(exclusions),
        'optional_blood_exclusions':dict(optional_exclusions),
        'endpoint':'SUSTAINED_CDR_GE_0P5_NEXT_DISTINCT_VISIT',
        'no_external_data':True,'legacy_models_not_yet_reuse_qualified':True,
        'status':'INTAKE_AUDIT_ONLY_NOT_TRAINING_READY'}
    return frame,private_lineage,summary,source_paths


def verify_mri_index_metadata(root):
    """Bind row selection to the actual cached keys, not only a CSV index."""
    import h5py
    root=Path(root);index=pd.read_csv(root/'docs/fasttrack_private/cache_full_index.csv')
    selected=index.loc[index.is_synthetic.astype(str).str.lower().isin(['0','false'])]
    def decode(v):
        return v.decode() if isinstance(v,bytes) else str(v)
    with h5py.File(root/'data/cached/t1_volume_cache_128.h5','r') as f:
        names=[decode(v) for v in f['sample_hash'][:]]
        for row in selected.itertuples():
            if names[int(row.row)]!=str(row.sample_hash):
                raise ValueError('MRI_ACTUAL_CACHE_INDEX_DRIFT')
        return {'real_rows_checked':len(selected),'cached_rows':len(names),
            'status':'CACHE_KEY_METADATA_VERIFIED_NOT_VOXEL_HASH_VERIFICATION'}


def audit_fold_compatibility(root):
    """A nonzero overlap proves reuse invalid; zero alone does not prove safety."""
    root=Path(root)
    real=pd.read_csv(root/'docs/fasttrack_private/cache_full_index.csv')
    real=real.loc[real.is_synthetic.astype(str).str.lower().isin(['0','false'])]
    available=set(real.sample_hash)
    x_train={}; x_test={}
    for f in range(5):
        for dest,role in ((x_train,'train'),(x_test,'test')):
            t=pd.read_csv(root/f'splits/fold_{f}_{role}.csv')
            dest[f]=set(t.loc[t.sample_hash.isin(available),'patient_hash'])
        if x_train[f]&x_test[f]:
            raise ValueError('X_TRAIN_TEST_SUBJECT_OVERLAP')
    report={}
    for branch in ('amyloid','tau'):
        split=json.loads((root/f'results/method_y_revival_v2/splits/{branch}_nested_splits.json').read_text())
        folds=split['outer_folds']
        if len(folds)!=5 or {int(f['fold']) for f in folds}!=set(range(5)):
            raise ValueError('Y_OUTER_FOLD_COUNT')
        matrix=[[len(x_train[xf]&set(yf['test_subjects'])) for yf in folds] for xf in range(5)]
        report[branch]={'x_development_ancestors_in_y_outer_test_counts':matrix,
            'same_number_fold_safe':all(matrix[f][f]==0 for f in range(5)),
            'any_direct_fold_mapping_possible':all(any(matrix[x][y]==0 for x in range(5)) for y in range(5)),
            'interpretation':'NONZERO_IS_FORBIDDEN_ANCESTOR_OVERLAP_ZERO_IS_NOT_COMPLETE_LINEAGE_PROOF'}
    return report


def baseline_mri_availability(root):
    """Metadata-only alternative; never silently substitute it for current X."""
    import re
    root=Path(root)
    e=pd.read_parquet(root/'results/method_y_revival_v2/endpoints/amyloid_progression_endpoint.parquet')
    e=e.loc[e.cohort.eq('A4')]
    vol=_table(RAW/'External Data/imaging_volumetric_mri.csv')
    vol=vol.loc[pd.to_numeric(vol.VISCODE,errors='coerce').eq(4)]
    vol=vol.dropna(subset=['Date_DAYS_T0']).drop_duplicates('patient_hash')
    matched=e.merge(vol[['patient_hash','Date_DAYS_T0']],on='patient_hash',validate='one_to_one')
    scan_ids=set(); count=0
    pattern=re.compile(r'^(A4|LEARN)_MR_T1_(B[0-9]+)_004\.nii(?:\.gz)?$',re.I)
    for path in RAW.parent.rglob('*MR_T1*004.nii*'):
        match=pattern.match(path.name)
        if match:
            count+=1
            scan_ids.add(stable_hash(match[1].upper()+'|'+match[2],prefix='pt'))
    raw_matched=e.loc[e.patient_hash.isin(scan_ids)]
    return {'baseline_volumetric_subjects':len(matched),'baseline_volumetric_historical_y_events':int(matched.event.sum()),
        'baseline_raw_t1_files':count,'baseline_raw_t1_unique_subjects':len(scan_ids),
        'baseline_raw_t1_a4_endpoint_subjects':len(raw_matched),
        'baseline_raw_t1_historical_y_events':int(raw_matched.event.sum()),
        'status':'AVAILABILITY_ONLY_NOT_VALIDATED_NEW_MRI_CACHE',
        'counts_are_not_final_fusion_sample':True,
        'new_baseline_cache_geometry_qc_and_joint_crossfitting_required':True}
