"""Exact baseline T1 inventory. Source-table dates are never taken from late X scans."""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from .audit import _table, RAW, build_intake
from .store import digest
from src.utils.privacy import stable_hash


def baseline_intake(root):
    root = Path(root)
    vol = _table(RAW/'External Data/imaging_volumetric_mri.csv')
    vol = vol.loc[pd.to_numeric(vol.VISCODE, errors='coerce').eq(4)].copy()
    vol['day'] = pd.to_numeric(vol.Date_DAYS_T0, errors='coerce')
    dates = vol.dropna(subset=['day'])[['patient_hash','day']].drop_duplicates()
    ambiguous = set(dates.loc[dates.patient_hash.duplicated(False), 'patient_hash'])
    dates = dates.loc[~dates.patient_hash.isin(ambiguous)].set_index('patient_hash').day.to_dict()
    pattern = re.compile(r'^(A4|LEARN)_MR_T1_(B[0-9]+)_004\.nii(?:\.gz)?$', re.I)
    records = []
    for p in sorted((RAW.parent/'T1').rglob('*004.nii*')):
        m = pattern.fullmatch(p.name)
        if not m:
            continue
        key = stable_hash(m[1].upper()+'|'+m[2], prefix='pt')
        if key not in dates:
            continue
        stat = p.stat()
        records.append({'patient_hash':key, 'raw_path':str(p.resolve()),
            't1_day_t0':float(dates[key]), 'bytes':stat.st_size, 'mtime_ns':stat.st_mtime_ns,
            'date_evidence':'EXACT_SUBJECT_VISIT004_SOURCE_VOLUMETRIC_DATE_DAYS_T0'})
    raw = pd.DataFrame(records)
    if raw.empty:
        raise ValueError('NO_EXACT_BASELINE_T1')
    # Multiple files are not resolved by arbitrary filesystem order.
    duplicate = set(raw.loc[raw.patient_hash.duplicated(False),'patient_hash'])
    raw = raw.loc[~raw.patient_hash.isin(duplicate)].sort_values('patient_hash').reset_index(drop=True)
    raw['sample_hash'] = ['mri_'+digest({'patient':r.patient_hash,'path':r.raw_path,'day':r.t1_day_t0})[:24]
                          for r in raw.itertuples()]
    raw['row'] = range(len(raw)); raw['x_fold'] = -1
    frame, lineage, summary, paths = build_intake(root, mri_override=raw)
    assays={r['blood']['assay'] for r in lineage if r['blood'] is not None}
    if len(assays)>1:raise ValueError('MULTIPLE_PTAU217_ASSAYS_REQUIRE_SEPARATE_FEATURE_CONTRACT')
    frame = frame.merge(raw[['patient_hash','raw_path','bytes','mtime_ns','date_evidence']],
                        on='patient_hash', validate='one_to_one')
    # Never use the row index of the old MRI cache for a baseline MRI.
    frame = frame.drop(columns=['mri_cache_row','x_fold']).sort_values('patient_hash').reset_index(drop=True)
    summary.update({'status':'BASELINE_CHRONOLOGY_READY_PENDING_IMAGE_QC',
                    'ambiguous_mri_dates':len(ambiguous),'ambiguous_raw_files':len(duplicate),
                    'source_mri':'RAW_VISIT004_NOT_LATE_X_CACHE'})
    return frame, lineage, summary, paths
