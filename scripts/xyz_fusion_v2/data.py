"""Exact source linkage and a read-only reuse of the outcome-blind MRI cache."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.xyz_fusion_v1.audit import _table, RAW
from scripts.xyz_fusion_v1.store import file_sha, digest
from scripts.xyz_postprimary_v1.context import load_parent
from .endpoints import endpoint_views
from .protocol import VOLUMES


def records(frame):
    return json.loads(frame.to_json(orient='records'))


def sidecar(raw):
    p = Path(raw)
    return p.with_name(p.name.removesuffix('.gz').removesuffix('.nii')+'.json')


def manufacturer(value):
    text = str(value or 'UNKNOWN').strip().upper()
    if 'SIEMENS' in text: return 'SIEMENS'
    if text.startswith(('GE ', 'GE_', 'GENERAL ELECTRIC')) or text == 'GE': return 'GE'
    if 'PHILIPS' in text: return 'PHILIPS'
    if 'TOSHIBA' in text or 'CANON' in text: return 'CANON_TOSHIBA'
    return 'UNKNOWN' if text in ('', 'NAN', 'NONE', 'UNKNOWN') else text


def exact_volumes(table, frame):
    table = table.loc[pd.to_numeric(table.VISCODE, errors='coerce').eq(4)].copy()
    table['source_day'] = pd.to_numeric(table.Date_DAYS_T0, errors='coerce')
    out = []
    for row in frame.itertuples():
        t = table.loc[table.patient_hash.eq(row.patient_hash) & table.source_day.eq(row.mri_day)]
        values = dict(patient_hash=row.patient_hash, **{c: None for c in VOLUMES})
        cols = ['LeftHippocampus', 'RightHippocampus', 'LeftLateralVentricle',
                'RightLateralVentricle', 'LeftEntorhinal', 'RightEntorhinal', 'IntraCranialVolume']
        t = t[cols].apply(pd.to_numeric, errors='coerce').drop_duplicates()
        if len(t) == 1:
            x = t.iloc[0]; icv = x.IntraCranialVolume
            if np.isfinite(icv) and icv > 0:
                for name, prefix in zip(VOLUMES, ['Hippocampus', 'LateralVentricle', 'Entorhinal']):
                    v = (x['Left'+prefix]+x['Right'+prefix])/icv
                    values[name] = float(v) if np.isfinite(v) and 0 < v < 1 else None
        out.append(values)
    return frame.merge(pd.DataFrame(out), on='patient_hash', validate='one_to_one')


def build_data(root, config):
    ctx = load_parent(root)
    frame = ctx['frame'].copy()
    cdr_path = RAW/'Raw Data/cdr.csv'
    cdr = _table(cdr_path)
    cdr['day'] = pd.to_numeric(cdr.CDADTC_DAYS_T0, errors='coerce')
    cdr['cdr'] = pd.to_numeric(cdr.CDGLOBAL, errors='coerce')
    visits = {k: list(zip(t.day, t.cdr)) for k, t in cdr.groupby('patient_hash')}
    endpoints = []
    for row in frame.itertuples():
        views = endpoint_views(visits.get(row.patient_hash, []), row.landmark_days_t0,
                               config['endpoint']['window_days'])
        endpoints.append(dict(patient_hash=row.patient_hash, **views))
    for i, view in enumerate(endpoints):
        for name in [config['endpoint']['primary'], *config['endpoint']['sensitivity']]:
            for col in ('event', 'time_days'):
                frame.loc[i, name+'_'+col] = view[name][col]
    for col in ('event', 'time_days'):
        frame[col] = frame[config['endpoint']['primary']+'_'+col]
    volume_path = RAW/'External Data/imaging_volumetric_mri.csv'
    frame = exact_volumes(_table(volume_path), frame)
    if (frame.blood_ptau217.dropna()<=0).any():raise ValueError('NONPOSITIVE_BLOOD_MEASUREMENT')
    frame['blood_log_ptau217'] = np.log(frame.blood_ptau217)
    scanner = []; source_hashes = {}
    for row in frame.itertuples():
        p = sidecar(row.raw_path)
        metadata = json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
        if p.exists(): source_hashes[str(p)] = file_sha(p)
        # A scanner console/station is NOT a recruiting site. Institution names
        # are retained as a separately labelled acquisition-center proxy only.
        center = metadata.get('InstitutionName')
        site = metadata.get('ClinicalTrialSiteID')
        scanner.append({'patient_hash': row.patient_hash,
                        'mri_manufacturer': manufacturer(metadata.get('Manufacturer')),
                        'mri_model': str(metadata.get('ManufacturersModelName', 'UNKNOWN')),
                        'mri_site': digest({'site': site})[:16] if site else 'UNKNOWN',
                        'mri_acquisition_center_proxy':digest({'center':center})[:16] if center else 'UNKNOWN',
                        'mri_metadata_linkage': 'EXACT_SCAN_STEM' if metadata else 'MISSING'})
    frame = frame.merge(pd.DataFrame(scanner), on='patient_hash', validate='one_to_one')
    for key, p in [('cdr', cdr_path), ('volumes', volume_path),
                   ('pacc', RAW/'Derived Data/PACC.csv'), ('visits', RAW/'Derived Data/SV.csv'),
                   ('tau', RAW/'External Data/imaging_SUVR_tau.csv'),
                   ('fbp', RAW/'External Data/imaging_SUVR_amyloid.csv'),
                   ('subject', RAW/'Derived Data/SUBJINFO.csv')]:
        source_hashes[key] = file_sha(p)
    dictionary=next((RAW/'Documents').rglob('external_datadic.csv'))
    source_hashes['external_data_dictionary']=file_sha(dictionary)
    primary = config['endpoint']['primary']
    summary = {'n': len(frame), 'events_primary': int(frame.event.sum()),
               'events_historical': int(frame.legacy_onset_event.sum()),
               'unconfirmed_final_positive': sum(v['unconfirmed_final_positive'] for v in endpoints),
               'endpoint_counts': {name: int(frame[name+'_event'].sum())
                                   for name in [primary, *config['endpoint']['sensitivity']]},
               'volume_complete': int(frame[VOLUMES].notna().all(axis=1).sum()),
               'blood_available': int(frame.blood_ptau217.notna().sum()),
               'mri_manufacturer_counts': frame.mri_manufacturer.value_counts().to_dict(),
               'population': 'HISTORICAL_A4_BASELINE_MRI_QC_COHORT_WITH_OBSERVED_FOLLOWUP',
               'selection_not_eliminated_by_endpoint_redefinition': True,
               'all_source_a4': int(_table(RAW/'Derived Data/SUBJINFO.csv').SUBSTUDY.eq('A4').sum())}
    return ctx, frame, endpoints, source_hashes, summary


def with_endpoint(frame, name):
    value = frame.copy()
    for col in ('event', 'time_days'): value[col] = value[name+'_'+col]
    # A pending positive without a post-origin negative has no positive follow-up
    # for this sensitivity; never invent a one-day observation.
    return value.loc[value.time_days > 0].copy()


def load_tau(root, ctx, frame, config):
    from scripts.xyz_postprimary_v1.feasibility import audit_tau_landmark
    audit = audit_tau_landmark(ctx)
    rows = pd.DataFrame(audit.pop('private_rows'))
    source = _table(RAW/'External Data/imaging_SUVR_tau.csv')
    source = source.loc[source.ligand.str.casefold().eq('flortaucipir') &
                        source.scan_analyzed.str.casefold().eq('yes')].copy()
    wanted = config['secondary']['tau_regions']
    column = config['secondary']['tau_value_column']
    features = []
    for row in rows.itertuples():
        t = source.loc[source.patient_hash.eq(row.patient_hash) &
                       pd.to_numeric(source.scan_date_DAYS_T0, errors='coerce').eq(row.ftp_day)]
        if len(t[['VISCODE','scan_number']].drop_duplicates()) != 1:
            continue  # Same-day scans must not be merged through identical intensities.
        values = []
        for region in wanted:
            v = pd.to_numeric(t.loc[t.brain_region.eq(region), column], errors='coerce').dropna().unique()
            if len(v) != 1 or v[0] <= 0: break
            values.append(float(v[0]))
        if len(values) == len(wanted) and row.origin == float(frame.set_index('patient_hash').loc[row.patient_hash, 'landmark_days_t0']):
            features.append({'patient_hash': row.patient_hash, 'tau_temporal': float(np.mean(values))})
    if not features:
        return frame.iloc[:0].assign(tau_temporal=np.nan), dict(audit, status='BLOCKED_EXACT_REGION_OR_LANDMARK_MAPPING')
    subset = frame.merge(pd.DataFrame(features), on='patient_hash', validate='one_to_one')
    return subset, dict(audit, status='EXACT_SOURCE_TABLE_FEATURES', n=len(subset), events=int(subset.event.sum()),
                        regions=wanted, reference_column=column, centaur_equivalence=False,
                        aggregation='ARITHMETIC_MEAN_OF_TWO_SOURCE_ROIS_NOT_VOLUME_WEIGHTED_COMPOSITE',
                        reference_documentation='EXTERNAL_DICTIONARY_CEREBELLAR_CRUS',
                        source_mask_equivalence_validated=False)
