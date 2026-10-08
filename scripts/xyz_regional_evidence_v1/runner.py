"""Source-table regional evidence without changing the frozen fusion or post-all."""
from pathlib import Path
import json
import sys
import importlib.metadata
import numpy as np
import pandas as pd
from scripts.xyz_fusion_postall_v2.core import (Units, ReadOnlyParent, read_json, digest, file_sha,
    atomic_json, parent_exclusion, writer_lock)
from scripts.xyz_fusion_postall_v2.context import parquet_atomic
from scripts.xyz_fusion_v2.protocol import CLINICAL
from .statistics import cox_association, holm_family, standardized, pacc_association
from .reporting import anatomical_mapping, write_report


def prepared_inputs(root, parent, post):
    from scripts.xyz_fusion_v1.audit import RAW, _table
    f02=post.read('f02');f06=post.read('f06')
    if f02 is None or f06 is None:raise ValueError('VERIFIED_F02_AND_F06_REQUIRED')
    frame=pd.DataFrame(parent.contract['manifest'])
    if frame.patient_hash.duplicated().any():raise ValueError('DUPLICATE_SUBJECT')
    source=RAW/'External Data/imaging_volumetric_mri.csv'
    if file_sha(source)!=parent.contract['input_sha256']['volumes']:
        raise ValueError('REGIONAL_RAW_VOLUME_DRIFT')
    dictionary=next((RAW/'Documents').rglob('external_datadic.csv'))
    if file_sha(dictionary)!=parent.contract['input_sha256']['external_data_dictionary']:
        raise ValueError('REGIONAL_DICTIONARY_DRIFT')
    raw=_table(source)
    raw=raw.loc[pd.to_numeric(raw.VISCODE,errors='coerce').eq(4)].copy()
    raw['source_day']=pd.to_numeric(raw.Date_DAYS_T0,errors='coerce')
    features=[]
    for record in frame.itertuples():
        t=raw.loc[raw.patient_hash.eq(record.patient_hash)&raw.source_day.eq(record.mri_day)]
        cols=['LeftHippocampus','RightHippocampus','LeftEntorhinal','RightEntorhinal',
              'LeftLateralVentricle','RightLateralVentricle','IntraCranialVolume']
        t=t[cols].apply(pd.to_numeric,errors='coerce').drop_duplicates()
        row={'patient_hash':record.patient_hash,'icv_volume':np.nan}
        if len(t)==1:
            row['icv_volume']=float(t.iloc[0].IntraCranialVolume)
            for key,prefix in [('hippocampus','Hippocampus'),('entorhinal','Entorhinal'),('lateral_ventricle','LateralVentricle')]:
                row[key+'_volume']=float(t.iloc[0]['Left'+prefix]+t.iloc[0]['Right'+prefix])
        features.append(row)
    frame=frame.merge(pd.DataFrame(features),validate='one_to_one')
    for key,ratio in [('hippocampus','hippocampus_icv'),('entorhinal','entorhinal_icv'),('lateral_ventricle','ventricle_icv')]:
        got=frame[key+'_volume']/frame.icv_volume
        valid=frame[ratio].notna()
        if not np.allclose(got[valid],frame.loc[valid,ratio],atol=1e-10,rtol=0):
            raise ValueError('SOURCE_VOLUME_PARENT_MAPPING_CHANGED:'+key)
    frame['fold']=-1
    for fold,split in enumerate(parent.job('splits')):
        selected=frame.patient_hash.isin(split['test'])
        if (frame.loc[selected,'fold']!=-1).any():raise ValueError('DUPLICATE_OUTER_MEMBERSHIP')
        frame.loc[selected,'fold']=fold
    if (frame.fold<0).any():raise ValueError('MISSING_OUTER_MEMBERSHIP')
    if (frame.mri_day>frame.landmark_days_t0).any():raise ValueError('POST_ORIGIN_MRI')
    pacc_path=post.root/'private/pacc_longitudinal.parquet'
    if file_sha(pacc_path)!=f06['outputs']['private/pacc_longitudinal.parquet']:
        raise ValueError('REGIONAL_PACC_INPUT_DRIFT')
    pacc=pd.read_parquet(pacc_path)[['patient_hash','years','pacc']]
    return frame,pacc,{'source_volume_sha256':file_sha(source),'dictionary_sha256':file_sha(dictionary),
        'f02_digest':digest(f02),'f06_digest':digest(f06),
        'pacc_sha256':file_sha(pacc_path),'input_frame_sha256':digest(json.loads(frame.to_json(orient='records')))}


def analyze(frame,pacc,units,options):
    raw_names=['hippocampus_volume','entorhinal_volume','lateral_ventricle_volume','icv_volume']
    required=[*raw_names,*CLINICAL,'amyloid_centiloid']
    valid=np.isfinite(frame[required].to_numpy(float)).all(axis=1)
    valid&=(frame[raw_names]>0).all(axis=1)
    for col in raw_names[:-1]:valid&=frame[col]<frame.icv_volume
    data=frame.loc[valid].copy()
    demographics=[]
    for name,sub in [('included',data),('excluded',frame.loc[~valid])]:
        demographics.append({'group':name,'n':len(sub),'events':int(sub.event.sum()),
            'mean_age':float(sub.age.mean()) if len(sub) else None,
            'mean_baseline_pacc':float(sub.baseline_pacc_raw.mean()) if len(sub) else None})
    for col in raw_names:data['log_'+col]=np.log(data[col])
    continuous=['age','education_years','baseline_pacc_raw','baseline_mmscore','amyloid_centiloid',
                *['log_'+c for c in raw_names]]
    data,scales=standardized(data,continuous)
    nuisance=['age_z','education_years_z','baseline_pacc_raw_z','baseline_mmscore_z',
              'amyloid_centiloid_z','log_icv_volume_z','sex_binary','apoe4_carrier','treatment_active']
    cdr,pacc_results={},{}
    for roi in options['regions']:
        local=data.copy();local['region_z']=local['log_'+roi+'_volume_z']
        features=['region_z',*nuisance]
        value=units.job(roi+'_cdr',lambda:cox_association(local,features,options))
        sensitivity=[]
        for fold in range(5):
            def check_fold():
                subset=local.loc[local.fold.ne(fold)]
                return dict(cox_association(subset,features,options,diagnostics=False),excluded_fold=fold,
                    independent_replication=False)
            sensitivity.append(units.job(roi+f'_leave_fold{fold}',check_fold))
        known=units.job(roi+'_known_scanner',lambda:cox_association(
            local.loc[local.mri_manufacturer.ne('UNKNOWN')],features,options,diagnostics=False))
        value=dict(value,leave_fold_out=sensitivity,unknown_scanner_excluded=known)
        cdr[roi]=value
        def longitudinal():
            table=pacc.merge(local[['patient_hash','region_z','mri_manufacturer',*nuisance]],validate='many_to_one')
            if table.patient_hash.nunique()<options['minimum_subjects']:
                return {'status':'NOT_ESTIMABLE_SUPPORT','p':None}
            return pacc_association(table,nuisance)
        pacc_results[roi]=units.job(roi+'_pacc',longitudinal)
    cdr=holm_family(cdr,options['alpha']);pacc_results=holm_family(pacc_results,options['alpha'])
    for roi,value in cdr.items():
        if value['status']!='ESTIMATED':value['interpretation']='NOT_ESTIMABLE';continue
        diag=all(value.get(k,{}).get('p',0)>=.05 for k in ('ph','linearity'))
        stable=all(v.get('status')=='ESTIMATED' and v['log_hr']*value['log_hr']>0 for v in value['leave_fold_out'])
        known=value['unknown_scanner_excluded']
        stable &= known.get('status')=='ESTIMATED' and known.get('log_hr',0)*value['log_hr']>0
        value['interpretation']=('INTERNAL_REGIONAL_PROGNOSTIC_ASSOCIATION_SUPPORTED' if value['multiplicity_supported'] and diag and stable else
            'ASSOCIATION_LIMITED_BY_DIAGNOSTICS_OR_STABILITY' if value['multiplicity_supported'] else
            'REGIONAL_ASSOCIATION_NOT_DEMONSTRATED_UNDER_MULTIPLICITY')
        value['disease_specificity_or_causality_proven']=False
    return {'status':'REGIONAL_ASSOCIATIONS_COMPUTED','cdr':cdr,'pacc':pacc_results,
        'counts':{'parent':len(frame),'complete_mri_family':len(data),'events':int(data.event.sum())},
        'attrition':demographics,'scale_parameters':scales,
        'complete_case_selection_not_generalized_to_all_recruited_participants':True,
        'same_primary_rows_for_all_three_rois':True,'pacc_is_secondary_same_cohort_not_replication':True,
        'no_cnn_training_or_new_model_selection':True}


def execute(root,mode):
    root=Path(root).resolve()
    if root.as_posix().lower()!='h:/new project':raise ValueError('AUTHORITATIVE_H_ROOT_REQUIRED')
    options=read_json(root/'configs/xyz_regional_evidence_v1.json')
    out=(root/options['output_root']).resolve();postroot=root/options['postall_root']
    protected=[postroot,root/'results/xyz_fusion_v2',root/'results/xyz_fusion_postall_v1']
    if (not out.is_relative_to(root/'results') or any(out.is_relative_to(p) or p.is_relative_to(out) for p in protected)):
        raise ValueError('SEPARATE_REGIONAL_OUTPUT_REQUIRED')
    with writer_lock(out),parent_exclusion(postroot),parent_exclusion(root/'results/xyz_fusion_v2'):
        parent=ReadOnlyParent(root/'results/xyz_fusion_v2')
        pc=read_json(postroot/'private/contract.json')
        for name,sha in pc['source_sha256'].items():
            if file_sha(root/name)!=sha:raise ValueError('POSTALL_SOURCE_DRIFT')
        if pc['parent_contract']!=parent.binding:raise ValueError('POSTALL_PARENT_DRIFT')
        for name,sha in parent.execution_contract['source_sha256'].items():
            if file_sha(root/name)!=sha:raise ValueError('PARENT_SOURCE_DRIFT')
        for old in (pc,parent.execution_contract):
            if old['python']!=sys.version:raise ValueError('REGIONAL_UPSTREAM_PYTHON_DRIFT')
            for name,version in old['runtime'].items():
                if importlib.metadata.version(name)!=version:raise ValueError('REGIONAL_UPSTREAM_RUNTIME_DRIFT')
        post=Units(postroot,pc)
        if post.read('f00') is None:raise ValueError('VERIFIED_PARENT_REQUIRED')
        frame,pacc,proof=prepared_inputs(root,parent,post)
        code=list(Path(__file__).parent.glob('*.py'))+[root/'scripts/run_xyz_regional_evidence_v1.py']
        contract={'options':options,'postall_contract':post.binding,'inputs':proof,
            'source_sha256':{p.relative_to(root).as_posix():file_sha(p) for p in code},
            'python':sys.version,'runtime':{p:importlib.metadata.version(p) for p in ['numpy','pandas','statsmodels','scipy']}}
        units=Units(out,contract)
        try:
            if mode=='verify':
                if units.read('inputs')!=proof:raise ValueError('REGIONAL_INPUT_RECEIPT_MISSING_OR_CHANGED')
            else:units.job('inputs',lambda:proof)
            if mode=='preflight':
                print(json.dumps({'status':'REGIONAL_INPUTS_READY','n':len(frame)}),flush=True);return 0
            if mode=='verify':
                associations=units.read('associations')
                if associations is None:raise ValueError('REGIONAL_ASSOCIATIONS_MISSING')
            else:
                associations=units.job('associations',lambda:analyze(frame,pacc,units,options))
            state={'status':'REGIONAL_ASSOCIATIONS_COMPLETE_SPATIAL_SYNTHESIS_PENDING','claim_status':'POST_HOC_INTERNAL_EXPLORATORY',
                   'contract':units.binding,'postall_final_required_for_thesis_dossier':True}
            if mode in ('all','verify'):
                from scripts.xyz_fusion_postall_v2.context import Context
                from scripts.xyz_fusion_postall_v2.release import verify_final
                ctx=Context(root,parent,post,pc['config'])
                final=verify_final(ctx)
                if not final.get('verifier','').startswith('PASS'):raise ValueError('POSTALL_FINAL_REQUIRED_FOR_REGIONAL_DOSSIER')
                maps=post.read('f04')
                registry=root/'results/xyz_fusion_v2_assets/source_registry.json'
                labels=read_json(registry)['labels']
                spatial={'requested_units':maps['requested_units'],'eligible_units':maps['eligible_units'],
                    'mri_unique_cases':sum(c.get('track')=='mri' for c in maps['cases']),
                    'interpretation':'ILLUSTRATIVE_PANEL_NOT_POPULATION_REGIONAL_DISCOVERY',
                    'mappings':anatomical_mapping(labels),
                    'anatomical_labels_approximate_not_subject_segmentation':True,
                    'combining_roi_association_and_attribution_is_not_independent_replication':True}
                if mode=='all':units.job('report',lambda:write_report(out,associations,spatial,options),
                                       {'f08':digest(post.read('f08')),'f04':digest(maps)})
                elif units.read('report',{'f08':digest(post.read('f08')),'f04':digest(maps)}) is None:
                    raise ValueError('REGIONAL_REPORT_MISSING')
                for p in (out/'private/units').glob('*.json'):units.read(p.stem)
                state.update(status='REGIONAL_POSTHOC_DOSSIER_COMPLETED',clinical_or_disease_region_claim_allowed=False)
            atomic_json(out/'orchestration_state.json',state);print(json.dumps(state),flush=True)
            return 0
        except BaseException as exc:
            atomic_json(out/'orchestration_state.json',{'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'BLOCKED',
                'contract':units.binding,'reason':type(exc).__name__+':'+str(exc)})
            raise
