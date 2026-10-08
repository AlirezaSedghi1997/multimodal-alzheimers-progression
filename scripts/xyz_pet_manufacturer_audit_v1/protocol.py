"""Prespecify every supported group and exclude it from selection as well as fitting."""
import numpy as np
from scripts.xyz_fusion_v1.pipeline import partition, stop_partition, assert_disjoint
from scripts.xyz_fusion_v1.statistics import ibs_score
from scripts.xyz_fusion_v2.protocol import CLINICAL, MRI, PET
from scripts.xyz_fusion_postall_v2.robustness import pet_scanner_splits

ARMS={'clinical_mri_amyloid':CLINICAL+MRI+['amyloid_centiloid'],
      'clinical_mri_amyloid_pet':CLINICAL+MRI+['amyloid_centiloid']+PET}


def blind(frame):
    return frame.drop(columns=[c for c in frame if c in ('event','time_days') or c.endswith(('_event','_time_days'))]).copy()


def select_inner_winner(candidates):
    if not candidates or any(c.get('metric') is None or not np.isfinite(c['metric']) or
            not 0<=c['metric']<=1 or c['epochs']<1 for c in candidates):
        raise ValueError('INNER_IBS_NOT_ESTIMABLE')
    return min(candidates,key=lambda c:(c['metric'],c['lr']))


def validate_partition(frame,manufacturer,pool,meta,held):
    if frame.patient_hash.duplicated().any():raise ValueError('DUPLICATE_SUBJECT')
    assert_disjoint(pool,meta,held)
    indexed=frame.set_index('patient_hash')
    if (not indexed.loc[list(held),'pet_manufacturer'].eq(manufacturer).all() or
            indexed.loc[list(pool)+list(meta),'pet_manufacturer'].eq(manufacturer).any()):
        raise ValueError('HELD_MANUFACTURER_IN_ANCESTRY')
    if set(pool)|set(meta)|set(held)!=set(frame.patient_hash):
        raise ValueError('LMO_PARTITION_COVERAGE')


def make_plan(frame,config,options):
    splits,excluded=pet_scanner_splits(frame,config)
    supported=[];grid=config['fusion']['ibs_grid_days']
    for split in splits:
        train=frame.loc[frame.patient_hash.isin(split['train'])].reset_index(drop=True)
        held=frame.loc[frame.patient_hash.isin(split['test'])].reset_index(drop=True)
        tasks=[]
        def check(a,b):
            ibs_score(a,b,np.tile(np.linspace(.98,.85,len(grid)),(len(b),1)),grid)
        try:
            check(train,held)
            for index,(u,v) in enumerate(partition(train,options['components'],config['seed'])):
                pool,meta=train.iloc[u],train.iloc[v]
                fit,stop=stop_partition(pool,config['seed']+index)
                validate_partition(frame,split['group'],pool.patient_hash,meta.patient_hash,held.patient_hash)
                assert_disjoint(fit.patient_hash,stop.patient_hash,meta.patient_hash,held.patient_hash)
                if set(fit.patient_hash)|set(stop.patient_hash)!=set(pool.patient_hash):raise ValueError('FIT_STOP_COVERAGE')
                for a,b in ((fit,stop),(pool,meta),(meta,held)):check(a,b)
                if int(meta.event.sum())<10:raise ValueError('FEWER_THAN_TEN_META_EVENTS')
                for features in ARMS.values():
                    raw=[x for x in features if not x.startswith(('mri_risk_','pet_risk_'))]
                    if meta[raw].notna().sum().eq(0).any():raise ValueError('ALL_MISSING_META_FEATURE')
                tasks.append({'component':index,'pool':list(pool.patient_hash),'meta':list(meta.patient_hash),
                              'fit':list(fit.patient_hash),'stop':list(stop.patient_hash)})
        except ValueError as exc:
            excluded.append({'manufacturer':split['group'],'n':len(held),'events':int(held.event.sum()),
                'status':'NOT_ESTIMABLE_NESTED_SUPPORT','reason':str(exc)})
            continue
        supported.append(dict(split,components=tasks,n=len(held),events=int(held.event.sum())))
    return {'status':'LOCKED_MANUFACTURER_SUPPORT_PLAN','supported':supported,'excluded':excluded,
            'expected_expert_fits':len(supported)*options['components']*2*(len(config['training']['learning_rates'])+1),
            'site_status':'NOT_ESTIMABLE_NO_LOCKED_EXACT_PET_SITE',
            'unknown_scanner_not_imputed':True,'no_manufacturer_merging':True,
            'selection_used_performance':False}
