"""Outcome support is an eligibility check, never selection of favorable performance."""
import numpy as np
from scripts.xyz_fusion_v1.pipeline import partition, stop_partition, assert_disjoint
from scripts.xyz_fusion_v1.statistics import ibs_score
from .protocol import arms, eligible


def support(frame, splits, config, track, *, image=False, require_five=True):
    if require_five and len(splits)!=5:raise ValueError('FIVE_LOCKED_SPLITS_REQUIRED')
    if frame.patient_hash.duplicated().any():raise ValueError('DUPLICATE_SUBJECT')
    grid=config['fusion']['ibs_grid_days'];checks=0;seen=set();all_ids=set(frame.patient_hash)
    def check(a,b,features=()):
        nonlocal checks
        if not len(a) or not len(b):raise ValueError('EMPTY_TRAIN_OR_TEST')
        if features and a[list(features)].notna().sum().eq(0).any():
            raise ValueError('ALL_MISSING_TRAIN_FEATURE')
        ibs_score(a,b,np.tile(np.linspace(.98,.85,len(grid)),(len(b),1)),grid);checks+=1
    for fold,s in enumerate(splits):
        tr=frame.loc[frame.patient_hash.isin(s['train'])].reset_index(drop=True)
        te=frame.loc[frame.patient_hash.isin(s['test'])].reset_index(drop=True)
        assert_disjoint(tr.patient_hash,te.patient_hash)
        if (set(tr.patient_hash)|set(te.patient_hash))!=all_ids or seen & set(te.patient_hash):
            raise ValueError('SPLIT_COVERAGE_OR_TEST_DUPLICATION')
        seen.update(te.patient_hash)
        for arm,features in arms(track).items():
            a=eligible(tr,arm).reset_index(drop=True);b=eligible(te,arm)
            raw=[f for f in features if not f.startswith(('mri_risk_','pet_risk_'))]
            check(a,b,raw)
            if not image:
                for u,v in partition(a,3,config['seed']+fold):check(a.iloc[u],a.iloc[v],raw)
        if image:
            for inner,(u,v) in enumerate(partition(tr,config['components'],config['seed']+fold)):
                pool,meta=tr.iloc[u],tr.iloc[v]
                fit,val=stop_partition(pool,config['seed']+100*fold+inner)
                check(fit,val);check(pool,meta)
                for arm,features in arms(track).items():
                    raw=[f for f in features if not f.startswith(('mri_risk_','pet_risk_'))]
                    check(eligible(meta,arm),eligible(te,arm),raw)
    if require_five and seen!=all_ids:raise ValueError('OUTER_TEST_COVERAGE_INCOMPLETE')
    return {'status':'SUPPORTED','checks':checks,'n':len(frame),'events':int(frame.event.sum())}


def optional_support(frame, splits, config, track, *, image=False, require_five=True):
    try:return support(frame,splits,config,track,image=image,require_five=require_five)
    except ValueError as exc:
        return {'status':'NOT_ESTIMABLE_SUPPORT','reason':str(exc),'n':len(frame),
                'events':int(frame.event.sum()) if len(frame) else 0}
