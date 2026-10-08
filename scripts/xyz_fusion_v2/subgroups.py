"""Fixed-cut, OOF-only aggregate subgroup diagnostics; no selection or recalibration."""
import numpy as np
from scripts.xyz_fusion_v1.report import cell_frame,individual_bs
from .protocol import contrasts


def subgroup_report(frame,cells,config,track):
    index=frame.set_index('patient_hash',verify_integrity=True)
    groups={'sex_0':index.sex_binary.eq(0),'sex_1':index.sex_binary.eq(1),
            'age_lt_70':index.age.lt(70),'age_ge_70':index.age.ge(70),
            'treatment_0':index.treatment_active.eq(0),'treatment_1':index.treatment_active.eq(1),
            'cognition_mmse_lt_29':index.baseline_mmscore.lt(29),'cognition_mmse_ge_29':index.baseline_mmscore.ge(29)}
    for column in ('mri_manufacturer','pet_manufacturer'):
        if column in index:
            for value in sorted(index[column].fillna('UNKNOWN').unique()):groups[column+'_'+str(value)]=index[column].eq(value)
    result={};rng=np.random.default_rng(config['seed'])
    for name,mask in groups.items():
        keys=set(index.index[mask]);row={'n':len(keys),'events':int(index.loc[list(keys),'event'].sum()) if keys else 0}
        if row['n']<30 or row['events']<5:
            result[name]=dict(row,status='NOT_ESTIMABLE_SUPPORT');continue
        values={};pairs={}
        for arm in cells[0]['arms']:
            losses=[]
            for cell in cells:
                a=cell['arms'][arm];selected=np.array([k in keys for k in a['subjects']])
                if not selected.any():continue
                tr,te=cell_frame(a)
                # Keep fold censoring distribution fixed: this is a conditional
                # subgroup diagnostic, not a claim of group-specific censor independence.
                loss=individual_bs(tr,te,np.asarray(a['curves']),config['fusion']['ibs_grid_days'])[0]
                losses.extend(loss[selected])
            if len(losses)<30:values[arm]={'status':'NOT_ESTIMABLE_MATCHED_SUBGROUP_N'};continue
            l=np.asarray(losses)
            d=[float(l[rng.integers(0,len(l),len(l))].mean()) for _ in range(config['fusion']['bootstrap_resamples'])]
            values[arm]={'n':len(l),'ibs':float(l.mean()),'conditional_ci95':np.quantile(d,[.025,.975]).tolist()}
        for base,aug in contrasts(track):
            differences=[]
            for cell in cells:
                a,b=cell['arms'][base],cell['arms'][aug]
                if a['subjects']!=b['subjects']:raise ValueError('SUBGROUP_UNMATCHED_CONTRAST')
                take=np.array([k in keys for k in a['subjects']]);tr,te=cell_frame(a)
                x=individual_bs(tr,te,np.asarray(a['curves']),config['fusion']['ibs_grid_days'])[0]
                y=individual_bs(tr,te,np.asarray(b['curves']),config['fusion']['ibs_grid_days'])[0]
                differences.extend((y-x)[take])
            if len(differences)>=30:
                v=np.asarray(differences);draws=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(config['fusion']['bootstrap_resamples'])]
                pairs[base+'__'+aug]={'delta_ibs':float(v.mean()),'conditional_ci95':np.quantile(draws,[.025,.975]).tolist()}
        result[name]=dict(row,status='CONDITIONAL_OOF_DIAGNOSTICS',arms=values,paired_contrasts=pairs)
    return {'groups':result,'thresholds_locked_before_new_training':True,
            'censoring_policy':'OUTER_TRAIN_KM_CONDITIONAL_NOT_GROUP_SPECIFIC',
            'external_group_generalization':False,'subgroup_comparisons_are_exploratory':True,
            'cohort_transport':'NOT_ESTIMABLE_SINGLE_A4_COHORT',
            'qc_sensitivity':'HISTORICAL_PASS_ONLY_SELECTION_NO_FAILED_IMAGE_TRAINING'}
