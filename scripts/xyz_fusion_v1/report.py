"""Aggregate-only release. Conditional paired subject bootstrap from committed predictions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv
from .statistics import metrics,support
from .store import atomic_text


CONTRASTS=[('clinical','clinical_mri'),('clinical_mri','clinical_mri_amyloid'),
           ('clinical_amyloid','clinical_mri_amyloid'),
           ('blood_matched_clinical_mri','clinical_mri_blood'),
           ('clinical_mri_blood','clinical_mri_blood_amyloid'),
           ('blood_matched_clinical_mri_amyloid','clinical_mri_blood_amyloid')]


def cell_frame(cell):
    test=pd.DataFrame({'event':cell['event'],'time_days':cell['time']})
    train=pd.DataFrame({'event':cell['censor_train']['event'],'time_days':cell['censor_train']['time']})
    return train,test


def individual_bs(train,test,p,grid):
    # Same right-continuous reverse-KM convention as sksurv IBS, not calibration.
    tr,te,_=support(train,test,grid)
    censor=CensoringDistributionEstimator().fit(tr)
    event_g=censor.predict_proba(te['time']);at_g=censor.predict_proba(np.asarray(grid))
    errors=[]
    for j,t in enumerate(grid):
        case=te['event']&(te['time']<=t);control=te['time']>t
        if np.any(event_g[case]<=0) or at_g[j]<=0:raise ValueError('BOOTSTRAP_CENSORING_SUPPORT')
        loss=np.zeros(len(te));loss[case]=p[case,j]**2/event_g[case]
        loss[control]=(1-p[control,j])**2/at_g[j]
        errors.append(loss)
    return np.trapezoid(np.array(errors).T,x=grid,axis=1)/(grid[-1]-grid[0]),tr,te


def paired_ci(cells,base,augmented,grid,seed,resamples):
    prepared=[]
    for fold in cells:
        a,b=fold['arms'][base],fold['arms'][augmented]
        if a['subjects']!=b['subjects'] or a['event']!=b['event'] or a['time']!=b['time'] or a['censor_train']!=b['censor_train']:
            raise ValueError('UNMATCHED_PAIRED_CONTRAST')
        train,test=cell_frame(a)
        ba,tr,te=individual_bs(train,test,np.asarray(a['curves']),grid)
        bb,_,_=individual_bs(train,test,np.asarray(b['curves']),grid)
        prepared.append((bb-ba,tr,te,1-np.asarray(a['curves'])[:,-1],1-np.asarray(b['curves'])[:,-1]))
    rng=np.random.default_rng(seed);bs=[];cs=[]
    for _ in range(resamples):
        sums=0.;n=0;deltas=[];weights=[]
        for delta,tr,te,ra,rb in prepared:
            idx=rng.integers(0,len(te),len(te));sums+=delta[idx].sum();n+=len(idx)
            try:
                ca=concordance_index_ipcw(tr,te[idx],ra[idx],tau=grid[-1])[0]
                cb=concordance_index_ipcw(tr,te[idx],rb[idx],tau=grid[-1])[0]
                if np.isfinite(ca) and np.isfinite(cb):deltas.append(cb-ca);weights.append(len(idx))
            except ValueError:pass
        bs.append(sums/n)
        if len(deltas)==len(prepared):cs.append(float(np.average(deltas,weights=weights)))
    def ci(values):
        return list(map(float,np.quantile(values,[.025,.975]))) if len(values)>=.95*resamples else None
    def simultaneous_ci(values):
        tail=.05/(2*len(CONTRASTS))
        return list(map(float,np.quantile(values,[tail,1-tail]))) if len(values)>=.95*resamples else None
    return {'base':base,'augmented':augmented,'delta_ibs_mean':float(np.mean(np.concatenate([x[0] for x in prepared]))),
            'delta_ibs_ci95':ci(bs),'delta_uno_ci95':ci(cs),'valid_ibs_resamples':len(bs),'valid_uno_resamples':len(cs),
            'delta_ibs_bonferroni_ci':simultaneous_ci(bs),'delta_uno_bonferroni_ci':simultaneous_ci(cs),
            'resamples':resamples,'resampling':'PAIRED_SUBJECT_WITHIN_OUTER_FOLD',
            'uncertainty':'CONDITIONAL_ON_FITTED_MODELS_AND_TRAIN_CENSORING_NOT_SELECTION_UNCERTAINTY',
            'multiplicity':'EXPLORATORY_NO_CONFIRMATORY_SIGNIFICANCE_CLAIM'}


def make_report(cells,frame,config):
    if len(cells)!=5 or {c['fold'] for c in cells}!=set(range(5)):raise ValueError('FINAL_REQUIRES_FIVE_COMMITTED_FOLDS')
    seen=set()
    for c in cells:
        if seen&set(c['test_subjects']) or set(c['train_subjects'])&set(c['test_subjects']):raise ValueError('FINAL_SUBJECT_OVERLAP')
        seen.update(c['test_subjects'])
    if seen!=set(frame.patient_hash):raise ValueError('FINAL_MEMBERSHIP_INCOMPLETE')
    arms={}
    rng=np.random.default_rng(config['seed']+40000)
    subgroup_results={}
    for arm in cells[0]['arms']:
        ms=[c['arms'][arm]['metrics'] for c in cells]
        cs=[m['uno_c_index'] for m in ms]
        arms[arm]={'n':sum(m['n'] for m in ms),'events':sum(m['events'] for m in ms),
                   'mean_fold_ibs':float(np.average([m['ibs'] for m in ms],weights=[m['n'] for m in ms])),
                   'mean_fold_uno_c_index':float(np.average(cs,weights=[m['n'] for m in ms])) if all(v is not None and np.isfinite(v) for v in cs) else None,
                   'fold_metrics':ms}
        individual=[];keys=[]
        for c in cells:
            cell=c['arms'][arm];tr,te=cell_frame(cell)
            loss,_,_=individual_bs(tr,te,np.asarray(cell['curves']),config['fusion']['ibs_grid_days'])
            individual.extend(loss);keys.extend(cell['subjects'])
        values=np.asarray(individual); draws=[]
        for _ in range(config['fusion']['bootstrap_resamples']):
            draws.append(float(values[rng.integers(0,len(values),len(values))].mean()))
        arms[arm]['ibs_conditional_ci95']=list(map(float,np.quantile(draws,[.025,.975])))
        info=frame.set_index('patient_hash').loc[keys]
        groups={'sex_code_0':info.sex_binary.eq(0),'sex_code_1':info.sex_binary.eq(1),
                'age_below_70':info.age.lt(70),'age_70_or_older':info.age.ge(70),
                'treatment_0':info.treatment_active.eq(0),'treatment_1':info.treatment_active.eq(1),
                'mmse_below_29':info.baseline_mmscore.lt(29),'mmse_29_or_higher':info.baseline_mmscore.ge(29)}
        if 'manufacturer' in info:
            # Fixed, separately reported manufacturers; UNKNOWN is not imputed.
            names=info.manufacturer.fillna('UNKNOWN').astype(str).str.upper()
            for name in ('SIEMENS','GE','PHILIPS','TOSHIBA','UNKNOWN'):
                groups['manufacturer_'+name]=names.eq(name)
        subgroup_results[arm]={}
        for name,mask in groups.items():
            idx=np.flatnonzero(mask.to_numpy());n=len(idx);events=int(info.iloc[idx].event.sum())
            r={'n':n,'events':events,'status':'NOT_ESTIMABLE'}
            if n>=30 and events>=10:
                v=values[idx];sample=[float(v[rng.integers(0,n,n)].mean()) for _ in range(config['fusion']['bootstrap_resamples'])]
                r.update(status='DESCRIPTIVE',ibs=float(v.mean()),ibs_conditional_ci95=list(map(float,np.quantile(sample,[.025,.975]))),
                         censoring_estimator_scope='OUTER_TRAIN_OVERALL_NOT_SUBGROUP_SPECIFIC')
            subgroup_results[arm][name]=r
    fc=config['fusion'];comparisons=[paired_ci(cells,a,b,fc['ibs_grid_days'],config['seed']+i,fc['bootstrap_resamples'])
                                   for i,(a,b) in enumerate(CONTRASTS)]
    return {'status':'PRIMARY_MRI_FBP_Z_INTERNAL_ANALYSIS_COMPLETED',
            'claim_status':'POST_HOC_INTERNAL_EXPLORATORY','clinical_claim_allowed':False,
            'external_claim_allowed':False,'external_status':'DEFERRED_NO_EXTERNAL_DATA',
            'positive_quantitative_equivalence_claim_allowed':False,
            'n_after_image_qc':len(frame),'events_after_new_landmark':int(frame.event.sum()),
            'outer_folds_completed':5,'arms':arms,'paired_comparisons':comparisons,
            'subgroup_ibs':subgroup_results,
            'manufacturer_metadata_scope':'FBP_IMAGE_MEMBERSHIP_WHERE_AVAILABLE_NOT_MRI_SCANNER_ROBUSTNESS',
            'not_completed_by_this_run':{'tau_paired_extension':fc['tau_extension'],
                                         'pet_image_fusion_extension':fc['pet_image_extension'],
                                         'native_anatomical_localization':'SEPARATE_XAI_VALIDATION_REQUIRED'},
            'limitations':['HISTORICAL_OUTCOME_EXPOSURE_DISCLOSED','NO_EXTERNAL_VALIDATION',
                           'SOURCE_TABLE_CENTILOID_NOT_INDEPENDENT_IMAGE_CALIBRATION',
                           'MRI_EXPERT_RETRAINED_FOR_PROGNOSIS_NOT_LEGACY_X_CLASSIFICATION',
                           'CALIBRATION_DESCRIPTIVE_NO_OUTER_REFIT','NO_DISEASE_LOCALIZATION_CLAIM']}


def render_report(report,path):
    lines=['# Primary MRI, FBP and baseline blood fusion','',
           'Status: '+report['status'], '', 'Claim ceiling: POST_HOC_INTERNAL_EXPLORATORY.', '',
           'This is a new common-landmark prognostic adjunct, not direct reuse of historical X/Y predictions.',
           'No external, diagnostic, causal, or quantitative image-equivalence claim is authorized.', '',
           '| Arm | Subjects | Events | Weighted fold IBS | Weighted fold Uno C |',
           '|---|---:|---:|---:|---:|']
    for name,r in report['arms'].items():
        uno='NOT_ESTIMABLE' if r['mean_fold_uno_c_index'] is None else f"{r['mean_fold_uno_c_index']:.4f}"
        lines.append(f"| {name} | {r['n']} | {r['events']} | {r['mean_fold_ibs']:.4f} | {uno} |")
    lines+=['','## Paired incremental-value analyses','',
            'Negative delta IBS favors the augmented arm. CIs condition on fitted models; they are not selection uncertainty.', '',
            '| Base -> augmented | Delta IBS | Paired 95% CI |', '|---|---:|---|']
    for r in report['paired_comparisons']:
        lines.append(f"| {r['base']} -> {r['augmented']} | {r['delta_ibs_mean']:.5f} | {r['delta_ibs_ci95']} |")
    lines+=['','Calibration slopes/intercepts, risk-bin summaries, event/at-risk counts, subgroup support and multiplicity-adjusted CIs are in results.json.',
            '', '## Still outside this execution', '']
    lines += [f'- {k}: {v}' for k,v in report['not_completed_by_this_run'].items()]
    lines += ['', 'The release contains aggregates only. Small/unsupported subgroups remain NOT_ESTIMABLE.', '']
    atomic_text(path,'\n'.join(lines))
