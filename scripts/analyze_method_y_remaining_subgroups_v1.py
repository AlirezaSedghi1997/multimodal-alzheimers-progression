"""No-inference subgroup closure using committed predictions, not a refit.

New cognition analysis is explicitly post-hoc, at 1095 days, with training-only
median definitions and training-only reverse-KM IPCW. It does not replace IBS.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'tmp')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='1'
import numpy as np
import pandas as pd
from sksurv.nonparametric import CensoringDistributionEstimator
from src.method_y_revival.atomic_io import atomic_write_json,read_json
from src.method_y_revival.local_validation_v5 import (
    file_sha,make_contract,analysis_lock,committed_job,verified_hashed_report)
from src.method_y_revival.posttraining_bootstrap_v4_2 import _prepare_branch
from src.method_y_revival.privacy import scan_public_bundle

OUT=ROOT/'results/method_y_remaining_subgroups_v2'
V4=ROOT/'results/method_y_revival_v4_image_factorial_v1'
CORRECTION=ROOT/'results/method_y_revival_v4_posttraining_correction_v4_2/locked_artifact_correction.json'
HORIZON=1095.0


def brier_contributions(train, rows, grid, horizon=HORIZON):
    grid=np.asarray(grid,float)
    where=np.flatnonzero(np.isclose(grid,horizon,rtol=0,atol=1e-9))
    if len(where)!=1:
        raise ValueError('LOCKED_GRID_LACKS_EXACT_HORIZON')
    time=np.array([r['time_days'] for r in rows],float)
    event=np.array([r['event'] for r in rows],bool)
    s=np.array([r['survival_probabilities'][where[0]] for r in rows],float)
    if not np.isfinite(s).all() or (s<0).any() or (s>1).any():
        raise ValueError('INVALID_SURVIVAL_PROBABILITY')
    censor=CensoringDistributionEstimator().fit(train)
    cases=event & (time<=horizon); controls=time>horizon
    g_h=float(censor.predict_proba(np.array([horizon]))[0])
    g_e=censor.predict_proba(time[cases])
    if g_h<.05 or (g_e<.05).any():
        raise ValueError('IPCW_SUPPORT_BELOW_0P05')
    loss=np.zeros(len(rows))
    loss[cases]=s[cases]**2/g_e
    loss[controls]=(1-s[controls])**2/g_h
    return loss,cases,controls


def conditional_ci(values, folds, *, seed=13, iterations=2000):
    values=np.asarray(values,float); folds=np.asarray(folds,int)
    groups=[values[folds==f] for f in range(5)]
    if any(len(g)<2 for g in groups):
        return {'status':'NOT_ESTIMABLE_EMPTY_OR_SINGLETON_FOLD'}
    rng=np.random.default_rng(seed)
    # Subject-weighted mean, not treating five overlapping training sets as n=5 independent trials.
    draws=np.array([sum(g[rng.integers(0,len(g),len(g))].sum() for g in groups)/len(values)
                    for _ in range(iterations)])
    return {'status':'ESTIMABLE_CONDITIONAL_DESCRIPTIVE','point_delta_brier':float(values.mean()),
        'delta_brier_95ci':np.quantile(draws,[.025,.975]).tolist(),'resamples':iterations,
        'fold_sizes':[len(g) for g in groups],
        'selection_training_and_censoring_fit_uncertainty_included':False,
        'positive_subgroup_claim_allowed':False}


def run(mode):
    correction=verified_hashed_report(CORRECTION)
    rc=read_json(V4/'run_contract.json')
    inputs={'correction':CORRECTION,'v4_contract':V4/'run_contract.json'}
    prepared={}; frames={}; splits={}
    for branch in ('amyloid','tau'):
        used={}; prepared[branch]=_prepare_branch(V4,correction,branch,used)
        for rel,digest in used.items():
            if correction['input_sha256'].get(rel)!=digest:
                raise ValueError('CORRECTION_INPUT_CHANGED')
            inputs[branch+rel]=V4/rel
        for name in ('membership','splits'):
            rec=rc['input_fingerprints'][branch+'_'+name]
            if file_sha(rec['path'])!=rec['sha256']:
                raise ValueError('SUBGROUP_INPUT_DRIFT')
            inputs[branch+name]=Path(rec['path'])
        frame=pd.read_parquet(inputs[branch+'membership'])
        frame=frame.loc[frame.role=='development'].copy()
        if frame.patient_hash.duplicated().any():
            raise ValueError('DUPLICATE_SUBJECT')
        frames[branch]=frame.set_index('patient_hash')
        splits[branch]=read_json(inputs[branch+'splits'])
    source=[Path(__file__),ROOT/'src/method_y_revival/posttraining_bootstrap_v4_2.py',
            ROOT/'src/method_y_revival/posttraining_correction_v4_2.py']
    contract=make_contract(inputs,{p.name:p for p in source},{'horizon':HORIZON,'resamples':2000,
        'cognition':'baseline_pacc_raw BELOW_TRAIN_MEDIAN vs AT_OR_ABOVE_TRAIN_MEDIAN',
        'groups_defined_without_outcome':True,'min_events_by_horizon':10,'min_observed_controls':10,
        'comparison':'WITHIN_OUTER_INNER_SELECTED_ARM_MINUS_A0',
        'claim_status':'POST_HOC_INTERNAL_EXPLORATORY'})
    with analysis_lock(OUT):
        cp=OUT/'private/contract.json'
        if cp.exists() and read_json(cp)!=contract:
            raise ValueError('RUN_CONTRACT_MISMATCH')
        if not cp.exists():
            if mode=='verify': raise ValueError('CONTRACT_MISSING')
            atomic_write_json(cp,contract)
        reports={}
        for branch in ('amyloid','tau'):
            def compute(branch=branch):
                frame=frames[branch]
                groups={name:[] for name in ('BELOW_TRAIN_MEDIAN','AT_OR_ABOVE_TRAIN_MEDIAN','MISSING')}
                thresholds=[]
                for fold in prepared[branch]:
                    sp=next(s for s in splits[branch]['outer_folds'] if s['fold']==fold['fold'])
                    # Splits describe the regional cohort; image training used
                    # its exact intersection with development image membership.
                    raw_train=set(map(str,sp['train_subjects']))
                    train=sorted(raw_train & set(frame.index))
                    test=[r['patient_hash'] for r in fold['baseline_rows']]
                    expected_test=set(map(str,sp['test_subjects'])) & set(frame.index)
                    if raw_train&set(test) or set(test)!=expected_test or not train:
                        raise ValueError('SUBGROUP_PARTITION_ERROR')
                    values=pd.to_numeric(frame.loc[train,'baseline_pacc_raw'],errors='coerce')
                    threshold=float(values.median())
                    if not np.isfinite(threshold): raise ValueError('COGNITION_TRAIN_MEDIAN_MISSING')
                    thresholds.append(threshold)
                    lb,cases,controls=brier_contributions(fold['train'],fold['baseline_rows'],fold['grid'])
                    ls,_,_=brier_contributions(fold['train'],fold['selected_rows'],fold['grid'])
                    for i,subject in enumerate(test):
                        value=float(frame.loc[subject,'baseline_pacc_raw'])
                        group='MISSING' if not np.isfinite(value) else 'BELOW_TRAIN_MEDIAN' if value<threshold else 'AT_OR_ABOVE_TRAIN_MEDIAN'
                        groups[group].append((float(ls[i]-lb[i]),fold['fold'],bool(cases[i]),bool(controls[i])))
                cognition={}
                for i,(name,rows) in enumerate(groups.items()):
                    n=len(rows); events=sum(r[2] for r in rows); controls=sum(r[3] for r in rows)
                    if n<30 or events<10 or controls<10:
                        cognition[name]={'status':'NOT_ESTIMABLE_INSUFFICIENT_SUPPORT','subject_count':n}
                        continue
                    cognition[name]={**conditional_ci([r[0] for r in rows],[r[1] for r in rows],seed=13+i),
                        'subject_count':n,'events_by_horizon':events,'observed_controls':controls}
                cohorts=frame.cohort.replace({'A4_placebo':'A4','A4_treatment':'A4'})
                qc=frame.qc_status.fillna('UNKNOWN')
                return {'cognition':cognition,'training_median_range':[min(thresholds),max(thresholds)],
                    'metric':'IPCW_BRIER_AT_1095_DAYS_NOT_IBS','horizon_days':HORIZON,
                    'qc':{'status':'NOT_ESTIMABLE_NO_VARIATION' if qc.nunique()==1 else 'REQUIRES_SEPARATE_CONTRAST',
                          'levels':sorted(qc.unique().tolist())},
                    'cohort':{'status':'NOT_ESTIMABLE_DEVELOPMENT_IS_A4_ONLY' if cohorts.nunique()==1 else 'REQUIRES_SEPARATE_CONTRAST',
                              'levels':sorted(cohorts.unique().tolist())},
                    'limitations':['POST_HOC','FIXED_MODEL_CONDITIONAL_CI','MARGINAL_TRAIN_CENSORING_REQUIRES_INDEPENDENCE_WITHIN_SUBGROUP',
                                   'NO_EXTERNAL_TRANSPORT','DESCRIPTIVE_NOT_MULTIPLICITY_ADJUSTED_SUPERIORITY'],
                    'positive_subgroup_claim_allowed':False}
            reports[branch]=committed_job(OUT/'private',branch,contract,compute,verify_only=mode=='verify')
        public={'status':'REMAINING_SUBGROUP_ANALYSES_COMPLETED_WITH_LIMITATIONS','branches':reports,
                'claim_status':'POST_HOC_INTERNAL_EXPLORATORY','new_training':False,'new_test_inference':False}
        if mode=='verify':
            if read_json(OUT/'public_bundle/subgroups.json')!=public: raise ValueError('PUBLIC_REPORT_DRIFT')
        else: atomic_write_json(OUT/'public_bundle/subgroups.json',public)
        if scan_public_bundle(OUT/'public_bundle'): raise ValueError('PRIVACY_FAIL')
        print(json.dumps({'status':public['status'],'branches':{b:{k:v['status'] for k,v in reports[b]['cognition'].items()} for b in reports}}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['run','verify'],default='run')
    run(parser.parse_args().mode)
