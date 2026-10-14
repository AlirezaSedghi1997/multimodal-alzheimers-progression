"""Ancestry-safe cross-fitting of a fresh MRI expert, then low-dimensional survival fusion."""
from __future__ import annotations
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from .store import Store,atomic_json,digest,file_sha
from .mri_training import ImageBank,train_expert,predict
from .statistics import fit_cox,cox_curves,metrics,ibs_score


def partition(frame,folds,seed):
    if frame.patient_hash.duplicated().any():raise ValueError('DUPLICATE_FUSION_SUBJECT')
    strata=frame.event.astype(int).astype(str)+'_'+frame.treatment_active.astype(int).astype(str)
    if strata.value_counts().min()<folds:raise ValueError('INSUFFICIENT_EVENT_TREATMENT_STRATUM')
    return list(StratifiedKFold(folds,shuffle=True,random_state=seed).split(frame,strata))


def stop_partition(frame,seed):
    strata=frame.event.astype(int).astype(str)+'_'+frame.treatment_active.astype(int).astype(str)
    if strata.value_counts().min()<2:raise ValueError('INSUFFICIENT_STOPPING_STRATUM')
    a,b=next(StratifiedShuffleSplit(1,test_size=.2,random_state=seed).split(frame,strata))
    return frame.iloc[a].copy(),frame.iloc[b].copy()


def assert_disjoint(*groups):
    sets=[set(g) for g in groups]
    if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError('FUSION_ANCESTOR_OVERLAP')


def validate_nested_support(frame,config):
    """Cheap protocol feasibility, never selection of a more favorable horizon."""
    grid=config['fusion']['ibs_grid_days']; checked=0
    def check(a,b):
        nonlocal checked
        curves=np.tile(np.linspace(.95,.75,len(grid)),(len(b),1))
        ibs_score(a,b,curves,grid);checked+=1
    for outer,(a,b) in enumerate(partition(frame,config['outer_folds'],config['seed'])):
        train,held=frame.iloc[a],frame.iloc[b]
        check(train,held)
        check(train.loc[train.blood_ptau217.notna()],held.loc[held.blood_ptau217.notna()])
        for inner,(u,v) in enumerate(partition(train,config['inner_folds'],config['seed']+outer+1)):
            pool,evaluation=train.iloc[u],train.iloc[v]
            fit,stop=stop_partition(pool,config['seed']+100*outer+inner)
            assert_disjoint(fit.patient_hash,stop.patient_hash,evaluation.patient_hash,held.patient_hash)
            check(fit,stop);check(pool,evaluation)
    return {'status':'FIXED_GRID_AND_NESTED_PARTITIONS_SUPPORTED','comparisons_checked':checked,
            'synthetic_curves_for_support_check_only':True}


def save_pickle(path,value):
    import os
    path=Path(path);tmp=path.with_suffix('.tmp');path.parent.mkdir(parents=True,exist_ok=True)
    with tmp.open('wb') as f:pickle.dump(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def cox_lock(path,frame,features,alpha,parent):
    path=Path(path);receipt=path.with_suffix('.json')
    binding=digest({'parent':parent,'features':features,'alpha':alpha,
                    'rows':json.loads(frame[['patient_hash','event','time_days',*features]].to_json(orient='records'))})
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['binding']!=binding or file_sha(path)!=r['sha256']:raise ValueError('COX_LOCK_MISMATCH')
        with path.open('rb') as f:model=pickle.load(f)  # trusted local, hash-verified artifact only
    else:
        model=fit_cox(frame,features,alpha);save_pickle(path,model)
        atomic_json(receipt,{'binding':binding,'sha256':file_sha(path),'features':features,'ancestry':sorted(frame.patient_hash)})
    return model,json.loads(receipt.read_text())


def branch_features(config):
    c=config['fusion']['clinical_features']
    return {'clinical':c,'mri_only':['mri_risk'],
            'clinical_amyloid':c+['amyloid_centiloid'],
            'clinical_mri':c+['mri_risk'],
            'clinical_mri_amyloid':c+['mri_risk','amyloid_centiloid'],
            'blood_matched_clinical_mri':c+['mri_risk'],
            'blood_matched_clinical_mri_amyloid':c+['mri_risk','amyloid_centiloid'],
            'clinical_mri_blood':c+['mri_risk','blood_log_ptau217'],
            'clinical_mri_blood_amyloid':c+['mri_risk','blood_log_ptau217','amyloid_centiloid']}


def crossfit(frame,bank,config,out,store):
    import torch
    out=Path(out);policy=config['training'];fc=config['fusion'];grid=fc['ibs_grid_days']
    splits=partition(frame,config['outer_folds'],config['seed'])
    split_payload=[{'fold':i,'train':sorted(frame.iloc[a].patient_hash),'test':sorted(frame.iloc[b].patient_hash)}
                   for i,(a,b) in enumerate(splits)]
    if store.job('splits',lambda:split_payload)!=split_payload:
        raise ValueError('CROSSFIT_SPLIT_CONTRACT_MISMATCH')
    done=[]
    for outer,(a,b) in enumerate(splits):
        complete=out/'private/jobs'/f'outer_{outer}.json'
        if complete.exists():
            done.append(store.job(f'outer_{outer}'));continue
        train=frame.iloc[a].copy();held=frame.iloc[b].copy()
        assert_disjoint(train.patient_hash,held.patient_hash)
        oof=np.full(len(train),np.nan);all_candidates=[]
        for inner,(u,v) in enumerate(partition(train,config['inner_folds'],config['seed']+outer+1)):
            pool=train.iloc[u].copy();evaluation=train.iloc[v].copy()
            fit,validation=stop_partition(pool,config['seed']+100*outer+inner)
            assert_disjoint(fit.patient_hash,validation.patient_hash,evaluation.patient_hash,held.patient_hash)
            candidates=[]
            for candidate,lr in enumerate(policy['learning_rates']):
                task=f'outer{outer}_inner{inner}_lr{candidate}_tune'
                model,r=train_expert(bank,fit,validation,policy,grid,out/'private/models'/task,store.binding,lr,
                                    config['seed']+1000*outer+100*inner+candidate)
                candidates.append({'candidate':candidate,'lr':lr,'metric':r['inner_ibs'],'epochs':r['best_epoch'],
                                   'checkpoint_sha256':r['sha256']})
                del model
            winner=min(candidates,key=lambda r:(r['metric'],r['candidate']))
            all_candidates.append(candidates)
            store.job(f'winner_o{outer}_i{inner}',lambda:{'winner':winner,'candidates':candidates,
                'selection_scope':'INNER_TRAIN_STOPPING_SUBSET_ONLY','never_seen':sorted(evaluation.patient_hash),
                'metric_name':'IPCW_INTEGRATED_BRIER_SCORE'})
            # Refit on the inner train pool. The inner heldout never selected LR or epoch.
            task=f'outer{outer}_inner{inner}_crossfit'
            model,r=train_expert(bank,pool,None,policy,grid,out/'private/models'/task,store.binding,winner['lr'],
                                 config['seed']+9000+100*outer+inner,fixed_epochs=winner['epochs'])
            def make_oof():
                curves=predict(model,bank,evaluation,policy,next(model.parameters()).device,grid)
                return {'subjects':list(evaluation.patient_hash),'mri_risk':(1-curves).mean(axis=1).tolist(),
                        'curves':curves.tolist(),'model_sha256':r['sha256'],
                        'ancestry':sorted(pool.patient_hash),'outer':outer,'inner':inner}
            record=store.job(f'mri_oof_o{outer}_i{inner}',make_oof)
            if record['subjects']!=list(evaluation.patient_hash) or set(record['ancestry'])&set(record['subjects']):
                raise ValueError('CROSSFIT_ANCESTRY_OR_ORDER_MISMATCH')
            oof[v]=record['mri_risk'];del model
        if not np.isfinite(oof).all():raise ValueError('INCOMPLETE_MRI_CROSSFIT')
        train['mri_risk']=oof
        # This outer model uses only outer-train stopping subsets for selection.
        rank=[float(np.mean([c[j]['metric'] for c in all_candidates])) for j in range(len(policy['learning_rates']))]
        best=int(np.argmin(rank));epochs=max(1,int(np.median([c[best]['epochs'] for c in all_candidates])))
        task=f'outer{outer}_final_mri'
        model,r=train_expert(bank,train,None,policy,grid,out/'private/models'/task,store.binding,policy['learning_rates'][best],
                            config['seed']+20000+outer,fixed_epochs=epochs)
        arm_models={};lock_meta={}
        for arm,features in branch_features(config).items():
            subset=train.loc[train.blood_ptau217.notna()].copy() if 'blood' in arm else train.copy()
            if 'blood' in arm:subset['blood_log_ptau217']=np.log(subset.blood_ptau217)
            arm_models[arm],lock_meta[arm]=cox_lock(out/'private/models'/f'outer{outer}_{arm}.pkl',subset,features,fc['ridge_alpha'],store.binding)
        store.job(f'outer_lock_{outer}',lambda:{'mri':r['sha256'],'cox':lock_meta,'lr':policy['learning_rates'][best],
                  'epochs':epochs,'ancestry':sorted(train.patient_hash),'forbidden':sorted(held.patient_hash)})
        # Prediction materialization is chunk-committed; outcomes are not passed to the model.
        rows=[]
        predictors=held.drop(columns=['event','time_days']).copy()
        for chunk,start in enumerate(range(0,len(predictors),16)):
            data=predictors.iloc[start:start+16].copy()
            def infer():
                journal=out/'private/access_journal'/f'outer_{outer}_chunk_{chunk}.json'
                old=json.loads(journal.read_text()) if journal.exists() else {'attempts':0}
                atomic_json(journal,{'attempts':old['attempts']+1,'contract':store.binding,
                    'model_sha256':r['sha256'],'subjects':list(data.patient_hash),
                    'state':'INFERENCE_IN_PROGRESS','outcomes_given_to_model':False,
                    'replay_policy':'ONLY_UNCOMMITTED_DETERMINISTIC_CHUNK_SAME_LOCKED_MODELS'})
                curves=predict(model,bank,data,policy,next(model.parameters()).device,grid)
                values=data.copy();values['mri_risk']=(1-curves).mean(axis=1)
                payload={'subjects':list(data.patient_hash),'mri_risk':values.mri_risk.tolist(),'arms':{},
                         'model_sha256':r['sha256'],'outer_lock_sha256':digest(store.job(f'outer_lock_{outer}'))}
                for arm,features in branch_features(config).items():
                    select=values.loc[values.blood_ptau217.notna()].copy() if 'blood' in arm else values.copy()
                    if select.empty:continue
                    if 'blood' in arm:select['blood_log_ptau217']=np.log(select.blood_ptau217)
                    payload['arms'][arm]={'subjects':list(select.patient_hash),
                                         'survival':cox_curves(arm_models[arm],select,features,grid).tolist()}
                return payload
            rows.append(store.job(f'outer_predictions_{outer}_{chunk}',infer))
            journal=out/'private/access_journal'/f'outer_{outer}_chunk_{chunk}.json'
            if not journal.exists():raise ValueError('OUTER_INFERENCE_JOURNAL_MISSING')
            audit=json.loads(journal.read_text())
            if audit['contract']!=store.binding or audit['model_sha256']!=r['sha256']:
                raise ValueError('OUTER_INFERENCE_JOURNAL_MISMATCH')
            audit.update(state='PREDICTIONS_COMMITTED',result_sha256=digest(rows[-1]))
            atomic_json(journal,audit)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
        def evaluate_committed_predictions():
            arms={}
            for arm in branch_features(config):
                ids=[];curves=[]
                for row in rows:
                    cell=row['arms'].get(arm)
                    if cell:ids+=cell['subjects'];curves+=cell['survival']
                if len(ids)!=len(set(ids)):raise ValueError('DUPLICATE_OUTER_PREDICTION')
                subset=held.set_index('patient_hash').loc[ids].reset_index()
                source=train.loc[train.blood_ptau217.notna()].copy() if 'blood' in arm else train
                arms[arm]={'metrics':metrics(source,subset,curves,grid,fc['horizons_days'],fc['min_at_risk']),
                           'subjects':ids,'curves':curves,
                           'event':subset.event.astype(int).tolist(),'time':subset.time_days.tolist(),
                           'censor_train':{'event':source.event.astype(int).tolist(),'time':source.time_days.tolist()}}
            return {'fold':outer,'status':'OUTER_EVAL_COMMITTED','arms':arms,
                    'test_subjects':list(held.patient_hash),'train_subjects':list(train.patient_hash)}
        done.append(store.job(f'outer_{outer}',evaluate_committed_predictions))
        print(json.dumps({'event':'OUTER_FOLD_COMMITTED','fold':outer,'completed':len(done),'total':5}),flush=True)
    if len(done)!=5 or {r['fold'] for r in done}!=set(range(5)):
        raise ValueError('INCOMPLETE_FIVE_FOLD_FUSION')
    return done


def verify_bound_outputs(out,store,config):
    """Verify actual model/cache bytes, not only report/state prose."""
    out=Path(out)
    for path in sorted((out/'private/jobs').glob('*.json')):
        store.job(path.stem)
    cache=json.loads((out/'private/mri_cache.json').read_text())
    if cache!=store.job('cache'):raise ValueError('FINAL_CACHE_RECEIPT_NOT_BOUND')
    if file_sha(out/'private/mri_cache.npy')!=cache['sha256']:
        raise ValueError('FINAL_MRI_CACHE_HASH_MISMATCH')
    for key,sha in cache.get('qc_receipt_sha256',{}).items():
        if file_sha(out/'private/mri_scans'/key/'receipt.json')!=sha:
            raise ValueError('FINAL_MRI_QC_RECEIPT_HASH_MISMATCH')
    expected=[];splits=store.job('splits');seen=set()
    for outer in range(config['outer_folds']):
        cell=store.job(f'outer_{outer}');outer_lock=store.job(f'outer_lock_{outer}')
        tr=set(cell['train_subjects']);te=set(cell['test_subjects'])
        if (cell['status']!='OUTER_EVAL_COMMITTED' or tr!=set(splits[outer]['train'])
                or te!=set(splits[outer]['test']) or seen&te or tr&te):
            raise ValueError('FINAL_OUTER_MEMBERSHIP_INVALID')
        seen|=te
        if outer_lock['ancestry']!=sorted(tr) or outer_lock['forbidden']!=sorted(te):
            raise ValueError('FINAL_OUTER_LOCK_ANCESTRY_MISMATCH')
        inner_seen=set()
        expected.append(f'outer{outer}_final_mri')
        for inner in range(config['inner_folds']):
            expected.append(f'outer{outer}_inner{inner}_crossfit')
            expected.extend(f'outer{outer}_inner{inner}_lr{j}_tune' for j in range(len(config['training']['learning_rates'])))
            w=store.job(f'winner_o{outer}_i{inner}')
            if w['winner']!=min(w['candidates'],key=lambda r:(r['metric'],r['candidate'])):
                raise ValueError('FINAL_INNER_WINNER_NOT_MINIMUM_IBS')
            oof=store.job(f'mri_oof_o{outer}_i{inner}')
            outer_test=store.job(f'outer_{outer}')['test_subjects']
            assert_disjoint(oof['ancestry'],oof['subjects'],outer_test)
            if inner_seen&set(oof['subjects']) or set(oof['ancestry'])|set(oof['subjects'])!=tr:
                raise ValueError('FINAL_INNER_COVERAGE_INVALID')
            inner_seen.update(oof['subjects'])
            cross_r=json.loads((out/'private/models'/f'outer{outer}_inner{inner}_crossfit/locked.json').read_text())
            if oof['model_sha256']!=cross_r['sha256'] or cross_r['train_subjects']!=oof['ancestry']:
                raise ValueError('FINAL_OOF_MODEL_LINK_MISMATCH')
            for candidate in w['candidates']:
                cr=json.loads((out/'private/models'/f'outer{outer}_inner{inner}_lr{candidate["candidate"]}_tune/locked.json').read_text())
                if cr['sha256']!=candidate['checkpoint_sha256'] or cr['inner_ibs']!=candidate['metric']:
                    raise ValueError('FINAL_HPO_SCORE_NOT_MODEL_BOUND')
                if set(cr['train_subjects'])|set(cr['selection_subjects'])!=set(oof['ancestry']):
                    raise ValueError('FINAL_HPO_ANCESTRY_INVALID')
                assert_disjoint(cr['train_subjects'],cr['selection_subjects'],oof['subjects'],outer_test)
        if inner_seen!=tr:raise ValueError('FINAL_INNER_COVERAGE_INCOMPLETE')
        fr=json.loads((out/'private/models'/f'outer{outer}_final_mri/locked.json').read_text())
        if fr['sha256']!=outer_lock['mri'] or fr['train_subjects']!=sorted(tr) or fr['selection_subjects']:
            raise ValueError('FINAL_MRI_ANCESTRY_MISMATCH')
        predicted=[]
        for chunk in range((len(te)+15)//16):
            p=store.job(f'outer_predictions_{outer}_{chunk}');predicted+=p['subjects']
            j=json.loads((out/'private/access_journal'/f'outer_{outer}_chunk_{chunk}.json').read_text())
            if (j['state']!='PREDICTIONS_COMMITTED' or j['result_sha256']!=digest(p)
                    or p['outer_lock_sha256']!=digest(outer_lock) or p['model_sha256']!=fr['sha256']):
                raise ValueError('FINAL_PREDICTION_MODEL_OR_JOURNAL_MISMATCH')
        if len(predicted)!=len(te) or set(predicted)!=te:raise ValueError('FINAL_PREDICTIONS_INCOMPLETE')
        if set(cell['arms'])!=set(branch_features(config)):raise ValueError('FINAL_ARMS_INCOMPLETE')
    actual={p.parent.name for p in (out/'private/models').rglob('locked.json')}
    if actual!=set(expected):raise ValueError('FINAL_MRI_MODELS_INCOMPLETE_OR_EXTRA')
    checked=0
    for task in expected:
        receipt=out/'private/models'/task/'locked.json'
        r=json.loads(receipt.read_text())
        if r.get('parent_contract')!=store.binding or not r['terminal']:
            raise ValueError('FINAL_MRI_MODEL_CONTRACT_MISMATCH')
        if file_sha(receipt.with_name('locked.pt'))!=r['sha256']:
            raise ValueError('FINAL_MRI_MODEL_HASH_MISMATCH')
        if task.endswith('_tune') and (r['inner_ibs'] is None or not 0<=r['inner_ibs']<=1):
            raise ValueError('FINAL_INNER_IBS_INVALID')
        checked+=1
    for outer in range(config['outer_folds']):
      for arm in branch_features(config):
        receipt=out/'private/models'/f'outer{outer}_{arm}.json'
        r=json.loads(receipt.read_text())
        if file_sha(receipt.with_suffix('.pkl'))!=r['sha256']:
            raise ValueError('FINAL_COX_MODEL_HASH_MISMATCH')
        lock=store.job(f'outer_lock_{outer}')
        if r!=lock['cox'][arm] or not set(r['ancestry']).issubset(lock['ancestry']):
            raise ValueError('FINAL_COX_ANCESTRY_MISMATCH')
        checked+=1
    return {'model_artifacts_verified':checked,'cache_sha256':cache['sha256']}
