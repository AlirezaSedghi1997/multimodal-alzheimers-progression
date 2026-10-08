"""Fresh paired MRI/PET experts and two matched heads for each held manufacturer."""
from pathlib import Path
import numpy as np
import torch
from scripts.xyz_fusion_postall_v2.core import digest, file_sha
from scripts.xyz_fusion_postall_v2.statistics import evaluate_cell
from scripts.xyz_fusion_v2.training import train_expert, predict
from scripts.xyz_fusion_v2.pipeline import locked_tabular, average_predictions
from scripts.xyz_fusion_v2.models import score_features, assert_pair, curves
from .protocol import ARMS, blind, select_inner_winner, validate_partition
from .access import evaluated_once, verify_access


def rows(frame,subjects):
    return frame.set_index('patient_hash').loc[subjects].reset_index()


def expert(store,task,bank,train,stop,policy,config,seed,lr,fixed=None):
    path=store.root/'private/models'/task
    model,receipt=train_expert(bank,train,stop,policy,config['fusion']['ibs_grid_days'],path,
                              store.binding,lr,seed,fixed_epochs=fixed)
    outputs={p.relative_to(store.root).as_posix():file_sha(p) for p in (path/'locked.pt',path/'locked.json')}
    record=store.job(task+'_receipt',lambda:dict(receipt,outputs=outputs))
    if {k:v for k,v in record.items() if k!='outputs'}!=receipt or record['outputs']!=outputs:
        raise ValueError('LMO_EXPERT_RECEIPT_DRIFT')
    return model,record


def run_group(ctx,group,options):
    store=ctx.units;config=ctx.config;grid=config['fusion']['ibs_grid_days']
    prefix='held_'+group['group'].lower();seed=options['training_seed']
    dependency={'group':digest(group)}
    if store.read(prefix+'_evaluation',dependency) is not None:
        return verify_access(store,prefix+'_evaluation',dependency)
    frame=ctx.pet;train=rows(frame,group['train']);held=rows(frame,group['test'])
    components=[]
    for specification in group['components']:
        i=specification['component'];task=prefix+f'_c{i}'
        pool=rows(frame,specification['pool']);meta=rows(frame,specification['meta'])
        validate_partition(frame,group['group'],pool.patient_hash,meta.patient_hash,held.patient_hash)
        saved=store.read(task+'_predictions',dependency)
        if saved is not None:
            if saved['lock_sha256']!=digest(store.job(task+'_lock')):raise ValueError('LMO_PREDICTION_LOCK_DRIFT')
            components.append(saved['arms']);continue
        fit=rows(frame,specification['fit']);stop=rows(frame,specification['stop'])
        models={};receipts={};meta_aug=meta.copy()
        for modality in ('mri','pet'):
            policy=dict(config['training'],modality=modality);bank=ctx.banks()[modality]
            candidates=[]
            for j,lr in enumerate(policy['learning_rates']):
                model,r=expert(store,f'{task}_{modality}_tune{j}',bank,fit,stop,policy,config,seed+i*100+j,lr)
                candidates.append({'lr':lr,'metric':r['inner_ibs'],'epochs':r['best_epoch'],'sha256':r['sha256']})
                del model
            winner=select_inner_winner(candidates)
            selection={'winner':winner,'candidates':candidates,'selection_subjects':list(stop.patient_hash),
                       'held_manufacturer_absent':True,'metric_name':'IPCW_INTEGRATED_BRIER_SCORE'}
            if store.job(f'{task}_{modality}_winner',lambda:selection)!=selection:raise ValueError('LMO_SELECTION_DRIFT')
            model,receipt=expert(store,f'{task}_{modality}_expert',bank,pool,None,policy,config,
                                 seed+50000+i,winner['lr'],fixed=winner['epochs'])
            assert_pair(receipt,meta.patient_hash,held.patient_hash)
            def meta_predict():
                p=predict(model,bank,blind(meta),policy,next(model.parameters()).device,grid)
                return {'subjects':list(meta.patient_hash),'survival':p.tolist(),'checkpoint':receipt['sha256']}
            value=store.job(f'{task}_{modality}_meta',meta_predict)
            if value['subjects']!=list(meta.patient_hash) or value['checkpoint']!=receipt['sha256']:
                raise ValueError('LMO_META_CHECKPOINT_OR_ROWS_DRIFT')
            meta_aug=score_features(meta_aug,value['survival'],grid,config['fusion']['score_times_days'],modality)
            models[modality]=model;receipts[modality]=receipt
        heads={};locks={}
        for arm,features in ARMS.items():
            heads[arm],locks[arm]=locked_tabular(store,task+'_'+arm,meta_aug,features,'cox_ridge',
                                                config['fusion']['ridge_alpha'],config,seed)
        lock={'experts':receipts,'heads':locks,'pool':list(pool.patient_hash),'meta':list(meta.patient_hash),
              'held':list(held.patient_hash),'held_manufacturer':group['group'],
              'whole_training_selection_ancestry_excluded':True,'parent_checkpoint_reused':False}
        if store.job(task+'_lock',lambda:lock)!=lock:raise ValueError('LMO_LOCK_DRIFT')
        def infer():
            values=blind(held)
            for modality,model in models.items():
                p=predict(model,ctx.banks()[modality],values,dict(config['training'],modality=modality),
                          next(model.parameters()).device,grid)
                values=score_features(values,p,grid,config['fusion']['score_times_days'],modality)
            return {'arms':{a:{'subjects':list(values.patient_hash),'survival':curves(heads[a],values,f,grid).tolist()}
                            for a,f in ARMS.items()},'lock_sha256':digest(lock),'outcomes_accessed':False}
        components.append(store.job(task+'_predictions',infer,dependency)['arms'])
        del models,heads,model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    combined=average_predictions(components)
    # Outcomes first enter evaluation INSIDE the once-only closure.
    return evaluated_once(store,prefix+'_evaluation',dependency,
        lambda:evaluate_cell(train,held,combined,config,group['group']))


def validate_completed(ctx,plan,options):
    cells=[]
    for group in plan['supported']:
        prefix='held_'+group['group'].lower();deps={'group':digest(group)}
        cell=verify_access(ctx.units,prefix+'_evaluation',deps)
        if (set(cell['arms'])!=set(ARMS) or cell['test_subjects']!=sorted(group['test']) or
                cell['train_subjects']!=sorted(group['train']) or cell['status']!='OUTER_EVAL_COMMITTED'):
            raise ValueError('LMO_ARM_OR_HELD_COVERAGE')
        component_predictions=[]
        for spec in group['components']:
            task=prefix+f'_c{spec["component"]}';lock=ctx.units.job(task+'_lock')
            validate_partition(ctx.pet,group['group'],lock['pool'],lock['meta'],lock['held'])
            if lock['pool']!=spec['pool'] or lock['meta']!=spec['meta']:raise ValueError('LMO_SPLIT_DRIFT')
            for modality in ('mri','pet'):
                r=ctx.units.job(f'{task}_{modality}_expert_receipt')
                assert_pair(r,lock['meta'],lock['held'])
                if set(r['train_subjects'])!=set(lock['pool']) or r['selection_subjects']:
                    raise ValueError('LMO_REFIT_ANCESTRY_DRIFT')
                winner=ctx.units.job(f'{task}_{modality}_winner')
                if winner['winner']!=select_inner_winner(winner['candidates']):raise ValueError('LMO_WINNER_DRIFT')
                for j in range(len(ctx.config['training']['learning_rates'])):
                    tuning=ctx.units.job(f'{task}_{modality}_tune{j}_receipt')
                    if set(tuning['train_subjects'])!=set(spec['fit']) or set(tuning['selection_subjects'])!=set(spec['stop']):
                        raise ValueError('LMO_HPO_ANCESTRY_DRIFT')
            for arm in ARMS:
                head=ctx.units.job(task+'_'+arm)
                if head!=lock['heads'][arm] or set(head['ancestry'])!=set(spec['meta']):
                    raise ValueError('LMO_HEAD_ANCESTRY_DRIFT')
            predictions=ctx.units.job(task+'_predictions',dependencies=deps)
            if predictions['lock_sha256']!=digest(lock) or predictions.get('outcomes_accessed') is not False:
                raise ValueError('LMO_PREDICTION_BINDING')
            component_predictions.append(predictions['arms'])
        if len(component_predictions)!=options['components']:raise ValueError('LMO_COMPONENT_COVERAGE')
        average=average_predictions(component_predictions)
        for arm in ARMS:
            if (cell['arms'][arm]['subjects']!=average[arm]['subjects'] or
                    not np.array_equal(cell['arms'][arm]['curves'],average[arm]['survival'])):
                raise ValueError('LMO_ENSEMBLE_RESULT_MISMATCH')
        from scripts.xyz_fusion_postall_v2.audit import metric_replay
        metric_replay([cell],ctx.config)
        cells.append(cell)
    for path in (ctx.out/'private/units').glob('*.json'):ctx.units.read(path.stem)
    return cells
