import json
import numpy as np
import pandas as pd
import pytest

from scripts.xyz_fusion_postall_v2.core import Units, atomic_json, digest
from scripts.xyz_pet_manufacturer_audit_v1.protocol import validate_partition, select_inner_winner, blind
from scripts.xyz_pet_manufacturer_audit_v1.access import evaluated_once


def test_manufacturer_excluded_from_every_ancestor():
    rows=pd.DataFrame({'patient_hash':['a','b','c','d'], 'pet_manufacturer':['GE','SIEMENS','PHILIPS','GE']})
    validate_partition(rows, 'GE', ['b'], ['c'], ['a','d'])
    with pytest.raises(ValueError,match='MANUFACTURER'):
        validate_partition(rows, 'GE', ['a'], ['c'], ['d'])
    with pytest.raises(ValueError,match='COVERAGE'):
        validate_partition(rows, 'GE', ['b'], [], ['a','d'])


def test_winner_only_valid_inner_ibs():
    candidates=[{'lr':.003,'metric':.14,'epochs':5},{'lr':.001,'metric':.14,'epochs':6}]
    assert select_inner_winner(candidates)['lr']==.001
    for bad in [None,np.nan,-.1,1.1]:
        with pytest.raises(ValueError,match='INNER_IBS'):
            select_inner_winner([dict(candidates[0],metric=bad)])


def test_no_outcomes_cross_inference_boundary():
    frame=pd.DataFrame({'event':[1],'time_days':[100],'last_event':[0], 'alt_time_days':[120],
        'amyloid_centiloid':[50.], 'patient_hash':['a']})
    assert list(blind(frame))==['amyloid_centiloid','patient_hash']


def test_actual_evaluator_called_once_and_receipt_recovered(tmp_path):
    store=Units(tmp_path,{'fixture':True});calls=[]
    def evaluate():
        calls.append(1);return {'status':'OUTER_EVAL_COMMITTED','metric':.1}
    assert evaluated_once(store,'held_ge',{'lock':'sha'},evaluate)['metric']==.1
    assert evaluated_once(store,'held_ge',{'lock':'sha'},evaluate)['metric']==.1
    journal=tmp_path/'private/access/held_ge.json'
    doc=json.loads(journal.read_text());doc['state']='IN_PROGRESS'
    atomic_json(journal,doc)
    evaluated_once(store,'held_ge',{'lock':'sha'},evaluate)
    assert calls==[1]
    assert json.loads(journal.read_text())['access_count']==1
    with pytest.raises(ValueError,match='ACCESS'):
        evaluated_once(store,'held_ge',{'lock':'changed'},evaluate)


def test_interruption_inside_evaluator_never_reopens_test(tmp_path):
    store=Units(tmp_path,{'fixture':True});calls=[]
    def interrupt():calls.append(1);raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):evaluated_once(store,'test',{},interrupt)
    with pytest.raises(ValueError,match='INCOMPLETE_OUTER_ACCESS'):
        evaluated_once(store,'test',{},interrupt)
    assert calls==[1]


def test_result_written_before_unit_commit_recovered(tmp_path):
    store=Units(tmp_path,{'fixture':True});value={'status':'OUTER_EVAL_COMMITTED'}
    lock={'binding':store.binding,'dependencies':{}}
    atomic_json(tmp_path/'private/access/test.json',dict(lock,state='IN_PROGRESS',access_count=1))
    atomic_json(tmp_path/'private/access/test.result.json',dict(lock,payload=value,payload_sha256=digest(value)))
    assert evaluated_once(store,'test',{},lambda:pytest.fail('second access'))==value


def test_full_paired_group_pipeline_resume_and_ancestry(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import torch
    from scripts.xyz_pet_manufacturer_audit_v1 import pipeline as p
    from scripts.xyz_fusion_v2.protocol import CLINICAL
    rng=np.random.default_rng(91);n=360
    frame=pd.DataFrame({k:rng.normal(size=n) for k in CLINICAL+['amyloid_centiloid']})
    frame['patient_hash']=[str(i) for i in range(n)]
    frame['cache_row']=np.arange(n);frame['pet_manufacturer']=['GE']*120+['SIEMENS']*240
    frame['event']=np.tile([0,1,1],120)
    frame['time_days']=rng.uniform(25,1900,n)
    frame.loc[frame.event.eq(0),'time_days']=1900.
    group={'group':'GE','train':list(frame.patient_hash[120:]),'test':list(frame.patient_hash[:120]),'components':[]}
    for i in range(3):
        meta=list(frame.patient_hash[120+i*80:200+i*80]);pool=[k for k in group['train'] if k not in meta]
        group['components'].append({'component':i,'pool':pool,'meta':meta,'fit':pool[:120],'stop':pool[120:]})
    calls=[]
    def fake_train(bank,train,val,policy,grid,out,binding,lr,seed,fixed_epochs=None):
        from pathlib import Path
        from scripts.xyz_fusion_postall_v2.core import file_sha
        calls.append(1);assert not set(train.patient_hash)&set(group['test'])
        if val is not None:assert not set(val.patient_hash)&set(group['test'])
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        net=torch.nn.Linear(1,1)
        with torch.no_grad():net.weight.fill_(.1);net.bias.zero_()
        torch.save({'model':net.state_dict()},out/'locked.pt')
        receipt={'binding':binding,'sha256':file_sha(out/'locked.pt'),'best_epoch':2,
            'inner_ibs':.1 if val is not None else None,'terminal':True,'history':[],
            'train_subjects':sorted(train.patient_hash),'selection_subjects':[] if val is None else sorted(val.patient_hash)}
        atomic_json(out/'locked.json',receipt);return net,receipt
    def fake_predict(model,bank,rows,policy,device,grid):
        assert not any(c in rows for c in ['event','time_days'])
        rate=(.00015+.0001*(rows.cache_row.to_numpy()%7)/7)[:,None]
        return np.exp(-rate*np.asarray(grid)[None,:])
    monkeypatch.setattr(p,'train_expert',fake_train);monkeypatch.setattr(p,'predict',fake_predict)
    units=Units(tmp_path,{'fixture':True})
    config={'training':{'learning_rates':[.0001,.0003]},'fusion':{'ibs_grid_days':[365.,730.,1095.],
        'score_times_days':[365.,730.,1095.],'horizons_days':[730.],'min_at_risk':20,'ridge_alpha':10.}}
    ctx=SimpleNamespace(units=units,out=tmp_path,config=config,pet=frame,banks=lambda:{'mri':None,'pet':None})
    opts={'training_seed':13,'components':3}
    first=p.run_group(ctx,group,opts)
    assert len(calls)==18 and len(first['arms'])==2
    assert p.run_group(ctx,group,opts)==first and len(calls)==18
    assert p.validate_completed(ctx,{'supported':[group]},opts)==[first]
    broken=dict(group,components=group['components'][:2])
    with pytest.raises(ValueError):p.validate_completed(ctx,{'supported':[broken]},opts)


def test_final_chain_does_not_rerun_parent_training():
    from scripts.run_xyz_postfusion_final_v1 import steps
    assert steps('all')==[
        ('run_xyz_fusion_postall_v2.py','all'),
        ('run_xyz_pet_manufacturer_audit_v1.py','all'),
        ('run_xyz_regional_evidence_v1.py','all')]
    assert all(mode=='verify' for _,mode in steps('verify'))


def test_ram_policy_uses_existing_banks_once_with_safe_caps(tmp_path,monkeypatch):
    from scripts.xyz_pet_manufacturer_audit_v1.resources import ResourceContext
    from scripts.xyz_fusion_v2 import training,inference
    from types import SimpleNamespace
    calls=[]
    monkeypatch.setattr(training,'ImageBank',lambda p,cap:(calls.append(('mri',cap)) or SimpleNamespace(mode='RAM_ONCE')))
    monkeypatch.setattr(inference,'PETBank',lambda f,max_gib:(calls.append(('pet',max_gib)) or SimpleNamespace(mode='RAM_ONCE')))
    ctx=object.__new__(ResourceContext);ctx.root=tmp_path;ctx._banks=None;ctx.pet=object()
    ctx.resource_options={'mri_ram_gib':10,'pet_ram_gib':6};ctx.ensure_inputs=lambda:None
    assert ctx.banks() is ctx.banks()
    assert calls==[('mri',10),('pet',6)]
