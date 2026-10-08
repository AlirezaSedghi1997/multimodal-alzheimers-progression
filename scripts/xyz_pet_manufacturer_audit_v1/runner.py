"""Bounded sensitivity only; never resume or mutate the completed parent model."""
import importlib.metadata
import json
from pathlib import Path
import sys
from scripts.xyz_fusion_postall_v2.core import (Units,ReadOnlyParent,read_json,digest,file_sha,
    atomic_json,writer_lock,parent_exclusion)
from .resources import ResourceContext
from .protocol import make_plan
from .pipeline import run_group,validate_completed
from .reporting import build_report,check_public


def bind_existing(root,parent,postroot):
    pc=read_json(postroot/'private/contract.json')
    if pc['parent_contract']!=parent.binding:raise ValueError('POSTALL_PARENT_DRIFT')
    for contract in (pc,parent.execution_contract):
        for name,sha in contract['source_sha256'].items():
            if file_sha(root/name)!=sha:raise ValueError('UPSTREAM_SOURCE_DRIFT:'+name)
        for name,version in contract['runtime'].items():
            if importlib.metadata.version(name)!=version:raise ValueError('UPSTREAM_RUNTIME_DRIFT:'+name)
        if contract['python']!=sys.version:raise ValueError('UPSTREAM_PYTHON_DRIFT')
    post=Units(postroot,pc)
    for key in ('f00','f02'):
        if post.read(key) is None:raise ValueError('UPSTREAM_VERIFIED_INPUTS_REQUIRED:'+key)
    return post,pc


def execute(root,mode):
    root=Path(root).resolve()
    if root.as_posix().lower()!='h:/new project':raise ValueError('AUTHORITATIVE_H_ROOT_REQUIRED')
    options=read_json(root/'configs/xyz_pet_manufacturer_audit_v1.json')
    out=(root/options['output_root']).resolve();postroot=(root/options['postall_root']).resolve()
    parentroot=(root/options['parent_root']).resolve()
    if (not out.is_relative_to(root/'results') or any(out.is_relative_to(p) or p.is_relative_to(out)
            for p in (postroot,parentroot,root/'results/xyz_fusion_postall_v1'))):
        raise ValueError('SEPARATE_LMO_OUTPUT_REQUIRED')
    with writer_lock(out),parent_exclusion(postroot),parent_exclusion(parentroot):
        parent=ReadOnlyParent(parentroot);post,pc=bind_existing(root,parent,postroot)
        # Context is used read-only until redirected to this separate output root.
        ctx=ResourceContext(root,parent,post,pc['config']);ctx.resource_options=options
        print(json.dumps({'event':'LMO_INPUT_HASH_VERIFICATION_STARTED'}),flush=True)
        proof=ctx.load_inputs();plan=make_plan(ctx.pet,ctx.config,options)
        if options['components']!=ctx.config['components'] or options['training_seed']!=ctx.config['training_seeds'][0]:
            raise ValueError('LMO_FIXED_BUDGET_POLICY_DRIFT')
        code=list(Path(__file__).parent.glob('*.py'))+[root/'scripts/run_xyz_pet_manufacturer_audit_v1.py']
        contract={'options':options,'parent':parent.binding,'postall':post.binding,
            'upstream_freeze':digest(post.read('f00')),'inputs':proof,'plan':plan,
            'source_sha256':{p.relative_to(root).as_posix():file_sha(p) for p in code},
            'python':sys.version,'runtime':pc['runtime']}
        units=Units(out,contract);ctx.units=units;ctx.out=out
        def state(status,**kw):atomic_json(out/'orchestration_state.json',dict(status=status,contract=units.binding,**kw))
        try:
            if mode=='verify':
                if units.read('plan')!=plan:raise ValueError('LMO_PLAN_MISSING_OR_CHANGED')
                cells=validate_completed(ctx,plan,options)
                if units.read('report') is None:raise ValueError('LMO_REPORT_MISSING')
                check_public(out/'public');print(json.dumps({'status':'PASS_LMO_EXECUTION_NOT_EXTERNAL_CLAIM','groups':len(cells)}),flush=True)
                return 0
            units.job('plan',lambda:plan)
            if mode=='preflight':
                status='LMO_INPUTS_READY' if plan['supported'] else 'LMO_NOT_ESTIMABLE'
                state(status,expected_expert_fits=plan['expected_expert_fits'],long_training_started=False)
                print(json.dumps({'status':status,'groups':[{'manufacturer':x['group'],'n':x['n'],'events':x['events']} for x in plan['supported']],
                    'excluded':plan['excluded'],'expected_expert_fits':plan['expected_expert_fits']}),flush=True);return 0
            if mode!='all':raise ValueError('UNKNOWN_LMO_MODE')
            if any(units.read('held_'+g['group'].lower()+'_evaluation') is None for g in plan['supported']):
                from scripts.xyz_fusion_v1.runtime import gpu_guard
                gpu_guard()
            state('RUNNING_MANUFACTURER_SENSITIVITY',expected_expert_fits=plan['expected_expert_fits'])
            for group in plan['supported']:
                run_group(ctx,group,options)
                print(json.dumps({'event':'PET_MANUFACTURER_CELL_COMMITTED','manufacturer':group['group']}),flush=True)
            cells=validate_completed(ctx,plan,options)
            report=units.job('report',lambda:build_report(ctx,plan,cells,options),{'plan':digest(plan),'cells':digest(cells)})
            check_public(out/'public');state('COMPLETED_INTERNAL_MANUFACTURER_SENSITIVITY',claim_status=options['claim_status'])
            print(json.dumps({'status':report['status'],'claim_status':options['claim_status']}),flush=True);return 0
        except BaseException as exc:
            state('INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'BLOCKED',reason=type(exc).__name__+':'+str(exc));raise
        finally:ctx.close()
