"""One manual command, resumable units, no production work in preflight/verify."""
from __future__ import annotations
import importlib.metadata
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
from scripts.xyz_fusion_v1.store import writer_lock, atomic_json, digest, file_sha
from .store import Store
from scripts.xyz_fusion_v1.runtime import gpu_guard
from .data import build_data, records, with_endpoint, load_tau
from .protocol import validate_config, CLAIM, CLINICAL, VOLUMES
from .review import historical_diagnostics, six_sources


def contract(root, config, ctx, frame, inputs, plan):
    import torch
    sources=[]
    for d in ['scripts/xyz_fusion_v2','scripts/xyz_fusion_v1','scripts/xyz_postprimary_v1']:
        sources+=sorted((root/d).glob('*.py'))
    sources += [root/'scripts/run_xyz_fusion_v2.py',root/'scripts/run_xyz_fusion_v2.ps1',
                root/'src/models/backbones/amuse_x_hybrid3d.py',root/'src/models/backbones/deterministic_pool.py',
                root/'src/evaluation/signed_spatial_validation_v1.py',root/'src/method_y_revival/privacy.py',
                root/'src/utils/privacy.py',root/'src/utils/pet_path_resolver.py']
    return {'protocol':config,'source_sha256':{p.relative_to(root).as_posix():file_sha(p) for p in sources},
            'input_sha256':inputs,'parent_cache':ctx['cache'],'manifest':records(frame),
            'parent_contract':ctx['store'].binding,'parent_splits':ctx['splits'],'analysis_plan':plan,'review_sources':six_sources(),
            'runtime':{p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy','scikit-learn','scikit-survival','statsmodels','nibabel','h5py','antspyx']},
            'torch_cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'python':sys.version,'policy':'DETERMINISTIC_AMP_NO_TF32','historical_outcome_exposure':True}


from .verification import verify


def execute(root, mode, config_path):
    root=Path(root);config=json.loads(Path(config_path).read_text(encoding='utf-8-sig'));validate_config(config)
    out=(root/config['output_root']).resolve()
    if not out.is_relative_to((root/'results').resolve()) or out==root/config['parent_root']:
        raise ValueError('NEW_VERSIONED_H_OUTPUT_REQUIRED')
    ctx,frame,endpoints,inputs,summary=build_data(root,config)
    from .feasibility import support,optional_support
    from .inference import pet_intake,PETBank
    print(json.dumps({'event':'V2_INTAKE_VERIFIED','subjects':len(frame),'events':int(frame.event.sum())}),flush=True)
    support(frame,ctx['splits'],config,'mri',image=True)
    plan={'tabular':{}}
    for endpoint in [config['endpoint']['primary'],*config['endpoint']['sensitivity']]:
        p=optional_support(with_endpoint(frame,endpoint),ctx['splits'],config,'scalar')
        if endpoint==config['endpoint']['primary'] and p['status']!='SUPPORTED':
            raise ValueError('PRIMARY_SCALAR_SUPPORT_FAILED:'+p['reason'])
        families=config['secondary']['families'] if endpoint==config['endpoint']['primary'] else config['secondary']['endpoint_sensitivity_families']
        plan['tabular']['scalar_'+endpoint]={'support':p,'families':families,'track':'scalar','endpoint':endpoint}
    tau,tau_audit=load_tau(root,ctx,frame,config)
    plan['tabular']['tau']={'support':optional_support(tau,ctx['splits'],config,'tau'),
            'track':'tau','families':config['secondary']['families'],'endpoint':config['endpoint']['primary']}
    pet,pet_audit=pet_intake(ctx,frame)
    plan['pet']=optional_support(pet,ctx['splits'],config,'pet',image=True)
    inputs.update({'pet_membership':pet_audit['membership_sha256'],**{'pet_cache:'+k:v for k,v in pet_audit['hdf5_sha256'].items()}})
    inputs['pet_manifest']=file_sha(root/'data/preprocessed_pet/_private/manifests/fbp_private_manifest.csv')
    from .atlas import source_assets
    atlas_assets=source_assets(root)
    inputs.update({'atlas:'+k:v for k,v in atlas_assets['files'].items()})
    plan['tau_features_sha256']=digest(records(tau));plan['pet_membership_sha256']=digest(records(pet))
    print(json.dumps({'event':'V2_ANALYSIS_SUPPORT_CHECKED','plan':plan}),flush=True)
    binding=contract(root,config,ctx,frame,inputs,plan)
    if mode=='preflight':
        from scripts.xyz_fusion_v1.statistics import ibs_score
        for split in ctx['splits']:
            a=frame.loc[frame.patient_hash.isin(split['train'])];b=frame.loc[frame.patient_hash.isin(split['test'])]
            ibs_score(a,b,np.tile(np.linspace(.97,.8,9),(len(b),1)),config['fusion']['ibs_grid_days'])
        audit=historical_diagnostics(ctx)
        path=root/'results/xyz_fusion_v2_checks'
        atomic_json(path/'preflight.json',{'status':'READINESS_CHECKED_NOT_PRODUCTION_COMPLETION','summary':summary,'analysis_plan':plan,
                    'contract_sha256':digest(binding),'production_training_started':False})
        atomic_json(path/'six_reviewer_adjudication.json',audit)
        print(json.dumps({'status':'PREFLIGHT_READY','summary':summary}),flush=True)
        return 0
    with writer_lock(out):
        store=Store(out,binding)
        state=out/'orchestration_state.json'
        terminal=state.exists() and json.loads(state.read_text()).get('stage')=='FINALIZED'
        if mode=='verify' or terminal:
            print(json.dumps(verify(store,config,final=True)),flush=True);return 0
        verify(store,config,final=False)
        # Every stage's input is receipt-bound. Parent source/cache are never modified.
        def stage(name):atomic_json(state,{'stage':name,'contract':store.binding,'claim_status':CLAIM})
        store.job('intake',lambda:{'summary':summary,'endpoint_rows':endpoints,'review':historical_diagnostics(ctx)})
        if file_sha(ctx['parent']/'private/mri_cache.npy')!=ctx['cache']['sha256']:
            raise ValueError('PARENT_CACHE_BYTES_CHANGED')
        store.job('analysis_plan',lambda:plan);store.job('splits',lambda:ctx['splits']);stage('INPUTS_VERIFIED')
        from .tabular import run_tabular
        from .reporting import summarize,write_report
        from .sensitivity import missingness,ph_diagnostic,scanner_splits,power_simulation,longitudinal_pacc
        reports={};diagnostics={'intake':summary,'analysis_support':plan}
        # Cheap tabular controls precede GPU experiments.
        for endpoint in [config['endpoint']['primary'],*config['endpoint']['sensitivity']]:
            spec=plan['tabular']['scalar_'+endpoint]
            if spec['support']['status']!='SUPPORTED':continue
            value=with_endpoint(frame,endpoint)
            output=run_tabular(value,config,store,'scalar_'+endpoint,'scalar',ctx['splits'],spec['families'])
            for family,cells in output.items():
                key=f'scalar_{endpoint}_{family}'
                reports[key]=store.job(key+'_report',lambda:summarize(cells,config,'scalar'))
            if endpoint==config['endpoint']['primary']:
                diagnostics['blood_selection']=store.job('blood_selection',lambda:missingness(value,output['cox_ridge'],config))
        diagnostics['ph']={}
        for i,split in enumerate(ctx['splits']):
            t=frame.loc[frame.patient_hash.isin(split['train'])]
            diagnostics['ph'][str(i)]=store.job(f'ph_o{i}',lambda:ph_diagnostic(t,CLINICAL+['amyloid_centiloid']+VOLUMES))
        diagnostics['pacc']=store.job('pacc',lambda:longitudinal_pacc(frame))
        stage('TABULAR_AND_ENDPOINT_SENSITIVITY_DONE')
        gpu_guard()
        from .training import ImageBank
        from .pipeline import paired_crossfit,ensemble_seeds
        bank=ImageBank(ctx['parent']/'private/mri_cache.npy',config['training']['ram_cache_max_gib'])
        runs=paired_crossfit(frame,{'mri':bank},config,store,'mri',ctx['splits'])
        cells=store.job('mri_ensemble_cells',lambda:ensemble_seeds(runs,frame,config))
        reports['mri']=store.job('mri_report',lambda:summarize(cells,config,'mri',runs))
        from .subgroups import subgroup_report
        diagnostics['mri_subgroups']=store.job('mri_subgroups',lambda:subgroup_report(frame,cells,config,'mri'))
        stage('MRI_FUSION_DONE')
        # Tables-only tau extension preserves its exact source scale and current landmark.
        diagnostics['tau_feasibility']=tau_audit
        if plan['tabular']['tau']['support']['status']=='SUPPORTED':
            diagnostics['tau_power']=store.job('tau_power',lambda:power_simulation(len(tau),float(tau.event.mean()),config))
            outputs=run_tabular(tau,config,store,'tau','tau',ctx['splits'])
            for family,cs in outputs.items():reports['tau_'+family]=store.job('tau_'+family+'_report',lambda:summarize(cs,config,'tau'))
        else:diagnostics['tau_status']='NOT_ESTIMABLE_EXACT_MAPPING_OR_SUPPORT'
        diagnostics['pet_feasibility']=pet_audit
        if plan['pet']['status']=='SUPPORTED':
            petbank=PETBank(pet)
            try:
                runs_pet=paired_crossfit(pet,{'mri':bank,'pet':petbank},config,store,'pet',ctx['splits'])
                pet_cells=store.job('pet_ensemble_cells',lambda:ensemble_seeds(runs_pet,pet,config))
                reports['pet']=store.job('pet_report',lambda:summarize(pet_cells,config,'pet',runs_pet))
                diagnostics['pet_subgroups']=store.job('pet_subgroups',lambda:subgroup_report(pet,pet_cells,config,'pet'))
            finally:petbank.close()
        else:diagnostics['pet_status']='NOT_ESTIMABLE_EXACT_LINKAGE_OR_SUPPORT'
        stage('PET_TAU_DONE')
        from .robustness import run_robustness
        diagnostics['scanner']=store.job('scanner_diagnostics',lambda:run_robustness(frame,bank,config,store))
        from .spatial import run_spatial
        diagnostics['spatial']=store.job('spatial_diagnostics',lambda:run_spatial(ctx,frame,bank,config,store,atlas_assets))
        store.job('diagnostics',lambda:diagnostics)
        stage('ROBUSTNESS_AND_SPATIAL_DONE')
        store.job('release',lambda:write_report(store,reports,diagnostics))
        stage('VERIFYING_RELEASE')
        verified=verify(store,config,final=True,allow_pending_state=True)
        stage('FINALIZED')
        print(json.dumps(verified),flush=True)
        return 0
