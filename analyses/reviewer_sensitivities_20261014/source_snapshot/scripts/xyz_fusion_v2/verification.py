"""Completion is derived from verified units, memberships and prediction ancestry."""
import io
import json
from contextlib import redirect_stdout
from scripts.xyz_fusion_v1.store import digest, file_sha
from .store import Store, checked_job
from .protocol import CLAIM, arms
from .models import assert_pair


def verify_integrity(store,require_complete=False):
    count=0
    for path in sorted((store.root/'private/jobs').glob('*.json')):
        with redirect_stdout(io.StringIO()):p=checked_job(store,path.stem)
        count+=1
        if isinstance(p,dict) and 'path' in p and 'sha256' in p:
            check_file(store,p['path'],p['sha256'])
        if isinstance(p,dict) and 'files' in p and path.stem!='release':
            for relative,sha in p['files'].items():check_file(store,relative,sha)
    for path in (store.root/'private/models').rglob('locked.json'):
        p=json.loads(path.read_text())
        if p['parent_contract']!=store.binding or file_sha(path.with_name('locked.pt'))!=p['sha256']:
            raise ValueError('IMAGE_ARTIFACT_HASH_MISMATCH')
    for path in (store.root/'private/access_journal').glob('*.json'):
        j=json.loads(path.read_text());name=path.stem
        if j['state']!='COMMITTED':
            if require_complete:raise ValueError('INCOMPLETE_PREDICTION_UNIT:'+name)
            continue
        p=store.job(name)
        if digest(p)!=j['payload_sha256']:raise ValueError('OUTER_JOURNAL_HASH_MISMATCH')
        lock=store.job(name.removesuffix('_predictions')+'_lock')
        if j['lock']!=digest(lock):raise ValueError('OUTER_MODEL_LOCK_DRIFT')
    return count


def check_file(store,relative,sha):
    target=(store.root/relative).resolve()
    if not target.is_relative_to(store.root.resolve()) or file_sha(target)!=sha:
        raise ValueError('ARTIFACT_HASH_MISMATCH:'+relative)


def verify_cells(cells,track,nfolds=5,*,expected_n=None,splits=None):
    seen=set()
    if len(cells)!=nfolds or {c['fold'] for c in cells}!=set(range(nfolds)):
        raise ValueError('INCOMPLETE_OUTER_CELLS')
    for c in cells:
        test=set(c['test_subjects']);train=set(c['train_subjects'])
        if train&test or seen&test or c['status']!='OUTER_EVAL_COMMITTED':
            raise ValueError('INVALID_CELL_MEMBERSHIP_OR_STATUS')
        seen|=test
        if splits is not None:
            locked=splits[c['fold']]
            if not test<=set(locked['test']) or not train<=set(locked['train']):
                raise ValueError('CELL_DEPARTED_FROM_LOCKED_SPLIT')
        if set(c['arms'])!=set(arms(track)):raise ValueError('MISSING_EXPECTED_ARM')
        for name,a in c['arms'].items():
            keys=a['subjects']
            if len(keys)!=len(set(keys)) or not set(keys)<=test or (not name.startswith('blood_') and set(keys)!=test):
                raise ValueError('ARM_MEMBERSHIP_INCOMPLETE')
            from scripts.xyz_fusion_v1.statistics import check_curves
            check_curves(a['curves'],len(keys),a['metrics']['grid_days'])
            if not 0<=a['metrics']['ibs']<=1:raise ValueError('INVALID_IBS')
    if expected_n is not None and len(seen)!=expected_n:raise ValueError('OUTER_POPULATION_COVERAGE_INCOMPLETE')
    return len(seen)


def verify_image_track(store,config,track,nfolds=5):
    for seed in config['training_seeds']:
        cs=[store.job(f'{track}_s{seed}_o{f}_evaluation') for f in range(nfolds)]
        verify_cells(cs,track,nfolds)
        for f,c in enumerate(cs):
            allmeta=[]
            for i in range(config['components']):
                task=f'{track}_s{seed}_o{f}_c{i}';lock=store.job(task+'_lock')
                meta=set(lock['meta_subjects']);expert=set(lock['expert_subjects']);test=set(c['test_subjects'])
                if meta&expert or meta|expert!=set(c['train_subjects']) or set(lock['outer_subjects'])!=test:
                    raise ValueError('EXPERT_META_PARTITION_NOT_EXACT')
                allmeta.extend(meta)
                for modality,r in lock['experts'].items():
                    assert_pair(r,meta,test)
                    if set(r['train_subjects'])!=expert:raise ValueError('EXPERT_ANCESTRY_DRIFT')
                    if file_sha(store.root/'private/models'/f'{task}_{modality}_expert/locked.pt')!=r['sha256']:
                        raise ValueError('EXPERT_LOCK_HASH_DRIFT')
                    scores=store.job(f'{task}_{modality}_meta')
                    if set(scores['subjects'])!=meta or scores['model_sha256']!=r['sha256']:
                        raise ValueError('META_EXPERT_SCORE_MISMATCH')
                    winner=store.job(f'{task}_{modality}_winner')
                    if set(winner['selection_subjects'])-expert:raise ValueError('SELECTION_OUTSIDE_EXPERT_POOL')
                    if any(not 0<=x['metric']<=1 for x in winner['candidates']):raise ValueError('INVALID_INNER_IBS')
                for arm,r in lock['heads'].items():
                    if not set(r['ancestry'])<=meta or (not arm.startswith('blood_') and set(r['ancestry'])!=meta):
                        raise ValueError('META_HEAD_ANCESTRY_DRIFT')
                    check_file(store,r['path'],r['sha256'])
                p=store.job(task+'_predictions')
                if p['lock_sha256']!=digest(lock):raise ValueError('PREDICTION_MODEL_BINDING_DRIFT')
            if len(allmeta)!=len(set(allmeta)) or set(allmeta)!=set(c['train_subjects']):
                raise ValueError('ROTATING_META_PARTITIONS_INCOMPLETE')


def verify(store,config,*,final,allow_pending_state=False):
    count=verify_integrity(store,require_complete=final)
    if final:
        state=json.loads((store.root/'orchestration_state.json').read_text())
        accepted={'FINALIZED','VERIFYING_RELEASE'} if allow_pending_state else {'FINALIZED'}
        if state.get('stage') not in accepted or state.get('contract')!=store.binding:
            raise ValueError('V2_FINALIZATION_INCOMPLETE')
        plan=store.job('analysis_plan')
        splits=store.job('splits')
        for label,spec in plan['tabular'].items():
            if spec['support']['status']!='SUPPORTED':continue
            for family in spec['families']:
                verify_cells([store.job(f'{label}_{family}_o{f}_evaluation') for f in range(5)],spec['track'],
                             expected_n=spec['support']['n'],splits=splits)
                store.job(f'{label}_{family}_report')
        verify_image_track(store,config,'mri')
        verify_cells(store.job('mri_ensemble_cells'),'mri',expected_n=store.job('intake')['summary']['n'],splits=splits)
        if plan['pet']['status']=='SUPPORTED':
            verify_image_track(store,config,'pet')
            verify_cells(store.job('pet_ensemble_cells'),'pet',expected_n=plan['pet']['n'],splits=splits)
        diagnostics=store.job('diagnostics')
        if 'scanner' not in diagnostics or 'spatial' not in diagnostics:raise ValueError('DIAGNOSTICS_MISSING')
        for key,job in [('scanner','scanner_diagnostics'),('spatial','spatial_diagnostics'),
                        ('mri_subgroups','mri_subgroups'),('pacc','pacc'),('blood_selection','blood_selection')]:
            if key not in diagnostics or digest(diagnostics[key])!=digest(store.job(job)):
                raise ValueError('DIAGNOSTIC_RECEIPT_MISSING_OR_MISMATCH:'+key)
        if set(diagnostics.get('ph',{}))!={str(f) for f in range(5)}:
            raise ValueError('PH_DIAGNOSTICS_INCOMPLETE')
        for f in range(5):
            if digest(diagnostics['ph'][str(f)])!=digest(store.job(f'ph_o{f}')):
                raise ValueError('PH_DIAGNOSTIC_RECEIPT_MISMATCH')
        if plan['tabular']['tau']['support']['status']=='SUPPORTED':
            if digest(diagnostics.get('tau_power'))!=digest(store.job('tau_power')):
                raise ValueError('TAU_POWER_DIAGNOSTIC_MISSING')
        if 'mri_subgroups' not in diagnostics:raise ValueError('SUBGROUP_DIAGNOSTICS_MISSING')
        if plan['pet']['status']=='SUPPORTED' and 'pet_subgroups' not in diagnostics:
            raise ValueError('PET_SUBGROUP_DIAGNOSTICS_MISSING')
        if plan['pet']['status']=='SUPPORTED' and digest(diagnostics['pet_subgroups'])!=digest(store.job('pet_subgroups')):
            raise ValueError('PET_SUBGROUP_RECEIPT_MISMATCH')
        for category in diagnostics['scanner'].values():
            for row in category['evaluated']:
                if 'child_root' not in row:continue
                root=store.root/row['child_root'];contract=json.loads((root/'private/contract.json').read_text())
                if contract['parent']!=store.binding:raise ValueError('ROBUSTNESS_PARENT_MISMATCH')
                child=Store(root,contract);verify_integrity(child,require_complete=True)
                verify_image_track(child,contract['config'],'mri',1)
        expected=5*config['spatial']['cases_per_fold']
        if diagnostics['spatial']['case_count']!=expected:raise ValueError('SPATIAL_DIAGNOSTICS_INCOMPLETE')
        baseline_names={'zero',*('blur_'+str(x).replace('.','p') for x in config['spatial']['baseline_sigma'])}
        for f in range(5):
            for i in range(config['spatial']['cases_per_fold']):
                name=f'v2_xai_o{f}_case{i}'
                case=store.job(name)
                if case.get('fold')!=f or set(case.get('baseline_status',{}))!=baseline_names:
                    raise ValueError('SPATIAL_CASE_COVERAGE_MISMATCH')
                if case.get('disease_localization_allowed') is not False or case.get('explains_entire_ensemble') is not False:
                    raise ValueError('SPATIAL_CLAIM_SCOPE_MISMATCH')
                for baseline in baseline_names:store.job(name+'_'+baseline)
        release=store.job('release')
        for name,sha in release['files'].items():check_file(store,'public/'+name,sha)
        from .privacy import scan_public_bundle
        if scan_public_bundle(store.root/'public'):raise ValueError('PUBLIC_PRIVACY_FAILED')
    return {'status':'FINAL_VERIFIED' if final else 'INTEGRITY_VERIFIED','verified_units':count,'claim_status':CLAIM}
