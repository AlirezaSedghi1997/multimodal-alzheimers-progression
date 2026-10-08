"""Private, resumable frozen-model X/Y attribution; never fits a model."""
from __future__ import annotations
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'tmp/mpl_spatial_v1')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from src.method_y_revival.atomic_io import atomic_write_bytes, atomic_write_json, read_json
from src.method_y_revival.contracts import sha256_mapping
from src.method_y_revival.local_validation_v5 import analysis_lock, committed_job, file_sha
from src.method_y_revival.privacy import scan_public_bundle
from src.evaluation.signed_spatial_validation_v1 import (
    integrated_gradients, signed_similarity, perturbation_controls, reset_model_in_memory)

OUT = ROOT / 'results/xy_spatial_validation_v3'


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def guard():
    # Standard Windows runtime; do not require/install psutil into a locked venv.
    text = subprocess.check_output(['powershell','-NoProfile','-Command',
        'Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress'],text=True)
    processes = json.loads(text)
    processes = processes if isinstance(processes,list) else [processes]
    by_pid={p['ProcessId']:p for p in processes}
    ignored={os.getpid()}; current=os.getpid()
    while current in by_pid:
        current=by_pid[current]['ParentProcessId']
        if current in ignored:
            break
        ignored.add(current)
    for p in processes:
        if p['ProcessId'] in ignored:
            continue
        if 'python' in (p['Name'] or '').lower():
            require(p['CommandLine'] is not None,'PROCESS_COMMAND_UNREADABLE')
            require('new project' not in p['CommandLine'].lower(),'OTHER_PROJECT_PYTHON_ACTIVE_USE_SERIAL_EXECUTION')
    require(torch.cuda.is_available(), 'CUDA_REQUIRED')
    free, _ = torch.cuda.mem_get_info()
    require(free >= 4 * 1024**3, 'GPU_FREE_MEMORY_BELOW_4GIB')


def prepare():
    from scripts import run_method_y_posttraining_xai_limited_v1 as y
    from scripts import run_method_y_posttraining_xai_limited_v2 as yp
    from scripts.mx_quick_validation_v1 import stress as x
    from scripts.mx_quick_validation_v1.audit import data_rows
    from scripts.mx_quick_validation_v1.common import CACHE, PANEL, REGISTRY, EXT, PARENT, INDEX
    import nibabel as nib
    _, factor, rc, cells = y._locked_inputs()
    prior_path = yp.OUTPUT / 'private/masked_inference_plan.json'
    prior = read_json(prior_path)
    require(prior['plan_sha256'] == sha256_mapping({k:v for k,v in prior.items() if k != 'plan_sha256'}), 'Y_PLAN_HASH')
    ym, ymc = yp.load_locked_brain_mask(yp.MASK_PATH, yp.REFERENCE_PATH)
    xm, xmc = x.load_mask(ROOT, read_json(ROOT/'configs/experiments/method_x_quick_validation_v1.json'))
    specs = x.checkpoint_specs(ROOT)
    _, real, tests = data_rows(ROOT)
    panel = {r['patient_hash']:r for r in read_json(ROOT/PANEL)['rows']}
    rows = []
    for fold in range(5):
        # Hash-order selection uses neither class nor model outcome nor map appearance.
        ordered = sorted(tests[fold], key=lambda r:r['patient_hash'])
        require(len({r['patient_hash'] for r in ordered}) == len(ordered), 'X_SUBJECT_DUPLICATE')
        path, sha = specs[fold, 'hybrid_selected_seed13']
        for i, r in enumerate(ordered[:2]):
            rows.append({'key': f'x_f{fold}_p{i}', 'branch':'x', 'fold':fold,
                'subject':r['patient_hash'], 'sample_hash':r['sample_hash'],
                'cache_row':int(real[r['sample_hash']]['row']), 'checkpoint':str(path),
                'checkpoint_sha256':sha,
                'expected_probability':panel[r['patient_hash']]['predictions']['hybrid_selected']['13']})
        for branch, arm in [('amyloid','A4'), ('tau','A1')]:
            chosen = next(r for r in prior['cells'] if (r['branch'],r['arm'],r['outer']) == (branch,arm,fold))
            cell = next(c for c in cells if (c['branch'],c['arm'],c['outer']) == (branch,arm,fold))
            for i, subject in enumerate(chosen['subjects'][:2]):
                require(subject not in cell['train_subjects'], 'Y_ATTRIBUTION_TRAIN_OVERLAP')
                rows.append({'key': f'{branch}_f{fold}_p{i}', 'branch':branch, 'arm':arm,
                    'fold':fold, 'subject':subject, 'checkpoint':str(cell['final_checkpoint']),
                    'checkpoint_sha256':chosen['checkpoint_sha256']})
    require(len(rows) == 30, 'EXPECTED_30_FIXED_CASES')
    import h5py
    selected_tensors = {}
    with h5py.File(ROOT/CACHE,'r') as h5:
        for row in rows:
            if row['branch']=='x':
                i=row['cache_row']; key=h5['sample_hash'][i]
                key=key.decode() if isinstance(key,bytes) else str(key)
                require(key==row['sample_hash'],'X_CACHE_INDEX_DRIFT')
                selected_tensors[row['key']]=[tensor_sha(torch.as_tensor(h5[n][i],dtype=torch.float32)) for n in ('image','tabular')]
    for row in rows:
        if row['branch']!='x':
            cell=next(c for c in cells if (c['branch'],c['arm'],c['outer'])==(row['branch'],row['arm'],row['fold']))
            batch=y._load_cell_batch(dict(cell,firewall_subjects=[row['subject']]),run_contract=rc,
                model_cfg=y._locked_model_cfg(),model_features=factor['branches'][row['branch']]['model_features'])
            require(len(batch)==1 and batch[0][0]==row['subject'],'Y_SELECTED_INPUT_MISMATCH')
            selected_tensors[row['key']]=[tensor_sha(t) for t in batch[0][1:]]
    inputs = [prior_path, ROOT/PANEL, ROOT/REGISTRY, ROOT/INDEX,
        ROOT/EXT/'reports/fold_local_hybrid_selection.v1.json',
        ROOT/PARENT/'reports/fasttrack_inner_lock.json', y.FACTOR_CONFIG, y.XAI_CONFIG]
    for branch in ('amyloid','tau'):
        for suffix in ('membership','splits'):
            rec = rc['input_fingerprints'][branch+'_'+suffix]
            require(file_sha(rec['path']) == rec['sha256'], 'Y_INPUT_DRIFT')
            inputs.append(Path(rec['path']))
    inputs += [ROOT/'splits'/f'fold_{f}_test.csv' for f in range(5)]
    sources = {Path(__file__), ROOT/'scripts/train_3d_mri.py', Path(y.__file__), Path(yp.__file__),
               ROOT/'scripts/run_method_y_pet_attribution_v4.py'}
    for folder in ('src/models','src/evaluation','src/method_y_revival','src/training',
                   'scripts/mx_quick_validation_v1','scripts/mx_science_extension_v1','scripts/mx_fasttrack_v2'):
        sources.update((ROOT/folder).rglob('*.py'))
    contract = {'schema_version':3, 'cases':rows, 'selected_tensor_sha256':selected_tensors,
        'predecessors_preserved':{'v1':'preflight only: psutil unavailable',
            'v2':'2 MRI diagnostics committed; PET blocked by deterministic CUDA cumprod backward'},
        'survival_target_device':'CPU_DIFFERENTIABLE_COPY_FROM_GPU_LOGITS_CUMPROD_BACKWARD',
        'inputs':{str(p):file_sha(p) for p in inputs},
        'sources':{p.relative_to(ROOT).as_posix():file_sha(p) for p in sorted(sources)},
        'y_run_digest':rc['digest'], 'x_mask':xmc, 'y_mask':ymc,
        'python':sys.version, 'torch':str(torch.__version__), 'numpy':np.__version__,
        'device':'cuda', 'tf32':False, 'deterministic_algorithms':True,
        'baseline_policy':['brain_zero_background_unchanged','brain_gaussian_sigma3_background_unchanged'],
        'ig_quadrature_steps':[16,32,64], 'completeness_tolerance':'1e-4 + 0.05*abs(target_delta)',
        'perturbation_fraction':.05, 'random_controls':20, 'random_seed':13,
        'target_x':'amyloid_logit_with_fixed_tabular',
        'target_y':'1095.75_day_event_risk_with_fixed_tabular',
        'selection':'2 fixed outcome-blind subjects per fold per branch; Y A4/A1 global post-hoc comparators, NOT all fold-local winners',
        'new_training':False, 'outer_performance_recomputed':False,
        'claims':'POST_HOC_INTERNAL_EXPLORATORY_NO_DISEASE_LOCALIZATION',
        'label_permutation_training':'NOT_EXECUTED_REQUIRES_SEPARATE_TRAINING_PROTOCOL'}
    context = dict(y=y, x=x, factor=factor, rc=rc, cells=cells, cache=ROOT/CACHE,
        masks={'x':xm,'amyloid':ym,'tau':ym},
        affines={'x':np.asarray(xmc['affine']), 'amyloid':nib.load(yp.REFERENCE_PATH).affine,
                 'tau':nib.load(yp.REFERENCE_PATH).affine})
    return contract, context


def load_case(row, ctx):
    require(file_sha(row['checkpoint']) == row['checkpoint_sha256'], 'CHECKPOINT_DRIFT')
    if row['branch'] == 'x':
        import h5py
        from scripts.train_3d_mri import normalize_tabular
        net, mean, std = ctx['x'].load_model(Path(row['checkpoint']), row['fold'])
        with h5py.File(ctx['cache'],'r') as h5:
            i = row['cache_row']
            key = h5['sample_hash'][i]
            key = key.decode() if isinstance(key,bytes) else str(key)
            require(key == row['sample_hash'], 'X_CACHE_INDEX_DRIFT')
            image = torch.as_tensor(h5['image'][i],dtype=torch.float32).reshape(1,1,128,128,128).cuda()
            raw_tab = torch.as_tensor(h5['tabular'][i],dtype=torch.float32,device='cuda').reshape(1,5)
        tab = normalize_tabular(raw_tab, mean, std)
        def forward(v):
            return net(v, tab, tab_presence=torch.ones(1,device='cuda')).reshape(())
        net.eval()
        with torch.no_grad():
            require(abs(float(torch.sigmoid(forward(image)))-row['expected_probability']) < 1e-4,
                    'X_LOCKED_PREDICTION_REPRODUCTION_FAILED')
    else:
        from src.method_y_revival.image_modeling_v3 import survival_at_times
        y, branch = ctx['y'], row['branch']
        cell = next(c for c in ctx['cells'] if (c['branch'],c['arm'],c['outer']) == (branch,row['arm'],row['fold']))
        prov = y.pa.verify_final_provenance(cell['final_checkpoint'],v4_root=y.V4_ROOT,
                    branch=branch,arm=row['arm'],outer=row['fold'])
        require(prov['checkpoint_sha256']==row['checkpoint_sha256'],'Y_CHECKPOINT_PROVENANCE')
        payload = y.pa.read_only_load_checkpoint(row['checkpoint'],expected_sha256=row['checkpoint_sha256'])
        features = ctx['factor']['branches'][branch]['model_features']
        net = y._build_real_arch(row['arm'],y._locked_model_cfg(),tabular_dim=len(features),
                                bins=y._infer_bins_from_payload(payload))
        net.load_state_dict(payload['model_state'],strict=True)
        batch = y._load_cell_batch(dict(cell,firewall_subjects=[row['subject']]),run_contract=ctx['rc'],
                                  model_cfg=y._locked_model_cfg(),model_features=features)
        require(len(batch)==1 and batch[0][0]==row['subject'],'Y_INPUT_SUBJECT_MISMATCH')
        image, tab = batch[0][1].reshape(1,1,128,128,128).cuda(), batch[0][2].unsqueeze(0).cuda()
        net = net.cuda()
        def forward(v):
            # The small survival head stays differentiable through the copy.
            # No relaxation of CUDA deterministic policy or model change.
            return (1-survival_at_times(net(v,tab).cpu(),ctx['factor']['shared']['hazard_bin_edges_days'],[1095.75]))[0,0]
    net.eval()
    # Gradients are needed only with respect to image, never parameters.
    for p in net.parameters():
        p.requires_grad_(False)
    return net, image, tab, forward


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_maps(directory, image, maps, affine):
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True)
    outputs = {}
    volumes = {'processed_individual_input':image, **maps}
    for name, arr in volumes.items():
        # Atomic uncompressed NIfTI (no unpredictable gzip timestamp).
        path = directory/(name+'.nii')
        atomic_write_bytes(path,nib.Nifti1Image(np.asarray(arr,dtype='f4'),affine).to_bytes())
        outputs[path.name] = file_sha(path)
    # The background is this person's actual normalized input, never a template.
    bg = nib.as_closest_canonical(nib.Nifti1Image(image,affine)).get_fdata()
    fig, axes = plt.subplots(len(maps),3,figsize=(12,4*len(maps)),squeeze=False)
    for r,(name,arr) in enumerate(maps.items()):
        z = nib.as_closest_canonical(nib.Nifti1Image(arr,affine)).get_fdata()
        limit = max(float(np.quantile(np.abs(z[z!=0]),.99)) if np.any(z!=0) else 0,1e-12)
        for ax,(axis,title) in zip(axes[r],enumerate(('Sagittal','Coronal','Axial'))):
            sl = np.take(bg,bg.shape[axis]//2,axis=axis)
            heat = np.take(z,z.shape[axis]//2,axis=axis)
            ax.imshow(np.rot90(sl),cmap='gray')
            shown = np.ma.masked_where(np.abs(heat)<.1*limit,heat)
            im = ax.imshow(np.rot90(shown),cmap='coolwarm',vmin=-limit,vmax=limit,alpha=.65)
            ax.set_title(name+' / '+title); ax.axis('off')
        fig.colorbar(im,ax=axes[r].tolist(),shrink=.5,label='Signed attribution (display clipped at p99)')
    fig.suptitle('Actual individual model-space input; RED increases / BLUE decreases model target\nNot a disease lesion map; normalized PET is not quantitative SUVR')
    stream=io.BytesIO(); fig.savefig(stream,format='png',dpi=120); plt.close(fig)
    path=directory/'individual_overlay.png'; atomic_write_bytes(path,stream.getvalue()); outputs[path.name]=file_sha(path)
    return outputs


def run(mode, limit):
    torch.set_num_threads(2); torch.manual_seed(13)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if mode == 'run':
        guard()
    contract,ctx=prepare()
    with analysis_lock(OUT):
        cp=OUT/'private/contract.json'
        if cp.exists():
            require(read_json(cp)==contract,'SPATIAL_CONTRACT_MISMATCH_NO_AUTOMATIC_MIGRATION')
        else:
            require(mode!='verify','SPATIAL_CONTRACT_MISSING'); atomic_write_json(cp,contract)
        if mode=='preflight':
            return {'status':'READY_FOR_BOUNDED_INFERENCE','cases':len(contract['cases'])}
        done=[]; new=0
        for row in contract['cases']:
            exists=(OUT/'private/jobs'/(row['key']+'.json')).exists()
            if mode=='run' and limit and new>=limit and not exists:
                continue
            directory=OUT/'private/cases'/row['key']
            def compute():
                guard(); started=time.perf_counter()
                net,image,tab,forward=load_case(row,ctx)
                # prepare() binds selected raw tensors; the model's normalized X
                # tabular input has a separate runtime digest below.
                require(tensor_sha(image)==contract['selected_tensor_sha256'][row['key']][0],
                        'SELECTED_IMAGE_CHANGED_DURING_RUN')
                mask=ctx['masks'][row['branch']]
                array=image.detach().cpu().numpy().squeeze()
                detail={}; maps={}; baselines={}
                for name in ('zero','blur'):
                    base=array.copy()
                    base[mask]=0 if name=='zero' else gaussian_filter(array,sigma=3,mode='reflect')[mask]
                    baseline=torch.as_tensor(base,device='cuda').reshape_as(image)
                    baselines[name]=baseline
                    def progress(n,i):
                        print(json.dumps({'case':row['key'],'baseline':name,'quadrature':n,'step':i}),flush=True)
                    a,info=integrated_gradients(forward,image,baseline,progress=progress)
                    require(np.count_nonzero(a[~mask])==0,'BACKGROUND_NOT_HELD_FIXED')
                    maps['ig_'+name]=a
                    detail[name]={'numerical':info,'perturbation':perturbation_controls(forward,image,baseline,a,mask)}
                detail['baseline_sensitivity']=signed_similarity(maps['ig_zero'],maps['ig_blur'],mask)
                detail['input_sha256']=tensor_sha(image); detail['tabular_sha256']=tensor_sha(tab)
                reset_count=reset_model_in_memory(net,seed=13)
                random_map,random_info=integrated_gradients(forward,image,baselines['blur'])
                detail['randomization']={'reset_modules':reset_count,'numerical':random_info,
                    'signed_map_similarity':signed_similarity(maps['ig_blur'],random_map,mask),
                    'label_permutation':'NOT_EXECUTED_REQUIRES_TRAINING'}
                maps['randomized_ig_blur']=random_map
                outputs=save_maps(directory,array,maps,ctx['affines'][row['branch']])
                require(file_sha(row['checkpoint'])==row['checkpoint_sha256'],'CHECKPOINT_MUTATED')
                detail.update(outputs=outputs,elapsed_seconds=time.perf_counter()-started,
                    spatial_scope='MODEL_INPUT_SPACE_NOT_NATIVE_SCANNER_SPACE',
                    claim_status='POST_HOC_INTERNAL_EXPLORATORY',disease_localization_allowed=False)
                del net,image,tab,forward,baselines
                gc.collect(); torch.cuda.empty_cache()
                return detail
            print(json.dumps({'event':'CASE_START_OR_RECOVER','case':row['key']}),flush=True)
            report=committed_job(OUT/'private',row['key'],contract,compute,verify_only=mode=='verify')
            for name,digest in report['outputs'].items():
                require(file_sha(directory/name)==digest,'SPATIAL_OUTPUT_HASH_MISMATCH')
            done.append(row['key']); new+=int(not exists)
            print(json.dumps({'event':'CASE_COMMITTED_OR_RECOVERED','case':row['key'],
                              'elapsed_seconds':report['elapsed_seconds']}),flush=True)
        summary={'status':'COMPLETED_DIAGNOSTICS_NOT_LOCALIZATION_PROOF' if len(done)==30 else 'PARTIAL',
            'cases_committed':len(done),'cases_required':30,'contract_sha256':sha256_mapping(contract),
            'claim_status':'POST_HOC_INTERNAL_EXPLORATORY','anatomical_claim_allowed':False,
            'external_validation':False,'training_performed':False}
        if mode!='verify':
            atomic_write_json(OUT/'public_bundle/summary.json',summary)
        else:
            require(read_json(OUT/'public_bundle/summary.json')==summary,'SPATIAL_SUMMARY_DRIFT')
        require(not scan_public_bundle(OUT/'public_bundle'),'PUBLIC_PRIVACY_FAILURE')
        return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['preflight','run','verify'],default='run')
    p.add_argument('--limit-new',type=int,default=0)
    args=p.parse_args()
    try:
        print(json.dumps(run(args.mode,args.limit_new)),flush=True)
    except Exception as exc:
        print(json.dumps({'status':'BLOCKED','reason':f'{type(exc).__name__}:{exc}'}),flush=True)
        raise SystemExit(3)
