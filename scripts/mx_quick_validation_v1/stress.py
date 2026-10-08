"""Fixed-weight paired stress tests, committed after every patient/condition."""
import hashlib
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .common import CACHE, REGISTRY, EXT, PARENT, PANEL, read, sha, require, progress


def load_mask(root, config):
    import nibabel as nib
    from nibabel.processing import resample_from_to
    registry = read(root/REGISTRY)
    reference = Path(registry['derived_model_space']['reference_path'])
    tpm_path = Path(config['mask_tpm'])
    require(tpm_path.is_file() and reference.is_file(), 'WHOLE_BRAIN_MASK_SOURCE_MISSING')
    require(sha(reference)==registry['derived_model_space']['reference_sha256'], 'MASK_REFERENCE_DRIFT')
    ref = nib.load(reference)
    geometry = registry['derived_model_space']
    require(list(ref.shape)==geometry['shape']==[128,128,128] and
            np.allclose(ref.affine,np.asarray(geometry['affine']),atol=1e-6), 'MASK_REFERENCE_GEOMETRY')
    tpm = nib.load(tpm_path)
    values = tpm.get_fdata(dtype=np.float32)
    require(values.ndim==4 and values.shape[-1]>=3 and np.isfinite(values).all(), 'INVALID_TPM')
    tissues = nib.Nifti1Image(values[...,:3].sum(axis=3),tpm.affine)
    mask = np.asarray(resample_from_to(tissues,(ref.shape,ref.affine),order=1).dataobj)>config['mask_threshold']
    require(.1<float(mask.mean())<.65,'WHOLE_BRAIN_MASK_COVERAGE_INVALID')
    import h5py
    with h5py.File(root/CACHE,'r') as h5:
        require(h5['image'].shape[1:]==(1,128,128,128) and
                not bool(h5.attrs.get('axis_flips_applied',True)) and
                h5.attrs.get('source_geometry_audit_status')=='PASS' and
                h5.attrs.get('source_nifti_orientation')=='LAS' and
                h5.attrs.get('source_nifti_unique_affine_count')==1, 'CACHE_MASK_GEOMETRY_UNPROVEN')
    metadata = {'tpm_sha256':sha(tpm_path),'reference_sha256':sha(reference),
                'registry_sha256':sha(root/REGISTRY),'affine':ref.affine.tolist(),
                'mask_voxels':int(mask.sum()),'mask_fraction':float(mask.mean()),
                'mask_array_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),
                'definition':'sum_SPM_GM_WM_CSF_probability_linear_resample_gt_0.5',
                'baseline_value':0.0,'individual_segmentation':False}
    return mask,metadata


def perturb(x, mask, condition, seed):
    x = np.asarray(x,dtype=np.float32)
    require(x.ndim==3 and x.shape==mask.shape and np.isfinite(x).all() and mask.any(), 'PERTURB_INPUT')
    out = x.copy()
    scale = float(np.subtract(*np.quantile(x[mask],[.75,.25])))
    require(scale>0,'PERTURB_IQR_ZERO')
    if condition in ('noise02','noise05'):
        sd = .02 if condition=='noise02' else .05
        out[mask] += np.random.default_rng(seed).normal(0,sd*scale,int(mask.sum())).astype('f4')
    elif condition=='bias10':
        axes = np.meshgrid(*(np.linspace(-1,1,n,dtype='f4') for n in x.shape),indexing='ij')
        field = 1 + .1*sum(axes)/3
        out[mask] *= field[mask]
    elif condition=='blur05':
        out[mask] = gaussian_filter(x,sigma=.5,mode='reflect')[mask]
    elif condition=='brain_only':
        out[~mask] = 0
    elif condition=='background_only':
        out[mask] = 0
    else:
        require(condition=='clean','UNKNOWN_PERTURBATION')
    require(np.isfinite(out).all(),'PERTURB_NONFINITE')
    return out


def select_sample(rows, per_fold, seed):
    require(len({r['patient_hash'] for r in rows})==len(rows),'SAMPLE_PATIENT_DUPLICATE')
    selected = []
    for fold in range(5):
        groups = [[r for r in rows if r['fold']==fold and r['label']==c] for c in (0,1)]
        require(min(map(len,groups))>=2 and per_fold>=4 and sum(map(len,groups))>=per_fold,'STRATIFIED_SAMPLE_TOO_SMALL')
        n0 = min(len(groups[0]),max(2,round(per_fold*len(groups[0])/sum(map(len,groups)))))
        n0 = max(n0,per_fold-len(groups[1])); n0=min(n0,per_fold-2)
        for c,n in enumerate((n0,per_fold-n0)):
            ordered = sorted(groups[c],key=lambda r:r['patient_hash'])
            rng = np.random.default_rng(seed+10*fold+c)
            chosen = rng.choice(len(ordered),n,replace=False)
            selected.extend({**ordered[int(i)],'weight':len(ordered)/n,
                             'inclusion_probability':n/len(ordered),'stratum_population':len(ordered)} for i in chosen)
    return sorted(selected,key=lambda r:(r['fold'],r['patient_hash']))


def checkpoint_specs(root):
    from scripts.mx_science_extension_v1.selection import arm_directory
    lock = read(root/PARENT/'reports/fasttrack_inner_lock.json')
    selection = read(root/EXT/'reports/fold_local_hybrid_selection.v1.json')
    choices = {(c['fold'],c['seed']):c['selected_variant'] for c in selection['choices']}
    specs = {}
    for fold in range(5):
        task = lock['final_task']
        specs[fold,'parent_final'] = (root/PARENT/'inner_tasks'/task/f'method_x_3d_fold_{fold}.pt',lock['tasks'][task]['checkpoint_sha256'][fold])
        variant = choices[fold,13]
        specs[fold,'hybrid_selected_seed13'] = (arm_directory(root,13,variant)/f'method_x_3d_fold_{fold}.pt',selection['arms'][f'13|{variant}']['checkpoint_sha256'][fold])
    for path,digest in specs.values():
        require(sha(path)==digest,'STRESS_CHECKPOINT_DRIFT')
    return specs


def load_model(path, fold):
    import torch
    from src.evaluation.method_x_xai_forward import build_model_from_checkpoint
    require(torch.cuda.is_available(),'CUDA_REQUIRED_NO_SILENT_CPU_FALLBACK')
    ckpt = torch.load(path,map_location='cpu',weights_only=True)
    require(ckpt['fold']==fold and ckpt['hpo_selection_only'] is True and
            ckpt['input_mode']=='mri_plus_tabular' and ckpt['drop_tabular_feature']=='none', 'STRESS_CHECKPOINT_IDENTITY')
    model,source = build_model_from_checkpoint(ckpt,device=torch.device('cuda'))
    require(source=='model_state_dict','STRESS_WEIGHT_SOURCE')
    mean,std = ckpt['tab_mean'],ckpt['tab_std']
    require(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std>0).all(),'TABULAR_NORMALIZATION_INVALID')
    return model,mean,std


def probability(model, image, tab, mean, std):
    import torch
    from scripts.train_3d_mri import normalize_tabular
    x = torch.as_tensor(np.asarray(image),dtype=torch.float32,device='cuda').reshape(1,1,*image.shape)
    tab = normalize_tabular(torch.as_tensor(tab,dtype=torch.float32,device='cuda').reshape(1,5),mean,std)
    with torch.inference_mode():
        p = float(torch.sigmoid(model(x,tab,tab_presence=torch.ones(1,device='cuda'))).item())
    require(np.isfinite(p) and 0<=p<=1,'STRESS_PROBABILITY_INVALID')
    return p


def benchmark(root):
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    specs = checkpoint_specs(root)
    timing = {}
    for arm in ('parent_final','hybrid_selected_seed13'):
        model,mean,std = load_model(specs[0,arm][0],0)
        x = np.random.default_rng(31).normal(size=(128,128,128)).astype('f4')
        probability(model,x,np.zeros(5,'f4'),mean,std)
        torch.cuda.synchronize(); start=time.perf_counter()
        for _ in range(3):
            probability(model,x,np.zeros(5,'f4'),mean,std)
        torch.cuda.synchronize()
        timing[arm]=(time.perf_counter()-start)/3
        del model; torch.cuda.empty_cache()
    return {'synthetic_seconds_per_forward':timing,
            'forward_only_2800_seconds':1400*sum(timing.values()), 'includes_disk_audit_or_bootstrap':False}


def run_stress(store,mask):
    import torch,h5py
    from .audit import data_rows
    from .paired_analysis import paired
    torch.set_num_threads(2); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    specs = checkpoint_specs(store.root)
    selected = store.protocol['stress_sample']
    _,real,tests = data_rows(store.root)
    cov = {r['patient_hash']:r for fold in tests for r in fold}
    panel = {r['patient_hash']:r for r in read(store.root/PANEL)['rows']}
    predictions = {arm:{} for arm in ('parent_final','hybrid_selected_seed13')}
    backgrounds = []
    with h5py.File(store.root/CACHE,'r') as h5:
        for fold in range(5):
            for arm in predictions:
                model=mean=std=None
                try:
                    for i,row in enumerate(selected):
                        if row['fold']!=fold: continue
                        index = int(real[cov[row['patient_hash']]['sample_hash']]['row'])
                        image = np.asarray(h5['image'][index],dtype='f4').squeeze()
                        tab = np.asarray(h5['tabular'][index],dtype='f4')
                        backgrounds.append(float(np.count_nonzero(image[~mask])/max(1,(~mask).sum())))
                        seed = int(hashlib.sha256((row['patient_hash']+str(store.config['bootstrap_seed'])).encode()).hexdigest()[:8],16)
                        probabilities = {}
                        for condition in store.config['stress_conditions']:
                            def compute(condition=condition):
                                nonlocal model,mean,std
                                if model is None:
                                    model,mean,std=load_model(specs[fold,arm][0],fold)
                                return {'p':probability(model,perturb(image,mask,condition,seed),tab,mean,std),
                                        'checkpoint_sha256':specs[fold,arm][1], 'patient_hash':row['patient_hash'],
                                        'condition':condition}
                            saved = store.receipt(f'stress/{arm}/patient_{i:03d}/{condition}',compute)
                            require(saved['checkpoint_sha256']==specs[fold,arm][1] and saved['patient_hash']==row['patient_hash']
                                    and saved['condition']==condition and np.isfinite(saved['p']) and 0<=saved['p']<=1,'STRESS_RECEIPT_INVALID')
                            probabilities[condition]=saved['p']
                        if arm=='hybrid_selected_seed13':
                            require(abs(probabilities['clean']-panel[row['patient_hash']]['predictions']['hybrid_selected']['13'])<1e-4,'STRESS_CLEAN_REPRODUCTION_FAILED')
                        predictions[arm][i]=probabilities
                        progress('stress',fold=fold,arm=arm,patient_in_fold=1+sum(1 for rr in selected[:i] if rr['fold']==fold),
                                 patients_per_fold=store.config['stress_patients_per_fold'],status='PATIENT_COMMITTED')
                finally:
                    if model is not None: del model
                    torch.cuda.empty_cache()
    output = {}
    y=np.array([r['label'] for r in selected]); w=np.array([r['weight'] for r in selected])
    strata=np.array([r['fold']*2+r['label'] for r in selected])
    for arm,values in predictions.items():
        require(set(values)==set(range(len(selected))),'STRESS_PATIENT_COVERAGE')
        clean=np.array([values[i]['clean'] for i in range(len(selected))])
        output[arm]={}
        for condition in store.config['stress_conditions'][1:]:
            def analyze(condition=condition):
                p=np.array([values[i][condition] for i in range(len(selected))])
                report=paired(y,p,clean,resamples=store.config['bootstrap_resamples'],seed=store.config['bootstrap_seed'],weights=w,strata=strata)
                report.update(mean_probability_change=float(np.average(p-clean,weights=w)),
                              mean_absolute_probability_change=float(np.average(abs(p-clean),weights=w)),
                              fixed_threshold_05_flip_rate=float(np.average((p>=.5)!=(clean>=.5),weights=w)))
                return report
            output[arm][condition]=store.receipt(f'stress_analysis/{arm}_{condition}',analyze,public=True)
    return store.receipt('stress_summary',lambda:{'status':'COMPLETED_FIXED_WEIGHT_STRESS','patient_n':len(selected),
        'conditions':store.config['stress_conditions'],'comparisons':output,'primary':'noise05_minus_clean_brier',
        'mean_nonzero_background_fraction':float(np.mean(backgrounds)),
        'background_may_already_be_zero_after_preprocessing':True,
        'clinical_scanner_robustness_proven':False,'disease_region_confirmation':False},public=True)
