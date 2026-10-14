"""Baseline-specific signed MRI attribution for a prespecified frozen fusion component.

This bounded panel explains one declared component, never the entire ensemble
or an Alzheimer's lesion. Both numerical successes and failures are published.
"""
from __future__ import annotations
import copy
import json
import pickle
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import gaussian_filter
from scripts.xyz_fusion_v1.store import digest, file_sha, atomic_json
from scripts.xyz_postprimary_v1.context import write_array
from scripts.xyz_postprimary_v1.rendering import overlay, save_nifti, native_maps
from src.evaluation.signed_spatial_validation_v1 import (
    integrated_gradients, signed_similarity, perturbation_controls, reset_model_in_memory)
from .training import MRISurvival, train_expert


def attribution_status(numerical, random_similarity, fidelity, random_numerical, perm_similarity, perm_numerical, threshold=.8):
    if numerical.get('numerical_status')!='PASS':return 'DIAGNOSTIC_ONLY_NUMERICAL_FAILURE'
    if any(n.get('numerical_status')!='PASS' for n in (random_numerical,perm_numerical)):
        return 'DIAGNOSTIC_ONLY_CONTROL_NUMERICAL_FAILURE'
    r=random_similarity.get('pearson')
    q=perm_similarity.get('pearson')
    if r is None or not np.isfinite(r) or abs(r)>=threshold:return 'DIAGNOSTIC_ONLY_RANDOMIZATION_FAILURE'
    if q is None or not np.isfinite(q) or abs(q)>=threshold:return 'DIAGNOSTIC_ONLY_LABEL_PERMUTATION_FAILURE'
    positive=fidelity.get('positive',{})
    if (positive.get('directional_margin_vs_random_mean',-1)<=0 or
            positive.get('descriptive_random_tail_fraction',1)>.05):
        return 'DIAGNOSTIC_ONLY_FIDELITY_FAILURE'
    return 'VALIDATED_FOR_THIS_BASELINE_AND_COMPONENT_ONLY'


def expert_target(model, image, policy, times, grid=None):
    # Cumprod backward on CPU has a deterministic implementation.
    logits=model(image).float().cpu()
    boundary=torch.cat([torch.ones((len(logits),1)),torch.cumprod(torch.sigmoid(-logits),dim=1)],dim=1)
    edges=np.asarray(policy['hazard_edges_days'])
    def interp(table,coordinates,query):
        values=[]
        for time in query:
            hi=min(int(np.searchsorted(coordinates,time,side='right')),len(coordinates)-1);lo=max(0,hi-1)
            weight=(time-coordinates[lo])/(coordinates[hi]-coordinates[lo])
            values.append(table[:,lo]*(1-weight)+table[:,hi]*weight)
        return torch.stack(values,dim=1)
    if grid is not None:
        boundary=interp(boundary,edges,grid);edges=np.asarray(grid)
    return -torch.log(interp(boundary,edges,times).clamp(1e-6,1))


def run_spatial(ctx,frame,bank,config,store,atlas_assets):
    from .protocol import MRI
    from .atlas import bridge,roi_summary
    atlas_receipt=bridge(ctx['root'],store,atlas_assets)
    atlas_image=nib.load(store.root/atlas_receipt['path'])
    atlas_labels=atlas_image.get_fdata().astype('int32')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');policy=config['training']
    cases=[];seed=config['training_seeds'][0]
    for fold,split in enumerate(ctx['splits']):
        task=f'mri_s{seed}_o{fold}_c0';folder=store.root/'private/models'/f'{task}_mri_expert'
        receipt=json.loads((folder/'locked.json').read_text())
        if file_sha(folder/'locked.pt')!=receipt['sha256']:raise ValueError('SPATIAL_CHECKPOINT_HASH_DRIFT')
        cp=torch.load(folder/'locked.pt',map_location='cpu',weights_only=True)
        model=MRISurvival(policy).to(device);model.load_state_dict(cp['model'],strict=True);model.eval()
        head_receipt=store.job(f'{task}_clinical_mri_amyloid')
        path=store.root/head_receipt['path']
        if file_sha(path)!=head_receipt['sha256']:raise ValueError('SPATIAL_COX_HASH_DRIFT')
        with path.open('rb') as f:head=pickle.load(f)
        coefficients=np.asarray(head[-1].coef_)/np.asarray(head[-2].scale_)
        ids=[head_receipt['features'].index(c) for c in MRI]
        beta=torch.tensor(coefficients[ids],dtype=torch.float32)
        def forward(x):return (expert_target(model,x,policy,config['fusion']['score_times_days'],config['fusion']['ibs_grid_days'])*beta).sum()
        # Matched label-shuffle control: joint event/time permutation on expert
        # training subjects, same fixed architecture and budget; no heldout outcomes.
        pool=frame.loc[frame.patient_hash.isin(receipt['train_subjects'])].copy().reset_index(drop=True)
        order=np.random.default_rng(seed+fold).permutation(len(pool))
        shuffled=pool.copy();shuffled[['event','time_days']]=pool.iloc[order][['event','time_days']].to_numpy()
        selected_lr=store.job(task+'_mri_winner')['winner']['lr']
        perm,perm_receipt=train_expert(bank,shuffled,None,dict(policy,modality='mri'),config['fusion']['ibs_grid_days'],
            store.root/'private/models'/f'spatial_labelperm_o{fold}',store.binding,selected_lr,
            seed+90000+fold,fixed_epochs=receipt['best_epoch'])
        def perm_forward(x):return (expert_target(perm,x,policy,config['fusion']['score_times_days'],config['fusion']['ibs_grid_days'])*beta).sum()
        selected=sorted(split['test'],key=lambda k:digest({'seed':config['seed'],'key':k}))[:config['spatial']['cases_per_fold']]
        for j,key in enumerate(selected):
            name=f'v2_xai_o{fold}_case{j}';directory=store.root/'private/spatial'/name
            if (store.root/'private/jobs'/f'{name}.json').exists():cases.append(store.job(name));continue
            row=frame.set_index('patient_hash').loc[key]
            if key in receipt['train_subjects'] or key in head_receipt['ancestry']:
                raise ValueError('SPATIAL_SUBJECT_SEEN_BY_MODEL')
            image=bank.batch([int(row.cache_row)],device);array=image.cpu().numpy().squeeze();mask=array!=0
            maps={};numerics={};fidelities={};similarities={};permutation={};controls={}
            baselines={'zero':np.zeros_like(array)}
            for sigma in config['spatial']['baseline_sigma']:
                b=gaussian_filter(array,sigma=sigma);b[~mask]=0;baselines['blur_'+str(sigma).replace('.','p')]=b
            for kind,baseline in baselines.items():
                b=torch.tensor(baseline,device=device).reshape_as(image)
                def compute():
                    attribution,numeric=integrated_gradients(forward,image,b,steps=tuple(config['spatial']['ig_steps']))
                    directory.mkdir(parents=True,exist_ok=True);write_array(directory/(kind+'.npy'),attribution)
                    f=perturbation_controls(forward,image,b,attribution,mask,seed=seed+fold,
                                            repeats=config['spatial']['random_controls'])
                    random=copy.deepcopy(model);reset_model_in_memory(random,seed+fold)
                    def random_forward(x):return (expert_target(random,x,policy,config['fusion']['score_times_days'],config['fusion']['ibs_grid_days'])*beta).sum()
                    rm,rn=integrated_gradients(random_forward,image,b,steps=tuple(config['spatial']['ig_steps']))
                    pm,pn=integrated_gradients(perm_forward,image,b,steps=tuple(config['spatial']['ig_steps']))
                    return {'numerical':numeric,'fidelity':f,'random_similarity':signed_similarity(attribution,rm,mask),
                            'random_numerical':rn,'label_permutation_similarity':signed_similarity(attribution,pm,mask),
                            'label_permutation_numerical':pn,'map_sha256':file_sha(directory/(kind+'.npy'))}
                r=store.job(name+'_'+kind,compute)
                if file_sha(directory/(kind+'.npy'))!=r['map_sha256']:raise ValueError('SPATIAL_MAP_HASH_DRIFT')
                maps[kind]=np.load(directory/(kind+'.npy'));numerics[kind]=r['numerical'];fidelities[kind]=r['fidelity']
                similarities[kind]=r['random_similarity'];permutation[kind]=r['label_permutation_similarity']
                controls[kind]=(r['random_numerical'],r['label_permutation_numerical'])
            def finish():
                folder_scan=ctx['parent']/'private/mri_scans'/row.mri_sample_hash
                reference=nib.load(folder_scan/'registered_t1.nii.gz')
                if reference.shape!=atlas_image.shape or not np.allclose(reference.affine,atlas_image.affine,atol=1e-5):
                    raise ValueError('SPATIAL_ATLAS_GEOMETRY_MISMATCH')
                for kind,a in maps.items():save_nifti(directory/(kind+'.nii'),a,reference.affine)
                overlay(reference,{k:nib.Nifti1Image(a,reference.affine) for k,a in maps.items()},
                        directory/'model_space.png','Actual MRI; fixed component log-hazard attribution')
                native=native_maps(row.raw_path,folder_scan/'registered_t1.nii.gz',folder_scan,
                                    maps,mask,directory/'native')
                status={k:attribution_status(numerics[k],similarities[k],fidelities[k],controls[k][0],
                    permutation[k],controls[k][1],config['spatial']['max_randomized_correlation']) for k in maps}
                return {'case':name,'fold':fold,'component':0,'seed':seed,'baseline_status':status,
                        'numerical':numerics,'control_numerical':controls,'random_similarity':similarities,'label_permutation_similarity':permutation,
                        'baseline_correlations':{k:signed_similarity(maps['zero'],a,mask) for k,a in maps.items() if k!='zero'},
                        'label_permutation_checkpoint':perm_receipt['sha256'],'native':native,
                        'explains_entire_ensemble':False,'disease_localization_allowed':False,
                        'atlas_status':atlas_receipt['status'],
                        'roi_attribution':{k:roi_summary(a,atlas_labels,atlas_assets['labels']) for k,a in maps.items()}
                            if atlas_receipt['status']=='APPROXIMATE_ANATOMICAL_LABELING_ONLY' else {},
                        'files':{p.relative_to(store.root).as_posix():file_sha(p) for p in directory.rglob('*') if p.is_file()}}
            cases.append(store.job(name,finish))
        del model,perm
        if device.type=='cuda':torch.cuda.empty_cache()
    # Public output never includes a patient key, path or map for an individual.
    return {'status':'BASELINE_SPECIFIC_COMPONENT_DIAGNOSTICS_COMPLETED','case_count':len(cases),
            'baseline_status_counts':{kind:{status:sum(c['baseline_status'][kind]==status for c in cases)
                 for status in sorted({c['baseline_status'][kind] for c in cases})} for kind in cases[0]['baseline_status']},
            'ensemble_attribution_validated':False,'anatomical_lesion_localization':False,
            'atlas_status':cases[0]['atlas_status'],
            'atlas_quality':{k:atlas_receipt[k] for k in ('mask_dice','positive_jacobian_fraction','present_labels','expected_labels')},
            'individual_images_public':False,
            'label_permutation_training_executed':True}
