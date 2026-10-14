"""Outcome-blind, scan-resumable anatomical registration and lossless float32 cache."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
import os
from pathlib import Path
import time
import numpy as np
from .store import atomic_json, digest, file_sha, writer_lock
from .runtime import cpu_workers,peak_rss


def render_qc(registered,reference,output):
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=nib.load(registered).get_fdata(dtype=np.float32)
    b=nib.load(reference).get_fdata(dtype=np.float32)
    fig,axes=plt.subplots(2,3,figsize=(9,6),dpi=110)
    for j,axis in enumerate((0,1,2)):
        sl=np.rot90(np.take(a,a.shape[axis]//2,axis=axis))
        ref=np.rot90(np.take(b,b.shape[axis]//2,axis=axis))
        for i,value in enumerate((sl,ref)):
            axes[i,j].imshow(value,cmap='gray',vmin=np.percentile(value,5),vmax=np.percentile(value,99))
            axes[i,j].axis('off')
        axes[0,j].set_title('Registered subject T1')
        axes[1,j].set_title('Locked template (reference only)')
    fig.tight_layout();fig.savefig(output,bbox_inches='tight');plt.close(fig)


def validate_reference(template_dir):
    d=Path(template_dir); lock=json.loads((d/'reference_128_1p5mm.lock.json').read_text())
    names={'reference_128_1p5mm.nii.gz':'output_template_sha256',
           'tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz':'source_brain_mask_sha256',
           'LICENSE':'license_text_sha256'}
    for name,key in names.items():
        if file_sha(d/name)!=lock[key]:
            raise ValueError('REFERENCE_ASSET_HASH_MISMATCH:'+name)
    import nibabel as nib
    image=nib.load(d/'reference_128_1p5mm.nii.gz')
    if tuple(image.shape)!=(128,128,128) or not np.allclose(image.affine,lock['output_affine'],atol=1e-6):
        raise ValueError('REFERENCE_GEOMETRY_MISMATCH')
    return lock


def normalize_image(array, mask, percentiles):
    a=np.asarray(array,dtype=np.float32); m=np.asarray(mask,dtype=bool)
    if a.shape!=m.shape or not np.isfinite(a).all() or m.sum()<100:
        raise ValueError('INVALID_MRI_NUMERIC_OR_MASK')
    values=a[m]; lo,hi=np.percentile(values,percentiles)
    values=np.clip(values,lo,hi)
    sd=float(values.std())
    if sd<1e-6:
        raise ValueError('MRI_CONSTANT_SIGNAL')
    result=np.zeros(a.shape,dtype=np.float32)
    result[m]=(values-values.mean())/sd
    return result


def checked_receipt(folder, expected):
    receipt=Path(folder)/'receipt.json'
    if not receipt.exists():
        return None
    r=json.loads(receipt.read_text())
    if r['binding']!=expected or digest(r['payload'])!=r['payload_sha256']:
        raise ValueError('MRI_SCAN_RECEIPT_MISMATCH')
    for name,sha in r['payload'].get('outputs',{}).items():
        if file_sha(Path(folder)/name)!=sha:
            raise ValueError('MRI_SCAN_OUTPUT_TAMPERED')
    return r['payload']


def process_scan(job):
    row,root,template_dir,policy,contract=job
    # Worker imports ANTs after setting its internal thread cap.
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']=str(policy['itk_threads_per_worker'])
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]=str(policy['itk_threads_per_worker'])
    root=Path(root); folder=root/'private/mri_scans'/row['mri_sample_hash']
    folder.mkdir(parents=True,exist_ok=True)
    import shutil
    if shutil.disk_usage(root).free<12*1024**3:
        raise ValueError('H_DISK_RESERVE_BELOW_12_GIB')
    raw=Path(row['raw_path']); st=raw.stat()
    if st.st_size!=row['bytes'] or st.st_mtime_ns!=row['mtime_ns']:
        raise ValueError('RAW_T1_CHANGED_SINCE_MANIFEST')
    raw_sha=file_sha(raw)
    binding=digest({'raw':raw_sha,'policy':policy,'contract':contract})
    with writer_lock(folder):
        old=checked_receipt(folder,binding)
        if old is not None:
            return old
        import ants
        import nibabel as nib
        import nibabel.processing as nip
        started=time.perf_counter()
        img=nib.load(raw)
        a=img.get_fdata(dtype=np.float32)
        if a.ndim!=3 or not np.isfinite(a).all() or abs(np.linalg.det(img.affine[:3,:3]))<1e-6:
            payload={'status':'EXCLUDED','reason':'INVALID_NATIVE_GEOMETRY_OR_SIGNAL','raw_sha256':raw_sha,'outputs':{}}
        else:
            moving=ants.image_read(str(raw))
            positive=a[a>0]
            if len(positive)<100 or np.std(positive)<1e-6:
                raise ValueError('INVALID_NATIVE_MRI_INTENSITY')
            if policy['n4']:
                moving=ants.n4_bias_field_correction(moving,shrink_factor=4,verbose=False)
            d=Path(template_dir); refpath=d/'reference_128_1p5mm.nii.gz'
            ref=nib.load(refpath); fixed=ants.image_read(str(refpath))
            mask=nip.resample_from_to(nib.load(d/'tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz'),ref,order=0).get_fdata()>0.5
            registration=ants.registration(fixed=fixed,moving=moving,type_of_transform=policy['registration'],
                reg_iterations=tuple(policy['registration_iterations']),random_seed=policy['seed'],
                outprefix=str(folder/'registration_'),singleprecision=True,verbose=False)
            warped=registration['warpedmovout']; image_path=folder/'registered_t1.nii.gz'
            ants.image_write(warped,str(image_path))
            ni=nib.load(image_path); w=ni.get_fdata(dtype=np.float32)
            if w.shape!=ref.shape or not np.allclose(ni.affine,ref.affine,atol=1e-4) or not np.isfinite(w).all():
                raise ValueError('REGISTERED_MRI_GEOMETRY_OR_SIGNAL_FAILED')
            target=ref.get_fdata(dtype=np.float32)
            corr=float(np.corrcoef(w[mask],target[mask])[0,1])
            coverage=float(np.mean(w[mask]>np.percentile(positive,1)))
            jac_path=next((p for p in registration['fwdtransforms'] if p.endswith('.nii.gz')),None)
            if jac_path is None:
                raise ValueError('NONLINEAR_TRANSFORM_MISSING')
            j=ants.create_jacobian_determinant_image(fixed,jac_path,do_log=False).numpy()
            jac_fraction=float(np.mean(np.isfinite(j)&(j>0)))
            qc={'template_correlation':corr if np.isfinite(corr) else None,'template_mask_coverage':coverage,
                'positive_jacobian_fraction':jac_fraction,'foreground_fraction':float(mask.mean()),
                'finite':bool(np.isfinite(w).all()),'geometry':True}
            passed=(np.isfinite(corr) and corr>=0.40 and coverage>=policy['qc_min_mask_overlap']
                    and jac_fraction>=policy['qc_min_positive_jacobian_fraction']
                    and policy['qc_min_foreground_fraction']<=mask.mean()<=policy['qc_max_foreground_fraction'])
            value=normalize_image(w,mask,policy['normalization_percentiles'])
            tmp=folder/'tensor.tmp'
            with tmp.open('wb') as f:
                np.save(f,value,allow_pickle=False); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,folder/'tensor.npy')
            render_qc(image_path,refpath,folder/'qc_montage.png')
            # Save original-resolution transforms for future native-space attribution.
            paths=[image_path,folder/'tensor.npy',folder/'qc_montage.png',*map(Path,registration['fwdtransforms']),*map(Path,registration['invtransforms'])]
            if any(p.parent.resolve()!=folder.resolve() for p in paths):
                raise ValueError('TRANSFORM_ESCAPED_PRIVATE_FOLDER')
            payload={'status':'PASS' if passed else 'EXCLUDED','reason':None if passed else 'AUTOMATED_ANATOMICAL_QC_FAILED',
                     'raw_sha256':raw_sha,'sample_key':row['mri_sample_hash'],'qc':qc,
                     'affine':ni.affine.tolist(),'outputs':{p.name:file_sha(p) for p in paths},
                     'seconds':time.perf_counter()-started,
                     'peak_rss_bytes':peak_rss(),
                     'qc_review_type':'AUTOMATED_NUMERIC_GEOMETRY_TEMPLATE_CONCORDANCE_NOT_HUMAN_REVIEW'}
        atomic_json(folder/'receipt.json',{'binding':binding,'payload':payload,'payload_sha256':digest(payload)})
        return payload


def prepare_cache(frame,out,template_dir,policy,contract,*,limit=0):
    out=Path(out); lock=validate_reference(template_dir)
    path=out/'private/mri_cache.npy'; receipt=out/'private/mri_cache.json'
    if receipt.exists() and not limit:
        r=json.loads(receipt.read_text())
        if r.get('parent_contract')!=contract or file_sha(path)!=r['sha256']:
            raise ValueError('MRI_CACHE_INTEGRITY_MISMATCH')
        # Training consumes this hash-verified consolidated cache, not each
        # auxiliary registration volume again. QC receipts remain byte-bound.
        for key,sha in r['qc_receipt_sha256'].items():
            if file_sha(out/'private/mri_scans'/key/'receipt.json')!=sha:
                raise ValueError('MRI_QC_RECEIPT_DRIFT')
        keys=r['selected_sample_keys']
        if len(keys)!=len(set(keys)) or not set(keys).issubset(frame.mri_sample_hash):
            raise ValueError('MRI_CACHED_MEMBERSHIP_MISMATCH')
        selected=frame.set_index('mri_sample_hash').loc[keys].reset_index()
        selected['cache_row']=range(len(selected))
        print(json.dumps({'event':'CONSOLIDATED_CACHE_VERIFIED_REUSED','scans':len(selected)}),flush=True)
        return selected,r
    rows=frame.to_dict('records'); rows=rows[:limit] if limit else rows
    workers=cpu_workers(policy)
    print(json.dumps({'event':'MRI_PREPROCESS_START','workers':workers,'itk_threads':policy['itk_threads_per_worker'],'scans':len(rows)}),flush=True)
    jobs=[(r,str(out),str(template_dir),policy,contract) for r in rows]
    results=[]
    def collect(batch,worker_count):
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            iterator=iter(batch)
            pending={pool.submit(process_scan,j) for j in __import__('itertools').islice(iterator,worker_count)}
            try:
                while pending:
                    finished,pending=wait(pending,return_when=FIRST_COMPLETED)
                    for future in finished:
                        r=future.result();results.append(r)
                        print(json.dumps({'event':'MRI_SCAN_COMMITTED_OR_RECOVERED','completed':len(results),'total':len(rows),'status':r['status']}),flush=True)
                    for job in __import__('itertools').islice(iterator,len(finished)):
                        pending.add(pool.submit(process_scan,job))
            except BaseException:
                for future in pending:future.cancel()
                print(json.dumps({'event':'STOPPING_NO_NEW_SCANS','remaining_inflight_at_most':worker_count,
                                  'resume':'FINISHED_SCAN_RECEIPTS_ARE_PRESERVED'}),flush=True)
                raise
    # First two outcome-blind scans measure actual memory, not an imagined 100%
    # utilization target. A 1.75x memory margin + 4 GiB OS reserve remains.
    collect(jobs[:2],min(2,workers))
    observed=max((r.get('peak_rss_bytes',0) for r in results),default=0)
    if observed:
        measured=dict(policy);measured['ram_gib_per_worker']=max(1.5,1.75*observed/1024**3)
        workers=cpu_workers(measured)
    print(json.dumps({'event':'MRI_WORKER_POLICY','workers':workers,'pilot_peak_rss_bytes':observed}),flush=True)
    if len(jobs)>2:collect(jobs[2:],workers)
    if limit:
        return {'scope':'SMOKE_ONLY','rows':results}
    good={r['sample_key']:r for r in results if r['status']=='PASS'}
    selected=frame.loc[frame.mri_sample_hash.isin(good)].copy().reset_index(drop=True)
    if len(selected)<0.9*len(frame):
        raise ValueError('MRI_QC_EXCLUSIONS_OVER_10_PERCENT_REVIEW_REQUIRED')
    binding=digest({'contract':contract,'rows':[(r.mri_sample_hash,good[r.mri_sample_hash]['outputs']['tensor.npy']) for r in selected.itertuples()]})
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['binding']!=binding or file_sha(path)!=r['sha256']:
            raise ValueError('MRI_CACHE_INTEGRITY_MISMATCH')
    else:
        tmp=path.with_suffix('.tmp.npy')
        array=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(selected),1,128,128,128))
        for i,row in enumerate(selected.itertuples()):
            array[i,0]=np.load(out/'private/mri_scans'/row.mri_sample_hash/'tensor.npy',allow_pickle=False)
        array.flush(); del array
        with tmp.open('r+b') as f:os.fsync(f.fileno())
        os.replace(tmp,path)
        atomic_json(receipt,{'binding':binding,'sha256':file_sha(path),'shape':[len(selected),1,128,128,128],
                            'template':lock['output_template_sha256'],'excluded':len(frame)-len(selected),
                            'parent_contract':contract,'selected_sample_keys':list(selected.mri_sample_hash),
                            'qc_receipt_sha256':{row['mri_sample_hash']:file_sha(
                                out/'private/mri_scans'/row['mri_sample_hash']/'receipt.json') for row in rows}})
    selected['cache_row']=range(len(selected))
    return selected,json.loads(receipt.read_text())
