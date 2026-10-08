"""Hash-verified private individual maps + aggregate-only diagnostic review.

Rendering does not smooth or improve a failed map, and never marks Alzheimer
lesions. A snapshot is immutable; further completed cases create a new snapshot.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/mpl_spatial_review_v1')
import numpy as np
import nibabel as nib
from src.method_y_revival.atomic_io import atomic_write_bytes,atomic_write_json,read_json
from src.method_y_revival.contracts import sha256_mapping
from src.method_y_revival.local_validation_v5 import committed_job,file_sha
from src.method_y_revival.privacy import scan_public_bundle

SOURCE=ROOT/'results/xy_spatial_validation_v3'
OUT=ROOT/'results/xy_spatial_review_v1'


def require(value,message):
    if not value: raise ValueError(message)


def render_case(directory, report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=nib.load(directory/'processed_individual_input.nii')
    bg=nib.as_closest_canonical(source).get_fdata()
    require(np.isfinite(bg).all(),'NONFINITE_BACKGROUND')
    figures={}
    for baseline in ('zero','blur'):
        source_map=nib.load(directory/f'ig_{baseline}.nii')
        require(source_map.shape==source.shape and np.allclose(source_map.affine,source.affine,atol=1e-7),
                'MAP_BACKGROUND_GEOMETRY_MISMATCH')
        a=nib.as_closest_canonical(source_map).get_fdata()
        require(np.isfinite(a).all(),'NONFINITE_MAP')
        state=report[baseline]['numerical']['numerical_status']
        support=np.abs(a[a!=0]); limit=max(float(np.quantile(support,.99)) if support.size else 0,1e-12)
        fig,axs=plt.subplots(3,3,figsize=(13,12),layout='constrained')
        for axis in range(3):
            for j,fraction in enumerate((.35,.5,.65)):
                index=int((bg.shape[axis]-1)*fraction)
                plane=np.take(bg,index,axis=axis)
                heat=np.take(a,index,axis=axis)
                ax=axs[axis,j]
                ax.imshow(np.rot90(plane),cmap='gray',origin='upper')
                im=ax.imshow(np.rot90(np.ma.masked_where(np.abs(heat)<.1*limit,heat)),
                    cmap='coolwarm',vmin=-limit,vmax=limit,alpha=.65,origin='upper')
                voxel=np.array([0,0,0,1.]);voxel[axis]=index
                mm=float((nib.as_closest_canonical(source).affine@voxel)[axis])
                ax.set_title(f"{('Sagittal','Coronal','Axial')[axis]} / {('x','y','z')[axis]}={mm:.1f} mm")
                # Canonical RAS, neurological convention, no silent left/right flip.
                ax.text(.01,.50,'P' if axis==0 else 'L',transform=ax.transAxes,color='yellow')
                ax.text(.97,.50,'A' if axis==0 else 'R',transform=ax.transAxes,color='yellow')
                ax.text(.5,.98,'A' if axis==2 else 'S',transform=ax.transAxes,color='yellow',va='top')
                ax.axis('off')
        fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.65,label='Signed model attribution; p99 display clipping only')
        fig.suptitle(f'Individual processed MRI/PET | IG baseline: {baseline} | numerical QC: {state}\n'
            'RED: increases model target; BLUE: decreases target relative to baseline\n'
            'Not Alzheimer lesions; not clinical diagnosis. Spatial fidelity NOT yet established.',fontsize=13)
        data=io.BytesIO();fig.savefig(data,format='png',dpi=115);plt.close(fig)
        path=output/f'{baseline}_individual_multislice.png'
        atomic_write_bytes(path,data.getvalue());figures[path.name]=file_sha(path)
    return figures


def main():
    contract=read_json(SOURCE/'private/contract.json')
    expected={r['key']:r for r in contract['cases']}
    files=list((SOURCE/'private/jobs').glob('*.json'))
    require(set(p.stem for p in files)<=set(expected),'UNEXPECTED_CASE_RECEIPT')
    binding={'parent_contract_sha256':sha256_mapping(contract),
        'receipts':{p.stem:file_sha(p) for p in sorted(files)},'source_sha256':file_sha(Path(__file__))}
    snapshot=OUT/sha256_mapping(binding)[:16]
    receipt=snapshot/'review_receipt.json'
    if receipt.exists():
        prior=read_json(receipt)
        require(prior['binding']==binding,'REVIEW_BINDING_MISMATCH')
        for path,digest in prior['outputs'].items(): require(file_sha(snapshot/path)==digest,'REVIEW_OUTPUT_DRIFT')
        print(json.dumps({'status':'REVIEW_RECOVERED','cases':len(files)}));return
    aggregate={b:{'cases':0,'zero_numerical_pass':0,'blur_numerical_pass':0,
                  'blur_positive_margin':0,'blur_negative_margin':0,'random_signed_abs_correlations':[]}
               for b in ('x','amyloid','tau')}
    outputs={};details={}
    for p in sorted(files):
        key=p.stem; row=expected[key]
        report=committed_job(SOURCE/'private',key,contract,None,verify_only=True)
        directory=SOURCE/'private/cases'/key
        for name,digest in report['outputs'].items(): require(file_sha(directory/name)==digest,'PARENT_OUTPUT_DRIFT')
        a=aggregate[row['branch']];a['cases']+=1
        for baseline in ('zero','blur'):
            a[baseline+'_numerical_pass']+=int(report[baseline]['numerical']['numerical_status']=='PASS')
        for sign in ('positive','negative'):
            a['blur_'+sign+'_margin']+=int(report['blur']['perturbation'][sign].get('directional_margin_vs_random_mean',-1)>0)
        corr=report['randomization']['signed_map_similarity']['pearson']
        if corr is not None: a['random_signed_abs_correlations'].append(abs(corr))
        figures=render_case(directory,report,snapshot/'private'/key)
        outputs.update({f'private/{key}/{n}':d for n,d in figures.items()})
        details[key]={'baseline_sensitivity':report['baseline_sensitivity'],
            'zero_numerical':report['zero']['numerical'],'blur_numerical':report['blur']['numerical'],
            'randomization':report['randomization'],'descriptive_perturbations':report['blur']['perturbation'],
            'elapsed_seconds':report['elapsed_seconds']}
    # Do not export individual correlations, maps, case identities or small-group diagnostics.
    public_branches={}
    for b,a in aggregate.items():
        if a['cases']<10:
            public_branches[b]={'status':'PENDING_TEN_CASES_AGGREGATE_SUPPRESSED'};continue
        correlations=a.pop('random_signed_abs_correlations')
        public_branches[b]={**a,'random_signed_abs_correlation_median':float(np.median(correlations)) if correlations else None,
                            'random_signed_abs_correlation_max':max(correlations) if correlations else None}
    public={'status':'FULL_DIAGNOSTIC_REVIEW_NOT_DISEASE_LOCALIZATION' if len(files)==30 else 'PARTIAL_DIAGNOSTIC_REVIEW',
        'cases_completed':len(files),'cases_required':30,'branches':public_branches,
        'map_selection_by_visual_appearance':False,'disease_localization_allowed':False,
        'science_gate_status':'NOT_ESTABLISHED','claims':'POST_HOC_INTERNAL_EXPLORATORY',
        'label_permutation_training':'NOT_EXECUTED','population_representativeness':'LIMITED_FIXED_PANEL',
        'equal_volume_controls_not_anatomical_roi_tests':True}
    atomic_write_json(snapshot/'private/diagnostics.json',details)
    atomic_write_json(snapshot/'public_bundle/review.json',public)
    require(not scan_public_bundle(snapshot/'public_bundle'),'PUBLIC_PRIVACY_FAILURE')
    for rel in ('private/diagnostics.json','public_bundle/review.json'):outputs[rel]=file_sha(snapshot/rel)
    atomic_write_json(receipt,{'binding':binding,'outputs':outputs,'status':'HASH_VERIFIED'})
    atomic_write_json(OUT/'latest.json',{'snapshot':snapshot.name,'review_receipt_sha256':file_sha(receipt),
                                    'cases_completed':len(files),'cases_required':30})
    print(json.dumps({'status':public['status'],'cases':len(files),'snapshot':snapshot.name}),flush=True)


if __name__=='__main__':main()
