"""Licensed anatomical labels with an explicit template-to-template transform.

No source mask becomes a quantitative PET biomarker or an individual diagnosis.
"""
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.request import Request,urlopen
import numpy as np
from scripts.xyz_fusion_v1.store import atomic_json,file_sha


def source_assets(root):
    root=Path(root);out=root/'results/xyz_fusion_v2_assets';lock=out/'source_registry.json'
    if lock.exists():
        r=json.loads(lock.read_text())
        for path,sha in r['files'].items():
            if file_sha(root/path)!=sha:raise ValueError('ATLAS_SOURCE_ASSET_HASH_DRIFT')
        return r
    out.mkdir(parents=True,exist_ok=True)
    old=root/'results/spatial_attribution_v1/provenance/atlas_registry.json'
    registry=json.loads(old.read_text());assets=old.parent/'assets';atlas=assets/'harvard_oxford/data/atlases'
    refs={}
    for name in ('HarvardOxford-cort-maxprob-thr25-2mm.nii.gz',
                 'HarvardOxford-sub-maxprob-thr25-2mm.nii.gz','Cerebellum-MNIflirt-maxprob-thr25-2mm.nii.gz'):
        p=atlas/'HarvardOxford'/name
        if file_sha(p)!=registry['input_sha256'][name]:raise ValueError('OFFICIAL_ATLAS_HASH_MISMATCH')
        refs[name]=p
    license=assets/'CC-BY-SA-4.0.txt'
    if file_sha(license)!=registry['input_sha256'][license.name]:raise ValueError('ATLAS_LICENSE_HASH_MISMATCH')
    refs['atlas_license']=license;refs['original_registry']=old
    urls={}
    def fetch(url,path):
        with urlopen(Request(url,headers={'User-Agent':'XYZ-research-asset-provenance/2'}),timeout=45) as response:
            data=response.read()
        temp=path.with_suffix(path.suffix+'.tmp')
        with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temp,path);urls[path.name]=url
        return data
    for kind in ('T1w','desc-brain_mask'):
        name=f'tpl-MNI152NLin6Asym_res-02_{kind}.nii.gz'
        pointer=fetch('https://raw.githubusercontent.com/templateflow/tpl-MNI152NLin6Asym/master/'+name,out/(name+'.annex.txt')).decode()
        m=re.search(r'MD5E-s(\d+)--([0-9a-f]{32})',pointer)
        if not m:raise ValueError('TEMPLATEFLOW_PUBLISHED_DIGEST_MISSING')
        payload=fetch('https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/'+name,out/name)
        if len(payload)!=int(m[1]) or hashlib.md5(payload).hexdigest()!=m[2]:
            raise ValueError('TEMPLATEFLOW_PUBLISHED_ASSET_MISMATCH')
        refs[kind]=out/name;refs[kind+'_pointer']=out/(name+'.annex.txt')
    description=fetch('https://raw.githubusercontent.com/templateflow/tpl-MNI152NLin6Asym/master/template_description.json',out/'template_description.json')
    if json.loads(description)['Identifier']!='MNI152NLin6Asym':raise ValueError('TEMPLATE_ID_MISMATCH')
    terms=fetch('https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html',out/'fsl_terms.html').decode()
    if 'non-commercial' not in terms or 'Harvard-Oxford' not in terms:raise ValueError('ASSET_TERMS_NOT_CONFIRMED')
    refs['description']=out/'template_description.json';refs['fsl_terms']=out/'fsl_terms.html'
    target=root/'data/templates/method_y_mni152nlin2009casym_v1'
    refs['target_template']=target/'reference_128_1p5mm.nii.gz'
    refs['target_mask']=target/'tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz'
    r={'status':'SOURCE_ASSETS_LOCKED','source_space':'MNI152NLin6Asym','target_space':'MNI152NLin2009cAsym',
       'atlas_license':'CC-BY-SA-4.0','template_use':'LOCAL_NONCOMMERCIAL_RESEARCH_ONLY',
       'redistribution':False,'urls':urls,'labels':registry['labels'],
       'paths':{k:p.relative_to(root).as_posix() for k,p in refs.items()},
       'files':{p.relative_to(root).as_posix():file_sha(p) for p in refs.values()}}
    atomic_json(lock,r);return r


def bridge(root,store,assets):
    import ants
    from nibabel import load
    folder=store.root/'private/atlas_bridge';folder.mkdir(parents=True,exist_ok=True)
    p={k:Path(root)/v for k,v in assets['paths'].items()}
    def compute():
        source=ants.image_read(str(p['T1w']));target=ants.image_read(str(p['target_template']))
        sm=ants.image_read(str(p['desc-brain_mask']))
        tm=ants.resample_image_to_target(ants.image_read(str(p['target_mask'])),target,interp_type='nearestNeighbor')
        reg=ants.registration(target*tm,source*sm,type_of_transform='SyN',reg_iterations=(60,40,20),
                random_seed=20261003,outprefix=str(folder/'bridge_'),verbose=False)
        moved=ants.apply_transforms(target,sm,reg['fwdtransforms'],interpolator='nearestNeighbor')
        a=moved.numpy()>.5;b=tm.numpy()>.5
        dice=float(2*(a&b).sum()/(a.sum()+b.sum()))
        jac=ants.create_jacobian_determinant_image(target,reg['fwdtransforms'][0]).numpy()
        positive=float((jac[b]>0).mean())
        names=['HarvardOxford-cort-maxprob-thr25-2mm.nii.gz','HarvardOxford-sub-maxprob-thr25-2mm.nii.gz','Cerebellum-MNIflirt-maxprob-thr25-2mm.nii.gz']
        images=[ants.image_read(str(p[n])) for n in names]
        mapped=[ants.apply_transforms(target,im,reg['fwdtransforms'],interpolator='nearestNeighbor').numpy().astype('int32') for im in images]
        labels=mapped[0].copy()
        for v in [4,5,6,7,8,9,10,11,15,16,17,18,19,20,21]:labels[mapped[1]==v]=100+v
        for v in [int(k)-200 for k in assets['labels'] if int(k)>200]:labels[mapped[2]==v]=200+v
        known=set(map(int,assets['labels']));present=set(np.unique(labels))-{0}
        if not present<=known:raise ValueError('UNKNOWN_TRANSFORMED_ATLAS_LABEL')
        path=folder/'labels.nii.gz';ants.image_write(target.new_image_like(labels.astype('float32')),str(path))
        status='APPROXIMATE_ANATOMICAL_LABELING_ONLY' if dice>=.90 and positive>=.999 and len(present)>=.95*len(known) else 'BLOCKED_ATLAS_BRIDGE_QC'
        # Geometry, deformability and label coverage are separate automatic passes.
        warped=folder/'warped_template.nii.gz';ants.image_write(reg['warpedmovout'],str(warped))
        render_template_qc(warped,p['target_template'],folder/'template_bridge_qc.png')
        return {'status':status,'mask_dice':dice,'positive_jacobian_fraction':positive,
                'present_labels':len(present),'expected_labels':len(known),'missing_labels':sorted(known-present),
                'clinical_anatomical_accuracy_proven':False,'quantitative_pet_mask':False,
                'path':path.relative_to(store.root).as_posix(),'sha256':file_sha(path),
                'files':{f.relative_to(store.root).as_posix():file_sha(f) for f in folder.iterdir() if f.is_file()}}
    r=store.job('atlas_bridge',compute)
    for name,sha in r['files'].items():
        if file_sha(store.root/name)!=sha:raise ValueError('ATLAS_BRIDGE_ARTIFACT_DRIFT')
    return r


def roi_summary(attribution,atlas,labels):
    total=float(np.abs(attribution).sum());rows=[]
    for key,name in labels.items():
        values=np.asarray(attribution)[atlas==int(key)]
        if len(values):rows.append({'label':int(key),'name':name,'voxels':len(values),
                    'mean_signed':float(values.mean()),'absolute_share':float(np.abs(values).sum()/max(total,1e-30))})
    return rows


def render_template_qc(warped,target,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nibabel as nib
    images=[nib.as_closest_canonical(nib.load(p)).get_fdata() for p in (warped,target)]
    fig,axes=plt.subplots(2,3,figsize=(10,7),layout='constrained')
    for row,(image,title) in enumerate(zip(images,['Warped FSL template (NOT a participant)','Target 2009c template (NOT a participant)'])):
        for axis in range(3):
            view=np.rot90(np.take(image,image.shape[axis]//2,axis=axis))
            axes[row,axis].imshow(view,cmap='gray');axes[row,axis].set_title(title,fontsize=8);axes[row,axis].axis('off')
    fig.savefig(path,dpi=120);plt.close(fig)
