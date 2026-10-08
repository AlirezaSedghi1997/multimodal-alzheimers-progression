"""Canonical-volume duplicate search; private candidates never imply diagnosis."""
import hashlib
from collections import defaultdict

import numpy as np

from scripts.mx_fasttrack_v2.audit_inputs import read_csv
from scripts.mx_science_extension_v1.audit_and_lock import build_audit
from .common import CACHE, INDEX, EXT, digest, read, require, progress


def image_signature(image):
    x = np.asarray(image, dtype='<f4').squeeze().copy()
    require(x.ndim == 3 and np.isfinite(x).all() and x.std() > 0, 'INVALID_VOLUME')
    x[x == 0] = 0  # canonicalize signed zero
    require(all(d % 8 == 0 for d in x.shape), 'FINGERPRINT_GRID')
    small = x.reshape(8,x.shape[0]//8,8,x.shape[1]//8,8,x.shape[2]//8).mean(axis=(1,3,5)).ravel()
    small -= small.mean()
    norm = float(np.linalg.norm(small))
    require(norm > 0, 'CONSTANT_FINGERPRINT')
    return {'voxel_sha256': hashlib.sha256(x.tobytes(order='C')).hexdigest(),
            'fingerprint': (small/norm).astype(float).tolist()}


def check_lineage(index, fold_fit):
    # source_base_index belongs to the pre-QC source cache, not the compacted
    # current row number (30 failed QC rows were removed when building it).
    real = [r for r in index if str(r['is_synthetic']).lower() in ('0','false')]
    by_index = {int(r['source_base_index']): r for r in real}
    require(len(by_index)==len(real),'REAL_SOURCE_INDEX_DUPLICATION')
    eligible = defaultdict(int)
    synthetic = 0
    for row in index:
        if str(row['is_synthetic']).lower() in ('0','false'):
            continue
        synthetic += 1
        parent = by_index.get(int(row['source_base_index']))
        require(parent is not None and str(parent['is_synthetic']).lower() in ('0','false'), 'AUGMENT_PARENT_INVALID')
        require(row['source_sample_hash'] == parent['sample_hash'] and
                row['source_patient_hash'] == parent['patient_hash'] and
                row['patient_hash'] == parent['patient_hash'] and
                str(row['source_train_only']).lower() in ('1','true'), 'AUGMENT_LINEAGE_INVALID')
        fold = int(row['source_fold'])
        require(fold in range(5), 'AUGMENT_SOURCE_FOLD_INVALID')
        # A derivative may exist in cache without being eligible for a new seed.
        for seed in (13,29,47):
            if parent['patient_hash'] in fold_fit[fold,seed]:
                eligible[f'{fold}|{seed}'] += 1
        require(parent['patient_hash'] in fold_fit[fold,13], 'AUGMENT_PARENT_OUTSIDE_ORIGINAL_FIT')
    return {'materialized_derivatives': synthetic, 'eligible_parent_counts_by_fold_seed': dict(eligible),
            'scope': 'cache_lineage_and_eligibility_not_reconstruction_of_every_historical_optimizer_batch'}


def data_rows(root):
    index = read_csv(root / INDEX)
    real = {r['sample_hash']: r for r in index if str(r['is_synthetic']).lower() in ('0','false')}
    tests = [[r for r in read_csv(root / 'splits' / f'fold_{f}_test.csv') if r['sample_hash'] in real] for f in range(5)]
    return index, real, tests


def run_audit(store):
    import h5py
    from .tabular import partitions
    root = store.root
    base = build_audit(root)
    require(base['status'] == 'READY_INTERNAL_ONLY', 'BOUNDARIES_BLOCKED')
    index, real, tests = data_rows(root)
    fit = {(f,s): {r['patient_hash'] for r in partitions(root,f,s,real)[0]} for f in range(5) for s in (13,29,47)}
    lineage = check_lineage(index, fit)
    rows = sorted(real.values(), key=lambda r:int(r['row']))
    signatures = []
    with h5py.File(root / CACHE,'r') as h5:
        for i,row in enumerate(rows):
            def compute(row=row):
                idx = int(row['row'])
                sample = h5['sample_hash'][idx]
                sample = sample.decode() if isinstance(sample,bytes) else str(sample)
                require(sample == row['sample_hash'], 'CACHE_INDEX_DRIFT')
                return {'patient_hash':row['patient_hash'], 'cache_row':idx, **image_signature(h5['image'][idx])}
            result = store.receipt(f'volume_signatures/{i:04d}', compute)
            require(result['patient_hash'] == row['patient_hash'] and result['cache_row'] == int(row['row']), 'SIGNATURE_MEMBERSHIP')
            signatures.append(result)
            if i%50==0 or i==len(rows)-1:
                progress('image_audit', done=i+1, total=len(rows))
        groups = defaultdict(list)
        for i,sig in enumerate(signatures):
            groups[sig['voxel_sha256']].append(i)
        exact = [g for g in groups.values() if len({rows[i]['patient_hash'] for i in g})>1]
        matrix = np.asarray([r['fingerprint'] for r in signatures],dtype=float)
        similarities = matrix @ matrix.T
        a,b = np.where(np.triu(similarities >= store.config['fingerprint_cosine_candidate_threshold'],k=1))
        candidates = []
        for i,j in zip(a,b):
            if signatures[i]['voxel_sha256'] == signatures[j]['voxel_sha256']:
                continue
            def compare(i=int(i),j=int(j)):
                x = np.asarray(h5['image'][int(rows[i]['row'])],dtype=float)
                y = np.asarray(h5['image'][int(rows[j]['row'])],dtype=float)
                return {'first':i,'second':j,'cosine':float(similarities[i,j]),
                        'normalized_rmse':float(np.sqrt(np.mean((x-y)**2))/max(x.std(),y.std(),1e-12)),
                        'status':'UNRESOLVED_SIMILARITY_NOT_PROOF_OF_DUPLICATION'}
            candidates.append(store.receipt(f'near_duplicates/{i}_{j}',compare))
    store.receipt('duplicate_private_inventory',lambda:{'exact_groups':exact,'candidates':candidates})
    # Source hash tokens are a second independent duplicate screen (not a fresh raw-file audit).
    files = defaultdict(set)
    split_rows = [r for fold in tests for r in fold] + read_csv(root/'splits/external_holdout.csv')
    for row in split_rows:
        if row['sample_hash'] in real and row.get('t1_file_hash'):
            files[row['t1_file_hash']].add(row['patient_hash'])
    repeated_tokens = sum(len(v)>1 for v in files.values())
    status = 'BLOCKED_DUPLICATE_REVIEW' if exact or repeated_tokens else ('COMPLETED_WITH_REVIEW_REQUIRED' if candidates else 'COMPLETED_CONTENT_AUDIT')
    report = {'status':status,'real_volumes':len(rows),'exact_duplicate_groups':len(exact),
              'repeated_source_hash_tokens':repeated_tokens,'near_duplicate_candidates':len(candidates),
              'lineage':lineage, 'patient_partitions':'VERIFIED_FIT_INNER_OUTER_PSEUDO',
              'site_scanner':'UNKNOWN', 'raw_MRI_PET_files_rehashed':False,
              'pseudo_predictions_accessed':False, 'external_validation':False}
    store.receipt('data_audit',lambda:report,public=True)
    require(not exact and not repeated_tokens, 'CONFIRMED_IDENTICAL_CONTENT_OR_SOURCE_TOKEN_REQUIRES_REVIEW_NO_RETRAIN_AUTOMATICALLY')
    return report
