"""Resolve current source data once and prove identity to the frozen parent."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from .core import digest, file_sha, read_json, atomic_json


def parquet_atomic(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.parquet.tmp')
    frame.to_parquet(temporary, index=False)
    with temporary.open('r+b') as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Context:
    def __init__(self, root, parent, units, options):
        self.root, self.parent, self.units = Path(root), parent, units
        self.out = units.root
        self.config = parent.contract['protocol']
        self.options = options
        self.frame = pd.DataFrame(parent.contract['manifest'])
        self.splits = parent.job('splits')
        self.tau = self.pet = None
        self.cache_ctx = None
        self._banks = None

    def load_inputs(self):
        from scripts.xyz_fusion_v2.data import build_data, records, load_tau
        from scripts.xyz_fusion_v2.inference import pet_intake
        cache_ctx, frame, endpoints, inputs, summary = build_data(self.root, self.config)
        if digest(records(frame)) != digest(self.parent.contract['manifest']):
            raise ValueError('RECONSTRUCTED_PARENT_FRAME_CHANGED')
        tau, tau_audit = load_tau(self.root, cache_ctx, frame, self.config)
        pet, pet_audit = pet_intake(cache_ctx, frame)
        inputs.update({'pet_membership': pet_audit['membership_sha256'],
                       **{'pet_cache:' + k: v for k, v in pet_audit['hdf5_sha256'].items()}})
        inputs['pet_manifest'] = file_sha(self.root / 'data/preprocessed_pet/_private/manifests/fbp_private_manifest.csv')
        # Never call source_assets() here: if missing it can download/write assets.
        assets = read_json(self.root / 'results/xyz_fusion_v2_assets/source_registry.json')
        for p, sha in assets['files'].items():
            if self.parent.hashes(self.root / p) != sha:
                raise ValueError('ATLAS_SOURCE_DRIFT')
            inputs['atlas:' + p] = sha
        if inputs != self.parent.contract['input_sha256']:
            keys = sorted(k for k in set(inputs) | set(self.parent.contract['input_sha256'])
                          if inputs.get(k) != self.parent.contract['input_sha256'].get(k))
            raise ValueError('RAW_INPUT_HASH_DRIFT:' + ','.join(keys[:5]))
        plan = self.parent.job('analysis_plan')
        if digest(records(tau)) != plan['tau_features_sha256'] or digest(records(pet)) != plan['pet_membership_sha256']:
            raise ValueError('OPTIONAL_SUBSET_MEMBERSHIP_DRIFT')
        cache = self.root / 'results/xyz_mri_baseline_fast_v1/private/mri_cache.npy'
        if self.parent.hashes(cache) != self.parent.contract['parent_cache']['sha256']:
            raise ValueError('MRI_CACHE_BYTES_CHANGED')
        self.cache_ctx, self.frame, self.tau, self.pet = cache_ctx, frame, tau, pet
        return {'status': 'INPUTS_RECONSTRUCTED_AND_BYTE_VERIFIED', 'summary': summary,
                'tau': tau_audit, 'pet': {k: v for k, v in pet_audit.items() if k not in ('hdf5_sha256',)},
                'input_digest': digest(inputs), 'endpoints_digest': digest(endpoints)}

    def ensure_inputs(self):
        if self.tau is None:
            self.load_inputs()

    def banks(self):
        if self._banks is None:
            self.ensure_inputs()
            from scripts.xyz_fusion_v2.training import ImageBank
            from scripts.xyz_fusion_v2.inference import PETBank
            self._banks = {'mri': ImageBank(self.root / 'results/xyz_mri_baseline_fast_v1/private/mri_cache.npy', 8),
                           'pet': PETBank(self.pet, max_gib=4)}
        return self._banks

    def close(self):
        if self._banks is not None:
            self._banks['pet'].close()
            self._banks = None


def support_rows(frame, keys):
    sub = frame.loc[frame.patient_hash.isin(keys)]
    return {'n': len(sub), 'events': int(sub.event.sum()),
            'non_events': int((sub.event == 0).sum())}


def eligibility(ctx):
    ctx.ensure_inputs()
    frame, tau, pet = ctx.frame, ctx.tau, ctx.pet
    if frame.patient_hash.duplicated().any():
        raise ValueError('DUPLICATE_FUSION_SUBJECT')
    dates = ['mri_day', 'fbp_day', 'cognition_day']
    for column in dates:
        if frame[column].isna().any() or (frame[column] > frame.landmark_days_t0).any():
            raise ValueError('FUTURE_OR_MISSING_PREDICTOR_DATE:' + column)
    blood = frame.loc[frame.blood_ptau217.notna()]
    if blood.blood_day.isna().any() or (blood.blood_day > blood.landmark_days_t0).any():
        raise ValueError('BLOOD_POST_ORIGIN')
    from scripts.xyz_postprimary_v1.feasibility import audit_tau_landmark
    dated_tau = pd.DataFrame(audit_tau_landmark({'frame': frame})['private_rows'])
    dated_tau = dated_tau.loc[dated_tau.patient_hash.isin(tau.patient_hash)]
    if len(dated_tau) != len(tau) or dated_tau.patient_hash.duplicated().any():
        raise ValueError('EXACT_TAU_DATE_COVERAGE_INCOMPLETE')
    dated_tau = dated_tau.merge(frame[['patient_hash', 'landmark_days_t0']], validate='one_to_one')
    if (dated_tau.ftp_day > dated_tau.landmark_days_t0).any() or not dated_tau.origin.eq(dated_tau.landmark_days_t0).all():
        raise ValueError('TAU_LANDMARK_CHANGED_REUSE_FORBIDDEN')
    groups = {'mri': set(frame.patient_hash), 'pet': set(pet.patient_hash), 'tau': set(tau.patient_hash),
              'blood': set(blood.patient_hash)}
    common = set.intersection(*groups.values())
    rows = []
    from scripts.xyz_fusion_v2.models import assert_pair
    for track in ('mri', 'pet'):
        for seed in ctx.config['training_seeds']:
            for fold in range(5):
                for component in range(ctx.config['components']):
                    task = f'{track}_s{seed}_o{fold}_c{component}'
                    lock = ctx.parent.job(task + '_lock')
                    meta, pool, held = (set(lock[k]) for k in ('meta_subjects', 'expert_subjects', 'outer_subjects'))
                    if meta & pool or meta & held or pool & held:
                        raise ValueError('ANCESTRY_OVERLAP')
                    for modality, expert in lock['experts'].items():
                        assert_pair(expert, meta, held)
                        winner = ctx.parent.job(task + '_' + modality + '_winner')
                        if set(winner['selection_subjects']) - pool:
                            raise ValueError('SELECTION_ANCESTRY_OUTSIDE_POOL')
                        score = ctx.parent.job(task + '_' + modality + '_meta')
                        if set(score['subjects']) != meta or len(score['subjects']) != len(meta) or score['model_sha256'] != expert['sha256']:
                            raise ValueError('META_SCORE_ANCESTRY_MISMATCH')
                    for name, keys in [('all', groups[track]), ('tau', groups['tau']), ('four_way', common)]:
                        for role, ids in [('meta', meta), ('outer', held)]:
                            selected = ids & keys
                            support = support_rows(frame, selected)
                            for key in sorted(selected):
                                rows.append({'task': task, 'track': track, 'seed': seed, 'fold': fold,
                                             'component': component, 'subset': name, 'role': role,
                                             'patient_hash': key, 'reuse_allowed': True,
                                             'pool_subject_digest': digest(sorted(pool)),
                                             'parent_lock_digest': digest(lock), **support})
    path = ctx.out / 'private/eligibility.parquet'
    dated = frame[['patient_hash', 'landmark_days_t0', 'mri_day', 'fbp_day', 'cognition_day', 'blood_day']]
    dated = dated.merge(dated_tau[['patient_hash', 'ftp_day']], how='left', validate='one_to_one')
    parquet_atomic(path, pd.DataFrame(rows).merge(dated, on='patient_hash', validate='many_to_one'))
    summary = {name: support_rows(frame, ids) for name, ids in groups.items()}
    summary['four_way'] = support_rows(frame, common)
    by_task = pd.DataFrame(rows).drop_duplicates(['task', 'subset', 'role'])
    return {'status': 'EXACT_INPUT_AND_ALL_COMPONENT_ANCESTRY_VERIFIED',
            'tasks_verified': 5 * len(ctx.config['training_seeds']) * ctx.config['components'] * 2,
            'intersection_counts': summary, 'eligibility_row_count': len(rows),
            'component_support': by_task.drop(columns=['patient_hash']).to_dict('records'),
            'outputs': {path.relative_to(ctx.out).as_posix(): file_sha(path)},
            'legacy_tau_negative_preserved': True, 'dates_at_or_before_locked_origin': True,
            'reuse_scope': 'SAME_EXPERT_FOR_META_AND_OUTER_NOT_ANY_OLD_OOF'}
