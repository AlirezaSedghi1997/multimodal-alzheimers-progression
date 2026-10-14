"""Read-only score reuse, exact ancestry checks and complete OOF row coverage."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.xyz_fusion_postall_v2.core import ReadOnlyParent, Hashes, read_json, digest
from scripts.xyz_fusion_v2.models import assert_pair, score_features
from scripts.xyz_fusion_v1.statistics import check_curves


def require_keys(keys, expected, label):
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError('EXACT_UNIQUE_MEMBERSHIP_REQUIRED:' + label)


def pool_oof(parts, expected):
    result = pd.concat(parts, ignore_index=True)
    require_keys(list(result.patient_hash), expected, 'POOLED_OOF')
    return result.set_index('patient_hash').loc[sorted(expected)].reset_index()


class FrozenUnits:
    def __init__(self, root, parent):
        self.root = Path(root).resolve()
        self.contract = read_json(self.root / 'private/contract.json')
        self.binding = digest(self.contract)
        if self.contract['parent_contract'] != parent.binding:
            raise ValueError('SCORE_PARENT_MISMATCH')
        if self.contract.get('parent_execution_contract') != parent.execution_binding:
            raise ValueError('SCORE_EXECUTION_PARENT_MISMATCH')
        math = self.contract['math']
        if math['cnn_precision'] != 'FP32' or math['amp'] or math['tf32']:
            raise ValueError('MIXED_PRECISION_SCORE_REUSE_FORBIDDEN')
        if self.contract['cnn_weights_frozen'] is not True:
            raise ValueError('FROZEN_CNN_REQUIRED')
        self.receipts = {}

    def read(self, name):
        p = self.root / 'private/units' / (name + '.json')
        d = read_json(p)
        if d.get('contract') != self.binding or digest(d['payload']) != d.get('payload_sha256'):
            raise ValueError('FROZEN_SCORE_RECEIPT_DRIFT:' + name)
        self.receipts[name] = digest(d)
        return d


class Inputs:
    def __init__(self, root, options):
        self.root, self.options = Path(root).resolve(), options
        self.parent = ReadOnlyParent(self.root / options['parent_root'])
        self.scores = FrozenUnits(self.root / options['score_root'], self.parent)
        self.config = self.parent.contract['protocol']
        if (self.config['outer_folds'], self.config['components'], self.config['training_seeds']) != (5, 3, [13, 29, 47]):
            raise ValueError('UNSUPPORTED_PARENT_SPLITS')
        self.frame = pd.DataFrame(self.parent.contract['manifest'])
        require_keys(list(self.frame.patient_hash), self.frame.patient_hash, 'PARENT_FRAME')
        self.splits = self.parent.job('splits')
        if len(self.splits) != 5:
            raise ValueError('FIVE_OUTER_FOLDS_REQUIRED')
        outer = []
        full = set(self.frame.patient_hash)
        for split in self.splits:
            tr, te = set(split['train']), set(split['test'])
            require_keys(split['train'], full - te, 'OUTER_TRAIN')
            require_keys(split['test'], full - tr, 'OUTER_TEST')
            if tr & te or tr | te != full:
                raise ValueError('INVALID_OUTER_PARTITION')
            outer.extend(split['test'])
        require_keys(outer, full, 'OUTER_COVERAGE')
        tau = pd.read_parquet(self.root / options['tau_frame'])
        from scripts.xyz_fusion_v2.data import records
        if digest(records(tau)) != self.parent.job('analysis_plan')['tau_features_sha256']:
            raise ValueError('TAU_FRAME_NOT_LOCKED_PARENT')
        require_keys(list(tau.patient_hash), set(tau.patient_hash), 'TAU')
        if not set(tau.patient_hash) <= full:
            raise ValueError('TAU_OUTSIDE_PARENT')
        # Use parent covariates/outcomes; only the hash-verified tau column is added.
        tau = self.frame.merge(tau[['patient_hash', 'tau_temporal']], validate='one_to_one')
        pet_cells = self.parent.job('pet_ensemble_cells')
        pet_keys = [k for c in pet_cells for k in c['test_subjects']]
        require_keys(pet_keys, set(pet_keys), 'PET')
        if not set(pet_keys) <= full:
            raise ValueError('PET_OUTSIDE_PARENT')
        self.frames = {'mri': self.frame,
                       'pet': self.frame.loc[self.frame.patient_hash.isin(pet_keys)].copy(),
                       'blood': self.frame.loc[self.frame.blood_ptau217.notna()].copy(),
                       'tau': tau}
        self.locks, self.features, self.input_hashes = {}, {}, {}

    def split_frame(self, track, fold, role):
        keys = self.splits[fold]['train' if role == 'train' else 'test']
        f = self.frames[track]
        return f.loc[f.patient_hash.isin(keys)].sort_values('patient_hash').reset_index(drop=True)

    def validate_task(self, task):
        if task in self.locks:
            return self.locks[task]
        track = task.split('_')[0]
        fold = int(task.split('_o')[1].split('_')[0])
        lock = self.parent.job(task + '_lock')
        pool, meta, held = (set(lock[k]) for k in ('expert_subjects', 'meta_subjects', 'outer_subjects'))
        for name in ('expert_subjects', 'meta_subjects', 'outer_subjects'):
            require_keys(lock[name], set(lock[name]), name)
        if not all((pool, meta, held)) or pool & meta or pool & held or meta & held:
            raise ValueError('EXPERT_META_OUTER_OVERLAP')
        train_keys = set(self.split_frame(track, fold, 'train').patient_hash)
        test_keys = set(self.split_frame(track, fold, 'test').patient_hash)
        if pool | meta != train_keys or held != test_keys:
            raise ValueError('COMPONENT_OUTER_PARTITION_DRIFT')
        if set(lock['experts']) != ({'mri'} if track == 'mri' else {'mri', 'pet'}):
            raise ValueError('EXPERT_MODALITY_MISMATCH')
        for modality, receipt in lock['experts'].items():
            assert_pair(receipt, meta, held)
            require_keys(receipt['train_subjects'], pool, 'EXPERT_TRAIN')
            if not set(receipt['selection_subjects']) <= pool:
                raise ValueError('EXPERT_SELECTION_LEAKAGE')
            winner = self.parent.job(task + '_' + modality + '_winner')
            if not set(winner['selection_subjects']) <= pool:
                raise ValueError('TUNING_SELECTION_LEAKAGE')
            require_keys(winner['never_seen'], meta | held, 'NEVER_SEEN')
            checkpoint = self.parent.root / 'private/models' / f'{task}_{modality}_expert/locked.pt'
            if self.parent.hashes(checkpoint) != receipt['sha256']:
                raise ValueError('LOCKED_CHECKPOINT_DRIFT')
            self.input_hashes[checkpoint.relative_to(self.root).as_posix()] = receipt['sha256']
        self.locks[task] = lock
        return lock

    def feature_frame(self, task, role):
        key = (task, role)
        if key in self.features:
            return self.features[key].copy()
        lock = self.validate_task(task)
        keys = lock[role + '_subjects']
        frame = self.frame.set_index('patient_hash').loc[sorted(keys)].reset_index()
        # Outcomes are used only later by the fit/evaluation layer.
        for mod, expert in lock['experts'].items():
            name = f'{task}_{mod}_{role}_fp32'
            doc = self.scores.read(name); value = doc['payload']
            if doc['dependencies'] != {'parent_lock': digest(lock), 'subjects': digest(value['subjects'])}:
                raise ValueError('SCORE_DEPENDENCY_DRIFT')
            require_keys(value['subjects'], keys, 'SCORE')
            if (value['model_sha256'] != expert['sha256'] or value['precision'] != 'FP32'
                    or value['outcomes_passed_to_cnn'] is not False):
                raise ValueError('SCORE_EXPERT_OR_PRECISION_DRIFT')
            pred = check_curves(value['survival'], len(keys), self.config['fusion']['ibs_grid_days'])
            indexed = pd.DataFrame(pred, index=value['subjects']).loc[frame.patient_hash].to_numpy()
            frame = score_features(frame, indexed, self.config['fusion']['ibs_grid_days'],
                                   self.config['fusion']['score_times_days'], mod)
        self.features[key] = frame
        return frame.copy()

    def frames_for_head(self, track, seed, fold):
        source = 'pet' if track == 'pet' else 'mri'
        tasks = [f'{source}_s{seed}_o{fold}_c{c}' for c in range(3)]
        train = self.split_frame(track, fold, 'train')
        parts = [self.feature_frame(t, 'meta') for t in tasks]
        parent_train = self.split_frame(source, fold, 'train')
        pooled = pool_oof(parts, set(parent_train.patient_hash))
        pooled = pooled.loc[pooled.patient_hash.isin(train.patient_hash)].copy()
        require_keys(list(pooled.patient_hash), train.patient_hash, 'ELIGIBLE_POOLED_OOF')
        if track == 'tau':
            pooled = pooled.merge(train[['patient_hash', 'tau_temporal']], validate='one_to_one')
        return pooled.sort_values('patient_hash').reset_index(drop=True), tasks

    def outer_for_head(self, track, task, fold):
        test = self.split_frame(track, fold, 'outer')
        value = self.feature_frame(task, 'outer')
        value = value.loc[value.patient_hash.isin(test.patient_hash)].copy()
        require_keys(list(value.patient_hash), test.patient_hash, 'ELIGIBLE_OUTER')
        if track == 'tau':
            value = value.merge(test[['patient_hash', 'tau_temporal']], validate='one_to_one')
        columns = [c for c in value if c not in ('event', 'time_days') and not c.endswith(('_event', '_time_days'))]
        return value[columns].sort_values('patient_hash').reset_index(drop=True)

    def preflight(self):
        support = []
        for track in self.options['tracks']:
            for seed in self.config['training_seeds']:
                for fold in range(5):
                    train, tasks = self.frames_for_head(track, seed, fold)
                    for task in tasks:
                        self.outer_for_head(track, task, fold)
                    if len(train) < 40 or int(train.event.sum()) < 10:
                        raise ValueError('INSUFFICIENT_POOLED_HEAD_SUPPORT')
                    support.append({'track': track, 'seed': seed, 'fold': fold,
                                    'n': len(train), 'events': int(train.event.sum())})
        # Include original FP32 split-head evaluations to compare pooling at one precision.
        for track in ('mri', 'pet'):
            for fold in range(5):
                self.scores.read(f'{track}_o{fold}_evaluation')
        from scripts.xyz_fusion_v1.store import file_sha
        self.input_hashes[self.options['tau_frame']] = file_sha(self.root / self.options['tau_frame'])
        return {'status': 'PREFLIGHT_PASSED_NO_MODEL_FITTING', 'locked_expert_components': len(self.locks),
                'score_receipts': len(self.scores.receipts) - 10, 'head_pools': len(support),
                'support': support, 'new_cnn_training': False,
                'cohorts': {k: {'n': len(v), 'events': int(v.event.sum())} for k, v in self.frames.items()}}
