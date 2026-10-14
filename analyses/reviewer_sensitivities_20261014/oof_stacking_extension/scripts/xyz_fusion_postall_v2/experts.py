"""Frozen experts: exact checkpoint, architecture and subject ancestry."""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
import torch

from .core import digest, file_sha, inside


def load_expert(ctx, task, modality, *, randomized=False):
    from scripts.xyz_fusion_v2.training import MRISurvival
    lock = ctx.parent.job(task + '_lock')
    receipt = lock['experts'][modality]
    path = ctx.parent.root / 'private/models' / f'{task}_{modality}_expert/locked.pt'
    if ctx.parent.hashes(path) != receipt['sha256']:
        raise ValueError('FROZEN_EXPERT_HASH_DRIFT')
    policy = dict(ctx.config['training'], modality=modality)
    model = MRISurvival(policy)
    cp = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(cp['model'], strict=True)
    if randomized:
        # Reconstruct instead of walking reset_parameters(): custom learned
        # parameters such as encoder.scale_gates would otherwise survive.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(ctx.config['seed'] + int(digest((task, modality))[:6], 16))
            model = MRISurvival(policy)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).eval()
    model.requires_grad_(False)  # input gradients remain available; no weight update in post-all
    return model, policy


def load_head(ctx, task, arm):
    receipt = ctx.parent.job(task + '_' + arm)
    path = inside(ctx.parent.root, receipt['path'])
    if ctx.parent.hashes(path) != receipt['sha256']:
        raise ValueError('FROZEN_HEAD_HASH_DRIFT')
    with path.open('rb') as stream:
        model = pickle.load(stream)  # only byte-verified project-owned artifacts
    return model, receipt


def meta_frame(ctx, task, frame, modalities):
    from scripts.xyz_fusion_v2.models import score_features
    lock = ctx.parent.job(task + '_lock')
    result = frame.loc[frame.patient_hash.isin(lock['meta_subjects'])].copy()
    if not len(result):
        raise ValueError('EMPTY_MATCHED_META_POOL')
    for modality in modalities:
        value = ctx.parent.job(task + '_' + modality + '_meta')
        if value['model_sha256'] != lock['experts'][modality]['sha256']:
            raise ValueError('META_MODEL_DRIFT')
        indexed = pd.Series(list(value['survival']), index=value['subjects'])
        if not indexed.index.is_unique:
            raise ValueError('DUPLICATE_META_SCORE')
        values = np.asarray(indexed.loc[result.patient_hash].tolist())
        result = score_features(result, values, ctx.config['fusion']['ibs_grid_days'],
                                ctx.config['fusion']['score_times_days'], modality)
    return result


def outer_features(ctx, task, frame, modalities):
    from scripts.xyz_fusion_v2.training import predict
    from scripts.xyz_fusion_v2.models import score_features, assert_pair
    lock = ctx.parent.job(task + '_lock')
    result = frame.loc[frame.patient_hash.isin(lock['outer_subjects'])].copy()
    for modality in modalities:
        assert_pair(lock['experts'][modality], lock['meta_subjects'], result.patient_hash)
        def compute():
            model, policy = load_expert(ctx, task, modality)
            # Outcome columns are physically unavailable to inference.
            values = result[['patient_hash', 'cache_row']]
            curves = predict(model, ctx.banks()[modality], values, policy,
                             next(model.parameters()).device, ctx.config['fusion']['ibs_grid_days'])
            del model
            return {'subjects': list(values.patient_hash), 'survival': curves.tolist(),
                    'model_sha256': lock['experts'][modality]['sha256'], 'outcomes_passed': False}
        dependencies = {'lock': digest(lock), 'subjects': digest(list(result.patient_hash))}
        value = ctx.units.job('expert_outer_' + task + '_' + modality, compute, dependencies)
        if value['subjects'] != list(result.patient_hash):
            raise ValueError('OUTER_EXPERT_SUBJECT_ORDER_DRIFT')
        result = score_features(result, value['survival'], ctx.config['fusion']['ibs_grid_days'],
                                ctx.config['fusion']['score_times_days'], modality)
    return result
