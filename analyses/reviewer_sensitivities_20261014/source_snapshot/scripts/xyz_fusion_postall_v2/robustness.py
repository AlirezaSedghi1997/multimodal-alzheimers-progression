"""PET-manufacturer audits, correctly paired modality banks and unseen groups."""
from __future__ import annotations

import warnings
import numpy as np
import torch

from .core import digest, file_sha
from .experts import load_expert


def scanner_support(ctx):
    from scripts.xyz_fusion_v2.feasibility import optional_support
    ctx.ensure_inputs()
    # Parent scanner_splits explicitly accepts MRI columns only. PET uses its
    # own exact metadata and the same locked n/event policy, not a column alias.
    splits, excluded = pet_scanner_splits(ctx.pet, ctx.config)
    rows = []
    for split in splits:
        proof = optional_support(ctx.pet, [split], ctx.config, 'pet', image=True, require_five=False)
        rows.append({'manufacturer': split['group'], 'support': proof, 'split': split})
    return {'status': 'FEASIBILITY_EVALUATED', 'eligible': rows, 'excluded': excluded,
            'unknown_imputed': False, 'ge_and_philips_merged': False,
            'site_lmo_status': 'NOT_ESTIMABLE_NO_LOCKED_EXACT_PET_SITE_VARIABLE',
            'two_manufacturers_have_nested_support': sum(r['support']['status'] == 'SUPPORTED' for r in rows) >= 2}


def pet_scanner_splits(frame, config):
    splits, excluded = [], []
    for name, group in frame.groupby('pet_manufacturer', dropna=False):
        other = frame.loc[frame.pet_manufacturer != name]
        n, events = len(group), int(group.event.sum())
        if (isinstance(name, str) and name != 'UNKNOWN' and n >= config['secondary']['min_group_subjects']
                and events >= config['secondary']['min_group_events'] and int(other.event.sum()) >= 30):
            splits.append({'group': name, 'train': sorted(other.patient_hash), 'test': sorted(group.patient_hash)})
        else:
            excluded.append({'manufacturer': str(name), 'n': n, 'events': events, 'status': 'NOT_ESTIMABLE_SUPPORT'})
    return splits, excluded


def probe_support(train_labels, test_labels):
    # Scanner classification needs labels per class, not future disease events.
    counts = {x: {'train': int(np.sum(train_labels == x)), 'test': int(np.sum(test_labels == x))}
              for x in sorted(set(train_labels) | set(test_labels))}
    classes = [x for x, c in counts.items() if x != 'UNKNOWN' and c['train'] >= 10 and c['test'] >= 5]
    return classes, counts


def probe_arrays(xtr, ytr, xte, yte, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.exceptions import ConvergenceWarning
    classes, counts = probe_support(ytr, yte)
    if len(classes) < 2:
        return {'status': 'NOT_ESTIMABLE_SCANNER_CLASS_SUPPORT', 'class_counts': counts}
    tr, te = np.isin(ytr, classes), np.isin(yte, classes)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=seed))
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(xtr[tr], ytr[tr])
    p = model.predict_proba(xte[te])
    labels = model[-1].classes_
    auc = (roc_auc_score(yte[te] == labels[1], p[:, 1]) if len(labels) == 2 else
           roc_auc_score(yte[te], p, labels=labels, average='macro', multi_class='ovr'))
    return {'status': 'HELDOUT_SCANNER_PROBE_COMPLETED', 'class_counts': counts,
            'balanced_accuracy': float(balanced_accuracy_score(yte[te], model.predict(xte[te]))),
            'macro_ovr_auc': float(auc), 'scaler_fit_on_meta_only': True,
            'cross_expert_embeddings_concatenated': False, 'proves_robustness': False,
            'interpretation': 'DESCRIPTIVE_SEPARABILITY_OF_FROZEN_FEATURES_NOT_CAUSAL_SHORTCUT_PROOF'}


@torch.no_grad()
def embedding(model, bank, rows, policy):
    device = next(model.parameters()).device
    values = []
    for start in range(0, len(rows), policy['microbatch_size']):
        selected = rows.cache_row.iloc[start:start + policy['microbatch_size']].to_numpy()
        with torch.autocast(device.type, enabled=policy['amp'] and device.type == 'cuda'):
            e = model.encoder(bank.batch(selected, device))
        values.append(e.float().cpu().numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError('NONFINITE_PROBE_EMBEDDING')
    return result


def scanner_probes(ctx):
    ctx.ensure_inputs()
    results = []
    for seed in ctx.config['training_seeds']:
        for fold in range(5):
            for component in range(ctx.config['components']):
                task = f'pet_s{seed}_o{fold}_c{component}'
                lock = ctx.parent.job(task + '_lock')
                tr = ctx.pet.loc[ctx.pet.patient_hash.isin(lock['meta_subjects'])]
                te = ctx.pet.loc[ctx.pet.patient_hash.isin(lock['outer_subjects'])]
                ytr = tr.pet_manufacturer.to_numpy(str)
                yte = te.pet_manufacturer.to_numpy(str)
                classes, counts = probe_support(ytr, yte)
                for modality in ('mri', 'pet'):
                    def compute():
                        if len(classes) < 2:
                            return {'status': 'NOT_ESTIMABLE_SCANNER_CLASS_SUPPORT', 'class_counts': counts,
                                    'embeddings_extracted': False}
                        model, policy = load_expert(ctx, task, modality)
                        # The bank is keyed by modality, never PET for both encoders.
                        a = embedding(model, ctx.banks()[modality], tr, policy)
                        b = embedding(model, ctx.banks()[modality], te, policy)
                        del model
                        from scripts.xyz_postprimary_v1.context import write_array
                        path = ctx.out / 'private/embeddings' / (task + '_' + modality + '.npy')
                        write_array(path, np.concatenate([a, b]))
                        value = probe_arrays(a, ytr, b, yte, ctx.config['seed'])
                        return dict(value, bank_modality=modality, train_subjects=list(tr.patient_hash),
                                    test_subjects=list(te.patient_hash), checkpoint=lock['experts'][modality]['sha256'],
                                    outputs={path.relative_to(ctx.out).as_posix(): file_sha(path)})
                    result = ctx.units.job('f05_probe_' + task + '_' + modality, compute, {'lock': digest(lock)})
                    results.append({'task': task, 'modality': modality,
                                    **{k: v for k, v in result.items() if k not in ('outputs', 'train_subjects', 'test_subjects')}})
    return {'status': 'COMPLETED_PROBE_AUDIT', 'units': results,
            'tested_experts': len(results), 'no_single_pooled_embedding_space': True,
            'not_a_manufacturer_transport_validation': True}


def pet_lmo(ctx):
    feasibility = scanner_support(ctx)
    outputs = [{'manufacturer': row['manufacturer'], 'support': row['support'],
                'status': 'NOT_EXECUTED_NEW_CNN_TRAINING_OUT_OF_SCOPE' if row['support']['status'] == 'SUPPORTED'
                else 'NOT_ESTIMABLE_NESTED_SUPPORT'} for row in feasibility['eligible']]
    return {'status': 'SUPPORT_AUDITED_NO_NEW_LMO_TRAINING', 'groups': outputs,
            'support': feasibility, 'site_lmo_status': feasibility['site_lmo_status'],
            'scanner_science_gate': 'NOT_ESTABLISHED_BY_PROBES',
            'reusing_manufacturer_exposed_parent_experts_as_lmo_is_forbidden': True}
