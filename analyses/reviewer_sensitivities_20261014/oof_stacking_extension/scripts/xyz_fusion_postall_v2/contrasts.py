"""Matched small-head extensions, fixed features and no outer reselection."""
from __future__ import annotations

import numpy as np
from .core import digest
from .experts import meta_frame, outer_features


def common_support(ctx, track, subset, columns):
    from scripts.xyz_fusion_v1.statistics import support
    from scripts.xyz_fusion_v2.protocol import MRI, PET
    rows = []
    for seed in ctx.config['training_seeds']:
        for fold in range(5):
            for comp in range(ctx.config['components']):
                task = f'{track}_s{seed}_o{fold}_c{comp}'
                lock = ctx.parent.job(task + '_lock')
                meta = subset.loc[subset.patient_hash.isin(lock['meta_subjects'])]
                held = subset.loc[subset.patient_hash.isin(lock['outer_subjects'])]
                raw = [c for c in columns if c not in MRI + PET]
                reason = None
                try:
                    if len(meta) < ctx.options['minimum_matched_meta_subjects'] or meta.event.sum() < ctx.options['minimum_matched_meta_events']:
                        raise ValueError('MATCHED_META_SUPPORT_BELOW_LOCKED_STABILITY_POLICY_30_SUBJECTS_10_EVENTS')
                    if meta[raw].notna().sum().eq(0).any():
                        raise ValueError('ALL_MISSING_META_FEATURE')
                    support(meta, held, ctx.config['fusion']['ibs_grid_days'])
                except ValueError as exc:
                    reason = str(exc)
                rows.append({'task': task, 'meta_n': len(meta), 'meta_events': int(meta.event.sum()),
                             'outer_n': len(held), 'outer_events': int(held.event.sum()),
                             'nominal_features': len(columns),
                             'events_per_nominal_feature_not_power': float(meta.event.sum() / len(columns)),
                             'reason': reason})
    return {'status': 'READY_FIXED_RIDGE_EXPLORATORY' if all(r['reason'] is None for r in rows)
            else 'NOT_ESTIMABLE_COMMON_META_SUPPORT', 'cells': rows,
            'support_rule_not_proof_of_power': True, 'all_45_components_required': True}


def four_way_decision(ctx):
    from scripts.xyz_fusion_v2.protocol import CLINICAL, MRI, PET
    ctx.ensure_inputs()
    frame = ctx.pet.merge(ctx.tau[['patient_hash', 'tau_temporal']], on='patient_hash', validate='one_to_one')
    frame = frame.loc[frame.blood_ptau217.notna()].copy()
    columns = CLINICAL + MRI + PET + ['amyloid_centiloid', 'tau_temporal', 'blood_log_ptau217']
    value = common_support(ctx, 'pet', frame, columns)
    return dict(value, subjects=len(frame), events=int(frame.event.sum()),
                no_zero_imputed_missing_modality=True,
                action='RUN_MATCHED_FOUR_HEAD_CHAIN' if value['status'].startswith('READY') else 'DO_NOT_FIT_UNSUPPORTED_FOUR_WAY_HEADS')


def matched_extension(ctx, name):
    from scripts.xyz_fusion_v2.protocol import CLINICAL, MRI, PET
    from scripts.xyz_fusion_v2.pipeline import locked_tabular, average_predictions
    from .statistics import evaluate_cell
    from scripts.xyz_fusion_v2.models import curves
    from scripts.xyz_fusion_v2.reporting import paired_ci
    from .audit import incremental_verdict
    ctx.ensure_inputs()
    if name == 'tau':
        frame, track, modalities = ctx.tau, 'mri', ['mri']
        base = CLINICAL + MRI + ['amyloid_centiloid']
        arms = {'base': base, 'plus_tau': base + ['tau_temporal']}
    elif name == 'four_way':
        frame = ctx.pet.merge(ctx.tau[['patient_hash', 'tau_temporal']], on='patient_hash', validate='one_to_one')
        frame = frame.loc[frame.blood_ptau217.notna()].copy()
        track, modalities = 'pet', ['mri', 'pet']
        base = CLINICAL + MRI + ['amyloid_centiloid']
        arms = {'base': base, 'plus_pet': base + PET,
                'plus_tau': base + PET + ['tau_temporal'],
                'plus_blood': base + PET + ['tau_temporal', 'blood_log_ptau217']}
    else:
        raise ValueError('UNKNOWN_MATCHED_EXTENSION')
    proof = common_support(ctx, track, frame, list(arms.values())[-1])
    if not proof['status'].startswith('READY'):
        return dict(proof, subjects=len(frame), events=int(frame.event.sum()),
                    status='NOT_ESTIMABLE_COMMON_META_SUPPORT', results_cherry_picked=False)
    cells = []
    grid = ctx.config['fusion']['ibs_grid_days']
    for fold in range(5):
        components = []
        for seed in ctx.config['training_seeds']:
            for comp in range(ctx.config['components']):
                task = f'{track}_s{seed}_o{fold}_c{comp}'
                lock = ctx.parent.job(task + '_lock')
                meta = meta_frame(ctx, task, frame, modalities)
                heads = {}
                receipts = {}
                # Units exposes the small Store interface without parent mutation.
                for arm, features in arms.items():
                    heads[arm], receipts[arm] = locked_tabular(ctx.units, name + '_' + task + '_' + arm,
                        meta, features, 'cox_ridge', ctx.config['fusion']['ridge_alpha'], ctx.config, seed)
                def infer():
                    held = outer_features(ctx, task, frame, modalities)
                    return {'arms': {arm: {'subjects': list(held.patient_hash),
                                          'survival': curves(heads[arm], held, features, grid).tolist()}
                                     for arm, features in arms.items()},
                            'matched_meta_digest': digest(list(meta.patient_hash)),
                            'heads': {k: v['sha256'] for k, v in receipts.items()}}
                component = ctx.units.job(name + '_' + task + '_predictions', infer,
                                           {'heads': digest(receipts), 'lock': digest(lock)})
                components.append(component['arms'])
        # Train censoring is fitted on this extension's eligible OUTER TRAIN,
        # never the full unmatched MRI population and never heldout outcomes.
        split = ctx.splits[fold]
        train = frame.loc[frame.patient_hash.isin(split['train'])]
        held = frame.loc[frame.patient_hash.isin(split['test'])]
        value = ctx.units.job(name + f'_fold{fold}_evaluation',
                             lambda: evaluate_cell(train, held, average_predictions(components), ctx.config, fold),
                             {'predictions': digest(components)})
        cells.append(value)
    names = list(arms)
    pairs = list(zip(names[:-1], names[1:]))
    comparisons = []
    for j, (base, augmented) in enumerate(pairs):
        def calculate():
            result = paired_ci(cells, base, augmented, grid, ctx.config['seed'] + j,
                               ctx.config['fusion']['bootstrap_resamples'], len(pairs))
            return dict(result, verdict=incremental_verdict(result), family_size=len(pairs))
        comparisons.append(ctx.units.job(name + f'_paired{j}', calculate, {'cells': digest(cells)}))
    return {'status': 'COMPLETED_MATCHED_INTERNAL_EXTENSION', 'subjects': len(frame),
            'events': int(frame.event.sum()), 'head_fits': 45 * len(arms), 'outer_folds': 5,
            'comparisons': comparisons, 'support': proof,
            'censoring_population': 'MATCHED_EXTENSION_OUTER_TRAIN',
            'quantitative_centaur_equivalence': False, 'external_validation': False,
            'uncertainty': 'CONDITIONAL_ON_FITTED_EXPERTS_AND_HEADS',
            'cnn_training_performed': False}
