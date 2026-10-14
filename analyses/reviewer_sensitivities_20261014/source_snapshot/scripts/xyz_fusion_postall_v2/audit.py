"""Independent structural audit; numerical replay never changes predictions."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys
import numpy as np

from .core import read_json, file_sha, digest, atomic_json


def verify_parent(root, parent):
    config = read_json(root / 'configs/xyz_fusion_v2.json')
    if config != parent.contract['protocol']:
        raise ValueError('PARENT_CONFIG_DRIFT')
    for relative, sha in parent.execution_contract['source_sha256'].items():
        if parent.hashes(root / relative) != sha:
            raise ValueError('PARENT_SOURCE_DRIFT:' + relative)
    for package, version in parent.contract['runtime'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('PARENT_RUNTIME_DRIFT:' + package)
    if sys.version != parent.contract['python']:
        raise ValueError('PARENT_PYTHON_DRIFT')
    # Windows byte-range locks disallow even this process reading that byte via
    # another handle. parent_exclusion verifies the lock before/after release.
    watched = [parent.root / 'orchestration_state.json']
    watched += list((parent.root / 'private/access_journal').glob('*.json'))
    before = {str(p): parent.hashes(p) for p in watched}
    proof = parent.verify()
    if any(parent.hashes(p) != sha for p, sha in before.items()):
        raise ValueError('READ_ONLY_VERIFIER_CHANGED_PARENT')
    # Freeze all scientific JSON and all files consumed by the full verifier.
    paths = set(watched) | set((parent.root / 'private/jobs').glob('*.json'))
    paths |= set((parent.root / 'private/models').rglob('locked.json'))
    paths |= set((parent.root / 'private').rglob('contract.json'))
    paths |= {p for p in parent.hashes.values if p.is_relative_to(parent.root)}
    paths.add(parent.root / 'private/contract.json')
    frozen = {p.relative_to(parent.root).as_posix(): parent.hashes(p) for p in sorted(paths)}
    return {'status': 'PARENT_STRUCTURAL_AND_ARTIFACT_VERIFICATION_PASSED',
            'parent_contract': parent.binding, 'parent_execution_contract': parent.execution_binding,
            'full_verifier': proof, 'files': frozen,
            'locked_image_checkpoints': len(list((parent.root / 'private/models').rglob('locked.pt'))),
            'prediction_journals': len(list((parent.root / 'private/access_journal').glob('*.json'))),
            'parent_writes': 0, 'input_reconstruction_required_in_f02': True}


def freeze_once(out, freeze):
    path = out / 'private/parent_freeze.json'
    if path.exists():
        if read_json(path) != freeze:
            raise ValueError('FROZEN_PARENT_DRIFT_NEW_VERSION_REQUIRED')
    else:
        atomic_json(path, freeze)


def analyses(parent):
    plan = parent.job('analysis_plan')
    result = {}
    for label, spec in plan['tabular'].items():
        if spec['support']['status'] != 'SUPPORTED':
            continue
        for family in spec['families']:
            name = label + '_' + family
            result[name] = (spec['track'], [parent.job(f'{name}_o{f}_evaluation') for f in range(5)])
    result['mri'] = ('mri', parent.job('mri_ensemble_cells'))
    if plan['pet']['status'] == 'SUPPORTED':
        result['pet'] = ('pet', parent.job('pet_ensemble_cells'))
    return result


def metric_replay(cells, config):
    """Compare all stored metrics, not just a favourable headline contrast."""
    from scripts.xyz_fusion_v1.report import cell_frame, individual_bs
    from .statistics import metrics
    fc = config['fusion']
    rows = []
    for cell in cells:
        for arm, value in cell['arms'].items():
            train, test = cell_frame(value)
            actual = metrics(train, test, value['curves'], fc['ibs_grid_days'],
                             fc['horizons_days'], fc['min_at_risk'])
            # Library IBS vs a separately constructed individual IPCW loss.
            individual = individual_bs(train, test, np.asarray(value['curves']), fc['ibs_grid_days'])[0]
            if not np.isclose(individual.mean(), actual['ibs'], rtol=0, atol=1e-12):
                raise ValueError('INDEPENDENT_IPCW_IDENTITY_FAILED')
            def equal(a, b):
                if isinstance(a, dict):
                    return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
                if isinstance(a, list):
                    return isinstance(b, list) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
                if isinstance(a, (int, float)) and not isinstance(a, bool):
                    return isinstance(b, (int, float)) and np.isfinite(a) and np.isclose(a, b, rtol=1e-8, atol=1e-10)
                return a == b
            if not equal(actual, value['metrics']):
                raise ValueError(f'PARENT_METRIC_REPLAY_MISMATCH:{cell["fold"]}:{arm}')
            rows.append({'fold': cell['fold'], 'arm': arm, **actual})
    return {'status': 'ALL_STORED_METRICS_REPRODUCED', 'folds': len(cells),
            'arm_fold_cells': len(rows), 'fold_metrics': rows,
            'recalibration_or_refit_performed': False}


def incremental_verdict(comparison):
    ibs = comparison.get('delta_ibs_bonferroni_ci')
    uno = comparison.get('delta_uno_bonferroni_ci')
    if ibs is None or uno is None:
        return 'NOT_ESTIMABLE'
    if not np.isfinite([*ibs, *uno]).all():
        raise ValueError('NONFINITE_COMPARISON_CI')
    return 'INTERNAL_EXPLORATORY_SUPPORT' if ibs[1] < 0 and uno[0] >= 0 else 'INCREMENTAL_VALUE_NOT_DEMONSTRATED'


def audit_metrics(ctx, *, bootstrap):
    from scripts.xyz_fusion_v2.protocol import contrasts
    from scripts.xyz_fusion_v2.reporting import paired_ci
    result = {}
    for label, (track, cells) in analyses(ctx.parent).items():
        dependency = {'cells': digest(cells)}
        replay = ctx.units.job('f01_' + label + '_metrics', lambda: metric_replay(cells, ctx.config), dependency)
        pairs = contrasts(track)
        comparisons = []
        if bootstrap:
            for j, (base, aug) in enumerate(pairs):
                def compute():
                    value = paired_ci(cells, base, aug, ctx.config['fusion']['ibs_grid_days'],
                                      ctx.config['seed'] + j, ctx.config['fusion']['bootstrap_resamples'], len(pairs))
                    return dict(value, verdict=incremental_verdict(value), family_size=len(pairs),
                                family_scope=label, family_was_not_chosen_from_outer_results=True)
                comparisons.append(ctx.units.job(f'f01_{label}_contrast{j}', compute, dependency))
        grouped = {}
        for row in replay['fold_metrics']:
            grouped.setdefault(row['arm'], []).append(row)
        summaries = {}
        for arm, rows in grouped.items():
            weights = [v['n'] for v in rows]
            unos = [v['uno_c_index'] for v in rows]
            summaries[arm] = {'n': sum(weights), 'events': sum(v['events'] for v in rows),
                              'ibs': float(np.average([v['ibs'] for v in rows], weights=weights)),
                              'uno_c_index': float(np.average(unos, weights=weights)) if all(v is not None for v in unos) else None,
                              'aggregation': 'TEST_SUBJECT_COUNT_WEIGHTED_FOLD_METRICS_NOT_UNSTRATIFIED_POOLED_CONCORDANCE'}
            if bootstrap:
                from scripts.xyz_fusion_v1.report import cell_frame, individual_bs
                from .subgroups import conditional_interval
                parts = []
                for cell in cells:
                    a = cell['arms'][arm]
                    tr, te = cell_frame(a)
                    parts.append(individual_bs(tr, te, np.asarray(a['curves']), ctx.config['fusion']['ibs_grid_days'])[0])
                interval = conditional_interval(parts, np.random.default_rng(ctx.config['seed']),
                                                 ctx.config['fusion']['bootstrap_resamples'])
                summaries[arm]['ibs_conditional_ci95'] = interval['ci95']
        result[label] = {'metrics': replay, 'arm_summary': summaries, 'paired_comparisons': comparisons,
                         'expected_contrasts': len(pairs)}
    subgroups = {}
    sensitivity = {}
    if bootstrap:
        from .subgroups import report as subgroup_report
        for track in ('mri', 'pet'):
            subgroups[track] = ctx.units.job('f01_' + track + '_subgroups', lambda: subgroup_report(ctx, track))
        from scripts.xyz_fusion_v2.sensitivity import missingness, ph_diagnostic
        from scripts.xyz_fusion_v2.protocol import CLINICAL, VOLUMES
        sensitivity['blood_selection'] = ctx.units.job('f01_blood_selection',
            lambda: dict(missingness(ctx.frame, ctx.parent.job('mri_ensemble_cells'), ctx.config),
                         interpretation='OBSERVED_BLOOD_WEIGHT_TILT_SCENARIOS_NOT_IDENTIFICATION_OF_UNOBSERVED_MNAR'))
        sensitivity['ph'] = {}
        for fold, split in enumerate(ctx.splits):
            def compute_ph():
                train = ctx.frame.loc[ctx.frame.patient_hash.isin(split['train'])]
                recomputed = ph_diagnostic(train, CLINICAL + ['amyloid_centiloid'] + VOLUMES)
                original = ctx.parent.job(f'ph_o{fold}')
                if recomputed.get('status') != original.get('status'):
                    raise ValueError('PH_DIAGNOSTIC_STATUS_REPLAY_MISMATCH')
                for key in ('statistic', 'p_approx'):
                    if key in original and not np.isclose(recomputed[key], original[key], rtol=1e-7, atol=1e-9):
                        raise ValueError('PH_DIAGNOSTIC_REPLAY_MISMATCH')
                return dict(recomputed, replayed_on_outer_train_only=True)
            sensitivity['ph'][str(fold)] = ctx.units.job(f'f01_ph_o{fold}', compute_ph)
    return {'status': 'COMPLETED' if bootstrap else 'METRIC_REPLAY_COMPLETE_BOOTSTRAP_PENDING',
            'analyses': result, 'positive_confirmatory_claim': False,
            'subgroups': subgroups,
            'sensitivity': sensitivity,
            'conditional_ci_not_full_training_uncertainty': True}
