"""Aggregate-only paired IBS inference; no seed pseudo-replication or selective reporting."""
import numpy as np
from scripts.xyz_fusion_v1.report import cell_frame, individual_bs
from scripts.xyz_fusion_postall_v2.core import digest, atomic_json, atomic_text, file_sha
from .protocol import contrasts


def paired_ibs(differences, resamples, seed, family):
    differences = [np.asarray(x, dtype=float) for x in differences]
    if not differences or any(x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all() for x in differences):
        raise ValueError('INVALID_PAIRED_FOLD_LOSSES')
    if resamples < 1 or family < 1:
        raise ValueError('INVALID_BOOTSTRAP_ARGUMENT')
    rng = np.random.default_rng(seed)
    draws = np.zeros(resamples)
    n = sum(map(len, differences))
    for loss in differences:
        # Chunking bounds RAM; subjects are resampled within their own outer fold.
        for start in range(0, resamples, 512):
            stop = min(start + 512, resamples)
            ids = rng.integers(0, len(loss), (stop - start, len(loss)))
            draws[start:stop] += loss[ids].sum(axis=1) / n
    tail = .05 / (2 * family)
    return {'delta_ibs': float(np.concatenate(differences).mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist(),
            'family_adjusted_ci': np.quantile(draws, [tail, 1 - tail]).tolist(),
            'family_size': family, 'resamples': resamples,
            'resampling': 'PAIRED_PARTICIPANT_WITHIN_OUTER_FOLD',
            'uncertainty': 'CONDITIONAL_ON_FROZEN_MODELS_AND_TRAIN_CENSORING',
            'model_development_uncertainty_included': False}


def report(inputs, units):
    cells = {}
    for track in inputs.options['tracks']:
        value = units.read(track + '_cells')
        if value is None:
            raise ValueError('ALL_FIVE_FOLDS_REQUIRED_BEFORE_REPORT:' + track)
        cells[track] = value['cells']
        if len(value['cells']) != 5 or {c['fold'] for c in value['cells']} != set(range(5)):
            raise ValueError('INVALID_REPORT_FOLD_COVERAGE')
        seen = [k for c in value['cells'] for k in c['test_subjects']]
        if len(seen) != len(set(seen)) or set(seen) != set(inputs.frames[track].patient_hash):
            raise ValueError('INVALID_REPORT_SUBJECT_COVERAGE')
    dependencies = {'cells': digest(cells)}
    def compute():
        grid = inputs.config['fusion']['ibs_grid_days']
        summary = {}; loss = {}
        for track, rows in cells.items():
            summary[track] = {}
            for arm in rows[0]['arms']:
                metrics = [c['arms'][arm]['metrics'] for c in rows]
                loss[track, arm] = []
                for c in rows:
                    value = c['arms'][arm]; tr, te = cell_frame(value)
                    loss[track, arm].append(individual_bs(tr, te, np.asarray(value['curves']), grid)[0])
                ns = [m['n'] for m in metrics]; cs = [m['uno_c_index'] for m in metrics]
                summary[track][arm] = {'n': sum(ns), 'events': sum(m['events'] for m in metrics),
                    'ibs': float(np.concatenate(loss[track, arm]).mean()),
                    'uno_c_fold_weighted': float(np.average(cs, weights=ns)) if all(c is not None for c in cs) else None,
                    'uno_ci_estimated': False, 'fold_metrics': metrics}
        comparisons = []
        for j, (track, base, aug, family, size) in enumerate(contrasts()):
            differences = []
            for idx, cell in enumerate(cells[track]):
                a, b = cell['arms'][base], cell['arms'][aug]
                for key in ('subjects', 'event', 'time', 'censor_train'):
                    if a[key] != b[key]:
                        raise ValueError('UNMATCHED_PAIRED_COMPARISON')
                differences.append(loss[track, aug][idx] - loss[track, base][idx])
            value = paired_ibs(differences, inputs.options['bootstrap_resamples'],
                               inputs.options['bootstrap_seed'] + j, size)
            lo, hi = value['family_adjusted_ci']
            value.update(track=track, reference=base, augmented=aug, family=family,
                         direction='LOWER_IBS_IS_BETTER',
                         verdict='INTERVAL_INCLUDES_ZERO' if lo <= 0 <= hi else
                                 ('LOWER_IBS_CONDITIONAL_SIGNAL' if hi < 0 else 'HIGHER_IBS_CONDITIONAL_SIGNAL'))
            comparisons.append(value)
        public = {'protocol': inputs.options['protocol'], 'claim_status': 'POST_HOC_INTERNAL_EXPLORATORY',
            'precision': inputs.options['precision'], 'arms': summary, 'comparisons': comparisons,
            'head_support': units.read('preflight')['support'], 'new_cnn_training': False,
            'best_seed_selection': False, 'pooled_inner_tuning': False,
            'historical_outcome_exposure': True, 'external_validation': False,
            'clinical_use_validated': False, 'hyperparameter_search_uncertainty_included': False,
            'limitations': ['One pooled head assumes comparable time-specific hazard scores across experts.',
                            'Base CNNs remain fitted on two thirds of each outer training set.',
                            'OOF feature rows share base models; bootstrap conditions on these fixed predictions.',
                            'Families address distinct post-hoc questions, not all historical multiplicity.',
                            'Tau timing, modality selection and endpoint definitions retain parent limitations.']}
        # Fail closed on accidental raw identifiers in public output.
        import json
        text = json.dumps(public, ensure_ascii=False, allow_nan=False)
        if any(key in text for key in inputs.frame.patient_hash):
            raise ValueError('PARTICIPANT_IDENTIFIER_IN_AGGREGATE_REPORT')
        target = units.root / 'public/aggregate_results.json'
        atomic_json(target, public)
        lines = ['# Pooled OOF stacking: exploratory internal evaluation', '',
                 'Frozen FP32 CNN scores; new pooled Cox heads, fixed alpha=10. No CNN retraining.', '',
                 'All eligible outer-training participants contribute one OOF feature row per seed. '
                 'Each outer subject is evaluated through the same three frozen component experts; '
                 'survival curves, not features, are averaged. All heads are locked before evaluation.', '',
                 '| Track / family | Comparison | Delta IBS | Adjusted interval |',
                 '|---|---|---:|---|']
        for v in comparisons:
            lo, hi = v['family_adjusted_ci']
            lines.append(f"| {v['track']} / {v['family']} | {v['reference']} -> {v['augmented']} | {v['delta_ibs']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
        lines += ['', 'Lower IBS is better. Intervals are conditional on fitted models, with paired resampling '
                  'within outer folds. They do not estimate training/selection uncertainty. Uno C is descriptive '
                  'without a new confidence interval. Each family is exploratory and separate. '
                  'Completion does not establish a benefit or clinical validity.']
        md = units.root / 'public/REPORT.md'; atomic_text(md, '\n'.join(lines) + '\n')
        return {'status': 'OOF_EXPERIMENT_COMPLETE_NOT_CLINICAL_VALIDATION',
                'outputs': {p.relative_to(units.root).as_posix(): file_sha(p) for p in (target, md)},
                'comparisons': len(comparisons), 'new_cnn_training': False}
    return units.job('aggregate_report', compute, dependencies)
