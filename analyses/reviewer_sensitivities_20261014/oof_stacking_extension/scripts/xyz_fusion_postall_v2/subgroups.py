"""Fixed subgroup losses, fold-stratified resampling and explicit limitations."""
import numpy as np
from scripts.xyz_fusion_v1.report import cell_frame, individual_bs
from scripts.xyz_fusion_v2.protocol import contrasts


def conditional_interval(parts, rng, count):
    parts = [np.asarray(p, float) for p in parts if len(p)]
    n = sum(map(len, parts))
    draws = [sum(p[rng.integers(len(p), size=len(p))].sum() for p in parts) / n for _ in range(count)]
    return {'n': n, 'estimate': float(sum(p.sum() for p in parts) / n),
            'ci95': np.quantile(draws, [.025, .975]).tolist()}


def report(ctx, track):
    frame = ctx.frame if track == 'mri' else ctx.pet
    cells = ctx.parent.job(track + '_ensemble_cells')
    index = frame.set_index('patient_hash', verify_integrity=True)
    groups = {'sex_0': index.sex_binary.eq(0), 'sex_1': index.sex_binary.eq(1),
              'age_lt_70': index.age.lt(70), 'age_ge_70': index.age.ge(70),
              'treatment_0': index.treatment_active.eq(0), 'treatment_1': index.treatment_active.eq(1),
              'cognition_mmse_lt_29': index.baseline_mmscore.lt(29),
              'cognition_mmse_ge_29': index.baseline_mmscore.ge(29)}
    for col in ['mri_manufacturer', 'pet_manufacturer']:
        if col in index:
            for value in sorted(index[col].fillna('UNKNOWN').unique()):
                groups[col + '_' + str(value)] = index[col].eq(value)
    losses = {}
    for cell in cells:
        for arm, value in cell['arms'].items():
            train, test = cell_frame(value)
            losses[cell['fold'], arm] = individual_bs(train, test, np.asarray(value['curves']), ctx.config['fusion']['ibs_grid_days'])[0]
    output = {}
    for name, mask in groups.items():
        ids = set(index.index[mask]); events = int(index.loc[list(ids), 'event'].sum()) if ids else 0
        result = {'n': len(ids), 'events': events, 'status': 'NOT_ESTIMABLE_SUPPORT'}
        if len(ids) < ctx.config['secondary']['min_group_subjects'] or events < ctx.config['secondary']['min_group_events']:
            output[name] = result
            continue
        rng = np.random.default_rng(ctx.config['seed'])
        by_arm = {}; pair = {}
        for arm in cells[0]['arms']:
            parts = [losses[c['fold'], arm][[key in ids for key in c['arms'][arm]['subjects']]] for c in cells]
            total = sum(map(len, parts))
            group_events = sum(sum(e for key, e in zip(c['arms'][arm]['subjects'], c['arms'][arm]['event']) if key in ids) for c in cells)
            if total >= 30 and group_events >= 10:
                by_arm[arm] = conditional_interval(parts, rng, ctx.config['fusion']['bootstrap_resamples'])
            else:
                by_arm[arm] = {'status': 'NOT_ESTIMABLE_ARM_SUPPORT', 'n': total, 'events': group_events}
        for base, aug in contrasts(track):
            if 'estimate' not in by_arm[base] or 'estimate' not in by_arm[aug]:
                continue
            parts = []
            for c in cells:
                a, b = c['arms'][base], c['arms'][aug]
                if any(a[k] != b[k] for k in ('subjects', 'event', 'time', 'censor_train')):
                    raise ValueError('SUBGROUP_CONTRAST_UNMATCHED')
                take = [key in ids for key in a['subjects']]
                parts.append((losses[c['fold'], aug] - losses[c['fold'], base])[take])
            pair[base + '__' + aug] = conditional_interval(parts, rng, ctx.config['fusion']['bootstrap_resamples'])
        result.update(status='CONDITIONAL_DESCRIPTIVE_SUBGROUP', arms=by_arm, paired_delta_ibs=pair)
        output[name] = result
    return {'status': 'COMPLETED_FIXED_SUBGROUP_AUDIT', 'groups': output,
            'bootstrap_unit': 'PAIRED_SUBJECT_WITHIN_OUTER_FOLD',
            'ci_scope': 'CONDITIONAL_ON_FITTED_MODELS_AND_OVERALL_OUTER_TRAIN_CENSORING',
            'multiplicity': 'NOMINAL_DESCRIPTIVE_NO_POSITIVE_CLAIM_FROM_SUBGROUP_SELECTION',
            'cohort_transport': 'NOT_ESTIMABLE_SINGLE_A4_COHORT',
            'qc_sensitivity': 'PASS_ONLY_SELECTED_COHORT_NOT_RECOVERY_OF_EXCLUDED_SCANS'}
