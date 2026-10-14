"""Evidence-derived closure and an explicit aggregate-only public schema."""
from __future__ import annotations

from collections import Counter
import re
from .core import CLAIM, atomic_json, atomic_text, digest, file_sha, read_json

TASKS = ['f00', 'f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07']
PUBLIC_FILES = {'results.json', 'CLAIM_POLICY.md', 'REPORTING_ALIGNMENT.md', 'REPORT.md', 'NEGATIVE_RESULTS.md',
                'fusion_ibs.svg', 'mri_calibration.svg', 'mri_paired_ibs.svg', 'pet_calibration.svg', 'pet_paired_ibs.svg'}
PRIVATE_KEYS = {'patient_hash', 'subject_key', 'raw_path', 'sidecar_path', 'source_subject_id',
                'event', 'time_days', 'time', 'censor_train', 'censoring_reference',
                'train_subjects', 'test_subjects', 'meta_subjects', 'outer_subjects', 'subjects_ids',
                'predictions', 'curves', 'survival', 'embeddings', 'outputs', 'map', 'files'}


def assert_public(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k).lower() in PRIVATE_KEYS or str(k).startswith('original_'):
                raise ValueError('PUBLIC_PRIVATE_FIELD:' + str(k))
            assert_public(v)
    elif isinstance(value, list):
        # Approved arrays are short CI/metric summaries, never long row-level data.
        if len(value) > 60 or (value and all(isinstance(v, list) for v in value)):
            raise ValueError('PUBLIC_POSSIBLE_ROW_LEVEL_ARRAY')
        for v in value:
            assert_public(v)
    elif isinstance(value, str):
        if re.search(r'(?i)(?<!\w)[a-z]:[\\/]|A4[ _]Learn|\b(?:sub|pt|scan)_[a-z0-9_-]+|\.(?:h5|ckpt|pkl|nii)(?:\b|\.)', value):
            raise ValueError('PUBLIC_PRIVATE_REFERENCE')


def require_public_file_whitelist(public, *, complete=False):
    present = set()
    for path in public.rglob('*'):
        if path.is_file():
            relative = path.relative_to(public).as_posix()
            if relative not in PUBLIC_FILES or not path.resolve().is_relative_to(public.resolve()):
                raise ValueError('UNAPPROVED_PUBLIC_FILE:' + relative)
            present.add(relative)
    if complete and present != PUBLIC_FILES:
        raise ValueError('INCOMPLETE_PUBLIC_FILE_SET')


def closure(units):
    tasks = {name: units.read(name) for name in TASKS}
    missing = [name for name, value in tasks.items() if value is None]
    return {'status': 'INCOMPLETE' if missing else 'COMPLETED_INTERNAL_POSTALL_AUDIT',
            'tasks': {name: value['status'] if value is not None else 'NOT_EXECUTED' for name, value in tasks.items()},
            'missing_tasks': missing, 'claim_status': CLAIM,
            'all_scientific_objectives_achieved': False,
            'clinical_claim_allowed': False, 'external_claim_allowed': False,
            'disease_localization_allowed': False, 'quantitative_pet_equivalence_allowed': False,
            'external_status': 'NO_EXTERNAL_DATA',
            'completion_is_not_evidence_of_incremental_value': True}


def public_report(units):
    result = closure(units)
    if result['missing_tasks']:
        raise ValueError('PUBLIC_FINALIZATION_REQUIRES_ALL_F00_F07')
    f0, f1, f2, f3, f4, f5, f6, f7 = [units.read(k) for k in TASKS]
    result['parent_verification'] = {k: f0[k] for k in ('status', 'locked_image_checkpoints', 'prediction_journals', 'parent_writes')}
    result['population'] = f2['intersection_counts']
    result['analysis'] = {}
    for name, analysis in f1['analyses'].items():
        # The source of these fields is a metric-only record, not a private cell.
        result['analysis'][name] = {'fold_metrics': analysis['metrics']['fold_metrics'],
                                    'arms': analysis['arm_summary'],
                                    'comparisons': analysis['paired_comparisons']}
    result['tau'] = {k: f3[k] for k in ('status', 'subjects', 'events', 'comparisons') if k in f3}
    result['subgroups'] = f1['subgroups']
    result['sensitivity'] = f1['sensitivity']
    statuses = Counter()
    for case in f4['cases']:
        for baseline in case.get('baselines', {}).values():
            statuses[baseline['status']] += 1
    result['spatial'] = {'requested_units': f4['requested_units'], 'eligible_units': f4['eligible_units'],
                          'baseline_status_counts': dict(statuses), 'disease_localization_allowed': False,
                          'roi_occlusion_units': sum(len(c.get('roi_occlusion', {})) for c in f4['cases']),
                          'roi_occlusion_is_nonadditive_model_perturbation_not_lesion_truth': True}
    probes = f5['probes']['units']
    result['scanner'] = {'probe_status_counts': dict(Counter(r['status'] for r in probes)),
                        'lmo': [{'manufacturer': r['manufacturer'], 'status': r.get('status', r.get('support', {}).get('status')),
                                 **({'metrics': r['metrics']} if 'metrics' in r else {})}
                                for r in f5['lmo']['groups']],
                        'site_status': f5['lmo']['site_lmo_status']}
    result['pacc'] = {k: f6[k] for k in ('status', 'counts', 'models', 'attrition', 'association_not_validation', 'uncertainty') if k in f6}
    result['four_way'] = {k: f7[k] for k in ('status', 'subjects', 'events', 'action', 'comparisons') if k in f7}
    assert_public(result)
    return result


def write_release(ctx):
    result = public_report(ctx.units)
    public = ctx.out / 'public'
    require_public_file_whitelist(public)
    atomic_json(public / 'results.json', result)
    text = ('# Internal fusion review\n\n' + CLAIM + '\n\n'
            'Completion describes an executed audit, not clinical efficacy. Outcomes are future sustained CDR progression; '
            'they are not a diagnosis of Alzheimer disease. Pattern-normalized PET is not an official quantitative standard. '
            'No external cohort was available. Attribution describes the frozen prediction function, not disease tissue.\n\n'
            'Prior post-all maps and positive tau claims are not primary evidence until their replacements pass the recorded checks. '
            'All prespecified baselines and failures are retained. Bootstrap intervals condition on fitted models and training censoring; '
            'they do not include retraining and model-selection uncertainty.\n')
    atomic_text(public / 'CLAIM_POLICY.md', text)
    atomic_text(public / 'REPORTING_ALIGNMENT.md',
        '# TRIPOD+AI and PROBAST+AI reporting alignment\n\n'
        'This is an alignment inventory, not certification or a low-risk-of-bias claim.\n\n'
        '| Domain | Evidence / remaining limitation |\n|---|---|\n'
        '| Participants | Locked internal A4 cohort; historical follow-up selection remains |\n'
        '| Predictors | Exact baseline dates, individual QC and shared-expert ancestry |\n'
        '| Outcome | Post-origin confirmed CDR change; no diagnostic or causal interpretation |\n'
        '| Analysis | Nested component separation, fixed heads, paired censored metrics, conditional uncertainty |\n'
        '| Missingness | Complete modality subsets; missing-at-random is not established |\n'
        '| Transportability | Manufacturer sensitivities only; no external validation |\n'
        '| Interpretation | Model attribution is not an anatomical lesion or clinical decision explanation |\n')
    write_numeric_report(public, result)
    from scripts.xyz_fusion_v2.privacy import scan_public_bundle
    findings = scan_public_bundle(public)
    if findings:
        raise ValueError('PUBLIC_PRIVACY_FAILED:' + str(findings[:3]))
    require_public_file_whitelist(public, complete=True)
    files = {p.relative_to(ctx.out).as_posix(): file_sha(p) for p in public.rglob('*') if p.is_file()}
    return {'status': 'AGGREGATE_RELEASE_VERIFIED', 'outputs': files, 'privacy': 'PASS',
            'closure': closure(ctx.units), 'positive_or_external_claim': False}


def write_numeric_report(public, result):
    """Tables and aggregate figures are derived only from the verified release."""
    rows = ['# Corrected internal post-fusion analysis', '', CLAIM, '',
            'This report does not rerun fusion training or reselect a model using outer outcomes.', '',
            '| Analysis | Arm | N | Events | IPCW IBS | Conditional IBS CI | Weighted fold Uno C |',
            '|---|---|---:|---:|---:|---|---:|']
    plots = {}
    negative = ['# Negative results and unresolved scientific limits', '',
                'A negative result is distinct from an invalid or unexecuted analysis.', '']
    for name, analysis in result['analysis'].items():
        arms = {}
        for arm, value in analysis['arms'].items():
            ci = value.get('ibs_conditional_ci95')
            rows.append(f"| {name} | {arm} | {value['n']} | {value['events']} | {value['ibs']:.6f} | {ci} | {value['uno_c_index']} |")
            arms[arm] = dict(value, uno_c=value['uno_c_index'],
                             fold_metrics=[r for r in analysis['fold_metrics'] if r['arm'] == arm])
        plots[name] = {'arms': arms, 'paired_comparisons': analysis['comparisons']}
        for contrast in analysis['comparisons']:
            if contrast['verdict'] != 'INTERNAL_EXPLORATORY_SUPPORT':
                negative.append(f"- {name}: {contrast['augmented']} vs {contrast['base']}: {contrast['verdict']}.")
    rows += ['', '## Additional matched analyses', '',
             'Tau: ' + str(result['tau']), '', 'Four-way: ' + str(result['four_way']), '',
             'PACC status: ' + result['pacc']['status'], '',
             'All confidence intervals condition on the already fitted predictors and training censoring. '
             'No clinical, external, prospective, or causal anatomical claim is licensed by these results.']
    negative += ['', '- No external cohort was available.',
                 '- PET leave-manufacturer-out training was not performed by this post-fusion recalculation.',
                 '- Whole-ensemble label-permutation training was not performed; maps remain diagnostic.',
                 '- Four-way support decision: ' + result['four_way']['status'] + '.',
                 '- Spatial baseline status counts: ' + str(result['spatial']['baseline_status_counts']) + '.',
                 '- PACC is secondary association, not independent model validation.',
                 '- Pattern PET does not establish quantitative Centiloid/CenTauR image equivalence.',
                 '- Complete-modality selection, historical outcome exposure, and censoring assumptions remain.']
    atomic_text(public / 'REPORT.md', '\n'.join(rows) + '\n')
    atomic_text(public / 'NEGATIVE_RESULTS.md', '\n'.join(negative) + '\n')
    from scripts.xyz_fusion_v2.reporting import draw_figures
    draw_figures(public, plots)


def verify_final(ctx):
    for path in (ctx.out / 'private/units').glob('*.json'):
        ctx.units.read(path.stem)
    value = closure(ctx.units)
    if value['missing_tasks']:
        return dict(value, verifier='FAIL_INCOMPLETE')
    f1, f2, f4 = [ctx.units.read(k) for k in ('f01', 'f02', 'f04')]
    from .audit import analyses
    expected_analyses = analyses(ctx.parent)
    if set(f1['analyses']) != set(expected_analyses):
        raise ValueError('NUMERICAL_ANALYSIS_COVERAGE_INCOMPLETE')
    for name, (_, cells) in expected_analyses.items():
        expected_cells = {(c['fold'], arm) for c in cells for arm in c['arms']}
        observed_cells = [(r['fold'], r['arm']) for r in f1['analyses'][name]['metrics']['fold_metrics']]
        if len(observed_cells) != len(expected_cells) or set(observed_cells) != expected_cells:
            raise ValueError('NUMERICAL_ARM_FOLD_COVERAGE_INCOMPLETE')
    if any(len(a['paired_comparisons']) != a['expected_contrasts'] for a in f1['analyses'].values()):
        raise ValueError('BOOTSTRAP_COVERAGE_INCOMPLETE')
    expected_tasks = 5 * len(ctx.config['training_seeds']) * ctx.config['components'] * 2
    expected_maps = 5 * ctx.config['spatial']['cases_per_fold'] * 3
    if f2['tasks_verified'] != expected_tasks or f4['requested_units'] != expected_maps or len(f4['cases']) != expected_maps:
        raise ValueError('POSTALL_TASK_COVERAGE_INCOMPLETE')
    expected = {(track, mod, f, c) for f in range(5) for c in range(ctx.config['spatial']['cases_per_fold'])
                for track, mod in [('mri', 'mri'), ('pet', 'mri'), ('pet', 'pet')]}
    observed = set()
    actual_cases = []
    baseline_names = {'zero', *('blur_' + str(s).replace('.', 'p') for s in ctx.config['spatial']['baseline_sigma'])}
    for track, modality, fold, case in expected:
        name = f'f04_{track}_{modality}_o{fold}_case{case}'
        receipt = ctx.units.read(name)
        if receipt is None:
            raise ValueError('SPATIAL_CASE_RECEIPT_MISSING')
        actual_cases.append(digest(receipt))
        observed.add((track, modality, fold, case))
        if receipt['status'] == 'NOT_ELIGIBLE_NO_LOCKED_PET_IMAGE':
            if track != 'pet':
                raise ValueError('PRIMARY_MRI_CASE_CANNOT_BE_SKIPPED')
            continue
        if ctx.units.read(name + '_parity') is None or set(receipt['baselines']) != baseline_names:
            raise ValueError('SPATIAL_PARITY_OR_BASELINES_INCOMPLETE')
        for baseline in baseline_names:
            if ctx.units.read(name + '_' + baseline + '_roi_occlusion') is None:
                raise ValueError('ROI_OCCLUSION_BASELINE_MISSING')
            for index in range(len(ctx.config['training_seeds']) * ctx.config['components']):
                for control in ('trained', 'random'):
                    if ctx.units.read(name + '_' + baseline + f'_component{index}_' + control) is None:
                        raise ValueError('SPATIAL_COMPONENT_OR_CONTROL_MISSING')
    if observed != expected:
        raise ValueError('SPATIAL_CASE_COVERAGE_CHANGED')
    if sorted(actual_cases) != sorted(digest(c) for c in f4['cases']):
        raise ValueError('SPATIAL_AGGREGATE_DOES_NOT_MATCH_CASE_RECEIPTS')
    f5 = ctx.units.read('f05')
    if len(f5['probes']['units']) != expected_tasks:
        raise ValueError('SCANNER_PROBE_COVERAGE_INCOMPLETE')
    summaries = {(r['task'], r['modality']): r for r in f5['probes']['units']}
    if len(summaries) != expected_tasks:
        raise ValueError('DUPLICATE_SCANNER_PROBE_SUMMARY')
    for seed in ctx.config['training_seeds']:
        for fold in range(5):
            for comp in range(ctx.config['components']):
                for modality in ('mri', 'pet'):
                    task = f'pet_s{seed}_o{fold}_c{comp}'
                    proof = ctx.units.read(f'f05_probe_{task}_{modality}')
                    if proof is None:
                        raise ValueError('SCANNER_PROBE_RECEIPT_MISSING')
                    expected_summary = {'task': task, 'modality': modality,
                        **{k: v for k, v in proof.items() if k not in ('outputs', 'train_subjects', 'test_subjects')}}
                    if summaries.get((task, modality)) != expected_summary:
                        raise ValueError('SCANNER_AGGREGATE_DOES_NOT_MATCH_PROBE_RECEIPTS')
    f3 = ctx.units.read('f03')
    if f3['status'] == 'COMPLETED_MATCHED_INTERNAL_EXTENSION':
        for seed in ctx.config['training_seeds']:
            for fold in range(5):
                for comp in range(ctx.config['components']):
                    task = f'tau_mri_s{seed}_o{fold}_c{comp}'
                    for suffix in ('base', 'plus_tau', 'predictions'):
                        if ctx.units.read(task + '_' + suffix) is None:
                            raise ValueError('MATCHED_TAU_COMPONENT_MISSING')
        for fold in range(5):
            if ctx.units.read(f'tau_fold{fold}_evaluation') is None:
                raise ValueError('MATCHED_TAU_OUTER_FOLD_MISSING')
        if f3['comparisons'] != [ctx.units.read('tau_paired0')]:
            raise ValueError('MATCHED_TAU_COMPARISON_RECEIPT_MISMATCH')
    release = ctx.units.read('f08')
    if release is None:
        return dict(value, verifier='FAIL_RELEASE_MISSING')
    from scripts.xyz_fusion_v2.privacy import scan_public_bundle
    require_public_file_whitelist(ctx.out / 'public', complete=True)
    actual = read_json(ctx.out / 'public/results.json')
    assert_public(actual)
    if actual != public_report(ctx.units):
        raise ValueError('PUBLIC_REPORT_NOT_DERIVED_FROM_VERIFIED_UNITS')
    if scan_public_bundle(ctx.out / 'public'):
        raise ValueError('PUBLIC_PRIVACY_FAILED')
    return dict(value, verifier='PASS_EXECUTION_AND_CONSISTENCY_NOT_POSITIVE_SCIENCE')
