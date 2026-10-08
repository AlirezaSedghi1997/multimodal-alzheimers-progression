"""CPU-only, resumable reanalysis of committed Method Y evidence; never trains CNNs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')

import numpy as np
import pandas as pd
from src.method_y_revival.atomic_io import atomic_write_json, read_json
from src.method_y_revival.contracts import sha256_mapping
from src.method_y_revival.local_validation_v5 import (
    analysis_lock, committed_job, file_sha, make_contract, verified_hashed_report,
)
from src.method_y_revival.privacy import scan_public_bundle
from src.method_y_revival.post_eval_v4 import calibration_at_horizon, PostEvalError
from src.method_y_revival.longitudinal_pacc import fit_longitudinal_pacc_models
from scripts.analyze_method_y_amyloid_lmo_a4_v2 import _paired_data, _bootstrap_one
from scripts.run_method_y_amyloid_lmo_a4_v2 import _preflight, _verify

OUTPUT = ROOT / 'results/method_y_local_validation_v5'
LMO = ROOT / 'results/method_y_revival_v4_amyloid_lmo_a4_v2'
CORRECTION = ROOT / 'results/method_y_revival_v4_posttraining_correction_v4_2'
PARENTS = {
    'lmo': LMO / 'reports/amyloid_lmo_a4_public_summary.json',
    'lmo_contract': LMO / 'run_contract.json',
    'status': CORRECTION / 'SCIENCE_STATUS_CORRECTED.json',
    'selection': CORRECTION / 'image_cohort_selection_flow.json',
    'xai': ROOT / 'results/method_y_revival_v4_posttraining_xai_limited_v2/public_bundle/masked_xai_aggregate.json',
    'tau_increment': ROOT / 'results/method_y_revival_v2/reports/paired_at_tau_beyond_amyloid_bootstrap.json',
    'quantitative': ROOT / 'results/method_y_revival_v2/provenance/scan_level_tau_quantitative_validation.json',
    'pacc_lock': ROOT / 'results/method_y_revival_v2/endpoints/longitudinal_pacc_endpoint_contract.json',
    'pacc_data': ROOT / 'results/method_y_revival_v2/endpoints/longitudinal_pacc_endpoint.csv',
}


def calibration_cell(metrics, resamples, seed):
    horizon = 1095.0  # existing locked grid; NOT the 1095.75-day AUC horizon
    grid = list(map(float, metrics['evaluation_grid_days']))
    if horizon not in grid:
        return {'status': 'NOT_ESTIMABLE', 'reason': 'LOCKED_GRID_LACKS_1095'}
    rows, train = metrics['oof_predictions'], metrics['censoring_reference']
    time = np.array([r['time_days'] for r in rows], float)
    event = np.array([r['event'] for r in rows], bool)
    probability = np.array([r['survival_probabilities'][grid.index(horizon)] for r in rows])
    train_time = np.array([r['time_days'] for r in train], float)
    train_event = np.array([r['event'] for r in train], bool)
    events, controls = int(np.sum(event & (time <= horizon))), int(np.sum(time > horizon))
    if min(events, controls) < 10:
        return {'status': 'NOT_ESTIMABLE', 'reason': 'FEWER_THAN_TEN_EVENTS_OR_CONTROLS'}
    def fit(indices):
        return calibration_at_horizon(time[indices], event[indices], probability[indices], horizon,
                                      train_time=train_time, train_event=train_event)
    result = fit(np.arange(len(rows)))
    result.update(horizon_days=horizon, subjects=len(rows), events_by_horizon=events,
                  observed_controls=controls)
    if result['status'] != 'PASS':
        return result
    result['status'] = 'ESTIMABLE_TECHNICAL_NOT_CALIBRATION_PASS'
    # Suppress small aggregate bins; private/source data remain untouched.
    result['calibration_bins'] = [b for b in result['calibration_bins'] if b['n'] >= 10]
    rng, slopes, intercepts, rejected = np.random.default_rng(seed), [], [], 0
    for _ in range(resamples):
        try:
            draw = fit(rng.integers(0, len(rows), len(rows)))
        except (PostEvalError, np.linalg.LinAlgError):
            draw = {'status': 'NOT_ESTIMABLE'}
        if draw['status'] == 'PASS':
            slopes.append(draw['slope'])
            intercepts.append(draw['intercept'])
        else:
            rejected += 1
    valid = len(slopes) >= int(.95 * resamples)
    result['conditional_subject_bootstrap'] = {
        'status': 'ESTIMABLE' if valid else 'NOT_ESTIMABLE',
        'valid_resamples': len(slopes), 'rejected_resamples': rejected,
        'slope_95ci': np.quantile(slopes, [.025, .975]).tolist() if valid else None,
        'intercept_95ci': np.quantile(intercepts, [.025, .975]).tolist() if valid else None,
        'scope': 'FIXED_MODEL_AND_TRAIN_CENSORING_NO_SELECTION_OR_REFIT_UNCERTAINTY',
    }
    return result


def run(mode, output, resamples):
    if resamples < 1000:
        raise ValueError('AT_LEAST_1000_RESAMPLES_REQUIRED')
    if not output.resolve().is_relative_to((ROOT / 'results').resolve()) or output.resolve() == (ROOT / 'results').resolve():
        raise ValueError('OUTPUT_MUST_BE_VERSIONED_PROJECT_RESULTS')
    if not output.name.startswith('method_y_local_validation_v5'):
        raise ValueError('OUTPUT_MUST_BE_DEDICATED_METHOD_Y_VALIDATION_ROOT')
    preflight = _preflight(ROOT / 'configs/method_y_amyloid_lmo_a4_v2.yaml')
    summary = _verify(preflight)
    if summary['outer_cells_finalized'] != 6:
        raise RuntimeError('INCOMPLETE_A4_LMO')
    for name in ('status', 'selection', 'xai'):
        verified_hashed_report(PARENTS[name])
    inputs = dict(PARENTS)
    for fold in range(3):
        for arm in ('A0', 'A4'):
            inputs[f'{arm}_{fold}'] = LMO / f'private/outer_evaluation/amyloid/{arm}/fold_{fold}/outer_result.json'
    source_names = ('local_validation_v5.py', 'post_eval_v4.py', 'longitudinal_pacc.py',
                    'privacy.py', 'atomic_io.py', 'contracts.py', 'posttraining_bootstrap_v4_2.py')
    sources = {name: ROOT / 'src/method_y_revival' / name for name in source_names}
    sources['runner'] = Path(__file__)
    sources['bootstrap_runner'] = ROOT / 'scripts/analyze_method_y_amyloid_lmo_a4_v2.py'
    policy = {'resamples': resamples, 'seed': 13, 'horizon': 1095.,
              'new_image_inference': False, 'new_cnn_training': False,
              'claim': 'POST_HOC_INTERNAL_EXPLORATORY'}
    contract = make_contract(inputs, sources, policy)
    if mode == 'preflight':
        return {'status': 'READY_CPU_ONLY_REANALYSIS', 'contract_sha256': sha256_mapping(contract)}
    with analysis_lock(output):
        contract_path = output / 'run_contract.json'
        if contract_path.exists():
            if read_json(contract_path) != contract:
                raise RuntimeError('LOCAL_VALIDATION_CONTRACT_MISMATCH')
        elif mode == 'verify':
            raise RuntimeError('LOCAL_VALIDATION_CONTRACT_MISSING')
        else:
            atomic_write_json(contract_path, contract)
        def job(name, compute):
            print(json.dumps({'job': name, 'status': 'CHECKING_COMMIT_OR_STARTING'}), flush=True)
            payload = committed_job(output, name, contract, compute, verify_only=mode == 'verify')
            print(json.dumps({'job': name, 'status': 'COMMITTED_OR_RECOVERED'}), flush=True)
            return payload
        calibration, uncertainty = {}, {}
        for outer in preflight['plan']['outer_folds']:
            fold, manufacturer = int(outer['fold']), outer['heldout_manufacturer']
            uncertainty[manufacturer] = job(f'bootstrap_{fold}', lambda: _bootstrap_one(
                _paired_data(LMO, fold), iterations=resamples, seed=13 + fold))
            calibration[manufacturer] = {}
            for arm in ('A0', 'A4'):
                calibration[manufacturer][arm] = job(f'calibration_{arm}_{fold}', lambda: calibration_cell(
                    read_json(inputs[f'{arm}_{fold}'])['metrics'], resamples, 13 + fold))
        def pacc_job(sensitivity=False):
            lock = read_json(PARENTS['pacc_lock'])
            if lock.get('locked') is not True or lock.get('endpoint_sha256') != file_sha(PARENTS['pacc_data']):
                raise RuntimeError('PACC_ENDPOINT_HASH_MISMATCH')
            frame = pd.read_csv(PARENTS['pacc_data'])
            if len(frame) != lock['row_count'] or frame.patient_hash.nunique() != lock['subject_count']:
                raise RuntimeError('PACC_ENDPOINT_CARDINALITY_MISMATCH')
            return fit_longitudinal_pacc_models(endpoint=frame, output_root=output, write_report=False,
                                               slope_covariate_sensitivity=sensitivity)
        pacc = job('pacc', pacc_job)
        pacc_sensitivity = job('pacc_slope_sensitivity', lambda: pacc_job(True))
        if make_contract(inputs, sources, policy) != contract:
            raise RuntimeError('INPUT_OR_SOURCE_CHANGED_DURING_ANALYSIS')
        xai, status = read_json(PARENTS['xai']), read_json(PARENTS['status'])
        tau = read_json(PARENTS['tau_increment'])
        facts = {
            'schema_version': 5, 'status': 'LOCAL_REANALYSIS_COMMITTED_WITH_LIMITATIONS',
            'claim_status': 'POST_HOC_INTERNAL_EXPLORATORY', 'science_gate_status': 'BLOCKED',
            'positive_claim_allowed': False, 'external_validation': 'DEFERRED_NO_EXTERNAL_DATA',
            'contract_sha256': sha256_mapping(contract),
            'manufacturer_holdout': {'technical_cells': 6, 'architecture_only_ablation': False,
                'conditional_paired_deltas': uncertainty, 'cross_grid_ibs_pooled': False},
            'calibration_1095_days': calibration,
            'pacc': {key: pacc[key] for key in ('subject_count', 'measurement_count', 'analysis_role',
                                               'clinical_model', 'amyloid_model', 'missingness_audit')},
            'pacc_slope_sensitivity': {key: pacc_sensitivity[key] for key in
                ('subject_count', 'measurement_count', 'model_formula', 'amyloid_model')},
            'tau_beyond_amyloid': {key: tau[key] for key in ('subject_count', 'delta_ibs_mean', 'delta_ibs_95ci')},
            'latest_limited_xai': {key: xai[key] for key in ('status', 'science_gate_status',
                'cells_completed', 'subject_maps_completed', 'gradcam_zero_inside_brain_total')},
            'xai_anatomical_claim_allowed': False,
            'quantitative_pet_claim_allowed': False,
            'tau_scanner_support': status['branch_results']['tau']['manufacturer_subject_event_support'],
            'selection_bias_audit': 'EXISTS_IN_PARENT_CORRECTION_NOT_SOLVED_BY_AUC',
            'remaining_work': ['PACC_INFORMATIVE_DROPOUT_NOT_IDENTIFIED_FROM_OBSERVED_DATA_ALONE',
                'XAI_ZERO_MAP_CAUSE_AND_FIDELITY_NOT_ESTABLISHED',
                'SITE_METADATA_EXACT_LINKAGE_AVAILABILITY_AUDIT',
                'QUANTITATIVE_PET_EQUIVALENCE_UNPROVEN',
                'ITEMIZED_REPORTING_STANDARD_ALIGNMENT'],
        }
        # No desired CI sign or desired scientific result is hard-coded as a verification requirement.
        bundle = output / 'public_bundle'
        target = bundle / 'facts.json'
        if target.exists() and read_json(target) != facts:
            raise RuntimeError('FINAL_FACTS_MISMATCH')
        if mode != 'verify':
            atomic_write_json(target, facts)
        elif not target.exists():
            raise RuntimeError('FINAL_FACTS_MISSING')
        findings = scan_public_bundle(bundle)
        if findings:
            raise RuntimeError('PUBLIC_PRIVACY_FAILURE:' + json.dumps(findings))
        receipt = {'status': 'PASS', 'facts_sha256': file_sha(target), 'findings': []}
        if mode != 'verify':
            atomic_write_json(bundle / 'privacy_scan.json', receipt)
        elif read_json(bundle / 'privacy_scan.json') != receipt:
            raise RuntimeError('PRIVACY_RECEIPT_MISMATCH')
        return {'status': facts['status'], 'claim_status': facts['claim_status'],
                'scientific_closure': 'NOT_COMPLETE', 'privacy': 'PASS'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'run', 'verify'), default='run')
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--resamples', type=int, default=1000)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.mode, args.output_root, args.resamples)), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'BLOCKED', 'reason': f'{type(exc).__name__}:{exc}'}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
