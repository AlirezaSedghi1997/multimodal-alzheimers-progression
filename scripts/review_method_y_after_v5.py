"""Read-only source audit + aggregate report. No model loading or fitting.

The v5 verifier is a prerequisite, not inferred from the presence of facts.json.
All new writes are confined to the dedicated H-side review root.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['MPLCONFIGDIR'] = str(ROOT / 'tmp/mpl_method_y_review')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import pandas as pd
from src.method_y_revival.atomic_io import atomic_write_json, read_json
from src.utils.io_helpers import atomic_write_text
from src.method_y_revival.local_validation_v5 import file_sha, analysis_lock
from src.method_y_revival.privacy import scan_public_bundle

OUT = ROOT / 'results/method_y_post_v5_review_20260929'
V5 = ROOT / 'results/method_y_local_validation_v5'
XAI = ROOT / 'results/method_y_revival_v4_posttraining_xai_limited_v2'
BLOOD = ROOT / 'data/processed/catalogs/bio_cog_method_z_catalog.csv'


def compare_calibration(new, old):
    rows = []
    for manufacturer, arms in new.items():
        for arm, value in arms.items():
            prior = old[manufacturer][arm]
            rows.append({
                'manufacturer': manufacturer, 'arm': arm,
                'slope': value['slope'], 'intercept': value['intercept'],
                'slope_change_v5_minus_v4': value['slope'] - prior['slope'],
                'intercept_change_v5_minus_v4': value['intercept'] - prior['intercept'],
                'binned_error_change_v5_minus_v4': value['ici'] - prior['ici'],
                'slope_bootstrap_ci95': value['conditional_subject_bootstrap']['slope_95ci'],
                'valid_bootstrap_draws': value['conditional_subject_bootstrap']['valid_resamples'],
                'published_bins': len(value['calibration_bins']),
                'fitted_bins': value['n_bins'],
                'claim': 'DESCRIPTIVE_NOT_CALIBRATION_PASS',
            })
    return rows


def candidate_overlap(endpoint, blood):
    """Exact existing hash intersection is availability only, never eligibility."""
    if endpoint.patient_hash.duplicated().any() or blood.patient_hash.duplicated().any():
        raise ValueError('DUPLICATE_SUBJECTS_NOT_INDEPENDENT_OBSERVATIONS')
    # Exact aliases documented in the locked tau endpoint summary; do not
    # infer cohorts from prefixes or silently report zero for unknown schemas.
    allowed = {'A4', 'A4_placebo', 'A4_treatment', 'LEARN'}
    if not set(endpoint.cohort).issubset(allowed):
        raise ValueError('UNMAPPED_ENDPOINT_COHORT')
    a4 = endpoint.loc[endpoint.cohort.isin({'A4', 'A4_placebo', 'A4_treatment'})]
    all_blood = set(blood.patient_hash)
    ptau = set(blood.loc[blood.pTau217.notna(), 'patient_hash'])
    result = {
        'a4_subjects': len(a4), 'a4_events': int(a4.event.sum()),
        'blood_available_subjects': int(a4.patient_hash.isin(all_blood).sum()),
        'ptau217_available_subjects': int(a4.patient_hash.isin(ptau).sum()),
        'ptau217_available_events': int(a4.loc[a4.patient_hash.isin(ptau), 'event'].sum()),
        'chronology_validated': False, 'assay_validated': False,
        'mri_intersection_validated': False, 'fusion_training_eligible_count': None,
        'status': 'CANDIDATE_AVAILABILITY_NOT_FUSION_ELIGIBILITY',
    }
    return result


def blood_source_audit(path):
    df = pd.read_csv(path, low_memory=False)
    code = next((c for c in ('LBTESTCD', 'TESTCD') if c in df), None)
    keys = ['SUBSTUDY', 'BID'] + ([code] if code else [])
    counts = df.groupby(keys, dropna=False).size()
    dates = [c for c in df if 'DAYS_T0' in c or 'DAYS_CONSENT' in c]
    methods = [c for c in df if c in ('METHOD', 'LBMETHOD', 'ORRESU', 'LABORESU', 'LBORRESU')]
    result = {
        'rows': len(df), 'subject_analyte_groups': len(counts),
        'groups_with_multiple_rows': int((counts > 1).sum()),
        'date_columns_nonmissing_counts': {c: int(df[c].notna().sum()) for c in dates},
        'assay_or_unit_columns_present': methods,
        'interpretation': 'REPEATS_REQUIRE_DATE_SAMPLE_ASSAY_RESOLUTION_NOT_LAST_ROW_WINS',
    }
    if path.stem == 'biomarker_pTau217':
        valid = df.loc[pd.to_numeric(df.ORRES, errors='coerce').notna()]
        last = valid.groupby(['SUBSTUDY', 'BID'], sort=False).tail(1)
        result['legacy_last_numeric_row_subjects'] = len(last)
        result['legacy_last_numeric_row_positive_t0_subjects'] = int(
            (pd.to_numeric(last.COLLECTION_DATE_DAYS_T0, errors='coerce') > 0).sum())
        result['t0_interpretation'] = 'POSITIVE_T0_IS_NOT_PROOF_OF_POST_PET_BUT_REQUIRES_ORIGIN_LINKAGE'
    return result


def xai_receipt_audit(root):
    counts = {'committed_receipts': 0, 'zero_inside_brain': 0,
              'zero_inside_and_zero_original_total': 0,
              'zero_inside_with_nonzero_original_total': 0}
    receipts = {}
    for path in sorted((root / 'private/maps').rglob('*.json')):
        meta = read_json(path)
        if meta.get('status') != 'COMMITTED_MASKED':
            continue
        receipts[path.relative_to(root).as_posix()] = file_sha(path)
        counts['committed_receipts'] += 1
        qc = meta['map_qc']['gradcam_primary']
        if not qc['inside_brain_nonzero']:
            counts['zero_inside_brain'] += 1
            key = ('zero_inside_and_zero_original_total'
                   if qc['outside_brain_absolute_mass_fraction_before_mask'] is None
                   else 'zero_inside_with_nonzero_original_total')
            counts[key] += 1
    counts.update(
        evidence_scope='STORED_QC_RECEIPTS_NOT_NEW_INFERENCE_OR_MAP_REHASH',
        cause='NOT_IDENTIFIABLE_WITHOUT_SIGNED_ACTIVATIONS_AND_GRADIENTS',
        anatomical_claim_allowed=False,
        warning='RELU_GRADCAM_ZERO_DOES_NOT_PROVE_NO_IMAGE_CONTRIBUTION',
    )
    return counts, receipts


def plot_calibration(facts, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True)
    for ax, (manufacturer, arms) in zip(axes, facts['calibration_1095_days'].items()):
        for arm, color in [('A0', '#0077BB'), ('A4', '#EE7733')]:
            b = arms[arm]['calibration_bins']
            # Points only: omitted small bins must not look like a full smooth curve.
            ax.scatter([v['expected'] for v in b], [v['observed'] for v in b],
                       c=color, marker='o' if arm == 'A0' else '^', label=arm)
        ax.plot([0, 1], [0, 1], '--', color='#999999', linewidth=1)
        ax.set(xlim=(0, .55), ylim=(0, .55), title=manufacturer,
               xlabel='Mean predicted 1095-day event risk')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('IPCW-weighted observed event proportion')
    axes[-1].legend()
    fig.suptitle('Held-manufacturer calibration: not validated for clinical use')
    fig.text(.5, .015, 'Fixed-model descriptive bins; n < 10 omitted. No bin CI; 1095 days is not the 1095.75-day AUC horizon.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .93))
    fig.savefig(target, dpi=170)
    plt.close(fig)


def main():
    from scripts.run_method_y_local_validation_v5 import run
    parent_verification = run('verify', V5, 1000)
    if parent_verification['status'] != 'LOCAL_REANALYSIS_COMMITTED_WITH_LIMITATIONS':
        raise RuntimeError('PARENT_V5_NOT_VERIFIED')
    with analysis_lock(OUT):
        inputs = {
            'v5_facts': V5 / 'public_bundle/facts.json',
            'v5_contract': V5 / 'run_contract.json',
            'v4_calibration': ROOT / 'results/method_y_amyloid_lmo_a4_calibration_v4/public_bundle/amyloid_lmo_a4_calibration.json',
            'z_catalog': BLOOD,
            'z_builder': ROOT / 'scripts/build_branch_catalogs.py',
            'xai_aggregate': XAI / 'public_bundle/masked_xai_aggregate.json',
            'review_source': Path(__file__),
        }
        facts = read_json(inputs['v5_facts'])
        blood = pd.read_csv(BLOOD)
        availability = {}
        for branch in ('amyloid', 'tau', 'paired_at'):
            p = ROOT / f'results/method_y_revival_v2/endpoints/{branch}_progression_endpoint.parquet'
            inputs['endpoint_' + branch] = p
            availability[branch] = candidate_overlap(pd.read_parquet(p), blood)
        raw_root = Path('H:/A4 Learn/Clinical')
        source_blood = {}
        for stem in ('biomarker_pTau217', 'biomarker_Plasma_Roche_Results', 'biomarker_AB_Test'):
            p = raw_root / 'External Data' / (stem + '.csv')
            inputs[stem] = p
            source_blood[stem] = blood_source_audit(p)
        site_candidates, headers = [], []
        for path in sorted(raw_root.rglob('*.csv')):
            with path.open(encoding='utf-8-sig', newline='') as f:
                cols = next(csv.reader(f), [])
            matches = [c for c in cols if any(k in c.upper() for k in ('SITE', 'CENTER', 'CENTRE', 'STATION'))]
            headers.append({'file_sha256': file_sha(path), 'column_names': cols})
            if matches:
                site_candidates.append({'source_table': path.name, 'column_names': matches})
        xai_audit, receipt_hashes = xai_receipt_audit(XAI)
        if xai_audit['committed_receipts'] != 100 or xai_audit['zero_inside_brain'] != 83:
            raise RuntimeError('XAI_RECEIPTS_DIFFER_FROM_LOCKED_AGGREGATE')
        comparison = compare_calibration(facts['calibration_1095_days'],
                                         read_json(inputs['v4_calibration'])['manufacturer_results'])
        report = {
            'status': 'POST_V5_REVIEW_COMPLETED_WITH_OPEN_LIMITATIONS',
            'v5_verifier': parent_verification, 'verified_v5_jobs': 11,
            'claim_status': 'POST_HOC_INTERNAL_EXPLORATORY',
            'positive_claim_allowed': False, 'new_training_or_inference': False,
            'calibration_comparison': comparison,
            'pacc_primary': facts['pacc']['amyloid_model'],
            'pacc_covariate_slope_sensitivity': facts['pacc_slope_sensitivity']['amyloid_model'],
            'pacc_interpretation': 'ASSOCIATION_NOT_OUT_OF_SAMPLE_PREDICTIVE_VALIDATION_OR_MAR_PROOF',
            'xai_receipt_audit': xai_audit,
            'site_availability': {
                'clinical_csv_headers_scanned': len(headers), 'candidate_tables': site_candidates,
                'status': 'NOT_ESTIMABLE_FROM_CURRENT_CLINICAL_TABLES' if not site_candidates else 'EXACT_LINKAGE_REQUIRED',
                'scope': 'HEADER_LEVEL_SCREEN_ONLY_NOT_ALL_DICOM_SIDECARS_OR_SITE_CODE_DICTIONARIES',
                'manufacturer_is_not_site': True,
            },
            'method_z': {
                'catalog_rows': len(blood), 'unique_subjects': int(blood.patient_hash.nunique()),
                'date_and_assay_lineage_missing_from_legacy_catalog': True,
                'legacy_scores_allowed_for_prognostic_fusion': False,
                'source_metadata': source_blood, 'candidate_overlap_a4': availability,
                'role': 'BASELINE_BLOOD_PROGNOSTIC_ADAPTER_AFTER_CHRONOLOGY_ASSAY_AND_CROSSFIT_LOCK',
            },
            'local_closure': 'NOT_COMPLETE',
            'external_status': 'DEFERRED_NO_EXTERNAL_DATA',
        }
        # Keep path-level provenance separate from aggregate release material.
        provenance = {'input_hashes': {k: file_sha(p) for k, p in inputs.items()},
                      'xai_receipt_hashes': receipt_hashes, 'clinical_headers': headers}
        bundle = OUT / 'public_bundle'
        atomic_write_json(OUT / 'provenance/review_inputs.json', provenance)
        atomic_write_json(bundle / 'review.json', report)
        findings = scan_public_bundle(bundle)
        if findings:
            raise RuntimeError('REVIEW_PUBLIC_PRIVACY_FAILURE:' + json.dumps(findings))
        atomic_write_json(bundle / 'privacy_scan.json', {
            'status': 'PASS', 'facts_sha256': file_sha(bundle / 'review.json'), 'findings': []})
        plot_calibration(facts, OUT / 'calibration_1095d.png')
        table = ['# Calibration v4 to v5\n',
                 'Technical estimation is not scientific calibration PASS. Fixed-model conditional intervals.\n',
                 '| Manufacturer | Arm | Slope | Change | Bootstrap 95% CI | Valid draws |',
                 '|---|---|---:|---:|---|---:|']
        for r in comparison:
            ci = r['slope_bootstrap_ci95']
            table.append(f"| {r['manufacturer']} | {r['arm']} | {r['slope']:.4f} | {r['slope_change_v5_minus_v4']:+.6f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {r['valid_bootstrap_draws']} |")
        atomic_write_text(OUT / 'CALIBRATION_COMPARISON.md', '\n'.join(table) + '\n')
        print(json.dumps({'status': report['status'], 'privacy': 'PASS', 'v5_jobs': 11,
                          'xai_receipts': xai_audit, 'candidate_overlap_a4': availability}, indent=2))


if __name__ == '__main__':
    main()
