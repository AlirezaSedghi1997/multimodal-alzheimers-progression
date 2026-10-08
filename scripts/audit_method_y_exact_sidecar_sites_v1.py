"""Metadata-only, exact-stem site screen; never guesses a site from a scanner.

The two inventories are scan inventories, not the final training population.
No NIfTI image, outcome, checkpoint, or existing artifact is modified.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import pandas as pd
from src.method_y_revival.atomic_io import atomic_write_json
from src.method_y_revival.privacy import scan_public_bundle

OUT = ROOT / 'results/method_y_exact_site_screen_v1'
SOURCES = {
    'amyloid': ('data/preprocessed_pet/_private/manifests/fbp_private_manifest.csv', 'raw_path_private'),
    'tau': ('results/method_y_revival_v2/manifests/private_ftp_t1_transform_eligible.parquet', 'ftp_path'),
}
SITE_KEYS = frozenset({'Site', 'SiteID', 'SiteName', 'Center', 'CenterID', 'CenterName',
                       'Centre', 'CentreID', 'InstitutionName', 'ClinicalTrialSiteID',
                       'ClinicalTrialSiteName'})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def site_evidence(payload):
    records = payload if isinstance(payload, list) else [payload]
    values, keys = set(), set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('UNSUPPORTED_SIDECAR_SCHEMA')
        for key in SITE_KEYS & record.keys():
            value = record[key]
            if value is None or str(value).strip().lower() in {'', 'unknown', 'n/a', 'na', 'none'}:
                continue
            if isinstance(value, (dict, list)):
                raise ValueError('NONSCALAR_SITE_FIELD')
            keys.add(key)
            # Different key types are not automatically synonyms or site codes.
            values.add((key, str(value).strip().casefold()))
    conflicts = any(sum(k == key for k, _ in values) > 1 for key in keys)
    return {
        'status': 'AMBIGUOUS' if conflicts else ('CANDIDATE_SITE_FIELDS' if values else 'NO_SITE_FIELD'),
        'keys': sorted(keys),
        'private_values_digest': hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest(),
    }


def main():
    private, summaries, inputs = {}, {}, {}
    for branch, (relative, column) in SOURCES.items():
        manifest = ROOT / relative
        before = sha(manifest)
        inputs[relative] = before
        frame = pd.read_parquet(manifest) if manifest.suffix == '.parquet' else pd.read_csv(manifest)
        paths = sorted(set(frame[column].dropna().astype(str)))
        counts, key_counts = Counter(), Counter()
        branch_rows = []
        for raw_path in paths:
            raw = Path(raw_path)
            sidecar = raw.with_suffix('').with_suffix('.json') if raw.name.endswith('.nii.gz') else raw.with_suffix('.json')
            row = {'raw_path': raw_path, 'sidecar_path': str(sidecar)}
            if not sidecar.is_file():
                row['status'] = 'EXACT_SIDECAR_MISSING'
            else:
                content = sidecar.read_bytes()
                row['sha256'] = hashlib.sha256(content).hexdigest()
                try:
                    row.update(site_evidence(json.loads(content.decode('utf-8-sig'))))
                except (ValueError, TypeError, UnicodeError) as exc:
                    row['status'] = 'UNREADABLE_OR_UNSUPPORTED'
                    row['exception_type'] = type(exc).__name__
                key_counts.update(row.get('keys', []))
                if sha(sidecar) != row['sha256']:
                    raise RuntimeError('SIDECAR_CHANGED_DURING_READ')
            counts[row['status']] += 1
            branch_rows.append(row)
        if sha(manifest) != before:
            raise RuntimeError('MANIFEST_CHANGED_DURING_READ')
        private[branch] = branch_rows
        summaries[branch] = {'manifest_rows': len(frame), 'unique_scan_paths': len(paths),
                             'status_counts': dict(counts), 'nonempty_site_key_counts': dict(key_counts)}
    report = {
        'status': 'EXACT_SIDECAR_METADATA_SCREEN_COMPLETED', 'branches': summaries,
        'scope': 'LISTED_SCAN_INVENTORIES_NOT_FINAL_TRAINING_MEMBERSHIP_OR_ALL_RAW_DICOM',
        'manufacturer_or_station_used_as_site': False,
        'site_code_dictionary_validated': False,
        'leave_site_out_status': 'NOT_ESTIMABLE_WITHOUT_VALIDATED_SITE_LINKAGE',
        'positive_claim_allowed': False,
    }
    atomic_write_json(OUT / 'private/sidecar_evidence.json', {'inputs': inputs, 'scans': private})
    atomic_write_json(OUT / 'public_bundle/site_screen.json', report)
    atomic_write_json(OUT / 'provenance.json', {'source_sha256': sha(__file__), 'inputs': inputs,
        'private_evidence_sha256': sha(OUT / 'private/sidecar_evidence.json'),
        'report_sha256': sha(OUT / 'public_bundle/site_screen.json')})
    violations = scan_public_bundle(OUT / 'public_bundle')
    privacy = {'status': 'FAIL' if violations else 'PASS', 'violations': violations}
    atomic_write_json(OUT / 'privacy_scan.json', privacy)
    print(json.dumps({'report': report, 'privacy': privacy}, ensure_ascii=False))
    if privacy.get('status') != 'PASS':
        raise RuntimeError('PUBLIC_PRIVACY_FAILED')


if __name__ == '__main__':
    main()
