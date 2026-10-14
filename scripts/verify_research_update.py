"""Check the research-update payload without accessing restricted inputs."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'analyses/reviewer_sensitivities_20261014'
BLOCKED_SUFFIXES = {'.docx', '.pdf', '.pptx', '.pkl', '.parquet', '.npy', '.npz',
                    '.pt', '.pth', '.ckpt', '.h5', '.hdf5', '.nii', '.gz',
                    '.zip', '.pem', '.key', '.joblib'}
PATTERNS = {
    'participant_key': re.compile(r'\bpt_[0-9a-f]{12,}\b', re.I),
    'source_subject_key': re.compile(r'\b(?:B\d{6,}|\d{3}_S_\d{4})\b'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})\b'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}
FORBIDDEN_RESULT_KEYS = {'patient_hash', 'subject_id', 'participant_id', 'BID',
                         'mri_sample_hash', 'image_path_private', 'subjects',
                         'train_subjects', 'test_subjects', 'patient_order'}


def walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def verify():
    inv = json.loads((ROOT/'docs/RESEARCH_UPDATE_INVENTORY_20261014.json').read_text())
    paths = {ROOT / row['path'] for row in inv['files']}
    paths.update(BUNDLE.rglob('*'))
    paths.update((ROOT/'scripts/cnn_jacobian_pilot_v1').glob('*.py'))
    paths.update(ROOT.glob('configs/cnn_jacobian_pilot_v1*.json'))
    paths.update((ROOT/'results/aggregate/12_cnn_jacobian_pilot').glob('*.json'))
    paths.update(ROOT.glob('docs/RESEARCH_UPDATE*'))
    paths.update(ROOT/p for p in ('README.md','CHANGELOG.md','docs/CNN_JACOBIAN_PILOT.md',
                                  'scripts/run_cnn_jacobian_pilot_v1.py',
                                  'tests/test_cnn_jacobian_pilot_v1.py'))
    files = sorted(p for p in paths if p.is_file())
    errors = []
    for p in files:
        rel=p.relative_to(ROOT).as_posix()
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT):
            errors.append((rel,'unsafe_path')); continue
        if p.suffix.lower() in BLOCKED_SUFFIXES or {'private','_private','.git','__pycache__'} & set(p.relative_to(ROOT).parts):
            errors.append((rel,'prohibited_payload')); continue
        text = p.read_text(encoding='utf-8-sig')
        for label, pattern in PATTERNS.items():
            if pattern.search(text): errors.append((rel,label))
        if p.suffix=='.py': ast.parse(text,filename=rel)
        if p.suffix=='.json':
            obj=json.loads(text)
            if {'aggregate_results','prior_aggregate_results','aggregate'} & set(p.parts):
                found=FORBIDDEN_RESULT_KEYS & set(walk_keys(obj))
                if found: errors.append((rel,'individual_result_keys:'+','.join(sorted(found))))
    for row in inv['files']:
        p=ROOT/row['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['distributed_sha256']:
            errors.append((row['path'],'inventory_hash_mismatch'))
    if any(not r['included'] or not r['identical'] for r in inv['result_coverage']):
        errors.append(('inventory','missing_or_modified_final_aggregate'))
    if any(not r['same_as_executed'] for r in inv['pilot_dependency_identity']):
        errors.append(('inventory','pilot_parent_dependency_changed'))
    # Check independent arithmetic relationships rather than just status strings.
    c=json.loads((BUNDLE/'aggregate_results/absolute_concordance.json').read_text())
    assert c['resamples']==2000
    assert len(c['table2'])==13
    assert len(c['pooled_oof'])==16
    logs=json.loads((BUNDLE/'aggregate_results/original_training_logs.json').read_text())
    assert logs['record_count']==len(logs['records'])==405 and logs['new_training'] is False
    oof=json.loads((BUNDLE/'aggregate_results/oof_aggregate_results.json').read_text())
    assert len(oof['comparisons'])==7 and oof['new_cnn_training'] is False
    decision=json.loads((ROOT/'results/aggregate/12_cnn_jacobian_pilot/decision.json').read_text())
    assert decision['status']=='PILOT_COMPLETED_NO_GO_FULL_DATASET_NOT_RUN'
    assert not decision['performance_numbers_distributed'] and not decision['full_cohort_run_performed']
    if errors:
        raise ValueError(json.dumps(errors))
    return {'status':'PASS_SCOPED_RESEARCH_PUBLICATION_CHECK', 'files_checked':len(files),
            'source_files_accounted_for':len(inv['files']),
            'covered_final_result_instances':len(inv['result_coverage']),
            'absolute_concordance_primary_models':13,'absolute_concordance_oof_entries':16,
            'original_training_histories':405,'oof_contrasts':7,
            'participant_artifacts_included':False,'manuscript_files_included':False,
            'pilot_performance_included':False,'scientific_validation':False,
            'scope':'Allowlist, hashes, prohibited extensions/keys, identifier/credential patterns and aggregate structure; not a legal or scientific approval'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=verify()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))
