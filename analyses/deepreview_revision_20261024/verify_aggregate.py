"""Safe offline checks; no imports of study runners or restricted data."""
from pathlib import Path
import json,hashlib,re,math

ROOT=Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT/'aggregate'/name).read_text('utf-8'),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('NONFINITE_JSON:'+x)))
def verify():
    manifest=json.loads((ROOT/'manifest.json').read_text('utf-8'))
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    assert actual==set(manifest['files']),'BUNDLE_FILESET_DRIFT'
    for name,want in manifest['files'].items():
        path=(ROOT/name).resolve();assert path.is_relative_to(ROOT) and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==want['sha256'],name
    for path in (ROOT/'aggregate').glob('*.json'):
        raw=path.read_bytes();json.loads(raw,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        assert not re.search(rb'pt_[a-f0-9]{16}|\b\d{3}_S_\d{4}\b',raw),'INDIVIDUAL_IDENTIFIER_PATTERN'
    tau=read('A01_tau_omission_robustness.json');assert tau['originals_reconciled']
    assert all(v['resamples']==40000 and v['valid_uno_resamples']==39361 and v['undefined_uno_resamples']==639 for v in tau['models'].values())
    overlap=read('A04_matched_biomarker_overlap.json');assert (overlap['n'],overlap['events'],overlap['events3y'])==(220,60,26)
    deformation=read('A09_deformation_control.json');assert deformation['n']==1059 and deformation['excluded']==0
    row=deformation['results']['base__deformation'];assert row['delta_ibs']>0 and row['intervals']['2']['ibs_all'][0]>0
    numerical=read('NUMERICAL_VERIFICATION.json');assert numerical['ibs_absolute_error']<1e-12 and numerical['edf_absolute_error']<1e-12
    assert math.isclose(numerical['zero_event_probability_70_draws_4_events'],(66/70)**70,abs_tol=1e-14)
    assert numerical['concordance_duplicate_resample_check']['max_absolute_error']<1e-12
    return {'status':'PASS_BUNDLE_INTEGRITY_AND_AGGREGATE_CONSISTENCY_ONLY','files':len(actual),'external_validation':False,'recomputed_restricted_data_results':False,'scientific_approval':False}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
