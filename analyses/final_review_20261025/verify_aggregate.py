"""Offline aggregate/package checks only; never import the research runner."""
from pathlib import Path
import json,hashlib,math
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_text('utf-8'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
manifest=read('manifest.json')
present={p.relative_to(P).as_posix() for p in P.rglob('*') if p.is_file() and p.name!='manifest.json'}
assert present==set(manifest['files'])
for n,h in manifest['files'].items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n
a=read('aggregate/A10_late_window_ibs.json');assert a['family_size']==20 and len(a['comparisons'])==20 and a['resamples']==40000
assert a['original_results_unchanged'] and a['max_library_absolute_error']<1e-12
for r in a['comparisons']:
 assert r['adjusted_20_ci'][0]<=0<=r['adjusted_20_ci'][1]
 assert abs((r['late_ibs_augmented']-r['late_ibs_reference'])-r['late_delta_ibs'])<1e-12
b=read('aggregate/A11_subset_nulls_and_ipa.json')
for r in b['models'].values():assert abs(100*(1-r['ibs_model']/r['ibs_training_km_null'])-r['ipa_percent'])<1e-10
c=read('aggregate/A12_component_standardization_invariance.json')
assert c['component_count']==90 and c['new_model_fits']==0 and not c['outer_outcomes_used']
assert c['max_training_difference']<c['tolerance'] and c['max_test_difference']<c['tolerance']
d=read('aggregate/all_draw_concordance_support.json')
for r in d['models']:assert r['pair_weighted_valid_draws']==r['resamples']==40000
assert round(100*(1-d['models'][-1]['all_fold_estimable']/40000),1)==37.3
for file in (P/'aggregate').glob('*.json'):
 obj=read(file.relative_to(P))
 def scan(x):
  if isinstance(x,dict):
   for k,v in x.items():
    assert k.lower() not in {'subjects','subject_id','rid','ptid','predictions','event_times','membership_digests'},k
    scan(v)
  elif isinstance(x,list):
   for v in x:scan(v)
 scan(obj)
print(json.dumps({'status':'PASS_AGGREGATE_INTEGRITY_AND_RECORDED_CHECKS_ONLY','files':len(present),'scientific_approval':False,'external_validation':False}))
