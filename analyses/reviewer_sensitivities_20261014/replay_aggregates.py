"""Validate distributed integrity and export the revised numerical claims."""
from pathlib import Path
import argparse,csv,hashlib,json
ROOT=Path(__file__).resolve().parent
def load(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest=load('MANIFEST_SHA256.json')
    for item in manifest:
        f=ROOT/item['path'];assert f.is_file(),item['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],item['path']
    tau=load('aggregate_results/tau_bootstrap.json');rows=[]
    for model,v in tau['models'].items():
        for family in ['1','3','20','40']:
            for metric in ['ibs','uno']:
                lo,hi=v['intervals'][family][metric];rows.append([model,metric,family,v['delta_'+metric],lo,hi])
        lo,hi=v['intervals']['3']['ibs'];assert lo<0<hi
    for name in ['cox','mri']:assert tau['models'][name]['intervals']['40']['uno'][0]>0
    replay=load('aggregate_results/tabular_sensitivities.json')['original_full_training_comparator_replay']
    assert abs(replay['ibs_error'])<1e-10 and abs(replay['uno_error'])<1e-10
    with (out/'tau_revised_intervals.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['model','metric','family','difference','lower','upper']);w.writerows(rows)
    horizon=load('aggregate_results/brier_3year.json')
    assert len(horizon['comparisons'])==20 and horizon['family_size']==20 and horizon['resamples']==40000
    for row in horizon['comparisons']:
        assert abs((row['bs_augmented']-row['bs_reference'])-row['delta_bs'])<1e-12
        assert row['adjusted_20_ci'][0]<0<row['adjusted_20_ci'][1] and row['all_resamples_finite']
    with (out/'three_year_brier_all_contrasts.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['track','reference','augmented','n','reference_brier','augmented_brier','difference','adjusted_lower','adjusted_upper'])
        w.writerows([[v['track'],v['reference'],v['augmented'],v['n'],v['bs_reference'],v['bs_augmented'],v['delta_bs'],*v['adjusted_20_ci']] for v in horizon['comparisons']])
    selection=load('aggregate_results/pet_selection.json')
    assert sum(v['n'] for v in selection['counts'])==1059
    assert {v['reason']:v['n'] for v in selection['counts']}=={'INCLUDED':544,'NONFINITE_SOURCE_ARRAY':288,'NOT_IN_ASSEMBLED_IMAGE_LIST':227}
    report={'status':'AGGREGATE_REPLAY_PASSED','verified_files':len(manifest),'rows_exported':len(rows),'tau_adjusted_ibs_all_include_zero':True,'two_cox_concordance_intervals_positive_family40':True,'scope':'Integrity and aggregate arithmetic, not participant-level verification or clinical validation'}
    (out/'aggregate_replay.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
