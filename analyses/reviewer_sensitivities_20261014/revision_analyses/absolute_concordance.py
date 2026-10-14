"""Conditional absolute Uno C intervals from frozen predictions; no model fitting.

Run beside selected_analyses.py. Only aggregate results are distributable.
The sibling private folder records restricted input hashes and local paths.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from sksurv.metrics import concordance_index_ipcw
from threadpoolctl import threadpool_limits
import selected_analyses as s

TABLE2 = {
    'mri': ['clinical', 'clinical_amyloid', 'clinical_mri_amyloid',
            'clinical_volumes_amyloid', 'clinical_volumes_mri_amyloid', 'mri_only',
            'blood_clinical_amyloid', 'blood_clinical_amyloid_blood',
            'blood_clinical_mri_amyloid_blood'],
    'pet': ['clinical_amyloid', 'clinical_mri_amyloid', 'clinical_mri_amyloid_pet'],
}


def estimate(cells, name, draws, seed):
    lengths = [len(c['arms'][name]['subjects']) for c in cells]
    counts = s.bootstrap_counts(lengths, draws, seed)
    samples = np.zeros(draws)
    points, undefined = [], {}
    for f, cell in enumerate(cells):
        arm = cell['arms'][name]
        _, tr, te = s.fold_arrays(arm)
        risk = 1 - np.asarray(arm['curves'], float)[:, -1]
        num, den = s.concordance_matrices(tr, te, risk, s.GRID[-1])
        values = s.ratio_quadratic(counts[f], num, den)
        point = s.ratio_quadratic(np.ones((1, len(te))), num, den)[0]
        library = concordance_index_ipcw(tr, te, risk, tau=s.GRID[-1])[0]
        assert abs(point - library) < 1e-12
        recorded = arm['metrics'].get('uno_c_index')
        if recorded is not None:
            assert abs(point - recorded) < 1e-12, (name, f, point, recorded)
        # Direct duplicated-record calculation also checks actual-data resamples.
        for b in range(3):
            ix = np.repeat(np.arange(len(te)), counts[f][b].astype(int))
            try:
                check = concordance_index_ipcw(tr, te[ix], risk[ix], tau=s.GRID[-1])[0]
            except ValueError:
                check = np.nan
            assert (not np.isfinite(check) and not np.isfinite(values[b])) or abs(check-values[b]) < 1e-12
        samples += values * lengths[f] / sum(lengths)
        points.append(point)
        undefined[str(f+1)] = int((~np.isfinite(values)).sum())
    return {'n':sum(lengths), 'uno_c':float(np.average(points,weights=lengths)),
            'nominal_ci95':s.interval(samples), 'resamples':draws,
            'valid_resamples':int(np.isfinite(samples).sum()),
            'undefined_by_fold':undefined}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--resamples',type=int,default=2000)
    a=p.parse_args();s.setup(a.repo);out=Path(a.output)
    assert a.resamples >= 2000
    target=out/'absolute_concordance.json'
    if target.exists():
        raise FileExistsError('Use a fresh output directory; existing results are preserved.')
    result={'status':'POST_HOC_CONDITIONAL_ABSOLUTE_PRECISION',
            'resamples':a.resamples,'seed':20261014,'horizon_days':float(s.GRID[-1]),
            'intervals':'NOMINAL_95_PERCENT_NOT_MULTIPLICITY_ADJUSTED',
            'scope':'Fixed predictions, outer folds and training censoring distributions. No CNN/head retraining. Not intervals for paired incremental differences.',
            'undefined_policy':'Omit a draw if any fold has zero IPCW comparable-pair support.',
            'metric_validation':s.validate_concordance(), 'table2':{},'pooled_oof':{}}
    started=time.perf_counter()
    for track, names in TABLE2.items():
        cells=s.load(s.JOB/f'{track}_ensemble_cells.json')
        for name in names:
            result['table2'][track+':'+name]=estimate(cells,name,a.resamples,20261014)
            print('Table2',track,name,flush=True)
    result['table2']['mri:null']={'n':1059,'uno_c':0.5,'nominal_ci95':[0.5,0.5],
        'scope':'All participants within each fold share the same KM risk; C is 0.5 whenever comparable pairs exist.'}
    for track in ['mri','pet','blood','tau']:
        payload=s.load(s.ROOT/f'results/xyz_oof_stacking_v1/private/units/{track}_cells.json')
        cells=payload['cells']
        for name in cells[0]['arms']:
            result['pooled_oof'][track+':'+name]=estimate(cells,name,a.resamples,20261014)
            print('OOF',track,name,flush=True)
    s.dump(target,result)
    s.dump(out.parent/'private'/'absolute_concordance_inputs.json',
           {'inputs':s.INPUTS,'elapsed_seconds':time.perf_counter()-started})
    print(json.dumps({'status':result['status'],'table2_models':len(result['table2']),
                      'oof_models':len(result['pooled_oof']),
                      'elapsed_seconds':round(time.perf_counter()-started,2)}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
