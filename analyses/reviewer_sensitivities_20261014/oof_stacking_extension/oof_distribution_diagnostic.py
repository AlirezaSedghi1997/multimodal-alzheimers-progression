"""Aggregate only: decompose pooled training-score variance; never refit a model."""
import argparse,json,sys
from pathlib import Path
import numpy as np

def decompose(parts):
 a=[np.asarray(x,dtype=float) for x in parts]
 if len(a)<2 or any(x.ndim!=1 or not len(x) or not np.isfinite(x).all() for x in a):raise ValueError('INVALID_PARTS')
 n=np.array([len(x) for x in a]);m=np.array([x.mean() for x in a]);v=np.array([x.var() for x in a]);w=n/n.sum()
 mu=w@m;between=float(w@((m-mu)**2));within=float(w@v);total=float(np.concatenate(a).var())
 if not np.isclose(total,between+within,rtol=1e-10,atol=1e-18):raise ValueError('VARIANCE_IDENTITY_FAILED')
 return {'between_component_variance_fraction':between/total if total>0 else None,'pooled_sd':float(np.sqrt(total)),'within_component_sd':[float(np.sqrt(x)) for x in v]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 sys.path.insert(0,str(args.repo))
 from scripts.xyz_oof_stacking_v1.inputs import Inputs
 x=Inputs(args.repo,json.loads((args.repo/'configs/xyz_oof_stacking_v1.json').read_text()))
 rows=[]
 for track,modalities in [('mri',['mri']),('pet',['mri','pet'])]:
  for mod in modalities:
   for seed in x.config['training_seeds']:
    for fold in range(5):
     parts=[x.feature_frame(f'{track}_s{seed}_o{fold}_c{c}','meta')[f'{mod}_risk_3'].to_numpy() for c in range(3)]
     rows.append(dict(track=track,modality=mod,seed=seed,fold=fold,**decompose(parts)))
 result={'scope':'POST_HOC_DIAGNOSTIC_NO_REFITTING_OR_SELECTION','feature':'cumulative_hazard_at_1095_75_days','rows':rows}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
 print('aggregate diagnostic',len(rows),'rows; no participant data exported')
if __name__=='__main__':main()
