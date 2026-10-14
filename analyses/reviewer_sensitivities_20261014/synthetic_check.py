"""No restricted data: verify installation, metric identities and model interfaces."""
from pathlib import Path
import json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source_snapshot'));sys.path.insert(0,str(HERE/'revision_analyses'))
import selected_analyses as s
from scripts.xyz_fusion_v2.models import fit_model,curves
from scripts.xyz_fusion_v1.statistics import ibs_score
from scripts.xyz_fusion_v1.report import individual_bs
from scripts.xyz_fusion_v2.training import MRISurvival,deterministic
import pandas as pd,torch

cfg=json.loads((HERE/'config/xyz_fusion_v2.json').read_text());s.GRID=np.array(cfg['fusion']['ibs_grid_days'])
rng=np.random.default_rng(1729);n=300
t=pd.DataFrame({'patient_hash':[f'SYNTHETIC_{i:04d}' for i in range(n)],'age':rng.normal(72,5,n),'sex_binary':rng.integers(0,2,n),'event':rng.random(n)<.4,'time_days':rng.uniform(30,1640,n)})
t.loc[0,'age']=np.nan
tr,te=t.iloc[:220].copy(),t.iloc[220:].copy();tr.loc[tr.index[-1],'time_days']=1640;te.loc[te.index[-1],'time_days']=1550
model=fit_model(tr,['age','sex_binary'],'cox_ridge',10,cfg,1729);pred=curves(model,te,['age','sex_binary'],s.GRID)
assert pred.shape==(80,9) and (np.diff(pred,axis=1)<=1e-6).all()
value=ibs_score(tr,te,pred,s.GRID);loss,_,_=individual_bs(tr,te,pred,s.GRID);assert abs(value-loss.mean())<1e-12
try:fit_model(tr,['event'],'cox_ridge',10,cfg,1729)
except ValueError:pass
else:raise AssertionError('OUTCOME_LEAKAGE_GUARD_FAILED')
metric=s.validate_concordance()
import statsmodels.api as sm
x=rng.normal(-2,.7,250);y=rng.binomial(1,1/(1+np.exp(-(0.2+0.8*x))));w=rng.uniform(.8,2,250)
fit=sm.GLM(y,sm.add_constant(x),family=sm.families.Binomial(),freq_weights=w).fit()
assert np.allclose(s.logistic_calibration(x,y,w),fit.params,atol=1e-7)
deterministic(1729,2);network=MRISurvival(cfg['training']);network.eval()
# Full published input geometry, deterministic random synthetic intensity only.
with torch.no_grad():logits=network(torch.randn(1,1,128,128,128))
assert logits.shape==(1,9) and torch.isfinite(logits).all()
receipt={'status':'SYNTHETIC_IMPLEMENTATION_CHECK_PASSED','not_scientific_validation':True,'ibs_metric_error':abs(value-loss.mean()),'concordance':metric,'network_parameters':sum(p.numel() for p in network.parameters()),'network_output_shape':list(logits.shape),'synthetic_only':True}
(HERE/'checks').mkdir(exist_ok=True);(HERE/'checks/synthetic_check.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(receipt,indent=2))
