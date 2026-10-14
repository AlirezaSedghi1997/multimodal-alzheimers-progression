from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import torch
from scripts.cnn_jacobian_pilot_v1.geometry import checked_logjac,geometry_selftest
from scripts.cnn_jacobian_pilot_v1.prepare import split_manifest
from scripts.cnn_jacobian_pilot_v1.training import Compact,individual_ibs,GRID,co_curves
from scripts.xyz_fusion_v1.loss import discrete_time_survival_nll

def test_physical_jacobian(tmp_path):
 assert len(geometry_selftest(tmp_path))==4

def test_jacobian_rejects_fold_and_nonfinite():
 for value in [0.,-1.,np.nan,np.inf]:
  with pytest.raises(ValueError):checked_logjac(np.full((3,3,3),value),np.ones((3,3,3),bool))

def test_logjac_does_not_individually_zscore():
 x,q=checked_logjac(np.full((3,3,3),np.exp(.5)),np.ones((3,3,3),bool))
 assert np.allclose(x,.5) and q['clipped_fraction']==0

def test_subject_split_and_excluded_outer():
 n=1000;f=pd.DataFrame({'patient_hash':[f'p{i}' for i in range(n)],'event':np.arange(n)%3==0,'time_days':np.where(np.arange(n)%2,900,1400)})
 cfg=json.loads(Path('configs/cnn_jacobian_pilot_v1.json').read_text())
 s=split_manifest(f,f.patient_hash[:800],f.patient_hash[800:],cfg)
 assert [len(s[k]) for k in ('fit','selection','audit')]==[320,64,128]
 assert not (set(sum(s.values(),[]))&set(f.patient_hash[800:]))
 assert s==split_manifest(f.sample(frac=1,random_state=4),f.patient_hash[:800],f.patient_hash[800:],cfg)
 with pytest.raises(ValueError):split_manifest(f,f.patient_hash[:800],f.patient_hash[799:],cfg)

def test_same_architecture_two_channels():
 m=Compact(9);x=torch.randn(2,2,64,64,64);assert m(x).shape==(2,9)
 assert sum(p.numel() for p in m.parameters())<100000

def test_loss_event_and_censor_at_edge():
 logits=torch.zeros(2,2);e=torch.tensor([1,0]);t=torch.tensor([10.,10.]);edges=torch.tensor([0.,10.,20.])
 v=discrete_time_survival_nll(logits,e,t,edges)
 assert float(v)==pytest.approx(np.log(2),abs=1e-6)

def test_individual_ipcw_identity():
 train=pd.DataFrame({'event':[1,0,1,0,1,0,1,0],'time_days':[400,600,800,1000,1200,1400,1500,1640]})
 test=pd.DataFrame({'event':[1,0,1,0],'time_days':[200,700,950,1600]})
 p=np.tile(np.exp(-GRID/5000),(4,1));v=individual_ibs(train,test,p)
 assert np.isfinite(v).all() and len(v)==4

def test_cox_tail_boundary_for_final_neural_bin():
 from sksurv.functions import StepFunction
 class MockCox:
  def predict_survival_function(self,x):return [StepFunction(np.array([100.,1000.,1626.]),np.array([.95,.8,.7]))]
 assert np.allclose(co_curves(MockCox(),None,np.array([365.,1095.75,1640.])),[[.95,.8,.7]])

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_deterministic_backward():
 import os
 from scripts.cnn_jacobian_pilot_v1.training import FixedSpatialPool
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
 x=torch.randn(2,32,8,8,8,device='cuda',requires_grad=True);p=FixedSpatialPool()(x)
 assert torch.allclose(p,torch.nn.functional.adaptive_avg_pool3d(x,(2,2,2)),atol=1e-6)
 p.square().sum().backward();assert torch.isfinite(x.grad).all()
 m=Compact(9).cuda();m(torch.randn(2,2,64,64,64,device='cuda')).square().mean().backward()
 assert all(torch.isfinite(v.grad).all() for v in m.parameters())
