"""Synthetic effective-batch throughput measurement; never reads research images."""
from __future__ import annotations
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from .training import MRISurvival,deterministic
from scripts.xyz_fusion_v1.loss import discrete_time_survival_nll
from scripts.xyz_fusion_v1.runtime import configure,gpu_guard
from scripts.xyz_fusion_v1.store import atomic_json


def run(root):
    root=Path(root);configure(root);gpu_guard()
    p=json.loads((root/'configs/xyz_fusion_v2.json').read_text())['training']
    deterministic(13,p['cpu_threads']);device=torch.device('cuda')
    x=torch.randn((p['batch_size'],1,128,128,128),device=device)
    event=torch.tensor([1,0,1,0],device=device)
    times=torch.tensor([600.,1000.,1200.,1500.],device=device)
    edges=torch.tensor(p['hazard_edges_days'],device=device,dtype=torch.float64)
    initial=copy.deepcopy(MRISurvival(p).state_dict());rows=[];reference=None
    for micro in (1,2,4):
        model=MRISurvival(p).to(device);model.load_state_dict(initial,strict=True);model.train()
        torch.cuda.reset_peak_memory_stats();elapsed=[];gradient=None;loss_total=None
        for repeat in range(6):
            model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            loss_total=0.
            for offset in range(0,len(x),micro):
                with torch.autocast('cuda',enabled=p['amp']):logits=model(x[offset:offset+micro])
                loss=discrete_time_survival_nll(logits.float(),event[offset:offset+micro],times[offset:offset+micro],edges)
                (loss*micro/len(x)).backward();loss_total+=float(loss)*micro/len(x)
            torch.cuda.synchronize()
            if repeat>=2:elapsed.append(time.perf_counter()-start)
        gradient=torch.cat([v.grad.detach().float().reshape(-1) for v in model.parameters() if v.grad is not None]).cpu()
        if not torch.isfinite(gradient).all():raise ValueError('NONFINITE_SYNTHETIC_GRADIENT')
        if reference is None:reference=gradient.clone()
        relative=float(torch.linalg.vector_norm(gradient-reference)/torch.linalg.vector_norm(reference).clamp_min(1e-12))
        rows.append({'microbatch_size':micro,'effective_batch_size':len(x),'seconds_per_effective_batch':float(np.median(elapsed)),
                     'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'relative_gradient_difference':relative,
                     'finite_loss':bool(np.isfinite(loss_total)),'augmentation_used':False})
        del model;torch.cuda.empty_cache()
    eligible=[r for r in rows if r['peak_allocated_mib']<6000 and r['relative_gradient_difference']<.01 and r['finite_loss']]
    if not eligible:raise ValueError('NO_SAFE_MICROBATCH_PROFILE')
    result={'status':'SYNTHETIC_THROUGHPUT_ONLY','rows':rows,
            'recommended_microbatch_size':min(eligible,key=lambda r:r['seconds_per_effective_batch'])['microbatch_size'],
            'production_training_started':False,'effective_batch_or_model_change':False,
            'device':torch.cuda.get_device_name(0),'torch_version':torch.__version__}
    atomic_json(root/'results/xyz_fusion_v2_checks/synthetic_throughput.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':run(Path(__file__).resolve().parents[2])
