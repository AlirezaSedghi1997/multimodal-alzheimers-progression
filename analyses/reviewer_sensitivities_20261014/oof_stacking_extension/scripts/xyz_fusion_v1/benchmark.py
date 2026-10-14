"""Short outcome-blind GPU microbatch benchmark; effective batch and loss stay fixed."""
from __future__ import annotations
import copy
import statistics
import time
import numpy as np
import torch
from .mri_training import MRISurvival, deterministic
from .loss import discrete_time_survival_nll


def benchmark_microbatch(policy):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA_REQUIRED_FOR_PRODUCTION_IMAGE_RUN')
    deterministic(1729,policy['cpu_threads']); device=torch.device('cuda')
    model=MRISurvival(policy).to(device); model.train()
    initial=copy.deepcopy(model.state_dict())
    x=torch.randn(policy['batch_size'],1,128,128,128,device=device)
    e=torch.tensor([i%2 for i in range(len(x))],device=device)
    t=torch.linspace(300,1500,len(x),device=device); edges=torch.tensor(policy['hazard_edges_days'],device=device)
    records=[];reference=None
    for micro in (1,2,4):
        if micro>len(x):continue
        try:
            times=[]; gradient=None; value=0.
            for repeat in range(4):
                model.load_state_dict(initial);model.zero_grad(set_to_none=True)
                torch.cuda.synchronize();start=time.perf_counter();value=0.
                for i in range(0,len(x),micro):
                    with torch.autocast('cuda',enabled=policy['amp']): logits=model(x[i:i+micro])
                    loss=discrete_time_survival_nll(logits.float(),e[i:i+micro],t[i:i+micro],edges)*len(x[i:i+micro])/len(x)
                    loss.backward();value+=float(loss.detach())
                torch.cuda.synchronize()
                if repeat:times.append(time.perf_counter()-start)
            gradient=torch.cat([p.grad.detach().float().flatten() for p in model.parameters()]).cpu()
            if reference is None:reference=(gradient.clone(),value)
            grad_rel=float(torch.linalg.vector_norm(gradient-reference[0])/(torch.linalg.vector_norm(reference[0])+1e-8))
            loss_rel=abs(value-reference[1])/max(abs(reference[1]),1e-6)
            records.append({'microbatch':micro,'median_seconds':statistics.median(times),'repeats_seconds':times,
                            'gradient_relative_l2':grad_rel,'loss_relative_error':loss_rel,
                            'equivalence_pass':grad_rel<=0.005 and loss_rel<=0.001})
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            records.append({'microbatch':micro,'status':'OOM_REJECTED'})
    good=[r for r in records if r.get('equivalence_pass')]
    if not good: raise RuntimeError('GPU_BENCHMARK_NO_VALID_CANDIDATE')
    base=good[0];best=min(good,key=lambda r:r['median_seconds'])
    selected=best if (best['median_seconds']<=0.85*base['median_seconds']
                     and max(best['repeats_seconds'])<min(base['repeats_seconds'])) else base
    return {'scope':'SYNTHETIC_128CUBE_KERNEL_BENCHMARK_NOT_REAL_COHORT_EPOCH_SPEEDUP',
            'selected_microbatch':selected['microbatch'],'records':records,
            'math_policy':'FP16_AUTOCAST_FP32_LOSS_DETERMINISTIC_NO_TF32',
            'effective_batch':policy['batch_size'],'minimum_accepted_speedup':0.15,
            'preprocessing_workers_are_ram_capped':True}
