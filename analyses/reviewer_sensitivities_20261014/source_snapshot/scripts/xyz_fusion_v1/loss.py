"""Versioned discrete hazard likelihood; right-closed event intervals.

An event at an upper edge belongs to that interval. A censor at the same
edge has survived that WHOLE interval. Do not change historical Y artifacts.
"""
import torch
import torch.nn.functional as F


def survival_from_logits(logits):
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('INVALID_HAZARD_LOGITS')
    return torch.cumprod(torch.sigmoid(-logits), dim=1)


def discrete_time_survival_nll(logits, event, observed_time, bin_edges):
    e=event.reshape(-1).to(logits.device)
    t=observed_time.reshape(-1).to(device=logits.device,dtype=torch.float64)
    edges=bin_edges.reshape(-1).to(device=logits.device,dtype=torch.float64)
    if (logits.ndim!=2 or logits.shape[0]!=len(e) or len(t)!=len(e)
            or len(edges)!=logits.shape[1]+1):
        raise ValueError('SURVIVAL_BATCH_SHAPE_MISMATCH')
    if (not torch.isfinite(logits).all() or not torch.isfinite(t).all()
            or not torch.isfinite(edges).all() or not ((e==0)|(e==1)).all()
            or not (edges[1:]>edges[:-1]).all() or not (t>edges[0]).all()
            or ((e==1)&(t>edges[-1])).any()):
        raise ValueError('INVALID_SURVIVAL_BATCH')
    event_bin=torch.bucketize(t,edges[1:],right=False).clamp(max=logits.shape[1]-1)
    fully_observed=torch.bucketize(t,edges[1:],right=True)
    j=torch.arange(logits.shape[1],device=logits.device)[None,:]
    is_event=e[:,None]==1
    at_risk=torch.where(is_event,j<=event_bin[:,None],j<fully_observed[:,None])
    event_cell=is_event&(j==event_bin[:,None])
    terms=torch.where(event_cell,F.logsigmoid(logits),F.logsigmoid(-logits))
    return -(terms*at_risk).sum(dim=1).mean()
