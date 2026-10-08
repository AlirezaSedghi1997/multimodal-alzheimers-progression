from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


def evidential_alpha(logits):
    if F is None:
        raise RuntimeError("PyTorch is required for evidential losses")
    return F.softplus(logits) + 1.0


def kl_anneal_factor(step: int, anneal_steps: int) -> float:
    if anneal_steps <= 0:
        return 1.0
    return max(0.0, min(float(step) / float(anneal_steps), 1.0))


def dirichlet_kl_to_uniform(alpha):
    if torch is None:
        raise RuntimeError("PyTorch is required for evidential losses")
    beta = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    sum_beta = beta.sum(dim=-1, keepdim=True)
    ln_b_alpha = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
    ln_b_beta = torch.lgamma(beta).sum(dim=-1, keepdim=True) - torch.lgamma(sum_beta)
    digamma = torch.digamma(alpha) - torch.digamma(sum_alpha)
    kl = ((alpha - beta) * digamma).sum(dim=-1, keepdim=True) + ln_b_alpha + ln_b_beta
    return kl.squeeze(-1)


def evidential_cross_entropy_loss(logits, target, *, step: int = 0, anneal_steps: int = 1000):
    if torch is None:
        raise RuntimeError("PyTorch is required for evidential losses")
    alpha = evidential_alpha(logits)
    strength = alpha.sum(dim=-1, keepdim=True)
    target = target.to(device=logits.device).long().view(-1)
    one_hot = torch.stack([(target == 1).float(), (target == 0).float()], dim=-1)
    expected_ce = (one_hot * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=-1)
    anneal = kl_anneal_factor(step, anneal_steps)
    kl = dirichlet_kl_to_uniform(alpha)
    return (expected_ce + anneal * kl).mean()


def centiloid_huber_loss(cl_pred, cl_true, *, synthetic_mask=None, missing_mask=None, delta: float = 1.0):
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for evidential losses")
    pred = cl_pred.reshape(-1)
    true = cl_true.to(device=pred.device, dtype=pred.dtype).reshape(-1)
    keep = torch.isfinite(true)
    if synthetic_mask is not None:
        keep = keep & (~synthetic_mask.to(device=pred.device).bool().reshape(-1))
    if missing_mask is not None:
        keep = keep & (~missing_mask.to(device=pred.device).bool().reshape(-1))
    if not bool(keep.any()):
        return pred.sum() * 0.0
    return F.huber_loss(pred[keep], true[keep], delta=delta)
