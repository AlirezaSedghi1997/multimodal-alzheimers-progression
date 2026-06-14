"""SynIB-inspired multimodal synergy loss (arXiv:2606.09853).

Penalizes confident predictions when one modality is masked, encouraging
cross-modal synergy rather than unimodal shortcuts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


@dataclass(frozen=True)
class SynIBConfig:
    lambda_syn: float = 0.5
    mask_prob: float = 0.5
    tabular_dim: int = 5
    mri_dim: int = 14


def confidence_penalty(probs: Any, eps: float = 1e-7) -> Any:
    """High confidence => low entropy => high penalty."""
    if torch is None:
        raise RuntimeError("PyTorch required for confidence_penalty")
    p = probs.clamp(eps, 1.0 - eps)
    entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    max_entropy = -(0.5 * torch.log(torch.tensor(0.5)) * 2.0)
    return 1.0 - entropy / max_entropy


def synib_multimodal_loss(
    logits_full: Any,
    logits_tab_masked: Any,
    logits_mri_masked: Any,
    targets: Any,
    *,
    config: SynIBConfig | None = None,
    pos_weight: Any = None,
) -> tuple[Any, dict[str, float]]:
    if torch is None or F is None:
        raise RuntimeError("PyTorch required for synib_multimodal_loss")
    cfg = config or SynIBConfig()
    logits_flat = logits_full.squeeze(-1) if logits_full.dim() > 1 else logits_full
    task = F.binary_cross_entropy_with_logits(logits_flat, targets.float(), pos_weight=pos_weight)
    p_tab = torch.sigmoid(logits_tab_masked.squeeze(-1) if logits_tab_masked.dim() > 1 else logits_tab_masked)
    p_mri = torch.sigmoid(logits_mri_masked.squeeze(-1) if logits_mri_masked.dim() > 1 else logits_mri_masked)
    syn = confidence_penalty(p_tab).mean() + confidence_penalty(p_mri).mean()
    total = task + cfg.lambda_syn * syn
    return total, {"task_loss": float(task.detach()), "syn_penalty": float(syn.detach()), "total": float(total.detach())}


def mask_modalities(
    x: Any,
    *,
    tabular_dim: int,
    mask_tab: bool,
    mask_mri: bool,
) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch required for mask_modalities")
    out = x.clone()
    if mask_tab:
        out[:, :tabular_dim] = 0.0
    if mask_mri:
        out[:, tabular_dim:] = 0.0
    return out
