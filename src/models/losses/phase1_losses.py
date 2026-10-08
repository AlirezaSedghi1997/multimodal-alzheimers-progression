from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


SUPPORTED_TASK_LOSSES: tuple[str, ...] = ("bce", "focal", "auc_pairwise", "bce_auc")


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for Phase 1 task losses")


def _binary_cross_entropy(logits, targets, *, pos_weight=None):
    _require_torch()
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def binary_focal_loss_with_logits(
    logits,
    targets,
    *,
    pos_weight=None,
    gamma: float = 2.0,
    alpha: float | None = None,
):
    """Binary focal loss for imbalanced Method X training, off unless selected."""

    _require_torch()
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_factor = (1.0 - p_t).clamp(0.0, 1.0).pow(float(gamma))
    if alpha is not None:
        alpha = float(alpha)
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal_factor = focal_factor * alpha_t
    return (focal_factor * bce).mean()


def pairwise_auc_surrogate_loss(
    logits,
    targets,
    *,
    margin: float = 1.0,
    pos_weight=None,
    fallback_to_bce: bool = True,
):
    """Dependency-free pairwise logistic AUC surrogate.

    The loss needs positive and negative labels in the active batch. With very
    small batches it falls back to BCE so opt-in smoke/debug runs remain stable.
    """

    _require_torch()
    labels = targets.reshape(-1)
    scores = logits.reshape(-1)
    positives = scores[labels >= 0.5]
    negatives = scores[labels < 0.5]
    pairs = int(positives.numel() * negatives.numel())
    if pairs == 0:
        if not fallback_to_bce:
            return scores.sum() * 0.0, {"pairs": 0, "fallback_to_bce": False}
        return _binary_cross_entropy(logits, targets, pos_weight=pos_weight), {
            "pairs": 0,
            "fallback_to_bce": True,
        }
    scale = max(float(margin), 1e-6)
    diff = positives.reshape(-1, 1) - negatives.reshape(1, -1)
    loss = F.softplus(-diff / scale).mean()
    return loss, {"pairs": pairs, "fallback_to_bce": False}


def compute_phase1_task_loss(
    logits,
    targets,
    *,
    loss_name: str = "bce",
    pos_weight=None,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    auc_margin: float = 1.0,
    auc_loss_weight: float = 1.0,
):
    """Return the selected task loss plus diagnostics for metrics manifests."""

    _require_torch()
    loss_name = str(loss_name or "bce").lower().strip()
    if loss_name not in SUPPORTED_TASK_LOSSES:
        raise ValueError(f"Unsupported task loss: {loss_name}")

    bce_loss = _binary_cross_entropy(logits, targets, pos_weight=pos_weight)
    diagnostics: dict[str, float | int | bool | str | None] = {
        "task_loss_name": loss_name,
        "bce_loss": float(bce_loss.detach()),
        "focal_loss": None,
        "auc_pairwise_loss": None,
        "auc_pairwise_pairs": 0,
        "auc_pairwise_fallback_to_bce": False,
    }
    if loss_name == "bce":
        return bce_loss, diagnostics

    alpha = None if focal_alpha is None or float(focal_alpha) < 0.0 else float(focal_alpha)
    if loss_name == "focal":
        focal = binary_focal_loss_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            gamma=focal_gamma,
            alpha=alpha,
        )
        diagnostics["focal_loss"] = float(focal.detach())
        return focal, diagnostics

    auc_loss, auc_diag = pairwise_auc_surrogate_loss(
        logits,
        targets,
        margin=auc_margin,
        pos_weight=pos_weight,
        fallback_to_bce=True,
    )
    diagnostics["auc_pairwise_loss"] = float(auc_loss.detach())
    diagnostics["auc_pairwise_pairs"] = int(auc_diag["pairs"])
    diagnostics["auc_pairwise_fallback_to_bce"] = bool(auc_diag["fallback_to_bce"])
    if loss_name == "auc_pairwise":
        return auc_loss, diagnostics

    total = bce_loss + float(auc_loss_weight) * auc_loss
    return total, diagnostics
