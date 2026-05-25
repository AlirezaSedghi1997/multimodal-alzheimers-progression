from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def latent_alignment_loss(mri_latent, tab_latent, presence=None):
    if torch is None:
        raise RuntimeError("PyTorch is required for latent_alignment_loss")
    diff = (mri_latent - tab_latent) ** 2
    if presence is not None:
        diff = diff * presence.reshape(-1, 1)
        denom = presence.sum().clamp_min(1.0)
        return diff.sum() / denom
    return diff.mean()

