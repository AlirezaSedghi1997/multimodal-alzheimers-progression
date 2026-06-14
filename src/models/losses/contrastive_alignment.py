from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


class ContrastiveAlignmentLoss(torch.nn.Module):
    """InfoNCE-style contrastive loss for MRI-tabular alignment.

    Encourages MRI and tabular representations of the same patient
    to be close in normalized embedding space while pushing apart
    representations from different patients.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for ContrastiveAlignmentLoss")
        super().__init__()
        self.temperature = temperature

    def forward(self, mri_features: torch.Tensor, tab_features: torch.Tensor) -> torch.Tensor:
        """Compute symmetric contrastive alignment loss.

        Args:
            mri_features: MRI latent representations (batch_size, dim)
            tab_features: Tabular latent representations (batch_size, dim)

        Returns:
            Scalar loss tensor.
        """
        mri_norm = F.normalize(mri_features, dim=-1)
        tab_norm = F.normalize(tab_features, dim=-1)
        logits = torch.mm(mri_norm, tab_norm.t()) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)
        loss_m2t = F.cross_entropy(logits, labels)
        loss_t2m = F.cross_entropy(logits.t(), labels)
        return (loss_m2t + loss_t2m) / 2.0
