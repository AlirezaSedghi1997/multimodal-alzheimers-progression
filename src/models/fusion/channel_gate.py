from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None

    class _NNStub:
        Module = object

    nn = _NNStub()


class LogisticChannelGate(nn.Module):
    """Optional AMUSE-X Q2 channel weighting over Method X MRI latents."""

    def __init__(
        self,
        *,
        channels: int,
        mode: str = "latent_logistic",
        condition_dim: int = 5,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for LogisticChannelGate")
        super().__init__()
        self.channels = int(channels)
        self.mode = str(mode).lower().strip().replace("-", "_")
        self.condition_dim = int(condition_dim)
        if self.mode == "latent_logistic":
            in_dim = self.channels
        elif self.mode == "phenotype":
            in_dim = self.condition_dim
        else:
            raise ValueError(f"Unsupported channel gate mode: {mode}")
        self.proj = nn.Linear(in_dim, self.channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, latent, phenotype_features=None):
        if self.mode == "latent_logistic":
            context = latent
        else:
            if phenotype_features is None:
                raise ValueError("phenotype_features are required for phenotype channel gating")
            if phenotype_features.shape[-1] < self.condition_dim:
                raise ValueError(
                    f"Expected at least {self.condition_dim} phenotype features; got {phenotype_features.shape[-1]}"
                )
            context = phenotype_features[..., : self.condition_dim]
        gate = torch.sigmoid(self.proj(context))
        return latent * (2.0 * gate), gate
