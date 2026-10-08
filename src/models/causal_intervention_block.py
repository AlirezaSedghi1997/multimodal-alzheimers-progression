"""AMUSE-X Pillar A4 channel-wise causal intervention block with EMA confounder bank."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None

    class _NNStub:
        Module = object

    nn = _NNStub()


class CausalInterventionBlock(nn.Module):
    """Apply channel-wise backdoor adjustment using an EMA confounder bank."""

    def __init__(
        self,
        *,
        feature_dim: int,
        confounder_dim: int,
        ema_momentum: float = 0.99,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for CausalInterventionBlock")
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.confounder_dim = int(confounder_dim)
        self.ema_momentum = float(ema_momentum)
        self.gate = nn.Linear(self.confounder_dim, self.feature_dim)
        self.register_buffer("ema_bank", torch.zeros(self.confounder_dim))

    def _update_ema(self, confounders: "torch.Tensor") -> None:
        with torch.no_grad():
            batch_mean = confounders.detach().mean(dim=0)
            self.ema_bank.mul_(self.ema_momentum).add_(batch_mean, alpha=1.0 - self.ema_momentum)

    def forward(self, features: "torch.Tensor", confounders: "torch.Tensor | None" = None) -> "torch.Tensor":
        if confounders is None:
            confounders = torch.zeros(
                features.shape[0],
                self.confounder_dim,
                dtype=features.dtype,
                device=features.device,
            )
        if confounders.ndim != 2:
            raise ValueError("confounders must have shape [batch, confounder_dim]")
        if confounders.shape[-1] != self.confounder_dim:
            raise ValueError(f"Expected {self.confounder_dim} confounders, got {confounders.shape[-1]}")
        confounders = confounders.to(device=features.device, dtype=features.dtype)
        if self.training:
            self._update_ema(confounders)
        centered = confounders - self.ema_bank.to(device=features.device, dtype=features.dtype)
        gate = torch.sigmoid(self.gate(centered))
        if features.ndim == 2:
            if features.shape[1] != self.feature_dim:
                raise ValueError(f"Expected {self.feature_dim} features, got {features.shape[1]}")
            return features * gate
        if features.ndim >= 3:
            if features.shape[1] != self.feature_dim:
                raise ValueError(f"Expected {self.feature_dim} channels, got {features.shape[1]}")
            view_shape = [features.shape[0], self.feature_dim] + [1] * (features.ndim - 2)
            return features * gate.reshape(*view_shape)
        raise ValueError("features must have shape [batch, feature_dim, ...]")


class ChannelCausalInterventionBlock(CausalInterventionBlock):
    """Backward-compatible name for the AMUSE-X A4 causal intervention block."""

    def __init__(
        self,
        *,
        channels: int,
        confound_dim: int,
        token_dim: int = 8,
        hidden_dim: int = 16,
        ema_momentum: float = 0.99,
    ) -> None:
        del token_dim, hidden_dim
        super().__init__(
            feature_dim=channels,
            confounder_dim=confound_dim,
            ema_momentum=ema_momentum,
        )

    def forward(
        self,
        features: "torch.Tensor",
        demographics: "torch.Tensor | None" = None,
        confounders: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        return super().forward(features, confounders if confounders is not None else demographics)
