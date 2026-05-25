from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object


class ConvFeatureFusion(nn.Module):
    """Convolutional fusion over projected MRI and tabular feature tokens."""

    def __init__(self, mri_dim: int, tab_dim: int, out_dim: int = 128) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for ConvFeatureFusion")
        super().__init__()
        self.mri_proj = nn.Linear(mri_dim, out_dim)
        self.tab_proj = nn.Linear(tab_dim, out_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=1),
        )
        self.out = nn.Sequential(nn.LayerNorm(out_dim), nn.ReLU())

    def forward(self, mri_feat, tab_feat, tab_presence=None):
        m = self.mri_proj(mri_feat)
        t = self.tab_proj(tab_feat)
        if tab_presence is not None:
            t = t * tab_presence.reshape(-1, 1)
        x = torch.stack([m, t], dim=1)
        return self.out(self.conv(x).squeeze(1))

