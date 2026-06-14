from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = object
    F = None


class AmplitudePhaseFusion(nn.Module):
    """Multi-scale amplitude+phase fusion for MRI-tabular features.

    Inspired by MSFL (arXiv:2603.24603) which fuses amplitude and phase
    of fMRI signals for brain disorder detection. Here we project MRI
    and tabular features to complex-valued space and fuse via
    amplitude modulation and phase interaction.
    """

    def __init__(
        self,
        mri_dim: int,
        tab_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for AmplitudePhaseFusion")
        super().__init__()
        self.mri_proj = nn.Linear(mri_dim, hidden_dim)
        self.tab_proj = nn.Linear(tab_dim, hidden_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_dim = hidden_dim

    def forward(self, mri_feat: torch.Tensor, tab_feat: torch.Tensor, tab_presence=None) -> torch.Tensor:
        mri_real = self.mri_proj(mri_feat)
        tab_real = self.tab_proj(tab_feat)
        if tab_presence is not None:
            tab_real = tab_real * tab_presence.reshape(-1, 1)
        d = mri_real.size(-1) // 2
        mri_a, mri_p = mri_real[:, :d], mri_real[:, d:]
        tab_a, tab_p = tab_real[:, :d], tab_real[:, d:]
        mri_amp = torch.sigmoid(mri_a)
        mri_phase = torch.tanh(mri_p)
        tab_amp = torch.sigmoid(tab_a)
        tab_phase = torch.tanh(tab_p)
        fused_amp = mri_amp * tab_phase + tab_amp * mri_phase
        fused_phase = mri_amp * tab_amp - mri_phase * tab_phase
        fused = torch.cat([fused_amp, fused_phase], dim=-1)
        return self.fusion_proj(fused)
