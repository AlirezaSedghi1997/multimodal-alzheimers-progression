from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object


class BottleneckCrossAttention(nn.Module):
    """Lightweight MRI-tabular cross attention for Method X fusion."""

    def __init__(
        self,
        mri_dim: int,
        tab_dim: int,
        out_dim: int | None = None,
        heads: int | None = None,
        hidden_dim: int | None = None,
        bottleneck_dim: int = 32,
        num_heads: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for BottleneckCrossAttention")
        super().__init__()
        hidden = int(hidden_dim if hidden_dim is not None else (out_dim if out_dim is not None else 128))
        attn_heads = int(num_heads if num_heads is not None else (heads if heads is not None else 2))
        if bottleneck_dim % attn_heads != 0:
            raise ValueError("bottleneck_dim must be divisible by the number of attention heads")
        self.mri_proj = nn.Linear(mri_dim, bottleneck_dim)
        self.tab_proj = nn.Linear(tab_dim, bottleneck_dim)
        self.tab_queries_mri = nn.MultiheadAttention(
            bottleneck_dim,
            attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mri_queries_tab = nn.MultiheadAttention(
            bottleneck_dim,
            attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_mri = nn.LayerNorm(bottleneck_dim)
        self.norm_tab = nn.LayerNorm(bottleneck_dim)
        self.out = nn.Sequential(
            nn.Linear(2 * bottleneck_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.out_dim = hidden

    def forward(self, mri_feat, tab_feat, tab_presence=None):
        m = self.mri_proj(mri_feat).unsqueeze(1)
        t = self.tab_proj(tab_feat).unsqueeze(1)
        if tab_presence is not None:
            t = t * tab_presence.reshape(-1, 1, 1)
        tab_attends_mri, _ = self.tab_queries_mri(t, m, m, need_weights=False)
        mri_attends_tab, _ = self.mri_queries_tab(m, t, t, need_weights=False)
        mri_out = self.norm_mri(m + mri_attends_tab).squeeze(1)
        tab_out = self.norm_tab(t + tab_attends_mri).squeeze(1)
        return self.out(torch.cat([mri_out, tab_out], dim=-1))
