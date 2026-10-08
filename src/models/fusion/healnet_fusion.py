from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None

    class _NNStub:
        Module = object

    nn = _NNStub()


class HEALNetFusion(nn.Module):
    """HEALNet/Perceiver-style learnable bottleneck fusion for Method X.

    The module only receives Method X MRI and approved tabular latents. It is
    intentionally lightweight so it can be ablated on 16GB-class GPUs before any
    larger Phase 2 training run is attempted.
    """

    def __init__(
        self,
        mri_dim: int,
        tab_dim: int,
        hidden_dim: int = 128,
        bottleneck_dim: int = 32,
        num_heads: int = 2,
        n_latents: int = 4,
        dropout: float = 0.1,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HEALNetFusion")
        super().__init__()
        if int(n_latents) < 1:
            raise ValueError("n_latents must be >= 1")
        if int(bottleneck_dim) % int(num_heads) != 0:
            raise ValueError("bottleneck_dim must be divisible by num_heads")
        self.n_latents = int(n_latents)
        self.out_dim = int(hidden_dim)
        self.mri_proj = nn.Linear(mri_dim, bottleneck_dim)
        self.tab_proj = nn.Linear(tab_dim, bottleneck_dim)
        self.modality_embed = nn.Parameter(torch.zeros(2, bottleneck_dim))
        self.latents = nn.Parameter(torch.randn(1, self.n_latents, bottleneck_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            bottleneck_dim,
            int(num_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            bottleneck_dim,
            int(num_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(bottleneck_dim)
        self.norm_self = nn.LayerNorm(bottleneck_dim)
        self.norm_mlp = nn.LayerNorm(bottleneck_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
        )
        self.out = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, mri_feat: torch.Tensor, tab_feat: torch.Tensor, tab_presence=None) -> torch.Tensor:
        batch = mri_feat.size(0)
        mri_token = self.mri_proj(mri_feat) + self.modality_embed[0]
        tab_token = self.tab_proj(tab_feat) + self.modality_embed[1]
        key_padding_mask = None
        if tab_presence is not None:
            presence = tab_presence.reshape(-1, 1).to(dtype=tab_token.dtype, device=tab_token.device)
            tab_token = tab_token * presence
            key_padding_mask = torch.zeros(batch, 2, dtype=torch.bool, device=tab_token.device)
            key_padding_mask[:, 1] = presence.squeeze(-1) <= 0

        tokens = torch.stack([mri_token, tab_token], dim=1)
        latents = self.latents.expand(batch, -1, -1)
        attended, _ = self.cross_attn(
            latents,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = self.norm_cross(latents + self.dropout(attended))
        attended, _ = self.self_attn(latents, latents, latents, need_weights=False)
        latents = self.norm_self(latents + self.dropout(attended))
        latents = self.norm_mlp(latents + self.dropout(self.mlp(latents)))
        return self.out(latents.mean(dim=1))
