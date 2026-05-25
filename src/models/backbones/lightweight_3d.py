from __future__ import annotations

import torch
import torch.nn as nn

from src.models.fusion.bottleneck_cross_attention import BottleneckCrossAttention


class Lightweight3DEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 8, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected MRI tensor shaped (B, C, D, H, W)")
        return self.net(x)


class MethodX3DClassifier(nn.Module):
    def __init__(
        self,
        tab_dim: int = 5,
        base_channels: int = 8,
        latent_dim: int = 64,
        tab_hidden_dim: int = 32,
        fusion: str = "cross_attention",
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.encoder = Lightweight3DEncoder(base_channels=base_channels, out_dim=latent_dim)
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, tab_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden_dim, tab_hidden_dim),
            nn.GELU(),
        )
        self.alignment_projection = nn.Identity() if tab_hidden_dim == latent_dim else nn.Linear(tab_hidden_dim, latent_dim)
        self.fusion = fusion
        if fusion == "cross_attention":
            self.fuser = BottleneckCrossAttention(
                mri_dim=latent_dim,
                tab_dim=tab_hidden_dim,
                hidden_dim=latent_dim,
                bottleneck_dim=32,
                num_heads=4,
                dropout=dropout,
            )
            head_in = self.fuser.out_dim
        elif fusion == "concat":
            self.fuser = None
            head_in = latent_dim + tab_hidden_dim
        else:
            raise ValueError(f"Unknown fusion mode: {fusion}")
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, 1))

    def encode_latents(self, image: torch.Tensor, tabular: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mri_latent = self.encoder(image)
        tab_latent = self.tab_encoder(tabular)
        return mri_latent, tab_latent

    def forward_with_latents(self, image: torch.Tensor, tabular: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mri_latent, tab_latent = self.encode_latents(image, tabular)
        if self.fuser is None:
            fused = torch.cat([mri_latent, tab_latent], dim=-1)
        else:
            fused = self.fuser(mri_latent, tab_latent)
        logits = self.head(fused).squeeze(-1)
        return logits, mri_latent, self.alignment_projection(tab_latent)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_with_latents(image, tabular)
        return logits
