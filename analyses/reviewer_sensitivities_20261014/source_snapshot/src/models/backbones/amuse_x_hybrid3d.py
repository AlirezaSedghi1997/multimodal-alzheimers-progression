"""Compact AMUSE-X-Hybrid3D MRI encoder.

This is an explicit, PET-free project architecture.  It combines residual
updates, capped dense reuse, depthwise large-kernel blocks and an optional
low-width token attention block in one shared encoder; it is not an inference
ensemble and is not presented as a proven clinical novelty.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.backbones.deterministic_pool import (
    DeterministicAdaptiveAvgPool3d,
    deterministic_adaptive_avg_pool3d,
)


class _Residual3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class _DenseReuse3D(nn.Module):
    def __init__(self, channels: int, growth: int = 8) -> None:
        super().__init__()
        self.new_features = nn.Sequential(
            nn.Conv3d(channels, growth, 1, bias=False),
            nn.InstanceNorm3d(growth, affine=True),
            nn.GELU(),
            nn.Conv3d(growth, growth, 3, padding=1, bias=False),
            nn.InstanceNorm3d(growth, affine=True),
            nn.GELU(),
        )
        self.project = nn.Conv3d(channels + growth, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([x, self.new_features(x)], dim=1))


class _LargeKernelSE3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.pointwise = nn.Conv3d(channels, channels, 1, bias=False)
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        squeeze = max(channels // 8, 4)
        self.se = nn.Sequential(
            DeterministicAdaptiveAvgPool3d(1),
            nn.Conv3d(channels, squeeze, 1),
            nn.GELU(),
            nn.Conv3d(squeeze, channels, 1),
            nn.Sigmoid(),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pointwise(self.depthwise(x))
        y = self.norm(y) * self.se(y)
        return self.act(x + y)


class _WindowAttention3D(nn.Module):
    """Low-width local attention with explicit padding/cropping.

    Attention is applied independently inside cubic windows, so the token
    count is bounded by ``window_size**3`` rather than the whole 3-D volume.
    """

    def __init__(self, channels: int, *, heads: int = 2, window_size: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("HYBRID_ATTENTION_CHANNELS_NOT_DIVISIBLE_BY_HEADS")
        self.window_size = int(window_size)
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("HYBRID_ATTENTION_EXPECTS_5D_FEATURE_MAP")
        b, c, d, h, w = x.shape
        ws = max(1, self.window_size)
        pd = (ws - d % ws) % ws
        ph = (ws - h % ws) % ws
        pw = (ws - w % ws) % ws
        padded = F.pad(x, (0, pw, 0, ph, 0, pd))
        _, _, dp, hp, wp = padded.shape
        tokens = padded.permute(0, 2, 3, 4, 1).reshape(
            b, dp // ws, ws, hp // ws, ws, wp // ws, ws, c
        )
        tokens = tokens.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, ws**3, c)
        normalized = self.norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = (tokens + attended).reshape(b, dp // ws, hp // ws, wp // ws, ws, ws, ws, c)
        tokens = tokens.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, dp, hp, wp, c)
        return tokens.permute(0, 4, 1, 2, 3)[:, :, :d, :h, :w]


class AMUSEXHybrid3DEncoder(nn.Module):
    """Memory-capped single encoder used by the AMUSE-X proposal."""

    feature_channels = 96

    VARIANT_FLAGS = {
        "baseline": {"dense": False, "large_kernel": False, "attention": False},
        "plus_dense_skip": {"dense": True, "large_kernel": False, "attention": False},
        "plus_large_kernel": {"dense": False, "large_kernel": True, "attention": False},
        "plus_attention": {"dense": False, "large_kernel": False, "attention": True},
        "plus_a3": {"dense": False, "large_kernel": False, "attention": False},
        "plus_a4": {"dense": False, "large_kernel": False, "attention": False},
        "full_model": {"dense": True, "large_kernel": True, "attention": True},
        "amuse_x_hybrid_lite": {"dense": True, "large_kernel": True, "attention": False},
    }

    def __init__(
        self,
        in_channels: int = 1,
        out_dim: int = 128,
        *,
        use_attention: bool = False,
        hybrid_variant: str = "baseline",
        attention_heads: int = 2,
        attention_window_size: int = 4,
    ) -> None:
        super().__init__()
        normalized_variant = str(hybrid_variant).strip().lower().replace("-", "_")
        if normalized_variant not in self.VARIANT_FLAGS:
            raise ValueError(f"HYBRID_VARIANT_UNSUPPORTED:{hybrid_variant}")
        flags = dict(self.VARIANT_FLAGS[normalized_variant])
        if use_attention:
            flags["attention"] = True
        self.hybrid_variant = normalized_variant
        self.use_dense = bool(flags["dense"])
        self.use_large_kernel = bool(flags["large_kernel"])
        self.use_attention = bool(flags["attention"])
        self.attention_heads = int(attention_heads)
        self.attention_window_size = int(attention_window_size)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(16, affine=True),
            nn.GELU(),
        )
        self.residual = nn.Sequential(_Residual3D(16), _Residual3D(16))
        self.down1 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(32, affine=True),
            nn.GELU(),
        )
        self.dense = _DenseReuse3D(32, growth=8) if self.use_dense else nn.Identity()
        self.down2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(64, affine=True),
            nn.GELU(),
        )
        self.large_kernel = _LargeKernelSE3D(64) if self.use_large_kernel else nn.Identity()
        self.down3 = nn.Sequential(
            nn.Conv3d(64, self.feature_channels, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(self.feature_channels, affine=True),
            nn.GELU(),
        )
        self.attention = (
            _WindowAttention3D(
                self.feature_channels,
                heads=self.attention_heads,
                window_size=self.attention_window_size,
            )
            if self.use_attention
            else None
        )
        # Each scale is pooled to the final grid and projected to a common
        # width.  Learnable softmax gates provide a bounded multi-scale fusion
        # without concatenating large intermediate volumes.
        self.scale_projections = nn.ModuleList(
            [nn.Conv3d(channels, self.feature_channels, 1, bias=False) for channels in (16, 32, 64, self.feature_channels)]
        )
        self.scale_gates = nn.Parameter(torch.zeros(4))
        self.pool = DeterministicAdaptiveAvgPool3d(1)
        self.project = nn.Linear(self.feature_channels, out_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.residual(x)
        scale0 = x
        x = self.down1(x)
        x = self.dense(x)
        scale1 = x
        x = self.down2(x)
        x = self.large_kernel(x)
        scale2 = x
        x = self.down3(x)
        target_size = x.shape[-3:]
        scales = [scale0, scale1, scale2, x]
        projected = [
            projection(deterministic_adaptive_avg_pool3d(scale, target_size))
            for projection, scale in zip(self.scale_projections, scales)
        ]
        weights = torch.softmax(self.scale_gates, dim=0)
        x = sum(weight * value for weight, value in zip(weights, projected))
        if self.attention is not None:
            x = self.attention(x)
        return x

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.project(self.pool(features).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_features(self.forward_features(x))
