from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    class _NNStub:
        Module = object

    nn = _NNStub()


AVAILABLE_ATTENTION: tuple[str, ...] = ("none", "se", "cbam")


class FeatureSEAttention(nn.Module):
    """Shape-preserving SE attention for encoder feature tensors."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for FeatureSEAttention")
        super().__init__()
        hidden = max(int(channels) // int(reduction), 1)
        self.fc = nn.Sequential(
            nn.Linear(int(channels), hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, int(channels)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.ndim == 2:
            scale = self.fc(x)
            return x * scale
        if x.ndim == 5:
            pooled = x.mean(dim=(2, 3, 4))
            scale = self.fc(pooled).reshape(x.size(0), x.size(1), 1, 1, 1)
            return x * scale
        return x


class FeatureCBAMAttention(nn.Module):
    """Shape-preserving CBAM-style attention for 3D or latent features."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for FeatureCBAMAttention")
        super().__init__()
        self.channel_attention = FeatureSEAttention(channels, reduction=reduction)
        padding = kernel_size // 2
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.channel_attention(x)
        if x.ndim != 5:
            return x
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True).values
        spatial = self.spatial_attention(torch.cat([avg_map, max_map], dim=1))
        return x * spatial


def build_attention(name: str, channels: int) -> nn.Module:
    mode = name.lower().strip()
    if mode == "none":
        return nn.Identity()
    if mode == "se":
        return FeatureSEAttention(channels)
    if mode == "cbam":
        return FeatureCBAMAttention(channels)
    raise ValueError(f"Unsupported attention mode: {name}")
