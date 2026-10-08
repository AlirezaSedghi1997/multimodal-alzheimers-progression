from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None

    class _NNStub:
        Module = object

    nn = _NNStub()


class ConvNeXt3DBlock(nn.Module):
    """Small 3D ConvNeXt-style block used for the local MedNeXt classifier adapter."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for ConvNeXt3DBlock")
        super().__init__()
        hidden = int(channels * expansion)
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=7, padding=3, groups=channels),
            nn.GroupNorm(1, channels),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class MedNeXt3DEncoder(nn.Module):
    """PET-free MedNeXt-style 3D ConvNeXt encoder for Method X.

    This is a compact classifier adapter inspired by MedNeXt/ConvNeXt design
    principles. It avoids external nnU-Net dependencies and remains fully
    opt-in through the backbone registry.
    """

    VARIANTS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
        "s": ((16, 32, 64, 128), (1, 1, 1, 1)),
        "b": ((24, 48, 96, 192), (1, 2, 2, 1)),
    }

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_dim: int = 128,
        variant: str = "s",
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for MedNeXt3DEncoder")
        super().__init__()
        variant = variant.lower().strip()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported MedNeXt3D variant: {variant}")
        channels, depths = self.VARIANTS[variant]
        stages: list[nn.Module] = []
        current = int(in_channels)
        for stage_idx, (width, depth) in enumerate(zip(channels, depths)):
            stride = 1 if stage_idx == 0 else 2
            stages.append(nn.Conv3d(current, width, kernel_size=3, stride=stride, padding=1))
            stages.append(nn.GroupNorm(1, width))
            stages.append(nn.GELU())
            for _ in range(depth):
                stages.append(ConvNeXt3DBlock(width))
            current = width
        self.features = nn.Sequential(*stages)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.project = nn.Linear(current, out_dim)
        self.out_dim = int(out_dim)
        self.variant = variant
        self.feature_channels = current

    def forward_features(self, image):
        return self.features(image)

    def forward_from_features(self, features):
        return self.project(self.pool(features))

    def forward(self, image):
        return self.forward_from_features(self.forward_features(image))


class SwinUNETR3DEncoder(nn.Module):
    """MONAI SwinUNETR classifier adapter with global pooling.

    The MONAI dependency is imported lazily so metadata and default Method X
    paths remain usable in lightweight environments. Selecting this backbone is
    an explicit Phase 2 architecture experiment.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_dim: int = 128,
        img_size: tuple[int, int, int] = (128, 128, 128),
        feature_size: int = 24,
        use_checkpoint: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for SwinUNETR3DEncoder")
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR
        except Exception as exc:
            raise ImportError("MONAI is required for swin_unetr_3d. Install monai or choose another backbone.") from exc

        kwargs = {
            "in_channels": int(in_channels),
            "out_channels": int(out_dim),
            "feature_size": int(feature_size),
            "use_checkpoint": bool(use_checkpoint),
            "spatial_dims": 3,
        }
        try:
            self.net = SwinUNETR(img_size=img_size, **kwargs)
        except TypeError:
            self.net = SwinUNETR(**kwargs)
        self.out_dim = int(out_dim)
        self.feature_channels = int(out_dim)
        self.img_size = tuple(int(v) for v in img_size)

    def forward(self, image):
        out = self.net(image)
        if out.ndim > 2:
            out = out.mean(dim=tuple(range(2, out.ndim)))
        return out
