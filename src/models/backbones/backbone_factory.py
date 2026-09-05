from __future__ import annotations

import torch.nn as nn


def available_3d_backbones() -> list[str]:
    return ["tiny", "resnet10_3d", "resnet18_3d", "densenet121_3d", "efficientnet3d_b0", "resnet34_3d", "resnet50_3d", "mednext_3d_s", "amuse_x_hybrid3d"]


AVAILABLE_BACKBONES = tuple(available_3d_backbones())


def build_3d_backbone(name: str, in_channels: int = 1, out_dim: int = 128, pretrained_path: str | None = None) -> nn.Module:
    """Build a PET-free 3D MRI encoder for Method X."""
    name = name.lower().strip()

    if name == "tiny":
        from src.models.backbones.lightweight_3d import Lightweight3DEncoder
        return Lightweight3DEncoder(in_channels=in_channels, base_channels=8, out_dim=out_dim)

    if name == "amuse_x_hybrid3d":
        from src.models.backbones.amuse_x_hybrid3d import AMUSEXHybrid3DEncoder
        return AMUSEXHybrid3DEncoder(in_channels=in_channels, out_dim=out_dim)

    if name == "resnet10_3d":
        try:
            from monai.networks.nets import resnet10
        except Exception as exc:
            raise ImportError("MONAI is required for resnet10_3d. Install monai or use --backbone tiny.") from exc
        return resnet10(spatial_dims=3, n_input_channels=in_channels, num_classes=out_dim)

    if name == "resnet18_3d":
        try:
            from monai.networks.nets import resnet18
        except Exception as exc:
            raise ImportError("MONAI is required for resnet18_3d. Install monai or use --backbone tiny.") from exc
        return resnet18(spatial_dims=3, n_input_channels=in_channels, num_classes=out_dim)

    if name == "resnet34_3d":
        try:
            from monai.networks.nets import resnet34
        except Exception as exc:
            raise ImportError("MONAI is required for resnet34_3d. Install monai or use --backbone tiny.") from exc
        return resnet34(spatial_dims=3, n_input_channels=in_channels, num_classes=out_dim)

    if name == "resnet50_3d":
        try:
            from monai.networks.nets import resnet50
        except Exception as exc:
            raise ImportError("MONAI is required for resnet50_3d. Install monai or use --backbone tiny.") from exc
        return resnet50(spatial_dims=3, n_input_channels=in_channels, num_classes=out_dim)

    if name == "densenet121_3d":
        try:
            from monai.networks.nets import DenseNet121
        except Exception as exc:
            raise ImportError("MONAI is required for densenet121_3d. Install monai or use --backbone tiny.") from exc
        return DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=out_dim)

    if name == "efficientnet3d_b0":
        try:
            from monai.networks.nets import EfficientNetBN
        except Exception as exc:
            raise ImportError("MONAI is required for efficientnet3d_b0. Install monai or use --backbone tiny.") from exc
        return EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=in_channels, num_classes=out_dim, pretrained=False)

    raise ValueError(f"Unknown 3D backbone: {name}")
