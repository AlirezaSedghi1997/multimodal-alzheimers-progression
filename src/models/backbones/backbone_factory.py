from __future__ import annotations

import torch.nn as nn


def _determinize_maxpool(model: nn.Module, *, backbone: str) -> nn.Module:
    """Replace every native 3D max-pool in an official Method X encoder."""

    from src.models.backbones.deterministic_pool import replace_native_maxpool3d

    if replace_native_maxpool3d(model) < 1:
        raise RuntimeError(f"{backbone} builder found no native MaxPool3d to determinize.")
    return model


def _determinize_maxavgpool(model: nn.Module, *, backbone: str) -> nn.Module:
    """Replace native 3D max- and average-pools (MONAI DenseNet transition blocks)."""

    from src.models.backbones.deterministic_pool import (
        replace_native_avgpool3d,
        replace_native_maxpool3d,
    )

    if replace_native_maxpool3d(model) < 1:
        raise RuntimeError(f"{backbone} builder found no native MaxPool3d to determinize.")
    if replace_native_avgpool3d(model) < 1:
        raise RuntimeError(f"{backbone} builder found no native AvgPool3d to determinize.")
    return model


def _determinize_adaptivepool(model: nn.Module, *, backbone: str) -> nn.Module:
    """Replace native adaptive 3-D pools and assert the post-condition.

    ``adaptive_avg_pool3d_backward_cuda`` is refused under the strict determinism
    protocol, so a surviving adaptive pool aborts the first training backward pass
    rather than silently degrading reproducibility.  This mutates ``model`` in
    place and then re-checks the invariant, so a future walker bug that leaves a
    pool behind still fails closed at build time instead of at hour three of a run.
    The module walk cannot see functional ``F.adaptive_*_pool3d`` calls, which the
    hybrid guards with an autograd-graph probe and a source-level test.
    """

    import torch

    from src.models.backbones.deterministic_pool import replace_native_adaptive_pool3d

    replace_native_adaptive_pool3d(model)
    remaining = sorted(
        {
            type(module).__name__
            for module in model.modules()
            if isinstance(module, (torch.nn.AdaptiveAvgPool3d, torch.nn.AdaptiveMaxPool3d))
        }
    )
    if remaining:
        raise RuntimeError(f"{backbone} builder left native adaptive pools: {','.join(remaining)}")
    return model


def available_3d_backbones() -> list[str]:
    return ["tiny", "resnet10_3d", "resnet18_3d", "densenet121_3d", "efficientnet3d_b0", "resnet34_3d", "resnet50_3d", "mednext_3d_s", "amuse_x_hybrid3d"]


AVAILABLE_BACKBONES = tuple(available_3d_backbones())


def build_3d_backbone(
    name: str,
    in_channels: int = 1,
    out_dim: int = 128,
    pretrained_path: str | None = None,
    *,
    pretrained_checkpoint: str | None = None,
    hybrid_variant: str = "baseline",
    hybrid_attention_heads: int = 2,
    hybrid_attention_window_size: int = 4,
) -> nn.Module:
    """Build a PET-free 3D MRI encoder for Method X."""
    name = name.lower().strip()
    if pretrained_path and pretrained_checkpoint and pretrained_path != pretrained_checkpoint:
        raise ValueError("pretrained_path and pretrained_checkpoint disagree")
    checkpoint = pretrained_checkpoint or pretrained_path
    if checkpoint:
        from src.models.backbones.pretrained_loader import load_encoder_weights

        model = build_3d_backbone(
            name, in_channels=in_channels, out_dim=out_dim,
            hybrid_variant=hybrid_variant,
            hybrid_attention_heads=hybrid_attention_heads,
            hybrid_attention_window_size=hybrid_attention_window_size,
        )
        model._pretrained_load_report = load_encoder_weights(model, checkpoint)
        return model

    if name == "tiny":
        from src.models.backbones.lightweight_3d import Lightweight3DEncoder
        return Lightweight3DEncoder(in_channels=in_channels, base_channels=8, out_dim=out_dim)

    if name == "amuse_x_hybrid3d":
        from src.models.backbones.amuse_x_hybrid3d import AMUSEXHybrid3DEncoder
        return _determinize_adaptivepool(
            AMUSEXHybrid3DEncoder(
                in_channels=in_channels, out_dim=out_dim,
                hybrid_variant=hybrid_variant,
                attention_heads=hybrid_attention_heads,
                attention_window_size=hybrid_attention_window_size,
            ),
            backbone=name,
        )

    if name == "resnet10_3d":
        try:
            from monai.networks.nets import resnet10
        except Exception as exc:
            raise ImportError("MONAI is required for resnet10_3d. Install monai or use --backbone tiny.") from exc
        return _determinize_maxpool(
            resnet10(spatial_dims=3, n_input_channels=in_channels, num_classes=out_dim),
            backbone=name,
        )

    if name == "resnet18_3d":
        try:
            from monai.networks.nets import resnet18
        except Exception as exc:
            raise ImportError("MONAI is required for resnet18_3d. Install monai or use --backbone tiny.") from exc
        return _determinize_maxpool(
            resnet18(spatial_dims=3, n_input_channels=in_channels, num_classes=out_dim),
            backbone=name,
        )

    if name == "resnet34_3d":
        try:
            from monai.networks.nets import resnet34
        except Exception as exc:
            raise ImportError("MONAI is required for resnet34_3d. Install monai or use --backbone tiny.") from exc
        return _determinize_maxpool(
            resnet34(spatial_dims=3, n_input_channels=in_channels, num_classes=out_dim),
            backbone=name,
        )

    if name == "resnet50_3d":
        try:
            from monai.networks.nets import resnet50
        except Exception as exc:
            raise ImportError("MONAI is required for resnet50_3d. Install monai or use --backbone tiny.") from exc
        return _determinize_maxpool(
            resnet50(spatial_dims=3, n_input_channels=in_channels, num_classes=out_dim),
            backbone=name,
        )

    if name == "densenet121_3d":
        try:
            from monai.networks.nets import DenseNet121
        except Exception as exc:
            raise ImportError("MONAI is required for densenet121_3d. Install monai or use --backbone tiny.") from exc
        return _determinize_maxavgpool(
            DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=out_dim),
            backbone=name,
        )

    if name == "mednext_3d_s":
        from src.models.backbones.modern_3d import MedNeXt3DEncoder

        return MedNeXt3DEncoder(in_channels=in_channels, out_dim=out_dim, variant="s")

    if name == "efficientnet3d_b0":
        try:
            from monai.networks.nets import EfficientNetBN
        except Exception as exc:
            raise ImportError("MONAI is required for efficientnet3d_b0. Install monai or use --backbone tiny.") from exc
        return EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=in_channels, num_classes=out_dim, pretrained=False)

    raise ValueError(f"Unknown 3D backbone: {name}")
