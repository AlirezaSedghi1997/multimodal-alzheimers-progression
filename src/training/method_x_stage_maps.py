"""Explicit Method X encoder stage semantics for fine-tuning.

Different backbone families expose different child names.  Fine-tuning by
``named_children()`` order alone silently changes meaning across families, so
the stage map is recorded in every optimizer manifest and used by freeze/
gradual-unfreeze code.
"""

from __future__ import annotations

from typing import Any


BACKBONE_STAGE_MAPS: dict[str, tuple[str, ...]] = {
    "resnet10_3d": ("conv1", "bn1", "act", "maxpool", "layer1", "layer2", "layer3", "layer4"),
    "resnet18_3d": ("conv1", "bn1", "act", "maxpool", "layer1", "layer2", "layer3", "layer4"),
    "resnet34_3d": ("conv1", "bn1", "act", "maxpool", "layer1", "layer2", "layer3", "layer4"),
    "resnet50_3d": ("conv1", "bn1", "act", "maxpool", "layer1", "layer2", "layer3", "layer4"),
    "densenet121_3d": ("features", "class_layers"),
    "efficientnet3d_b0": ("_conv_stem", "_bn0", "_blocks", "_conv_head", "_bn1"),
    "mednext_3d": ("features", "pool", "project"),
    "mednext_3d_s": ("features", "pool", "project"),
    "mednext_3d_b": ("features", "pool", "project"),
    "tiny": ("stem", "block1", "block2", "block3", "pool"),
    "amuse_x_hybrid3d": ("stem", "residual", "down1", "dense", "down2", "large_kernel", "down3", "attention"),
}


def ordered_encoder_stages(model: Any) -> list[tuple[str, Any]]:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return []
    available = dict(encoder.named_children())
    backbone = str(getattr(model, "backbone_name", "")).lower().strip()
    preferred = BACKBONE_STAGE_MAPS.get(backbone, ())
    ordered: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for name in preferred:
        if name in available:
            ordered.append((name, available[name]))
            seen.add(name)
    for name, child in encoder.named_children():
        if name not in seen:
            ordered.append((name, child))
    return ordered


def stage_map_manifest(model: Any) -> dict[str, Any]:
    stages = ordered_encoder_stages(model)
    return {
        "backbone": str(getattr(model, "backbone_name", "unknown")),
        "stage_names": [name for name, _ in stages],
        "declared_stage_names": list(BACKBONE_STAGE_MAPS.get(str(getattr(model, "backbone_name", "")).lower(), ())),
        "fallback_children_appended": [
            name
            for name, _ in stages
            if name not in BACKBONE_STAGE_MAPS.get(str(getattr(model, "backbone_name", "")).lower(), ())
        ],
    }
