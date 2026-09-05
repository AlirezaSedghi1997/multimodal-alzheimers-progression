from __future__ import annotations

FUSION_REGISTRY: dict[str, dict[str, object]] = {
    "concat": {
        "implemented": True,
        "contract": "MRI latent + tabular latent -> binary logits",
        "method_x_pet_free": True,
    },
    "cross_attention": {
        "implemented": True,
        "contract": "MRI latent + tabular latent -> binary logits",
        "method_x_pet_free": True,
    },
    "moe": {
        "implemented": True,
        "contract": "MRI latent + tabular latent -> binary logits",
        "method_x_pet_free": True,
    },
    "amplitude_phase": {
        "implemented": True,
        "contract": "MRI latent + tabular latent -> binary logits",
        "method_x_pet_free": True,
        "stage": "Q1_resolved_optional_mode",
    },
    "healnet": {
        "implemented": True,
        "contract": "MRI latent + tabular latent -> learnable bottleneck latents -> binary logits",
        "method_x_pet_free": True,
        "stage": "P2_1_optional_healnet_perceiver_bottleneck",
        "missing_modality": "tab_presence masks the tabular token while preserving MRI-only inference.",
    },
}


def available_fusion_modes() -> tuple[str, ...]:
    return tuple(FUSION_REGISTRY)


def validate_fusion_mode(name: str) -> str:
    mode = name.lower().strip().replace("-", "_")
    if mode not in FUSION_REGISTRY:
        raise ValueError(f"Unsupported fusion mode: {name}")
    return mode
