"""Run a fail-closed 128^3 backbone smoke without starting training."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HYBRID_VARIANT_FLAGS = {
    "baseline": {"dense_skip": False, "large_kernel": False, "attention": False, "a3_alignment": False, "a4_channel_gate": False},
    "plus_dense_skip": {"dense_skip": True, "large_kernel": False, "attention": False, "a3_alignment": False, "a4_channel_gate": False},
    "plus_large_kernel": {"dense_skip": False, "large_kernel": True, "attention": False, "a3_alignment": False, "a4_channel_gate": False},
    "plus_attention": {"dense_skip": False, "large_kernel": False, "attention": True, "a3_alignment": False, "a4_channel_gate": False},
    "plus_a3": {"dense_skip": False, "large_kernel": False, "attention": False, "a3_alignment": True, "a4_channel_gate": False},
    "plus_a4": {"dense_skip": False, "large_kernel": False, "attention": False, "a3_alignment": False, "a4_channel_gate": True},
    "full_model": {"dense_skip": True, "large_kernel": True, "attention": True, "a3_alignment": True, "a4_channel_gate": True},
    "amuse_x_hybrid_lite": {"dense_skip": True, "large_kernel": True, "attention": False, "a3_alignment": True, "a4_channel_gate": True},
}
CANONICAL_TABULAR_FEATURES = (
    "age",
    "sex",
    "education_years",
    "apoe4_carrier",
    "mmse_base",
)


def inspect_hybrid_component_evidence(model: object, *, hybrid_variant: str) -> dict:
    """Inspect the constructed full classifier instead of echoing YAML claims.

    This intentionally relies on live module attributes after construction.  A
    constant dictionary would let an encoder-only or partially wired model
    claim that A3/A4 and fusion were exercised, which is exactly what the
    legacy smoke artifact did.
    """

    variant = str(hybrid_variant).strip().lower().replace("-", "_")
    expected_flags = HYBRID_VARIANT_FLAGS.get(variant)
    encoder = getattr(model, "encoder", None)
    scale_projections = getattr(encoder, "scale_projections", None)
    scale_gates = getattr(encoder, "scale_gates", None)
    scale_gate_count = None
    if scale_gates is not None and callable(getattr(scale_gates, "numel", None)):
        scale_gate_count = int(scale_gates.numel())
    observed_flags = {
        "dense_skip": bool(getattr(encoder, "use_dense", False)),
        "large_kernel": bool(getattr(encoder, "use_large_kernel", False)),
        "attention": bool(getattr(encoder, "use_attention", False)),
        "a3_alignment": bool(getattr(model, "hybrid_a3_alignment_enabled", False)),
        "a4_channel_gate": bool(getattr(model, "hybrid_a4_channel_gate_enabled", False)),
    }
    checks = {
        "encoder_is_amuse_x_hybrid3d": type(encoder).__name__ == "AMUSEXHybrid3DEncoder",
        "variant_identity_matches": str(getattr(encoder, "hybrid_variant", "")) == variant,
        "stem_present": getattr(encoder, "stem", None) is not None,
        "residual_stages_present": getattr(encoder, "residual", None) is not None,
        "multiscale_projection_count_is_four": (
            scale_projections is not None
            and callable(getattr(scale_projections, "__len__", None))
            and len(scale_projections) == 4
        ),
        "multiscale_gate_count_is_four": scale_gate_count == 4,
        "tabular_branch_present": getattr(model, "tab_encoder", None) is not None,
        "cross_attention_bottleneck_present": type(getattr(model, "fuser", None)).__name__
        == "BottleneckCrossAttention",
        "binary_head_present": getattr(model, "head", None) is not None,
        "variant_flags_match": expected_flags is not None and observed_flags == expected_flags,
    }
    return {
        "schema_version": 1,
        "evidence_mode": "LIVE_MODULE_INSPECTION",
        "hybrid_variant": variant,
        "encoder_class": type(encoder).__name__ if encoder is not None else None,
        "observed_variant_flags": observed_flags,
        "expected_variant_flags": expected_flags,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def validate_hybrid_component_evidence(evidence: object, *, expected_variant: str) -> list[str]:
    """Validate the live structural receipt without importing torch."""

    if not isinstance(evidence, dict):
        return ["COMPONENT_EVIDENCE_MISSING"]
    variant = str(expected_variant).strip().lower().replace("-", "_")
    errors: list[str] = []
    if evidence.get("schema_version") != 1:
        errors.append("COMPONENT_EVIDENCE_SCHEMA_INVALID")
    if evidence.get("evidence_mode") != "LIVE_MODULE_INSPECTION":
        errors.append("COMPONENT_EVIDENCE_MODE_INVALID")
    if evidence.get("hybrid_variant") != variant:
        errors.append("COMPONENT_EVIDENCE_VARIANT_MISMATCH")
    if evidence.get("expected_variant_flags") != HYBRID_VARIANT_FLAGS.get(variant):
        errors.append("COMPONENT_EVIDENCE_EXPECTED_FLAGS_MISMATCH")
    if evidence.get("observed_variant_flags") != HYBRID_VARIANT_FLAGS.get(variant):
        errors.append("COMPONENT_EVIDENCE_OBSERVED_FLAGS_MISMATCH")
    checks = evidence.get("checks")
    if not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()):
        errors.append("COMPONENT_EVIDENCE_CHECKS_FAILED")
    if evidence.get("all_checks_pass") is not True:
        errors.append("COMPONENT_EVIDENCE_NOT_PASS")
    return errors


def smoke(
    backbone: str,
    device_name: str,
    resolution: int,
    hard_stop_gb: float,
    *,
    hybrid_variant: str = "baseline",
    hybrid_attention_heads: int = 2,
    hybrid_attention_window_size: int = 4,
) -> dict:
    import torch

    from src.models.backbones.lightweight_3d import MethodX3DClassifier

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return {"status": "BLOCKED_CUDA_UNAVAILABLE", "backbone": backbone}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    constructor = {
        "backbone": backbone,
        "in_channels": 1,
        "latent_dim": 128,
        "tab_hidden_dim": 32,
        "tab_dim": len(CANONICAL_TABULAR_FEATURES),
        "fusion": "cross_attention",
        "dropout": 0.1,
    }
    if backbone == "amuse_x_hybrid3d":
        required_hybrid_parameters = {
            "hybrid_variant",
            "hybrid_attention_heads",
            "hybrid_attention_window_size",
        }
        available_parameters = set(inspect.signature(MethodX3DClassifier).parameters)
        missing = sorted(required_hybrid_parameters - available_parameters)
        if missing:
            raise RuntimeError(f"HYBRID_FULL_CLASSIFIER_WIRING_MISSING:{','.join(missing)}")
        constructor.update(
            {
                "hybrid_variant": hybrid_variant,
                "hybrid_attention_heads": int(hybrid_attention_heads),
                "hybrid_attention_window_size": int(hybrid_attention_window_size),
                "alignment_head": "linear",
                "channel_gate": (
                    "phenotype" if hybrid_variant in {"plus_a4", "full_model"} else "none"
                ),
            }
        )
    model = MethodX3DClassifier(**constructor).to(device).eval()
    component_evidence = (
        inspect_hybrid_component_evidence(model, hybrid_variant=hybrid_variant)
        if backbone == "amuse_x_hybrid3d"
        else None
    )
    if backbone == "amuse_x_hybrid3d":
        component_errors = validate_hybrid_component_evidence(
            component_evidence, expected_variant=hybrid_variant
        )
        if component_errors:
            raise RuntimeError("HYBRID_COMPONENT_EVIDENCE_INVALID:" + ",".join(component_errors))
    total_parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    with torch.inference_mode():
        output = model(
            torch.zeros((1, 1, resolution, resolution, resolution), device=device),
            torch.zeros((1, len(CANONICAL_TABULAR_FEATURES)), device=device),
            tab_presence=torch.ones((1, 1), device=device),
        )
    if list(output.shape) != [1]:
        raise RuntimeError(f"HYBRID_FULL_CLASSIFIER_OUTPUT_SHAPE_INVALID:{list(output.shape)}")
    output_finite = bool(torch.isfinite(output).all().item())
    if not output_finite:
        raise RuntimeError("HYBRID_FULL_CLASSIFIER_OUTPUT_NONFINITE")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated = float(torch.cuda.max_memory_allocated(device) / (1024**3))
        reserved = float(torch.cuda.max_memory_reserved(device) / (1024**3))
    else:
        allocated = reserved = None
    status = "PASS"
    if device.type == "cuda" and max(allocated or 0.0, reserved or 0.0) >= float(hard_stop_gb):
        status = "BLOCKED_VRAM_CAP"
    return {
        "status": status,
        "backbone": backbone,
        "hybrid_variant": hybrid_variant,
        "device": str(device),
        "resolution": int(resolution),
        "model_scope": "FULL_CLASSIFIER_WITH_TABULAR_FUSION",
        "canonical_tabular_features": list(CANONICAL_TABULAR_FEATURES),
        "component_evidence": component_evidence,
        "total_parameter_count": total_parameter_count,
        "output_shape": list(output.shape),
        "output_finite": output_finite,
        "max_memory_allocated_gb": allocated,
        "max_memory_reserved_gb": reserved,
        "hard_stop_vram_gb": float(hard_stop_gb),
        "fallback_backbone_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="amuse_x_hybrid3d")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--hard-stop-vram-gb", type=float, default=7.5)
    parser.add_argument(
        "--hybrid_variant",
        choices=(
            "baseline",
            "plus_dense_skip",
            "plus_large_kernel",
            "plus_attention",
            "plus_a3",
            "plus_a4",
            "full_model",
            "amuse_x_hybrid_lite",
        ),
        default="baseline",
    )
    parser.add_argument("--hybrid_attention_heads", type=int, default=2)
    parser.add_argument("--hybrid_attention_window_size", type=int, default=4)
    args = parser.parse_args()
    try:
        report = smoke(
            args.backbone,
            args.device,
            args.resolution,
            args.hard_stop_vram_gb,
            hybrid_variant=args.hybrid_variant,
            hybrid_attention_heads=args.hybrid_attention_heads,
            hybrid_attention_window_size=args.hybrid_attention_window_size,
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        report = {"status": "BLOCKED_SMOKE_FAILED", "backbone": args.backbone, "blocker": str(exc)}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
