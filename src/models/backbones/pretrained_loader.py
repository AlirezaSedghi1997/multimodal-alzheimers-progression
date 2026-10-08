"""Load provenance-bound weights into 3D MRI encoders for fine-tuning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = object


def _strip_prefix(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    plen = len(prefix)
    return {k[plen:]: v for k, v in state.items() if k.startswith(prefix)}


def _looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and any(isinstance(k, str) and hasattr(v, "shape") for k, v in obj.items())


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: expected a dictionary")
    for key in (
        "encoder_state_dict",
        "model_state_dict",
        "state_dict",
        "network",
        "net",
        "model",
    ):
        nested = checkpoint.get(key)
        if _looks_like_state_dict(nested):
            return nested
    if _looks_like_state_dict(checkpoint):
        return checkpoint
    raise ValueError("Unsupported checkpoint format: no tensor state_dict found")


def _candidate_states(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    prefixes = (
        "module.encoder.",
        "model.encoder.",
        "student.encoder.",
        "encoder.",
        "module.backbone.",
        "model.backbone.",
        "backbone.",
        "module.",
        "model.",
        "net.",
    )
    candidates: list[tuple[str, dict[str, Any]]] = [("direct", state)]
    seen = {tuple(state.keys())}
    for prefix in prefixes:
        stripped = _strip_prefix(state, prefix)
        if not stripped:
            continue
        marker = tuple(stripped.keys())
        if marker in seen:
            continue
        seen.add(marker)
        candidates.append((f"strip:{prefix}", stripped))
    return candidates


def _shape_list(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_provenance_manifest(path: Path, checkpoint_sha256: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded_sha = str(payload.get("sha256", payload.get("checkpoint_sha256", ""))).lower()
    license_name = str(payload.get("license", "")).strip()
    if recorded_sha != checkpoint_sha256:
        raise RuntimeError(
            "Pretrained provenance SHA256 mismatch: "
            f"manifest={recorded_sha or 'MISSING'} actual={checkpoint_sha256}"
        )
    if not license_name:
        raise RuntimeError("Pretrained provenance manifest must declare a non-empty license")
    return {"path": str(path), "sha256": recorded_sha, "license": license_name}


def _compatible_subset(candidate: dict[str, Any], target: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    compatible: dict[str, Any] = {}
    unexpected: list[str] = []
    non_tensor: list[str] = []
    shape_mismatch: list[dict[str, Any]] = []
    for key, value in candidate.items():
        if not isinstance(key, str):
            continue
        if not hasattr(value, "shape"):
            non_tensor.append(key)
            continue
        if key not in target:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            shape_mismatch.append(
                {
                    "key": key,
                    "checkpoint_shape": _shape_list(value),
                    "target_shape": _shape_list(target[key]),
                }
            )
            continue
        compatible[key] = value
    skipped = {
        "unexpected_keys": unexpected,
        "skipped_non_tensor": non_tensor,
        "skipped_shape_mismatch": shape_mismatch,
    }
    return compatible, skipped


def load_encoder_weights(
    encoder: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    provenance_manifest_path: str | Path | None = None,
    expected_sha256: str | None = None,
    min_coverage: float = 0.0,
    require_provenance: bool = False,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required to load pretrained weights")
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    if not 0.0 <= float(min_coverage) <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")
    checkpoint_sha256 = _sha256(path)
    expected = str(expected_sha256 or "").strip().lower()
    if expected and expected != checkpoint_sha256:
        raise RuntimeError(
            f"Pretrained checkpoint SHA256 mismatch: expected={expected} actual={checkpoint_sha256}"
        )
    provenance: dict[str, Any] | None = None
    if provenance_manifest_path:
        provenance = _read_provenance_manifest(Path(provenance_manifest_path), checkpoint_sha256)
    elif require_provenance:
        raise RuntimeError(
            "Fine-tuning requires --pretrained_manifest with checkpoint SHA256 and license"
        )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = _extract_state_dict(ckpt)
    target = encoder.state_dict()

    best: tuple[str, dict[str, Any], dict[str, Any]] | None = None
    for label, candidate in _candidate_states(state):
        compatible, skipped = _compatible_subset(candidate, target)
        if best is None or len(compatible) > len(best[1]):
            best = (label, compatible, skipped)

    if best is None or not best[1]:
        attempted = [label for label, _ in _candidate_states(state)]
        sample_keys = [str(key) for key in list(state.keys())[:20]]
        raise RuntimeError(
            f"Could not load encoder weights from {path}: no compatible tensor keys found. "
            f"attempted_candidates={attempted}; checkpoint_key_sample={sample_keys}"
        )

    label, compatible, skipped = best
    coverage = len(compatible) / max(1, len(target))
    if coverage < float(min_coverage):
        raise RuntimeError(
            "Pretrained encoder coverage below policy: "
            f"loaded={len(compatible)} target={len(target)} coverage={coverage:.6f} "
            f"minimum={float(min_coverage):.6f}"
        )
    missing, unexpected_after_load = encoder.load_state_dict(compatible, strict=False)
    report = {
        "path": str(path),
        "checkpoint_sha256": checkpoint_sha256,
        "provenance": provenance,
        "provenance_status": "VERIFIED" if provenance is not None else "NOT_REQUIRED_BY_CALLER",
        "status": "LOADED_FULL" if len(compatible) == len(target) and not missing else "LOADED_PARTIAL",
        "selected_candidate": label,
        "loaded_keys": len(compatible),
        "target_keys": len(target),
        "coverage": coverage,
        "minimum_coverage": float(min_coverage),
        "missing_keys": list(missing),
        "unexpected_keys": list(skipped["unexpected_keys"]) + list(unexpected_after_load),
        "skipped_non_tensor": list(skipped["skipped_non_tensor"]),
        "skipped_shape_mismatch": list(skipped["skipped_shape_mismatch"]),
    }
    return report
