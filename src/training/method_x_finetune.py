"""Fail-closed MX-04 fine-tuning contracts shared by the trainer and tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PRESPECIFIED_UNFREEZE_SCHEDULES = frozenset(
    {"none", "head_8_then_last_stage", "full_after_warmup"}
)

_LAST_STAGE_PREFIXES: dict[str, tuple[str, ...]] = {
    "resnet10_3d": ("layer4", "fc"),
    "resnet18_3d": ("layer4", "fc"),
    "resnet34_3d": ("layer4", "fc"),
    "densenet121_3d": (
        "features.denseblock4",
        "features.norm5",
        "class_layers",
    ),
    "mednext_3d_s": (
        "features.12",
        "features.13",
        "features.14",
        "features.15",
        "project",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _read_json_object(path: Path, *, error: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(error) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(error)
    return payload


def _resolve_under_root(value: Any, *, root: Path, error: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(error)
    candidate = Path(text)
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(error) from exc
    if not resolved.is_file():
        raise RuntimeError(error)
    return resolved


def validate_pretrained_fold_manifest(
    manifest_path: str | Path,
    *,
    folds: int,
    expected_backbone: str,
    allowed_root: str | Path,
    minimum_coverage: float = 0.95,
) -> dict[str, Any]:
    """Validate all same-fold checkpoints before any training/cache I/O."""

    root = Path(allowed_root).resolve(strict=True)
    path = _resolve_under_root(
        manifest_path,
        root=root,
        error="PRETRAINED_FOLD_MANIFEST_PATH_INVALID",
    )
    payload = _read_json_object(path, error="PRETRAINED_FOLD_MANIFEST_INVALID_JSON")
    if payload.get("schema_version") != 1:
        raise RuntimeError("PRETRAINED_FOLD_MANIFEST_SCHEMA_INVALID")
    if str(payload.get("backbone") or "") != str(expected_backbone):
        raise RuntimeError("PRETRAINED_FOLD_MANIFEST_BACKBONE_MISMATCH")
    if payload.get("outer_test_used_for_selection") is not False:
        raise RuntimeError("PRETRAINED_FOLD_MANIFEST_OUTER_POLICY_INVALID")
    source_contract_digest = str(payload.get("source_contract_digest") or "").lower()
    if not _is_sha256(source_contract_digest):
        raise RuntimeError("PRETRAINED_FOLD_MANIFEST_CONTRACT_DIGEST_INVALID")
    try:
        declared_coverage = float(payload.get("coverage_minimum"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("PRETRAINED_FOLD_MANIFEST_COVERAGE_INVALID") from exc
    if not 0.0 <= declared_coverage <= 1.0 or declared_coverage < float(minimum_coverage):
        raise RuntimeError("PRETRAINED_FOLD_MANIFEST_COVERAGE_INVALID")

    raw_entries = payload.get("folds")
    expected_keys = {str(fold) for fold in range(int(folds))}
    if not isinstance(raw_entries, Mapping) or set(map(str, raw_entries)) != expected_keys:
        raise RuntimeError("PRETRAINED_FOLD_MANIFEST_INCOMPLETE")

    entries: dict[int, dict[str, Any]] = {}
    for fold in range(int(folds)):
        raw = raw_entries.get(str(fold))
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"PRETRAINED_FOLD_MANIFEST_ENTRY_INVALID:{fold}")
        expected_sha256 = str(raw.get("sha256") or "").lower()
        if not _is_sha256(expected_sha256):
            raise RuntimeError(f"PRETRAINED_FOLD_CHECKPOINT_HASH_INVALID:{fold}")
        checkpoint = _resolve_under_root(
            raw.get("checkpoint"),
            root=root,
            error=f"PRETRAINED_FOLD_CHECKPOINT_PATH_INVALID:{fold}",
        )
        sidecar = _resolve_under_root(
            raw.get("manifest"),
            root=root,
            error=f"PRETRAINED_FOLD_SIDECAR_PATH_INVALID:{fold}",
        )
        actual_sha256 = _sha256(checkpoint)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"PRETRAINED_FOLD_CHECKPOINT_HASH_MISMATCH:{fold}")
        sidecar_payload = _read_json_object(
            sidecar,
            error=f"PRETRAINED_FOLD_SIDECAR_INVALID_JSON:{fold}",
        )
        if (
            str(sidecar_payload.get("checkpoint_sha256") or "").lower() != actual_sha256
            or str(sidecar_payload.get("sha256") or "").lower() != actual_sha256
        ):
            raise RuntimeError(f"PRETRAINED_FOLD_SIDECAR_HASH_MISMATCH:{fold}")
        if not str(sidecar_payload.get("license") or "").strip():
            raise RuntimeError(f"PRETRAINED_FOLD_SIDECAR_LICENSE_MISSING:{fold}")
        if sidecar_payload.get("source_type") != "FOLD_LOCAL_METHOD_X_BASELINE":
            raise RuntimeError(f"PRETRAINED_FOLD_SIDECAR_SOURCE_INVALID:{fold}")
        if isinstance(sidecar_payload.get("source_fold"), bool) or sidecar_payload.get("source_fold") != fold:
            raise RuntimeError(f"PRETRAINED_FOLD_SIDECAR_FOLD_MISMATCH:{fold}")
        if str(sidecar_payload.get("source_backbone") or "") != str(expected_backbone):
            raise RuntimeError(f"PRETRAINED_FOLD_SIDECAR_BACKBONE_MISMATCH:{fold}")
        if (
            sidecar_payload.get("external_download") is not False
            or sidecar_payload.get("raw_identifiers_stored") is not False
        ):
            raise RuntimeError(f"PRETRAINED_FOLD_SIDECAR_POLICY_INVALID:{fold}")
        if str(sidecar_payload.get("source_run_contract_digest") or "").lower() != source_contract_digest:
            raise RuntimeError(f"PRETRAINED_FOLD_SIDECAR_CONTRACT_MISMATCH:{fold}")
        if not _is_sha256(sidecar_payload.get("source_split_membership_digest")):
            raise RuntimeError(f"PRETRAINED_FOLD_SPLIT_DIGEST_INVALID:{fold}")
        entries[fold] = {
            "checkpoint": checkpoint,
            "sidecar": sidecar,
            "sha256": actual_sha256,
            "source_split_membership_digest": str(
                sidecar_payload["source_split_membership_digest"]
            ).lower(),
        }

    return {
        "manifest": path,
        "manifest_sha256": _sha256(path),
        "backbone": str(expected_backbone),
        "source_contract_digest": source_contract_digest,
        "minimum_coverage": declared_coverage,
        "entries": entries,
    }


def load_fold_local_encoder_weights(
    encoder: Any,
    validated_manifest: Mapping[str, Any],
    *,
    fold: int,
) -> dict[str, Any]:
    """Load only the checkpoint bound to ``fold`` and return auditable coverage."""

    from src.models.backbones.pretrained_loader import load_encoder_weights

    entries = validated_manifest.get("entries")
    if not isinstance(entries, Mapping) or fold not in entries:
        raise RuntimeError(f"PRETRAINED_FOLD_ENTRY_MISSING:{fold}")
    entry = entries[fold]
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"PRETRAINED_FOLD_ENTRY_INVALID:{fold}")
    report = load_encoder_weights(
        encoder,
        entry["checkpoint"],
        device="cpu",
        provenance_manifest_path=entry["sidecar"],
        expected_sha256=str(entry["sha256"]),
        min_coverage=float(validated_manifest.get("minimum_coverage", 0.95)),
        require_provenance=True,
    )
    return {
        **report,
        "fold": int(fold),
        "source_backbone": str(validated_manifest.get("backbone") or ""),
        "fold_manifest": str(validated_manifest.get("manifest") or ""),
        "fold_manifest_sha256": str(validated_manifest.get("manifest_sha256") or ""),
        "source_split_membership_digest": str(
            entry.get("source_split_membership_digest") or ""
        ),
        "fold_local": True,
    }


def encoder_stage_map(backbone: str, encoder: Any) -> dict[str, Any]:
    """Return and validate the explicit final-stage map for each MX-04 family."""

    normalized = str(backbone).lower().strip()
    if normalized not in _LAST_STAGE_PREFIXES:
        raise RuntimeError(f"UNFREEZE_STAGE_MAP_UNSUPPORTED:{normalized}")
    prefixes = _LAST_STAGE_PREFIXES[normalized]
    parameter_names = [name for name, _ in encoder.named_parameters()]
    if not any(
        name == prefix or name.startswith(prefix + ".")
        for name in parameter_names
        for prefix in prefixes
    ):
        raise RuntimeError(f"UNFREEZE_STAGE_MAP_DOES_NOT_MATCH_ENCODER:{normalized}")
    return {
        "backbone": normalized,
        "last_stage_prefixes": list(prefixes),
        "encoder_children": [name for name, _ in encoder.named_children()],
        "mapping_policy": "explicit_backbone_family_map_v1",
    }


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def apply_prespecified_unfreeze_schedule(
    model: Any,
    *,
    backbone: str,
    schedule: str,
    unfreeze_epoch: int,
    epoch: int,
) -> dict[str, Any]:
    """Apply an idempotent MX-04 transition, including exact-resume epochs."""

    normalized = str(schedule).strip().lower()
    if normalized not in PRESPECIFIED_UNFREEZE_SCHEDULES:
        raise RuntimeError(f"UNFREEZE_SCHEDULE_INVALID:{normalized}")
    if int(unfreeze_epoch) < 0 or int(epoch) < 0:
        raise RuntimeError("UNFREEZE_EPOCH_INVALID")
    stage_map = None
    target_prefixes: tuple[str, ...] | None = ()
    if normalized == "head_8_then_last_stage":
        stage_map = encoder_stage_map(backbone, model.encoder)
        target_prefixes = tuple(stage_map["last_stage_prefixes"])
    elif normalized == "full_after_warmup":
        target_prefixes = None

    transition_applied = normalized != "none" and int(epoch) >= int(unfreeze_epoch)
    newly_unfrozen: list[str] = []
    if transition_applied:
        for name, parameter in model.encoder.named_parameters():
            selected = target_prefixes is None or _matches_prefix(name, target_prefixes)
            if selected and not parameter.requires_grad:
                parameter.requires_grad = True
                newly_unfrozen.append(f"encoder.{name}")
    trainable = [
        f"encoder.{name}"
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad
    ]
    frozen = [
        f"encoder.{name}"
        for name, parameter in model.encoder.named_parameters()
        if not parameter.requires_grad
    ]
    return {
        "schedule": normalized,
        "configured_unfreeze_epoch": int(unfreeze_epoch),
        "observed_epoch": int(epoch),
        "transition_applied": bool(transition_applied),
        "newly_unfrozen": newly_unfrozen,
        "trainable_encoder_parameters": trainable,
        "remaining_frozen_encoder_parameters": frozen,
        "stage_map": stage_map,
    }
