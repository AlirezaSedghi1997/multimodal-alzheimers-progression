"""Fail-closed contracts for the raw-T1 BrainIAC preprocessing route.

The primary BrainIAC route is intentionally separate from Method X's locked
128-cubed SPM cache.  It starts from raw T1, reads no outcome label, records
only pseudonymous sample hashes, and binds every retained 96-cubed image to its
rigid transform and step-level evidence.  The locally available SPM avg152
template is protocol-compatible but is *not* claimed to be the undistributed
template used during BrainIAC pretraining.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
OUTPUT_SHAPE = (96, 96, 96)
ROUTE_ROLE = "PROTOCOL_COMPATIBLE_INTERNAL_POSTHOC_PRIMARY"
UPSTREAM_TEMPLATE_IDENTITY = "UNKNOWN_NOT_DISTRIBUTED"

_FORBIDDEN_RECEIPT_FIELDS = frozenset(
    {
        "raw_path",
        "raw_source_path",
        "source_path",
        "patient_id",
        "subject_id",
        "bid",
        "ab_label",
        "label",
        "pet",
        "suvr",
        "centiloid",
    }
)


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_reparse_between(path: Path, root: Path) -> bool:
    current = path
    while True:
        try:
            stat_result = os.lstat(current)
        except OSError:
            return True
        attributes = int(getattr(stat_result, "st_file_attributes", 0))
        if current.is_symlink() or attributes & 0x400:
            return True
        if current == root:
            return False
        if current.parent == current:
            return True
        current = current.parent


def _resolve_artifact(root: Path, relative_value: object) -> Path:
    relative = Path(str(relative_value or ""))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("BRAINIAC_PREPROCESS_ARTIFACT_PATH_INVALID")
    try:
        resolved_root = root.resolve(strict=True)
        candidate = (resolved_root / relative).resolve(strict=True)
        candidate.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError("BRAINIAC_PREPROCESS_ARTIFACT_PATH_INVALID") from exc
    if not candidate.is_file() or _has_reparse_between(candidate, resolved_root):
        raise RuntimeError("BRAINIAC_PREPROCESS_ARTIFACT_UNSAFE")
    return candidate


def normalize_nonzero_channelwise(volume: np.ndarray) -> np.ndarray:
    """Match BrainIAC's nonzero, channel-wise validation normalization."""

    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1:
        raise RuntimeError("BRAINIAC_OFFICIAL_TRANSFORM_CHANNEL_SHAPE_INVALID")
    if not np.isfinite(array).all():
        raise RuntimeError("BRAINIAC_OFFICIAL_TRANSFORM_NONFINITE_INPUT")
    output = np.zeros_like(array, dtype=np.float32)
    for channel_index in range(array.shape[0]):
        channel = array[channel_index]
        mask = channel != 0
        if not bool(mask.any()):
            raise RuntimeError("BRAINIAC_OFFICIAL_TRANSFORM_EMPTY_FOREGROUND")
        foreground = channel[mask].astype(np.float64, copy=False)
        mean = float(foreground.mean())
        std = float(foreground.std())
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-8:
            raise RuntimeError("BRAINIAC_OFFICIAL_TRANSFORM_FOREGROUND_DEGENERATE")
        output[channel_index, mask] = ((foreground - mean) / std).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("BRAINIAC_OFFICIAL_TRANSFORM_NONFINITE_OUTPUT")
    return output


def validate_completed_item(
    receipt: Mapping[str, Any], *, artifact_root: Path
) -> dict[str, Any]:
    """Validate one completed item against immutable bytes on disk."""

    record = dict(receipt)
    forbidden = sorted(_FORBIDDEN_RECEIPT_FIELDS.intersection(record))
    if forbidden:
        raise RuntimeError(
            "BRAINIAC_PREPROCESS_PRIVACY_FIELD_FORBIDDEN:" + ",".join(forbidden)
        )
    body = {key: value for key, value in record.items() if key != "integrity_sha256"}
    if record.get("integrity_sha256") != _canonical_sha(body):
        raise RuntimeError("BRAINIAC_PREPROCESS_RECEIPT_INTEGRITY_INVALID")
    if (
        record.get("status") != "COMPLETED"
        or record.get("source_role") != "raw_t1"
        or record.get("label_read") is not False
        or record.get("direct_128_to_96_used") is not False
        or tuple(record.get("output_shape", ())) != OUTPUT_SHAPE
        or record.get("finite") is not True
    ):
        raise RuntimeError("BRAINIAC_PREPROCESS_RECEIPT_CONTRACT_INVALID")
    sample_hash = str(record.get("sample_hash") or "")
    if not sample_hash.startswith("s_"):
        raise RuntimeError("BRAINIAC_PREPROCESS_SAMPLE_HASH_INVALID")
    required_hash_fields = (
        "output_sha256",
        "output_affine_sha256",
        "source_affine_sha256",
        "n4_artifact_sha256",
        "registered_artifact_sha256",
        "brain_mask_sha256",
        "rigid_transform_sha256",
        "template_sha256",
    )
    if any(not _is_sha256(record.get(field)) for field in required_hash_fields):
        raise RuntimeError("BRAINIAC_PREPROCESS_RECEIPT_HASH_MISSING")
    output = _resolve_artifact(artifact_root, record.get("output_path"))
    transform = _resolve_artifact(artifact_root, record.get("rigid_transform_path"))
    if output.stat().st_size != int(record.get("output_bytes", -1)):
        raise RuntimeError("BRAINIAC_PREPROCESS_OUTPUT_SIZE_MISMATCH")
    if sha256_file(output) != record["output_sha256"]:
        raise RuntimeError("BRAINIAC_PREPROCESS_OUTPUT_HASH_MISMATCH")
    if transform.stat().st_size != int(record.get("rigid_transform_bytes", -1)):
        raise RuntimeError("BRAINIAC_PREPROCESS_TRANSFORM_SIZE_MISMATCH")
    if sha256_file(transform) != record["rigid_transform_sha256"]:
        raise RuntimeError("BRAINIAC_PREPROCESS_TRANSFORM_HASH_MISMATCH")
    foreground_fraction = float(record.get("foreground_fraction", float("nan")))
    foreground_mean = float(record.get("foreground_mean", float("nan")))
    foreground_std = float(record.get("foreground_std", float("nan")))
    if (
        not all(np.isfinite([foreground_fraction, foreground_mean, foreground_std]))
        or not 0.01 <= foreground_fraction <= 0.95
        or abs(foreground_mean) > 1e-3
        or not 0.95 <= foreground_std <= 1.05
    ):
        raise RuntimeError("BRAINIAC_PREPROCESS_NUMERIC_QC_INVALID")
    return {
        "status": "PASS",
        "sample_hash": sample_hash,
        "output_sha256": record["output_sha256"],
        "rigid_transform_sha256": record["rigid_transform_sha256"],
    }


def build_aggregate_manifest(
    *,
    receipts: Sequence[Mapping[str, Any]],
    expected_sample_hashes: Sequence[str],
    artifact_root: Path,
    template_sha256: str,
    runtime_lock_sha256: str,
    strict_receipt_sha256: str,
    forward_receipt_sha256: str,
) -> dict[str, Any]:
    """Create a completed primary-route manifest from exact item membership."""

    binding_hashes = (
        template_sha256,
        runtime_lock_sha256,
        strict_receipt_sha256,
        forward_receipt_sha256,
    )
    if any(not _is_sha256(value) for value in binding_hashes):
        raise RuntimeError("BRAINIAC_PREPROCESS_AGGREGATE_BINDING_INVALID")
    expected = [str(value) for value in expected_sample_hashes]
    if not expected or len(expected) != len(set(expected)):
        raise RuntimeError("BRAINIAC_PREPROCESS_EXPECTED_MEMBERSHIP_INVALID")
    records = [dict(receipt) for receipt in receipts]
    observed = [str(record.get("sample_hash") or "") for record in records]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise RuntimeError("BRAINIAC_PREPROCESS_MEMBERSHIP_MISMATCH")
    for record in records:
        validate_completed_item(record, artifact_root=artifact_root)
        if record.get("template_sha256") != template_sha256:
            raise RuntimeError("BRAINIAC_PREPROCESS_TEMPLATE_BINDING_MISMATCH")
    records.sort(key=lambda record: str(record["sample_hash"]))

    def digest_list(field: str) -> str:
        return _canonical_sha([str(record[field]) for record in records])

    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "route_role": ROUTE_ROLE,
        "source_role": "raw_t1",
        "label_read": False,
        "direct_128_to_96_used": False,
        "real_only": True,
        "rows": len(records),
        "output_shape": list(OUTPUT_SHAPE),
        "affine_aware": True,
        "membership_sha256": _canonical_sha(sorted(expected)),
        "item_receipts_sha256": _canonical_sha(
            [str(record["integrity_sha256"]) for record in records]
        ),
        "source_affine_sha256": digest_list("source_affine_sha256"),
        "n4_bias_correction": {
            "status": "PASS",
            "artifact_sha256": digest_list("n4_artifact_sha256"),
        },
        "rigid_registration": {
            "status": "PASS",
            "transform_sha256": digest_list("rigid_transform_sha256"),
            "template_sha256": template_sha256,
            "mode": "Euler3D_MattesMI_fixed_seed",
        },
        "brain_extraction": {
            "status": "PASS",
            "mask_sha256": digest_list("brain_mask_sha256"),
            "implementation": "PINNED_BRAINIAC_VENDOR_HD_BET_FAST_FOLD0",
        },
        "official_96_transform": {
            "status": "PASS",
            "output_shape": list(OUTPUT_SHAPE),
            "output_affine_sha256": digest_list("output_affine_sha256"),
            "transform": "Resized_trilinear_then_NormalizeIntensity_nonzero_channelwise",
        },
        "runtime_lock_sha256": runtime_lock_sha256,
        "strict_load_receipt_sha256": strict_receipt_sha256,
        "forward_receipt_sha256": forward_receipt_sha256,
        "template_sha256": template_sha256,
        "upstream_template_identity": UPSTREAM_TEMPLATE_IDENTITY,
        "canonical_reproduction_claim_allowed": False,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        "external_validation_status": "OPTIONAL_NOT_AVAILABLE_NOT_EXECUTION_BLOCKER",
    }
    body["manifest_sha256"] = _canonical_sha(body)
    return body


__all__ = [
    "OUTPUT_SHAPE",
    "ROUTE_ROLE",
    "UPSTREAM_TEMPLATE_IDENTITY",
    "build_aggregate_manifest",
    "normalize_nonzero_channelwise",
    "sha256_file",
    "validate_completed_item",
]
