"""Schema, privacy boundary and validators for BrainIAC preprocessing v2.

Receipts are the only scientific authority of the schema-v2 raw-T1 pipeline.
Every artifact is byte-bound, fail-closed and free of identifiers: no label,
PET, raw path or raw identifier may appear anywhere, recursively.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.research.brainiac_readiness import _beneath, _is_reparse


RECEIPT_SCHEMA_VERSION = 2
AGGREGATE_SCHEMA_VERSION = 2
RECEIPT_ARTIFACT_TYPE = "BRAINIAC_RAW_T1_PREPROCESSING_RECEIPT"
AGGREGATE_ARTIFACT_TYPE = "BRAINIAC_RAW_T1_PREPROCESSING_AGGREGATE_V2"
THRESHOLD_MANIFEST_ARTIFACT_TYPE = "BRAINIAC_PREPROCESSING_QC_THRESHOLD_MANIFEST_V2"

RECEIPT_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "artifact",
    "sample_hash",
    "generation",
    "status",
    "source_sha256",
    "source_size",
    "output_relative_path",
    "output_sha256",
    "output_size",
    "transform_relative_path",
    "transform_sha256",
    "transform_size",
    "policy_sha256",
    "config_sha256",
    "pipeline_contract_sha256",
    "dependency_closure_sha256",
    "membership_sha256",
    "cache_sha256",
    "catalog_sha256",
    "runtime_lock_sha256",
    "vendor_commit",
    "vendor_tree_sha256",
    "hd_bet_weight_sha256",
    "template_sha256",
    "template_lock_sha256",
    "official_transform_source_sha256",
    "affine_before",
    "affine_after",
    "spacing_before",
    "spacing_after",
    "origin_before",
    "origin_after",
    "direction_before",
    "direction_after",
    "com_coordinate_frame",
    "rigid_determinant",
    "orthogonality_error",
    "rotation_deg",
    "translation_norm_mm",
    "foreground_fraction",
    "finite",
    "normalized_nonzero_mean",
    "normalized_nonzero_std",
    "center_of_mass_before",
    "center_of_mass_after",
    "registration_similarity_before",
    "registration_similarity_after",
    "qc_threshold_manifest_sha256",
    "labels_read",
    "pet_read",
    "raw_identifier_persisted",
    "raw_path_persisted",
    "direct_128_to_96_used",
)

AUDIT_FALSE_FIELDS: tuple[str, ...] = (
    "labels_read",
    "pet_read",
    "raw_identifier_persisted",
    "raw_path_persisted",
    "direct_128_to_96_used",
)

# Exact-match denylist: carrying any of these data keys anywhere in an
# artifact is a privacy violation.  The audit boolean flags above are the
# only permitted label/PET/raw mentions and must be exactly False.
FORBIDDEN_DATA_KEYS: frozenset[str] = frozenset(
    {
        "label",
        "labels",
        "ab_label",
        "pet",
        "pet_input",
        "pet_path",
        "raw_path",
        "raw_identifier",
        "subject_id",
        "patient_id",
        "subject",
        "subjects",
        "subject_rows",
        "age",
        "sex",
        "education_years",
        "apoe4_carrier",
        "mmse_base",
        "tabular",
        "predictors",
        "cohort",
        "site",
        "scanner",
        "manufacturer",
        "field_strength",
    }
)

FIXED_QC_LIMITS: Mapping[str, tuple[float, float]] = {
    "rigid_determinant": (0.999, 1.001),
    "foreground_fraction": (0.05, 0.50),
    "normalized_nonzero_std": (0.95, 1.05),
}
FIXED_QC_MAXIMA: Mapping[str, float] = {
    "orthogonality_error": 1e-3,
    "rotation_deg": 45.0,
    "translation_norm_mm": 100.0,
    "normalized_nonzero_mean": 1e-3,
}

_HEX = frozenset("0123456789abcdef")


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    # Project integrity convention: compact canonical JSON without newline
    # (matches brainiac_readiness._integrity_matches_compact and the sealed
    # runtime/strict/forward receipts this module must revalidate).
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hex64(value: Any) -> bool:
    text = str(value or "").casefold()
    return len(text) == 64 and all(char in _HEX for char in text)


def assert_privacy(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in FORBIDDEN_DATA_KEYS:
                raise RuntimeError(
                    f"BRAINIAC_PREPROCESSING_PRIVACY_FIELD_FORBIDDEN:{key}"
                )
            assert_privacy(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            assert_privacy(nested)


def seal_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the static contract and attach the integrity digest."""

    missing = [field for field in RECEIPT_REQUIRED_FIELDS if field not in body]
    if missing:
        raise RuntimeError(f"BRAINIAC_PREPROCESSING_RECEIPT_FIELD_MISSING:{missing[0]}")
    if body["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise RuntimeError("BRAINIAC_PREPROCESSING_SCHEMA_V1_FORBIDDEN_IN_V2")
    if body["artifact"] != RECEIPT_ARTIFACT_TYPE:
        raise RuntimeError("BRAINIAC_PREPROCESSING_RECEIPT_ARTIFACT_INVALID")
    if body["status"] not in ("COMPLETED", "QUARANTINED"):
        raise RuntimeError("BRAINIAC_PREPROCESSING_RECEIPT_STATUS_INVALID")
    for field in AUDIT_FALSE_FIELDS:
        if body[field] is not False:
            raise RuntimeError(f"BRAINIAC_PREPROCESSING_RECEIPT_AUDIT_FLAG_INVALID:{field}")
    for field in (
        "source_sha256",
        "output_sha256",
        "transform_sha256",
        "policy_sha256",
        "config_sha256",
        "pipeline_contract_sha256",
        "dependency_closure_sha256",
        "membership_sha256",
        "cache_sha256",
        "catalog_sha256",
        "runtime_lock_sha256",
        "hd_bet_weight_sha256",
        "template_sha256",
        "template_lock_sha256",
        "official_transform_source_sha256",
        "qc_threshold_manifest_sha256",
    ):
        if not _is_hex64(body[field]):
            raise RuntimeError(f"BRAINIAC_PREPROCESSING_RECEIPT_SHA_INVALID:{field}")
    if int(body["generation"]) < 1:
        raise RuntimeError("BRAINIAC_PREPROCESSING_RECEIPT_GENERATION_INVALID")
    payload = {key: value for key, value in body.items() if key != "integrity_sha256"}
    assert_privacy(payload)
    _check_qc(payload)
    sealed = dict(payload)
    sealed["integrity_sha256"] = canonical_sha(payload)
    return sealed


def _check_qc(receipt: Mapping[str, Any]) -> None:
    import numpy as np

    if receipt["finite"] is not True:
        raise RuntimeError("BRAINIAC_PREPROCESSING_QC_NONFINITE")
    if receipt.get("com_coordinate_frame") != "TEMPLATE_PHYSICAL_MM_RELATIVE":
        raise RuntimeError("BRAINIAC_PREPROCESSING_COM_FRAME_INVALID")
    for role in ("before", "after"):
        try:
            affine = np.asarray(receipt[f"affine_{role}"], dtype=np.float64).reshape(4, 4)
            spacing = np.asarray(receipt[f"spacing_{role}"], dtype=np.float64).reshape(3)
            origin = np.asarray(receipt[f"origin_{role}"], dtype=np.float64).reshape(3)
            direction = np.asarray(receipt[f"direction_{role}"], dtype=np.float64).reshape(3, 3)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("BRAINIAC_PREPROCESSING_GEOMETRY_SHAPE_INVALID") from exc
        if (
            not np.isfinite(affine).all()
            or not np.isfinite(spacing).all()
            or not np.isfinite(origin).all()
            or not np.isfinite(direction).all()
            or np.any(spacing <= 0)
            or abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12
        ):
            raise RuntimeError("BRAINIAC_PREPROCESSING_GEOMETRY_INVALID")
    for field in ("center_of_mass_before", "center_of_mass_after"):
        try:
            values = np.asarray(receipt[field], dtype=np.float64).reshape(3)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("BRAINIAC_PREPROCESSING_COM_INVALID") from exc
        if not np.isfinite(values).all():
            raise RuntimeError("BRAINIAC_PREPROCESSING_COM_INVALID")
    for field in ("rigid_determinant", "foreground_fraction", "normalized_nonzero_std"):
        low, high = FIXED_QC_LIMITS[field]
        value = float(receipt[field])
        if not math.isfinite(value) or not low <= value <= high:
            raise RuntimeError(f"BRAINIAC_PREPROCESSING_QC_OUT_OF_BOUNDS:{field}")
    for field, maximum in FIXED_QC_MAXIMA.items():
        value = float(receipt[field])
        if not math.isfinite(value) or abs(value) > maximum:
            raise RuntimeError(f"BRAINIAC_PREPROCESSING_QC_OUT_OF_BOUNDS:{field}")


def verify_receipt_artifacts(receipt: Mapping[str, Any], *, artifact_root: Path) -> None:
    """Bind receipt hashes/sizes to live, non-reparse, non-temp bytes."""

    from src.research import brainiac_preprocessing_v2 as v2

    root = Path(artifact_root)
    for role in ("output", "transform"):
        relative = str(receipt[f"{role}_relative_path"])
        path = root / relative
        try:
            resolved = _beneath(path, root, must_exist=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError("BRAINIAC_PREPROCESSING_ARTIFACT_INVALID") from exc
        if (
            v2._is_reparse(resolved)
            or not resolved.is_file()
            or resolved.stat().st_size == 0
            or ".tmp" in resolved.name
        ):
            raise RuntimeError("BRAINIAC_PREPROCESSING_ARTIFACT_INVALID")
        raw = resolved.read_bytes()
        if (
            hashlib.sha256(raw).hexdigest() != str(receipt[f"{role}_sha256"]).casefold()
            or len(raw) != int(receipt[f"{role}_size"])
        ):
            raise RuntimeError(
                f"BRAINIAC_PREPROCESSING_{role.upper()}_BINDING_MISMATCH"
            )


def validate_receipt(
    receipt: Mapping[str, Any],
    *,
    artifact_root: Path,
    expected_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise RuntimeError("BRAINIAC_PREPROCESSING_RECEIPT_INVALID")
    assert_privacy(receipt)
    expected = canonical_sha(
        {key: value for key, value in receipt.items() if key != "integrity_sha256"}
    )
    if receipt.get("integrity_sha256") != expected:
        raise RuntimeError("BRAINIAC_PREPROCESSING_RECEIPT_INTEGRITY_MISMATCH")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise RuntimeError("BRAINIAC_PREPROCESSING_SCHEMA_V1_FORBIDDEN_IN_V2")
    seal_receipt(dict(receipt))
    for field, expected_value in sorted((expected_bindings or {}).items()):
        if str(receipt.get(field) or "").casefold() != str(expected_value).casefold():
            raise RuntimeError(
                f"BRAINIAC_PREPROCESSING_EXPECTED_BINDING_MISMATCH:{field}"
            )
    verify_receipt_artifacts(receipt, artifact_root=artifact_root)
    return dict(receipt)


def quarantine_decision(
    *,
    com_before: Sequence[float],
    com_after: Sequence[float],
    similarity_before: float,
    similarity_after: float,
) -> str:
    """Quarantine only when geometry AND similarity both degrade.

    ``similarity_*`` is the SimpleITK Mattes-MI registration metric (negative
    mutual information, empirically verified on this exact SimpleITK build:
    perfect alignment -2.31 vs 10 mm shift -0.62).  Lower is better, so
    worsening means the metric value INCREASES.
    """

    before_norm = math.dist((0.0, 0.0, 0.0), tuple(map(float, com_before)))
    after_norm = math.dist((0.0, 0.0, 0.0), tuple(map(float, com_after)))
    return (
        "QUARANTINE_BOTH_WORSENED"
        if after_norm > before_norm + 1e-6 and similarity_after > similarity_before
        else "OK"
    )


def validate_threshold_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_sample_hashes: Sequence[str] | None = None,
    expected_receipt_integrity_sha256s: Mapping[str, str] | None = None,
    expected_contract_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise RuntimeError("BRAINIAC_PREPROCESSING_THRESHOLD_MANIFEST_INVALID")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("artifact") != THRESHOLD_MANIFEST_ARTIFACT_TYPE
    ):
        raise RuntimeError("BRAINIAC_PREPROCESSING_THRESHOLD_MANIFEST_INVALID")
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise RuntimeError("BRAINIAC_PREPROCESSING_THRESHOLD_MANIFEST_INVALID")
    for metric, row in thresholds.items():
        if (
            not isinstance(row, Mapping)
            or not math.isfinite(float(row.get("low", float("nan"))))
            or not math.isfinite(float(row.get("high", float("nan"))))
            or float(row["low"]) > float(row["high"])
        ):
            raise RuntimeError(f"BRAINIAC_PREPROCESSING_THRESHOLD_MANIFEST_INVALID:{metric}")
    sample_hashes = manifest.get("sample_hashes")
    receipt_hashes = manifest.get("receipt_integrity_sha256s")
    contract_bindings = manifest.get("contract_bindings")
    if (
        not isinstance(sample_hashes, list)
        or len(sample_hashes) != int(manifest.get("sample_count", -1))
        or len(sample_hashes) != len(set(map(str, sample_hashes)))
        or not isinstance(receipt_hashes, Mapping)
        or set(map(str, sample_hashes)) != set(map(str, receipt_hashes))
        or not isinstance(contract_bindings, Mapping)
        or manifest.get("pipeline_contract_sha256")
        != canonical_sha(dict(sorted(contract_bindings.items())))
        or any(not _is_hex64(value) for value in receipt_hashes.values())
        or any(not _is_hex64(value) for value in contract_bindings.values())
    ):
        raise RuntimeError("BRAINIAC_PREPROCESSING_THRESHOLD_MANIFEST_BINDING_INVALID")
    expected = canonical_sha(
        {key: value for key, value in manifest.items() if key != "integrity_sha256"}
    )
    if manifest.get("integrity_sha256") != expected:
        raise RuntimeError("BRAINIAC_PREPROCESSING_THRESHOLD_MANIFEST_INVALID")
    if expected_sample_hashes is not None and sorted(map(str, expected_sample_hashes)) != sorted(
        map(str, sample_hashes)
    ):
        raise RuntimeError("BRAINIAC_PREPROCESSING_THRESHOLD_MANIFEST_BINDING_MISMATCH")
    if expected_receipt_integrity_sha256s is not None and dict(
        sorted((str(k), str(v)) for k, v in expected_receipt_integrity_sha256s.items())
    ) != dict(sorted((str(k), str(v)) for k, v in receipt_hashes.items())):
        raise RuntimeError("BRAINIAC_PREPROCESSING_THRESHOLD_MANIFEST_BINDING_MISMATCH")
    if expected_contract_bindings is not None and dict(
        sorted((str(k), str(v)) for k, v in expected_contract_bindings.items())
    ) != dict(sorted((str(k), str(v)) for k, v in contract_bindings.items())):
        raise RuntimeError("BRAINIAC_PREPROCESSING_THRESHOLD_MANIFEST_BINDING_MISMATCH")
    return dict(manifest)


def build_aggregate(
    *,
    receipts: Sequence[Mapping[str, Any]],
    expected_sample_hashes: Sequence[str],
    artifact_root: Path,
) -> dict[str, Any]:
    seen: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        sample_hash = str(receipt.get("sample_hash") or "")
        if sample_hash in seen:
            raise RuntimeError("BRAINIAC_PREPROCESSING_DUPLICATE_SAMPLE")
        if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
            raise RuntimeError("BRAINIAC_PREPROCESSING_SCHEMA_V1_FORBIDDEN_IN_V2")
        seen[sample_hash] = receipt
    if sorted(seen) != sorted(set(expected_sample_hashes)):
        raise RuntimeError("BRAINIAC_PREPROCESSING_AGGREGATE_COUNT_MISMATCH")
    quarantined = [
        sample for sample, row in seen.items() if row.get("status") == "QUARANTINED"
    ]
    body: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "artifact": AGGREGATE_ARTIFACT_TYPE,
        "status": "COMPLETED" if not quarantined else "BLOCKED_QUARANTINE",
        "sample_count": len(seen),
        "expected_sample_count": len(set(expected_sample_hashes)),
        "quarantined_sample_hashes": sorted(quarantined),
        "sample_hashes": sorted(seen),
        "receipt_integrity_sha256s": sorted(
            str(row["integrity_sha256"]) for row in seen.values()
        ),
        "labels_read": False,
        "pet_read": False,
        "raw_paths_persisted": False,
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }
    assert_privacy(body)
    body["integrity_sha256"] = canonical_sha(
        {key: value for key, value in body.items() if key != "integrity_sha256"}
    )
    return body
