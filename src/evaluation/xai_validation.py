from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


class XAIValidationError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_locked_atlas(
    *, atlas_path: str | Path, reference_path: str | Path, expected_sha256: str
) -> dict[str, Any]:
    atlas_file = Path(atlas_path)
    if not atlas_file.is_file():
        raise XAIValidationError("MISSING_LICENSED_ATLAS")
    if sha256_file(atlas_file) != expected_sha256:
        raise XAIValidationError("ATLAS_HASH_MISMATCH")
    atlas = nib.load(str(atlas_file))
    reference = nib.load(str(reference_path))
    if atlas.shape != reference.shape or not np.allclose(
        atlas.affine, reference.affine, atol=1e-5, rtol=0
    ):
        raise XAIValidationError("ATLAS_GEOMETRY_MISMATCH")
    labels = np.unique(np.asarray(atlas.dataobj))
    labels = labels[labels > 0]
    if labels.size < 2:
        raise XAIValidationError("ATLAS_LABEL_COVERAGE_INVALID")
    return {"status": "PASS", "label_count": int(labels.size), "atlas_sha256": expected_sha256}


def validate_registered_atlas(
    *,
    registry_path: str | Path,
    atlas_path: str | Path | None = None,
    reference_path: str | Path,
) -> dict[str, Any]:
    """Validate a registry-locked anatomical atlas before XAI labeling."""
    registry_file = Path(registry_path)
    if not registry_file.is_file():
        raise XAIValidationError("MISSING_ATLAS_REGISTRY")
    try:
        registry = json.loads(registry_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise XAIValidationError("INVALID_ATLAS_REGISTRY") from exc
    if registry.get("status") != "ATLAS_LOCKED":
        raise XAIValidationError("ATLAS_REGISTRY_NOT_LOCKED")
    if registry.get("role") != "anatomical_xai_atlas":
        raise XAIValidationError("ATLAS_ROLE_INVALID")
    if registry.get("quantitative_biomarker_use_allowed") is not False:
        raise XAIValidationError("ATLAS_QUANTITATIVE_ROLE_FORBIDDEN")
    selected_atlas = Path(atlas_path) if atlas_path is not None else Path(registry["output_atlas_path"])
    result = validate_locked_atlas(
        atlas_path=selected_atlas,
        reference_path=reference_path,
        expected_sha256=str(registry["output_atlas_sha256"]),
    )
    model_space = registry.get("derived_model_space", {})
    reference_shape = list(nib.load(str(reference_path)).shape[:3])
    if list(model_space.get("shape", [])) != reference_shape:
        raise XAIValidationError("ATLAS_REGISTRY_REFERENCE_SHAPE_MISMATCH")
    return {**result, "registry_status": "PASS", "registry_sha256": sha256_file(registry_file)}


def _read_json_object(path: str | Path, error_code: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise XAIValidationError(error_code)
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise XAIValidationError(error_code) from exc
    if not isinstance(value, dict):
        raise XAIValidationError(error_code)
    return value


def load_registered_region_names(
    *, registry_path: str | Path, label_table_path: str | Path
) -> dict[int, str]:
    """Load names only from the hash-locked atlas label table.

    The older Method X path carried an AAL dictionary in source code.  That
    dictionary is not interchangeable with the Harvard--Oxford atlas used by
    the locked image.  This helper verifies both the table hash and the exact
    registry mapping before allowing an attribution report to name a region.
    """
    registry = _read_json_object(registry_path, "INVALID_ATLAS_REGISTRY")
    if registry.get("status") != "ATLAS_LOCKED":
        raise XAIValidationError("ATLAS_REGISTRY_NOT_LOCKED")
    if registry.get("role") != "anatomical_xai_atlas":
        raise XAIValidationError("ATLAS_ROLE_INVALID")
    if registry.get("quantitative_biomarker_use_allowed") is not False:
        raise XAIValidationError("ATLAS_QUANTITATIVE_ROLE_FORBIDDEN")
    asset_id = str(registry.get("asset_id", "")).lower()
    if "harvard_oxford" not in asset_id or "aal" in asset_id:
        raise XAIValidationError("ATLAS_REGION_NAME_SOURCE_MISMATCH")

    table_file = Path(label_table_path)
    if not table_file.is_file():
        raise XAIValidationError("MISSING_ATLAS_LABEL_TABLE")
    table = _read_json_object(table_file, "INVALID_ATLAS_LABEL_TABLE")
    expected_hash = str(registry.get("label_table_sha256", "")).lower()
    # lock_spatial_atlas_v1 hashes the canonical label object rather than the
    # pretty-printed file bytes; this keeps harmless JSON formatting changes
    # from invalidating a registry while still detecting label edits.
    try:
        canonical_table = {int(key): str(value) for key, value in table.items()}
    except (TypeError, ValueError) as exc:
        raise XAIValidationError("ATLAS_LABEL_ID_INVALID") from exc
    canonical_table_hash = hashlib.sha256(
        json.dumps(canonical_table, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if len(expected_hash) != 64 or canonical_table_hash != expected_hash:
        raise XAIValidationError("ATLAS_LABEL_TABLE_HASH_MISMATCH")
    names: dict[int, str] = {}
    for raw_id, raw_name in table.items():
        try:
            region_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise XAIValidationError("ATLAS_LABEL_ID_INVALID") from exc
        name = str(raw_name).strip()
        if region_id <= 0 or not name:
            raise XAIValidationError("ATLAS_LABEL_VALUE_INVALID")
        names[region_id] = name

    registry_labels = registry.get("labels")
    if not isinstance(registry_labels, dict):
        raise XAIValidationError("ATLAS_REGISTRY_LABELS_MISSING")
    expected_names = {int(key): str(value) for key, value in registry_labels.items()}
    if names != expected_names:
        raise XAIValidationError("ATLAS_LABEL_TABLE_REGISTRY_MISMATCH")
    return names


def validate_region_name_source(
    *, registry_path: str | Path, region_names: dict[int, str]
) -> dict[str, Any]:
    """Reject a caller-provided AAL (or other) name map for this atlas."""
    registry = _read_json_object(registry_path, "INVALID_ATLAS_REGISTRY")
    labels = registry.get("labels")
    expected = (
        {int(key): str(value) for key, value in labels.items()}
        if isinstance(labels, dict)
        else {}
    )
    supplied = {int(key): str(value) for key, value in region_names.items()}
    if not expected or supplied != expected:
        raise XAIValidationError("ATLAS_REGION_NAME_SOURCE_MISMATCH")
    return {"status": "PASS", "source": "registry_locked_label_table"}


def validate_xai_checkpoint_binding(
    *,
    checkpoint: dict[str, Any],
    expected_fold: int,
    expected_cache_sha256: str,
    expected_run_contract_digest: str,
    expected_split_membership_digest: str,
    expected_backbone: str | None = None,
) -> dict[str, Any]:
    """Validate the immutable Method X final-fold contract consumed by XAI."""
    contract = checkpoint.get("run_contract")
    if not isinstance(contract, dict):
        raise XAIValidationError("XAI_RUN_CONTRACT_MISSING")
    parameters = contract.get("parameters")
    if not isinstance(parameters, dict):
        raise XAIValidationError("XAI_RUN_CONTRACT_PARAMETERS_MISSING")
    # Check semantic scope before digest so stale 96^3 artifacts get a useful,
    # deterministic blocker instead of a generic digest error.
    if int(parameters.get("resolution", -1)) != 128:
        raise XAIValidationError("XAI_CACHE_RESOLUTION_NOT_128")
    if str(parameters.get("phase", "")) != "final":
        raise XAIValidationError("XAI_PHASE_NOT_FINAL")
    if expected_backbone is not None:
        if str(parameters.get("backbone", "")) != str(expected_backbone):
            raise XAIValidationError("XAI_BACKBONE_CONTRACT_MISMATCH")
        if str(checkpoint.get("backbone", "")) != str(expected_backbone):
            raise XAIValidationError("XAI_BACKBONE_CHECKPOINT_MISMATCH")
    if bool(parameters.get("hpo_selection_only", checkpoint.get("hpo_selection_only", True))):
        raise XAIValidationError("XAI_HPO_SELECTION_CHECKPOINT_FORBIDDEN")
    if int(contract.get("folds", -1)) != 5:
        raise XAIValidationError("XAI_FOLD_COUNT_INVALID")
    if str(contract.get("cache_sha256", "")) != str(expected_cache_sha256):
        raise XAIValidationError("XAI_CACHE_HASH_MISMATCH")
    contract_body = {key: value for key, value in contract.items() if key != "digest"}
    actual_digest = _canonical_digest(contract_body)
    if (
        contract.get("digest") != actual_digest
        or checkpoint.get("run_contract_digest") != actual_digest
        or actual_digest != expected_run_contract_digest
    ):
        raise XAIValidationError("XAI_RUN_CONTRACT_DIGEST_MISMATCH")
    if int(checkpoint.get("fold", -1)) != int(expected_fold):
        raise XAIValidationError("XAI_CHECKPOINT_FOLD_MISMATCH")
    if len(str(expected_split_membership_digest)) != 64 or any(
        character not in "0123456789abcdef" for character in str(expected_split_membership_digest).lower()
    ):
        raise XAIValidationError("XAI_EXPECTED_SPLIT_MEMBERSHIP_DIGEST_INVALID")
    if str(checkpoint.get("split_membership_digest", "")) != str(expected_split_membership_digest):
        raise XAIValidationError("XAI_SPLIT_MEMBERSHIP_DIGEST_MISMATCH")
    return {
        "status": "PASS",
        "fold": int(expected_fold),
        "resolution": 128,
        "cache_sha256": str(expected_cache_sha256),
        "run_contract_digest": actual_digest,
        "split_membership_digest": str(expected_split_membership_digest),
        "backbone": str(expected_backbone) if expected_backbone is not None else None,
    }


def _canonical_digest(value: Any) -> str:
    try:
        from src.utils.reproducibility import canonical_digest

        return str(canonical_digest(value))
    except Exception:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def evaluate_fold_xai_gates(
    fold_results: list[dict[str, Any]],
    *,
    expected_folds: int = 5,
    required_randomization_seeds: int = 3,
) -> dict[str, Any]:
    """Aggregate fold-level faithfulness gates, fail-closed and non-diagnostic."""
    folds = {int(row.get("fold", -1)) for row in fold_results if isinstance(row, dict)}
    expected = set(range(int(expected_folds)))
    base = {
        "region_claim_allowed": False,
        "anatomical_attribution_available": True,
        "public_granularity": "aggregate_region_level",
        "contains_raw_identifiers": False,
        "contains_subject_level_outputs": False,
    }
    if folds != expected or len(fold_results) != expected_folds:
        return {**base, "status": "BLOCKED_INCOMPLETE_FOLD_COVERAGE", "folds_observed": sorted(folds)}
    for row in fold_results:
        occlusion = row.get("occlusion_fidelity")
        if not isinstance(occlusion, dict) or occlusion.get("status") != "PASS":
            return {**base, "status": "BLOCKED_FAITHFULNESS_GATE_FAILED", "failed_fold": row.get("fold")}
        randomization = row.get("randomization_sanity")
        if not isinstance(randomization, list) or len({item.get("seed") for item in randomization if isinstance(item, dict)}) < required_randomization_seeds:
            return {**base, "status": "BLOCKED_RANDOMIZATION_SANITY_INCOMPLETE", "failed_fold": row.get("fold")}
        if any(not isinstance(item, dict) or item.get("status") != "PASS" for item in randomization):
            return {**base, "status": "BLOCKED_RANDOMIZATION_SANITY_FAILED", "failed_fold": row.get("fold")}
    return {
        **base,
        "status": "COMPLETED",
        "folds": expected_folds,
        "randomization_seeds_required": required_randomization_seeds,
    }


def evaluate_occlusion_fidelity(
    *, original_prediction: float, top_roi_prediction: float, random_roi_predictions: list[float]
) -> dict[str, Any]:
    if not random_roi_predictions:
        raise XAIValidationError("RANDOM_ROI_CONTROLS_REQUIRED")
    top_drop = float(original_prediction - top_roi_prediction)
    random_drops = np.asarray(
        [original_prediction - value for value in random_roi_predictions], dtype=float
    )
    return {
        "status": "PASS" if top_drop > float(np.median(random_drops)) else "FAIL",
        "top_roi_drop": top_drop,
        "random_roi_drop_median": float(np.median(random_drops)),
    }


def evaluate_randomization_sanity(
    original_attribution: Any, randomized_attribution: Any, *, max_abs_correlation: float = 0.90
) -> dict[str, Any]:
    original = np.asarray(original_attribution, dtype=float).ravel()
    randomized = np.asarray(randomized_attribution, dtype=float).ravel()
    if original.shape != randomized.shape or original.size < 2:
        raise XAIValidationError("SANITY_MAP_SHAPE_INVALID")
    if np.std(original) == 0 or np.std(randomized) == 0:
        correlation = 1.0 if np.allclose(original, randomized) else 0.0
    else:
        correlation = float(np.corrcoef(original, randomized)[0, 1])
    return {
        "status": "PASS" if abs(correlation) < max_abs_correlation else "FAIL",
        "absolute_correlation": abs(correlation),
        "threshold": float(max_abs_correlation),
    }
