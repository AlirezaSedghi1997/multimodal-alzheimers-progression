"""Fail-closed loader for the current Method X completion scope.

This module is intentionally outside ``src/training/method_x_*.py``. Planning and
claim-governance changes must not invalidate an in-flight model checkpoint's
semantic trainer source digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCOPE_PATH = (
    PROJECT_ROOT / "governance" / "method_x_current_completion_scope_v1.json"
)

OFFICIAL_CNN_ORDER = (
    "resnet10_3d",
    "resnet18_3d",
    "resnet34_3d",
    "densenet121_3d",
    "mednext_3d_s",
)
PROTOCOL_EXCLUSIONS = ("resnet50_3d", "efficientnet3d_b0")
C_STAGING_BASENAMES = ("resume_current.pt", "resume_previous.pt", "*.tmp")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, error: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(error)
    return value


def _project_file(relative: Any, *, error: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise RuntimeError(error)
    candidate = PROJECT_ROOT / Path(relative)
    try:
        candidate.resolve(strict=True).relative_to(PROJECT_ROOT.resolve(strict=True))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(error) from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError(error)
    return candidate


def validate_completion_scope(scope: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the locked project/storage/scientific boundary and return it."""

    if not isinstance(scope, Mapping) or scope.get("schema_version") != 1:
        raise RuntimeError("COMPLETION_SCOPE_SCHEMA_INVALID")
    if tuple(scope.get("official_cnn_order", ())) != OFFICIAL_CNN_ORDER:
        raise RuntimeError("OFFICIAL_CNN_SET_INVALID")
    if tuple(scope.get("protocol_exclusions", ())) != PROTOCOL_EXCLUSIONS:
        raise RuntimeError("PROTOCOL_EXCLUSION_SET_INVALID")

    brainiac = _mapping(scope.get("brainiac"), "BRAINIAC_BATTERY_ROLE_INVALID")
    if (
        brainiac.get("role") != "POST_HOC_SIXTH_MODEL_FAMILY"
        or brainiac.get("official_battery_member") is not False
        or brainiac.get("blocks_cnn_battery") is not False
    ):
        raise RuntimeError("BRAINIAC_BATTERY_ROLE_INVALID")

    storage = _mapping(scope.get("storage"), "STORAGE_SCOPE_INVALID")
    if storage.get("project_write_root") != r"H:\New project":
        raise RuntimeError("PROJECT_WRITE_ROOT_INVALID")
    if storage.get("repeat_temporary_write_probe") is not False:
        raise RuntimeError("REPEATED_WRITE_PROBE_POLICY_INVALID")
    c_staging = _mapping(
        storage.get("c_checkpoint_staging"), "C_STAGING_POLICY_INVALID"
    )
    if c_staging.get("root") != (
        r"C:\Users\USER\AppData\Local\AMUSE-X\checkpoint_staging"
    ) or c_staging.get("purpose") != "RUNTIME_RESUME_CHECKPOINTS_ONLY":
        raise RuntimeError("C_STAGING_POLICY_INVALID")
    if tuple(c_staging.get("allowed_basenames", ())) != C_STAGING_BASENAMES:
        raise RuntimeError("C_STAGING_BASENAME_POLICY_INVALID")

    external = _mapping(
        scope.get("external_evaluation"), "EXTERNAL_EVALUATION_POLICY_INVALID"
    )
    if external.get("required_for_internal_completion") is not False:
        raise RuntimeError("EXTERNAL_EVALUATION_MUST_BE_OPTIONAL")
    if external.get("default_status") != "EXTERNAL_UNAVAILABLE_NONBLOCKING":
        raise RuntimeError("EXTERNAL_EVALUATION_DEFAULT_INVALID")
    if (
        external.get("unavailable_evidence_label")
        != "ROBUSTNESS_EVIDENCE_NOT_EXTERNAL_VALIDATION"
    ):
        raise RuntimeError("INTERNAL_ROBUSTNESS_LABEL_INVALID")

    claims = _mapping(scope.get("claims"), "CLAIM_POLICY_INVALID")
    if claims.get("external_generalization_without_external_data") is not False:
        raise RuntimeError("EXTERNAL_GENERALIZATION_CLAIM_INVALID")
    if claims.get("diagnostic_claim_allowed") is not False:
        raise RuntimeError("DIAGNOSTIC_CLAIM_INVALID")

    supervision = _mapping(scope.get("supervision"), "SUPERVISION_POLICY_INVALID")
    if (
        supervision.get("human_approval_required") is not False
        or supervision.get("manual_override_allowed") is not False
        or supervision.get("unknown_is_pass") is not False
    ):
        raise RuntimeError("SUPERVISION_POLICY_INVALID")

    plan = _project_file(scope.get("canonical_plan"), error="CANONICAL_PLAN_INVALID")
    expected_plan_hash = scope.get("canonical_plan_sha256")
    if not isinstance(expected_plan_hash, str) or _sha256(plan) != expected_plan_hash:
        raise RuntimeError("CANONICAL_PLAN_HASH_MISMATCH")

    supersedes = scope.get("supersedes")
    if not isinstance(supersedes, list) or not supersedes:
        raise RuntimeError("SUPERSESSION_REGISTRY_INVALID")
    for item in supersedes:
        row = _mapping(item, "SUPERSESSION_REGISTRY_INVALID")
        historical = _project_file(
            row.get("path"), error="SUPERSESSION_REGISTRY_INVALID"
        )
        if row.get("status") != "SUPERSEDED_BY_20260821_SCOPE":
            raise RuntimeError("SUPERSESSION_REGISTRY_INVALID")
        if not isinstance(row.get("sha256"), str) or _sha256(historical) != row["sha256"]:
            raise RuntimeError("SUPERSESSION_HASH_MISMATCH")
    return scope


def load_completion_scope(path: Path = DEFAULT_SCOPE_PATH) -> dict[str, Any]:
    """Load and validate the H-side current scope registry."""

    candidate = Path(path)
    try:
        candidate.resolve(strict=True).relative_to(PROJECT_ROOT.resolve(strict=True))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("COMPLETION_SCOPE_PATH_INVALID") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError("COMPLETION_SCOPE_PATH_INVALID")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("COMPLETION_SCOPE_LOAD_FAILED") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("COMPLETION_SCOPE_SCHEMA_INVALID")
    validate_completion_scope(payload)
    return payload

