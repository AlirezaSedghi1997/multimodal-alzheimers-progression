#!/usr/bin/env python3
"""Fold-local, PET-free anatomical attribution runner for Method X.

This stage intentionally refuses legacy checkpoints, 96^3 caches, AAL label
aliases, and partial corrected-v1 runs.  It emits aggregate region summaries
only; no subject-level heatmap, identifier, or diagnosis artifact is written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.xai_validation import (
    XAIValidationError,
    evaluate_fold_xai_gates,
    evaluate_occlusion_fidelity,
    evaluate_randomization_sanity,
    load_registered_region_names,
    sha256_file,
    validate_registered_atlas,
    validate_xai_checkpoint_binding,
)
from src.evaluation.method_x_battery_gate import (
    EFFECTIVE_FIVE as SHARED_EFFECTIVE_FIVE,
    EXCLUDED_BACKBONES as SHARED_EXCLUDED_BACKBONES,
    TERMINAL_TRAINING_STATUSES as SHARED_TERMINAL_TRAINING_STATUSES,
    MethodXBatteryGateError,
    validate_xai_battery_terminal as _validate_xai_battery_terminal_pure,
)
from src.method_y_revival.strict_yaml import load_unique_yaml
from src.training.method_x_autosupervisor import enforce_autonomous_supervision, evaluate_autonomous_supervision
from src.training.method_x_spm_snapshot import verify_spm_gate_snapshot
from src.utils.io_helpers import atomic_write_json, atomic_write_text


DEFAULT_CORRECTED_ROOT = ROOT / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729"
DEFAULT_CACHE = ROOT / "data/cached/t1_volume_cache_128.h5"
DEFAULT_SPLITS = ROOT / "splits"
DEFAULT_REGISTRY = ROOT / "results/spatial_attribution_v1/provenance/atlas_registry.json"
DEFAULT_LABELS = ROOT / "results/spatial_attribution_v1/atlases/harvard_oxford_method_x_labels.json"
DEFAULT_OUTPUT = ROOT / "results/final_results/method_x_anatomical_xai_v3"
DEFAULT_CONFIG = ROOT / "configs/experiments/method_x_xai_v1.yaml"

# Battery terminality canon, transcribed from the definitions in
# scripts/run_method_x_finetune_orchestrator.py and
# scripts/run_method_x_hybrid_orchestrator.py.  The *predicate* is transcribed
# too, not merely the constants: an earlier revision of this gate used counting
# (len==7, len(effective)==5) instead of the canon's exact identity set, which
# silently accepted a battery where a required model had been replaced by an
# arbitrary COMPLETED row.  Counting is not equivalent to identity.
#
# Kept as a local copy rather than a shared import so this stage cannot be
# silently loosened by an edit elsewhere; drift between the copies is checked by
# scripts/audit_mx06_mx07_readiness.py.
EFFECTIVE_FIVE = (
    "resnet10_3d",
    "resnet18_3d",
    "resnet34_3d",
    "densenet121_3d",
    "mednext_3d_s",
)
EXCLUDED_BACKBONES = ("resnet50_3d", "efficientnet3d_b0")
TERMINAL_TRAINING_STATUSES = frozenset(
    {
        "COMPLETED",
        "COMPLETED_VALIDATED_REPAIRED",
        "COMPLETED_VALIDATED_RECONCILED",
    }
)
ALLOWED_FINAL_XAI_MODELS = (*EFFECTIVE_FIVE, "amuse_x_hybrid3d")
FINAL_SELECTION_MANIFEST = "reports/final_model_selection_manifest.json"
FINAL_SUMMARY_SCHEMA_VERSION = 2
FINAL_SELECTION_PROMOTE = "PROMOTE_MX05_HYBRID"
FINAL_SELECTION_RETAIN = "RETAIN_MX04_NO_HYBRID_PROMOTION"
CANONICAL_XAI_CHECKPOINT_SEED = 13
FINAL_SELECTION_REQUIRED_ARTIFACT_ROLES = frozenset(
    {
        "metrics",
        "run_contract",
        "replication_report",
        "outer_evaluation_report",
        "paired_bootstrap_report",
        "calibration_report",
        "cohort_probe_report",
        "promotion_report",
        *(f"fold_{fold}_checkpoint" for fold in range(5)),
    }
)
_ABSOLUTE_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "subject_id",
        "subject_ids",
        "sample_id",
        "sample_ids",
        "raw_id",
        "raw_ids",
        "patient_id",
        "patient_ids",
    }
)


def _sha256_file_local(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_absolute_path_text(value: str) -> bool:
    return bool(value.startswith("/") or value.startswith("\\\\") or _ABSOLUTE_WINDOWS_PATH.match(value))


def _public_payload(value: Any) -> Any:
    """Remove machine-local absolute paths from a publishable XAI packet."""

    if isinstance(value, Mapping):
        return {str(key): _public_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_public_payload(item) for item in value]
    if isinstance(value, str) and _is_absolute_path_text(value):
        return _relative_path(Path(value))
    return value


def _assert_public_summary_safe(value: Any, *, key: str = "") -> None:
    normalized_key = key.strip().lower()
    if normalized_key in _FORBIDDEN_PUBLIC_KEYS:
        raise XAIValidationError("XAI_PUBLIC_SUMMARY_CONTAINS_RAW_IDENTIFIER_FIELD")
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            _assert_public_summary_safe(child_value, key=str(child_key))
        return
    if isinstance(value, list):
        for child_value in value:
            _assert_public_summary_safe(child_value, key=key)
        return
    if isinstance(value, str) and _is_absolute_path_text(value):
        raise XAIValidationError("XAI_PUBLIC_SUMMARY_CONTAINS_ABSOLUTE_PATH")


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_locked_atlas(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    label_table_path: Path = DEFAULT_LABELS,
) -> dict[str, Any]:
    """Resolve and verify the only atlas/reference allowed by the registry.

    The caller cannot supply an atlas or reference path.  Both paths and
    their hashes come from the locked registry, and the permitted use is
    restricted to aggregate anatomical attribution labeling.
    """

    registry_path = _project_path(registry_path).resolve()
    label_table_path = _project_path(label_table_path).resolve()
    registry = _read_json(registry_path)
    if str(registry.get("status", "")).upper() != "ATLAS_LOCKED":
        raise XAIValidationError("XAI_ATLAS_REGISTRY_NOT_LOCKED")
    if str(registry.get("allowed_use", "")) != "AGGREGATE_ANATOMICAL_ATTRIBUTION_LABELING_ONLY":
        raise XAIValidationError("XAI_ATLAS_USE_POLICY_INVALID")
    license_id = str(registry.get("license_id", "")).strip()
    if not license_id:
        raise XAIValidationError("XAI_ATLAS_LICENSE_MISSING")
    atlas_path = _project_path(str(registry.get("output_atlas_path", ""))).resolve()
    reference_path = _project_path(
        str((registry.get("derived_model_space") or {}).get("reference_path", ""))
    ).resolve()
    if not atlas_path.is_file() or not reference_path.is_file() or not label_table_path.is_file():
        raise XAIValidationError("XAI_LOCKED_ATLAS_ASSET_MISSING")
    expected_atlas = str(registry.get("output_atlas_sha256", "")).lower()
    expected_reference = str((registry.get("derived_model_space") or {}).get("reference_sha256", "")).lower()
    actual_atlas = _sha256_file_local(atlas_path)
    actual_reference = _sha256_file_local(reference_path)
    if expected_atlas != actual_atlas:
        raise XAIValidationError("XAI_ATLAS_HASH_MISMATCH")
    if expected_reference != actual_reference:
        raise XAIValidationError("XAI_REFERENCE_HASH_MISMATCH")
    return {
        "registry_path": str(registry_path),
        "registry_sha256": _sha256_file_local(registry_path),
        "labels_path": str(label_table_path),
        "labels_sha256": _sha256_file_local(label_table_path),
        "atlas_path": str(atlas_path),
        "atlas_sha256": actual_atlas,
        "reference_path": str(reference_path),
        "reference_sha256": actual_reference,
        "asset_id": registry.get("asset_id"),
        "license_id": license_id,
        "allowed_use": registry.get("allowed_use"),
    }


def _validate_final_selection_manifest(
    *, manifest_path: Path, corrected_root: Path, allowed_models: set[str]
) -> dict[str, Any]:
    selection = _read_json(manifest_path)
    if type(selection.get("schema_version")) is not int or selection.get("schema_version") != 1:
        raise XAIValidationError("XAI_SELECTION_MANIFEST_SCHEMA_INVALID")
    if str(selection.get("status", "")).upper() != "COMPLETED":
        raise XAIValidationError("XAI_SELECTION_MANIFEST_NOT_FINALIZED")
    run_id = str(selection.get("run_id") or "").strip()
    if not run_id:
        raise XAIValidationError("XAI_SELECTION_RUN_ID_MISSING")
    selected = str(selection.get("selected_backbone") or "").strip()
    kind = str(selection.get("model_kind") or "").strip()
    if selected not in allowed_models:
        raise XAIValidationError("XAI_SELECTED_BACKBONE_NOT_ALLOWED")
    if kind not in {"MX04_FINETUNE", "MX05_HYBRID"}:
        raise XAIValidationError("XAI_SELECTION_MODEL_KIND_INVALID")
    if (kind == "MX05_HYBRID") != (selected == "amuse_x_hybrid3d"):
        raise XAIValidationError("XAI_SELECTION_MODEL_KIND_BACKBONE_MISMATCH")
    decision = str(selection.get("selection_decision") or "").strip()
    canonical_seed = selection.get("canonical_checkpoint_seed")
    if type(canonical_seed) is not int or canonical_seed != CANONICAL_XAI_CHECKPOINT_SEED:
        raise XAIValidationError("XAI_SELECTION_CANONICAL_CHECKPOINT_SEED_INVALID")
    if selection.get("replication_seeds") != [13, 29, 47]:
        raise XAIValidationError("XAI_SELECTION_REPLICATION_SEEDS_INCOMPLETE")
    if str(selection.get("replication_status")) != "COMPLETED":
        raise XAIValidationError("XAI_SELECTION_REPLICATION_NOT_COMPLETED")
    if selection.get("outer_test_used_for_selection") is not False:
        raise XAIValidationError("XAI_SELECTION_USED_OUTER_TEST")
    outer = selection.get("outer_evaluation")
    if (
        not isinstance(outer, Mapping)
        or str(outer.get("status")) != "COMPLETED"
        or outer.get("count") != 1
        or isinstance(outer.get("count"), bool)
    ):
        raise XAIValidationError("XAI_SELECTION_OUTER_EVALUATION_NOT_EXACTLY_ONCE")
    promotion = selection.get("promotion_gate")
    if not isinstance(promotion, Mapping):
        raise XAIValidationError("XAI_SELECTION_PROMOTION_GATE_INVALID")
    promotion_status = str(promotion.get("status") or "").upper()
    promotion_blockers = promotion.get("blockers")
    if decision == FINAL_SELECTION_PROMOTE:
        if kind != "MX05_HYBRID" or selected != "amuse_x_hybrid3d":
            raise XAIValidationError("XAI_SELECTION_DECISION_MODEL_MISMATCH")
        if (
            promotion_status != "PASS"
            or promotion.get("would_promote") is not True
            or promotion_blockers != []
        ):
            raise XAIValidationError("XAI_SELECTION_PROMOTION_GATE_INVALID")
    elif decision == FINAL_SELECTION_RETAIN:
        if kind != "MX04_FINETUNE" or selected not in EFFECTIVE_FIVE:
            raise XAIValidationError("XAI_SELECTION_DECISION_MODEL_MISMATCH")
        if (
            promotion_status != "NO_PROMOTION"
            or promotion.get("would_promote") is not False
            or not isinstance(promotion_blockers, list)
            or not promotion_blockers
            or any(not isinstance(item, str) or not item.strip() for item in promotion_blockers)
        ):
            raise XAIValidationError("XAI_SELECTION_PROMOTION_GATE_INVALID")
    else:
        raise XAIValidationError("XAI_SELECTION_DECISION_INVALID")
    selected_root_text = str(selection.get("selected_model_root") or "").strip()
    if not selected_root_text:
        raise XAIValidationError("XAI_SELECTED_MODEL_ROOT_MISSING")
    selected_root = Path(selected_root_text)
    selected_root = selected_root if selected_root.is_absolute() else corrected_root / selected_root
    try:
        selected_root = selected_root.resolve()
        selected_root.relative_to(corrected_root.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise XAIValidationError("XAI_SELECTED_MODEL_ROOT_OUTSIDE_CORRECTED_ROOT") from exc
    if not selected_root.is_dir():
        raise XAIValidationError("XAI_SELECTED_MODEL_ROOT_MISSING")

    artifacts = selection.get("artifacts")
    if not isinstance(artifacts, list):
        raise XAIValidationError("XAI_SELECTION_ARTIFACTS_INVALID")
    observed_roles: set[str] = set()
    artifacts_by_role: dict[str, Path] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise XAIValidationError("XAI_SELECTION_ARTIFACTS_INVALID")
        role = str(item.get("role") or "")
        path_text = str(item.get("path") or "")
        expected_sha = str(item.get("sha256") or "").lower()
        if not role or role in observed_roles or len(expected_sha) != 64 or any(
            char not in "0123456789abcdef" for char in expected_sha
        ):
            raise XAIValidationError("XAI_SELECTION_ARTIFACTS_INVALID")
        artifact_path = Path(path_text)
        artifact_path = artifact_path if artifact_path.is_absolute() else corrected_root / artifact_path
        try:
            artifact_path = artifact_path.resolve()
            artifact_path.relative_to(corrected_root.resolve())
        except (OSError, RuntimeError, ValueError) as exc:
            raise XAIValidationError("XAI_SELECTION_ARTIFACT_OUTSIDE_CORRECTED_ROOT") from exc
        if not artifact_path.is_file() or _sha256_file_local(artifact_path) != expected_sha:
            raise XAIValidationError("XAI_SELECTION_ARTIFACT_HASH_MISMATCH")
        observed_roles.add(role)
        artifacts_by_role[role] = artifact_path
    if not FINAL_SELECTION_REQUIRED_ARTIFACT_ROLES.issubset(observed_roles):
        raise XAIValidationError("XAI_SELECTION_ARTIFACT_COVERAGE_INCOMPLETE")

    exact_model_artifacts = {
        "metrics": selected_root / "three_d_metrics.json",
        "run_contract": selected_root / "run_contract.json",
        **{
            f"fold_{fold}_checkpoint": selected_root / f"method_x_3d_fold_{fold}.pt"
            for fold in range(5)
        },
    }
    for role, expected_path in exact_model_artifacts.items():
        if artifacts_by_role.get(role) != expected_path.resolve():
            raise XAIValidationError("XAI_SELECTION_MODEL_ARTIFACT_PATH_MISMATCH")

    run_contract_document = _read_json(artifacts_by_role["run_contract"])
    run_contract = run_contract_document.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise XAIValidationError("XAI_SELECTION_RUN_CONTRACT_INVALID")
    run_parameters = run_contract.get("parameters")
    if not isinstance(run_parameters, Mapping):
        raise XAIValidationError("XAI_SELECTION_RUN_CONTRACT_INVALID")
    if (
        type(run_contract.get("seed")) is not int
        or run_contract.get("seed") != CANONICAL_XAI_CHECKPOINT_SEED
        or type(run_parameters.get("seed")) is not int
        or run_parameters.get("seed") != CANONICAL_XAI_CHECKPOINT_SEED
    ):
        raise XAIValidationError("XAI_SELECTION_RUN_CONTRACT_SEED_MISMATCH")
    if str(run_parameters.get("backbone") or "") != selected:
        raise XAIValidationError("XAI_SELECTION_RUN_CONTRACT_BACKBONE_MISMATCH")

    evidence_statuses = {
        "replication_report": {"COMPLETED"},
        "outer_evaluation_report": {"COMPLETED"},
        "paired_bootstrap_report": {"PASS", "COMPLETED"},
        "calibration_report": {"PASS", "COMPLETED"},
        "cohort_probe_report": {"PASS", "COMPLETED"},
    }
    for role, accepted_statuses in evidence_statuses.items():
        evidence = _read_json(artifacts_by_role[role])
        if str(evidence.get("status", "")).upper() not in accepted_statuses:
            raise XAIValidationError(f"XAI_SELECTION_EVIDENCE_NOT_TERMINAL:{role}")
        if str(evidence.get("run_id") or "") != run_id:
            raise XAIValidationError(f"XAI_SELECTION_EVIDENCE_RUN_ID_MISMATCH:{role}")
        evidence_backbone = str(evidence.get("selected_backbone") or evidence.get("backbone") or "")
        if evidence_backbone != selected:
            raise XAIValidationError(f"XAI_SELECTION_EVIDENCE_BACKBONE_MISMATCH:{role}")
    replication_evidence = _read_json(artifacts_by_role["replication_report"])
    if replication_evidence.get("replication_seeds") != [13, 29, 47]:
        raise XAIValidationError("XAI_SELECTION_REPLICATION_EVIDENCE_INVALID")
    promotion_evidence = _read_json(artifacts_by_role["promotion_report"])
    if str(promotion_evidence.get("status") or "").upper() != promotion_status:
        raise XAIValidationError("XAI_SELECTION_PROMOTION_EVIDENCE_STATUS_MISMATCH")
    if promotion_evidence.get("would_promote") is not promotion.get("would_promote"):
        raise XAIValidationError("XAI_SELECTION_PROMOTION_EVIDENCE_VERDICT_MISMATCH")
    if promotion_evidence.get("blockers") != promotion_blockers:
        raise XAIValidationError("XAI_SELECTION_PROMOTION_EVIDENCE_BLOCKERS_MISMATCH")
    if str(promotion_evidence.get("selection_decision") or "") != decision:
        raise XAIValidationError("XAI_SELECTION_PROMOTION_EVIDENCE_DECISION_MISMATCH")
    if str(promotion_evidence.get("run_id") or "") != run_id:
        raise XAIValidationError("XAI_SELECTION_EVIDENCE_RUN_ID_MISMATCH:promotion_report")
    promotion_backbone = str(
        promotion_evidence.get("selected_backbone") or promotion_evidence.get("backbone") or ""
    )
    if promotion_backbone != selected:
        raise XAIValidationError("XAI_SELECTION_EVIDENCE_BACKBONE_MISMATCH:promotion_report")
    bootstrap_evidence = _read_json(artifacts_by_role["paired_bootstrap_report"])
    bootstrap_count = bootstrap_evidence.get("resamples")
    if type(bootstrap_count) is not int or bootstrap_count < 1000:
        raise XAIValidationError("XAI_SELECTION_BOOTSTRAP_EVIDENCE_INVALID")
    outer_evidence = _read_json(artifacts_by_role["outer_evaluation_report"])
    if type(outer_evidence.get("count")) is not int or outer_evidence.get("count") != 1:
        raise XAIValidationError("XAI_SELECTION_OUTER_EVIDENCE_NOT_EXACTLY_ONCE")
    return {**selection, "selected_model_root": str(selected_root)}


def resolve_xai_selection(
    *,
    config_path: Path = DEFAULT_CONFIG,
    corrected_root: Path,
    require_completed_manifest: bool = False,
) -> dict[str, Any]:
    """Resolve the formal XAI model without an operator-selected backbone.

    ResNet34 is the pre-registered reference until an explicit, locked
    inner-validation manifest exists.  Outer-test metrics and CLI choices are
    never allowed to select the model used for anatomical attribution.
    """

    if not config_path.is_file():
        raise XAIValidationError("XAI_SELECTION_CONFIG_MISSING")
    try:
        payload = load_unique_yaml(config_path) or {}
    except (OSError, UnicodeError, ValueError) as exc:
        raise XAIValidationError("XAI_SELECTION_CONFIG_INVALID") from exc
    if not isinstance(payload, dict):
        raise XAIValidationError("XAI_SELECTION_CONFIG_INVALID")
    policy = str(payload.get("selection_policy", ""))
    allowed = [str(value) for value in payload.get("allowed_backbones", [])]
    allowed_final = [str(value) for value in payload.get("allowed_final_models", [])]
    reference = str(payload.get("reference_backbone", "")).strip()
    if "require_all_baselines_completed" in payload:
        raise XAIValidationError("XAI_DEAD_FLAG_PRESENT")
    if payload.get("require_effective_five_terminal") is not True:
        raise XAIValidationError("XAI_EFFECTIVE_FIVE_REQUIREMENT_MISSING")
    if policy != "PRESPECIFIED_REFERENCE_UNTIL_INNER_LOCK":
        raise XAIValidationError("XAI_SELECTION_POLICY_INVALID")
    if len(allowed) != len(EFFECTIVE_FIVE) or set(allowed) != set(EFFECTIVE_FIVE):
        raise XAIValidationError("XAI_ALLOWED_BACKBONES_NOT_EFFECTIVE_FIVE")
    if len(allowed_final) != len(ALLOWED_FINAL_XAI_MODELS) or set(allowed_final) != set(
        ALLOWED_FINAL_XAI_MODELS
    ):
        raise XAIValidationError("XAI_ALLOWED_FINAL_MODELS_NOT_LOCKED")
    if not reference or reference not in allowed:
        raise XAIValidationError("XAI_REFERENCE_BACKBONE_INVALID")
    if str(payload.get("attribution_method", "")) != "INPUT_GRADIENT_ABSOLUTE_NORMALIZED":
        raise XAIValidationError("XAI_ATTRIBUTION_METHOD_NOT_LOCKED")
    raw_seeds = payload.get("randomization_seeds", [])
    if (
        not isinstance(raw_seeds, list)
        or raw_seeds != [13, 29, 47]
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_seeds)
    ):
        raise XAIValidationError("XAI_RANDOMIZATION_SEEDS_NOT_LOCKED")
    try:
        bootstrap_resamples = int(payload.get("bootstrap_resamples", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise XAIValidationError("XAI_BOOTSTRAP_RESAMPLES_INVALID") from exc
    if isinstance(payload.get("bootstrap_resamples"), bool) or bootstrap_resamples < 1000:
        raise XAIValidationError("XAI_BOOTSTRAP_RESAMPLES_INVALID")
    try:
        max_samples_per_fold = int(payload.get("max_samples_per_fold", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise XAIValidationError("XAI_FORMAL_SAMPLE_LIMIT_FORBIDDEN") from exc
    if isinstance(payload.get("max_samples_per_fold"), bool) or max_samples_per_fold != 0:
        raise XAIValidationError("XAI_FORMAL_SAMPLE_LIMIT_FORBIDDEN")
    configured_registry = payload.get("atlas_registry")
    if configured_registry and _project_path(str(configured_registry)).resolve() != DEFAULT_REGISTRY.resolve():
        raise XAIValidationError("XAI_ATLAS_REGISTRY_PATH_NOT_LOCKED")
    configured_labels = payload.get("atlas_labels")
    if configured_labels and _project_path(str(configured_labels)).resolve() != DEFAULT_LABELS.resolve():
        raise XAIValidationError("XAI_ATLAS_LABELS_PATH_NOT_LOCKED")

    final_manifest_value = str(payload.get("final_selection_manifest") or "").strip()
    if final_manifest_value != FINAL_SELECTION_MANIFEST:
        raise XAIValidationError("XAI_SELECTION_MANIFEST_PATH_NOT_LOCKED")
    selection_path = corrected_root / final_manifest_value
    try:
        selection_path = selection_path.resolve()
        selection_path.relative_to(corrected_root.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise XAIValidationError("XAI_SELECTION_MANIFEST_PATH_INVALID") from exc
    selected = reference
    selection_scope = "PRESPECIFIED_REFERENCE"
    if selection_path.is_file():
        selection = _validate_final_selection_manifest(
            manifest_path=selection_path,
            corrected_root=corrected_root,
            allowed_models=set(ALLOWED_FINAL_XAI_MODELS),
        )
        selected = str(selection["selected_backbone"])
        selection_scope = "FINAL_SELECTION_LOCK"
    elif any(
        (corrected_root / "reports" / name).is_file()
        for name in ("finetune_selection_manifest.json", "hybrid_selection_manifest.json")
    ):
        raise XAIValidationError("XAI_SELECTION_MANIFEST_NOT_FINALIZED")
    elif require_completed_manifest:
        raise XAIValidationError("XAI_SELECTION_MANIFEST_MISSING_AFTER_UPSTREAM_COMPLETED")
    return {
        "backbone": selected,
        "selection_scope": selection_scope,
        "selection_policy": policy,
        "outer_test_used_for_selection": False,
        "config": _relative_path(config_path),
        "config_sha256": _sha256_file_local(config_path),
        "selection_manifest": _relative_path(selection_path) if selection_path.is_file() else None,
        "selection_manifest_sha256": _sha256_file_local(selection_path) if selection_path.is_file() else None,
        "selected_model_root": selection.get("selected_model_root") if selection_path.is_file() else None,
        "model_kind": selection.get("model_kind") if selection_path.is_file() else None,
        "selection_decision": selection.get("selection_decision") if selection_path.is_file() else None,
        "canonical_checkpoint_seed": selection.get("canonical_checkpoint_seed") if selection_path.is_file() else None,
        "attribution_method": "INPUT_GRADIENT_ABSOLUTE_NORMALIZED",
        "randomization_seeds": list(raw_seeds),
        "bootstrap_resamples": bootstrap_resamples,
        "max_samples_per_fold": max_samples_per_fold,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise XAIValidationError(f"INVALID_JSON:{path.name}") from exc
    if not isinstance(value, dict):
        raise XAIValidationError(f"INVALID_JSON_OBJECT:{path.name}")
    return value


def ensure_output_is_new(out_dir: Path) -> None:
    if (out_dir / "anatomical_xai_summary.json").exists():
        raise XAIValidationError("XAI_OUTPUT_ALREADY_EXISTS")


def blocked_summary(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "BLOCKED_STALE_OR_MISMATCHED_ARTIFACT",
        "reason": str(reason),
        "region_claim_allowed": False,
        "anatomical_attribution_available": False,
        "public_granularity": "aggregate_region_level",
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "contains_subject_level_outputs": False,
    }


def _cache_resolution(cache_path: Path) -> int:
    try:
        import h5py

        with h5py.File(cache_path, "r") as h5:
            value = h5.attrs.get("resolution")
            if hasattr(value, "item"):
                value = value.item()
            if value is None and "image" in h5 and len(h5["image"].shape) >= 4:
                value = h5["image"].shape[-1]
            return int(value)
    except Exception as exc:
        raise XAIValidationError("INVALID_MRI_CACHE") from exc


def validate_xai_battery_terminal(
    *,
    battery: Mapping[str, Any],
    backbone: str,
    selection: Mapping[str, Any] | None = None,
    model_root: Path | None = None,
) -> None:
    """Evaluate the shipped 5+2 battery gate without touching large inputs.

    Keeping this predicate pure lets the readiness auditor exercise the exact
    production gate without hashing the 6.5 GB MRI cache or walking through
    unrelated preflight guards.  ``preflight_corrected_v1`` calls this same
    function, so the audit and the formal runner cannot drift silently.
    """

    if (
        tuple(SHARED_EFFECTIVE_FIVE) != tuple(EFFECTIVE_FIVE)
        or tuple(SHARED_EXCLUDED_BACKBONES) != tuple(EXCLUDED_BACKBONES)
        or frozenset(SHARED_TERMINAL_TRAINING_STATUSES) != TERMINAL_TRAINING_STATUSES
    ):
        raise XAIValidationError("XAI_BATTERY_GATE_CONSTANT_DRIFT")
    try:
        _validate_xai_battery_terminal_pure(
            battery=battery,
            backbone=backbone,
            selection=selection,
            model_root=model_root,
        )
    except MethodXBatteryGateError as exc:
        raise XAIValidationError(str(exc)) from exc


def preflight_corrected_v1(
    *,
    corrected_root: Path,
    cache_path: Path,
    split_dir: Path,
    backbone: str = "resnet34_3d",
    folds: int = 5,
    selection: dict[str, Any] | None = None,
    model_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the immutable final artifacts before any model forward pass."""
    if model_root is not None:
        candidate = model_root.resolve()
        if not candidate.is_dir():
            raise XAIValidationError("XAI_SELECTED_MODEL_ROOT_MISSING")
        try:
            candidate.relative_to(corrected_root.resolve())
        except ValueError as exc:
            raise XAIValidationError("XAI_SELECTED_MODEL_ROOT_OUTSIDE_CORRECTED_ROOT") from exc
        final_dir = candidate
    else:
        candidate = corrected_root / "models" / backbone
        if not candidate.is_dir():
            raise XAIValidationError("CORRECTED_V1_FINAL_ARTIFACT_MISSING")
        final_dir = candidate
    metrics_path = final_dir / "three_d_metrics.json"
    if not metrics_path.is_file():
        raise XAIValidationError("CORRECTED_V1_FINAL_ARTIFACT_MISSING")
    metrics = _read_json(metrics_path)
    aggregate = metrics.get("aggregate") if isinstance(metrics.get("aggregate"), dict) else metrics
    if (
        aggregate.get("status") != "COMPLETED"
        or int(aggregate.get("folds", 0)) != folds
        or int(aggregate.get("completed_folds", 0)) != folds
    ):
        raise XAIValidationError("CORRECTED_V1_FINAL_ARTIFACT_NOT_COMPLETED")
    if str(aggregate.get("backbone", "")) != str(backbone):
        raise XAIValidationError("XAI_METRICS_BACKBONE_MISMATCH")
    battery_path = corrected_root / "battery_manifest.json"
    if not battery_path.is_file():
        raise XAIValidationError("XAI_BATTERY_MANIFEST_MISSING")
    battery = _read_json(battery_path)
    # Battery terminality follows the project's exclusion canon, identical to
    # `_battery_terminal` in the MX-04 fine-tune and MX-05 hybrid orchestrators:
    # the five in-protocol backbones must be terminally trained, and the two
    # protocol exclusions must REMAIN excluded.
    #
    # The previous predicate required every row to be COMPLETED, which is
    # unsatisfiable whenever the protocol exclusions are present, and which also
    # rejected the post-repair statuses COMPLETED_VALIDATED_REPAIRED /
    # COMPLETED_VALIDATED_RECONCILED that the live battery actually carries. A
    # gate that can never pass is not stricter than the canon, it is broken.
    #
    # The canon is transcribed exactly, including its use of the identity set
    # EFFECTIVE_FIVE and its rejection of duplicate backbone names.  Checking
    # only counts (7 rows, 5 non-excluded) would accept a battery in which a
    # required model was swapped for a different COMPLETED row, which is the same
    # class of error in the opposite direction.
    validate_xai_battery_terminal(
        battery=battery,
        backbone=backbone,
        selection=selection,
        model_root=model_root,
    )
    contract_payload = final_dir / "run_contract.json"
    if not contract_payload.is_file():
        raise XAIValidationError("CORRECTED_V1_RUN_CONTRACT_MISSING")
    contract_doc = _read_json(contract_payload)
    contract = contract_doc.get("run_contract")
    if not isinstance(contract, dict):
        raise XAIValidationError("CORRECTED_V1_RUN_CONTRACT_MISSING")
    contract_body = {key: value for key, value in contract.items() if key != "digest"}
    canonical_contract_digest = _canonical_sha256(contract_body)
    cache_sha = str(contract.get("cache_sha256", ""))
    if not cache_path.is_file():
        raise XAIValidationError("XAI_CACHE_MISSING")
    if _cache_resolution(cache_path) != 128:
        raise XAIValidationError("XAI_CACHE_RESOLUTION_NOT_128")
    if cache_sha != sha256_file(cache_path):
        raise XAIValidationError("XAI_CACHE_HASH_MISMATCH")
    if int(contract.get("folds", -1)) != folds:
        raise XAIValidationError("XAI_FOLD_COUNT_INVALID")
    if (
        contract.get("digest") != canonical_contract_digest
        or contract_doc.get("run_contract_digest") != canonical_contract_digest
    ):
        raise XAIValidationError("XAI_RUN_CONTRACT_DIGEST_MISMATCH")
    if not split_dir.is_dir() or any(not (split_dir / f"fold_{fold}_test.csv").is_file() for fold in range(folds)):
        raise XAIValidationError("XAI_OUTER_TEST_SPLITS_MISSING")
    from scripts.train_3d_mri import method_x_split_contract_digest

    split_digest = method_x_split_contract_digest(split_dir, folds, selection_only=False)
    if str(contract.get("split_digest", "")) != split_digest:
        raise XAIValidationError("XAI_SPLIT_CONTRACT_DIGEST_MISMATCH")
    locked_inputs = battery.get("locked_inputs") if isinstance(battery.get("locked_inputs"), dict) else {}
    if str(locked_inputs.get("cache_sha256", "")) != cache_sha:
        raise XAIValidationError("XAI_BATTERY_CACHE_BINDING_MISMATCH")
    spm_binding = battery.get("spm_gate_snapshot") if isinstance(battery.get("spm_gate_snapshot"), dict) else {}
    spm_sha = str(spm_binding.get("snapshot_sha256", "")).lower()
    if len(spm_sha) != 64 or str(contract.get("spm_snapshot_sha256", "")).lower() != spm_sha:
        raise XAIValidationError("XAI_SPM_SNAPSHOT_BINDING_MISMATCH")
    spm_path = _project_path(str(spm_binding.get("path", ""))).resolve()
    # ``snapshot_sha256`` is the canonical digest of the snapshot payload,
    # not the byte hash of its pretty-printed JSON file.  Recompute it with
    # the shared verifier and also re-check all bound source inventories.
    spm_verification = verify_spm_gate_snapshot(spm_path, root=ROOT, verify_sources=True)
    if (
        not spm_verification.get("verified")
        or not spm_verification.get("ready")
        or str(spm_verification.get("snapshot_sha256", "")).lower() != spm_sha
    ):
        raise XAIValidationError("XAI_SPM_SNAPSHOT_HASH_MISMATCH")
    fold_bindings: list[dict[str, Any]] = []
    for fold in range(folds):
        checkpoint_path = final_dir / f"method_x_3d_fold_{fold}.pt"
        if not checkpoint_path.is_file():
            raise XAIValidationError(f"CORRECTED_V1_FINAL_CHECKPOINT_MISSING:FOLD_{fold}")
        state_path = final_dir / f"method_x_3d_fold_{fold}_state.json"
        if not state_path.is_file():
            raise XAIValidationError(f"CORRECTED_V1_FOLD_STATE_MISSING:FOLD_{fold}")
        state = _read_json(state_path)
        if str(state.get("stage", "")).upper() != "FINALIZED":
            raise XAIValidationError(f"CORRECTED_V1_FOLD_NOT_FINALIZED:FOLD_{fold}")
        details = state.get("details") if isinstance(state.get("details"), dict) else {}
        split_membership_digest = str(details.get("split_membership_digest", "")).lower()
        if len(split_membership_digest) != 64 or any(
            character not in "0123456789abcdef" for character in split_membership_digest
        ):
            raise XAIValidationError(f"CORRECTED_V1_SPLIT_MEMBERSHIP_DIGEST_INVALID:FOLD_{fold}")
        try:
            import torch

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise XAIValidationError(f"CORRECTED_V1_CHECKPOINT_UNREADABLE:FOLD_{fold}") from exc
        binding = validate_xai_checkpoint_binding(
            checkpoint=checkpoint,
            expected_fold=fold,
            expected_cache_sha256=cache_sha,
            expected_run_contract_digest=str(contract.get("digest")),
            expected_split_membership_digest=split_membership_digest,
            expected_backbone=backbone,
        )
        fold_bindings.append(
            {
                "fold": fold,
                "checkpoint": _relative_path(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "fold_state": _relative_path(state_path),
                "fold_state_sha256": sha256_file(state_path),
                **binding,
            }
        )
    return {
        "status": "PASS",
        "cache": _relative_path(cache_path),
        "cache_sha256": cache_sha,
        "split_digest": split_digest,
        "split_manifest_sha256": str(locked_inputs.get("split_sha256", "")),
        "spm_snapshot_sha256": spm_sha,
        "spm_snapshot": _relative_path(spm_path),
        "spm_verification": {
            "status": spm_verification.get("status"),
            "verified": bool(spm_verification.get("verified")),
            "ready": bool(spm_verification.get("ready")),
            "checks": spm_verification.get("checks", {}),
        },
        "battery_manifest": _relative_path(battery_path),
        "battery_manifest_sha256": sha256_file(battery_path),
        "run_contract": _relative_path(contract_payload),
        "run_contract_sha256": sha256_file(contract_payload),
        "run_contract_digest": str(contract.get("digest")),
        "backbone": backbone,
        "selection": selection or {
            "backbone": backbone,
            "selection_scope": "LEGACY_CALLER_UNSPECIFIED",
            "outer_test_used_for_selection": False,
        },
        "final_dir": str(final_dir.resolve()),
        "model_root": str(candidate.resolve()),
        "folds": fold_bindings,
    }


def assert_xai_upstream_stages_completed(corrected_root: Path) -> dict[str, str]:
    """Require terminal validity, fine-tune and hybrid stages before XAI."""

    required = {
        "MX-03": corrected_root / "reports/mx03_validity_orchestrator.json",
        "MX-04": corrected_root / "reports/mx04_finetune_orchestrator.json",
        "MX-05": corrected_root / "reports/mx05_hybrid_orchestrator.json",
    }
    statuses: dict[str, str] = {}
    for stage, path in required.items():
        if not path.is_file():
            raise XAIValidationError(f"XAI_UPSTREAM_STAGE_NOT_COMPLETED:{stage}:MISSING")
        payload = _read_json(path)
        status = str(payload.get("status", "")).upper()
        statuses[stage] = status
        if status != "COMPLETED":
            raise XAIValidationError(
                f"XAI_UPSTREAM_STAGE_NOT_COMPLETED:{stage}:{status or 'UNKNOWN'}"
            )
    return statuses


def build_fold_gate_result(
    *,
    fold: int,
    original_predictions: list[float],
    top_roi_predictions: list[float],
    random_roi_predictions: list[list[float]],
    original_attribution: Any,
    randomized_attributions: dict[int, Any],
) -> dict[str, Any]:
    """Run the required occlusion and randomized-weight checks for one fold."""
    if not original_predictions or len(original_predictions) != len(top_roi_predictions):
        raise XAIValidationError("XAI_OCCLUSION_INPUTS_INVALID")
    flattened_random = [value for row in random_roi_predictions for value in row]
    occlusion = evaluate_occlusion_fidelity(
        original_prediction=sum(original_predictions) / len(original_predictions),
        top_roi_prediction=sum(top_roi_predictions) / len(top_roi_predictions),
        random_roi_predictions=flattened_random,
    )
    sanity = []
    for seed, randomized in sorted(randomized_attributions.items()):
        result = evaluate_randomization_sanity(original_attribution, randomized)
        sanity.append({"seed": int(seed), **result})
    return {"fold": int(fold), "occlusion_fidelity": occlusion, "randomization_sanity": sanity}


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def run_forward_xai(
    *,
    corrected_root: Path,
    cache_path: Path,
    split_dir: Path,
    atlas_path: Path,
    region_names: dict[int, str],
    preflight: dict[str, Any],
    max_samples_per_fold: int = 0,
    device_name: str = "auto",
    randomization_seeds: tuple[int, ...] = (13, 29, 47),
    bootstrap_resamples: int = 1000,
) -> dict[str, Any]:
    """Run fold-wise input-gradient attribution after the immutable guard."""

    import nibabel as nib
    import numpy as np
    import torch

    from scripts.train_3d_mri import H5MethodXDataset, cache_index, ok_indices, split_indices
    from src.evaluation.method_x_xai_forward import aggregate_fold_regions, build_model_from_checkpoint, run_fold_attribution
    from src.evaluation.xai_validation import evaluate_fold_xai_gates

    if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise XAIValidationError("XAI_CUDA_REQUIRED")
        device = torch.device("cuda")
    else:
        raise XAIValidationError("XAI_CUDA_REQUIRED_FOR_FORMAL_FORWARD")
    atlas_data = np.asarray(nib.load(str(atlas_path)).dataobj).astype(np.int32)
    atlas_ids = {int(value) for value in np.unique(atlas_data) if int(value) > 0}
    label_ids = {int(value) for value in region_names}
    if not atlas_ids or atlas_ids != label_ids:
        raise XAIValidationError("XAI_ATLAS_LABEL_COVERAGE_MISMATCH")
    fold_results: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    sample_to_idx = cache_index(cache_path)
    valid_indices = ok_indices(cache_path)
    for binding in preflight.get("folds", []):
        fold = int(binding.get("fold", -1))
        if fold < 0:
            raise XAIValidationError("XAI_FOLD_BINDING_INVALID")
        final_dir = Path(str(preflight.get("final_dir")))
        checkpoint_path = final_dir / f"method_x_3d_fold_{fold}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model, state_source = build_model_from_checkpoint(checkpoint, device=device)
        split_path = split_dir / f"fold_{fold}_test.csv"
        indices, labels = split_indices(split_path, sample_to_idx, valid_indices, limit=max_samples_per_fold)
        if not indices:
            raise XAIValidationError(f"XAI_FOLD_TEST_INPUTS_EMPTY:FOLD_{fold}")
        dataset = H5MethodXDataset(
            cache_path,
            indices,
            labels,
            input_mode=str(checkpoint.get("input_mode", "mri_plus_tabular")),
        )
        mean = checkpoint.get("tab_mean")
        std = checkpoint.get("tab_std")
        if mean is not None and std is not None:
            mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
            std_tensor = torch.as_tensor(std, dtype=torch.float32).clamp_min(1.0)
        else:
            mean_tensor = torch.zeros(5)
            std_tensor = torch.ones(5)
        class _NormalizedFoldBatches:
            """Re-iterable, lazy CPU sequence; never materializes a fold on GPU."""

            def __len__(self) -> int:
                return len(dataset)

            def __getitem__(self, item: int):
                if item < 0 or item >= len(dataset):
                    raise IndexError(item)
                image, tabular, _label, presence = dataset[item]
                normalized = (tabular - mean_tensor) / std_tensor
                return image.unsqueeze(0), normalized.unsqueeze(0), presence.reshape(1)

        batches = _NormalizedFoldBatches()
        fold_result = run_fold_attribution(
            model=model,
            batches=batches,
            atlas_data=atlas_data,
            region_names=region_names,
            randomization_seeds=randomization_seeds,
            bootstrap_resamples=bootstrap_resamples,
        )
        fold_result.update(
            {
                "fold": fold,
                "checkpoint": _relative_path(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "state_source": state_source,
            }
        )
        fold_results.append(fold_result)
        gate_rows.append({"fold": fold, **fold_result["gate"]})
    gates = evaluate_fold_xai_gates(gate_rows, expected_folds=5, required_randomization_seeds=3)
    return {
        "status": "COMPLETED" if gates.get("status") == "COMPLETED" else str(gates.get("status", "BLOCKED_XAI_GATE")),
        "forward_method": "INPUT_GRADIENT_ABSOLUTE_NORMALIZED",
        "device": str(device),
        "max_samples_per_fold": int(max_samples_per_fold),
        "folds": fold_results,
        "fold_gates": gates,
        "aggregate_region_attribution": aggregate_fold_regions(
            fold_results, bootstrap_resamples=bootstrap_resamples
        ),
        "aggregate_only": True,
        "contains_raw_identifiers": False,
        "contains_subject_level_outputs": False,
        "region_claim_allowed": False,
        "interpretation": (
            "Regions are higher-attribution locations for the model's PET-free biomarker-status decision; "
            "they are not disease regions, causal explanations, or quantitative biomarkers."
        ),
        "provenance": {
            "cache": _relative_path(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "atlas": _relative_path(atlas_path),
            "atlas_sha256": sha256_file(atlas_path),
            "run_contract_digest": preflight.get("run_contract_digest"),
            "split_digest": preflight.get("split_digest"),
            "spm_snapshot_sha256": preflight.get("spm_snapshot_sha256"),
            "selection": preflight.get("selection"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the intentionally small, non-operator XAI interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "run"), default="audit")
    parser.add_argument("--corrected_root", default=str(DEFAULT_CORRECTED_ROOT))
    return parser


def _xai_output_from_config(payload: dict[str, Any]) -> Path:
    # Early-track 034: a YAML edit must never divert formal output outside
    # the locked provenance tree. Only the locked output_root is accepted.
    value = payload.get("output_root")
    if not value:
        return DEFAULT_OUTPUT
    candidate = _project_path(str(value)).resolve()
    if candidate != DEFAULT_OUTPUT.resolve():
        raise XAIValidationError("XAI_OUTPUT_ROOT_NOT_LOCKED")
    return DEFAULT_OUTPUT


def _render_summary_markdown(summary: Mapping[str, Any]) -> str:
    return (
        "# Method X Anatomical XAI\n\n"
        f"- status: `{summary.get('status', '')}`\n"
        f"- reason: `{summary.get('reason', '')}`\n"
        "- granularity: `aggregate_region_level`\n"
        "- diagnostic_claim_allowed: `false`\n"
    )


def _summary_integrity_digest(summary: Mapping[str, Any]) -> str:
    payload = dict(summary)
    integrity = dict(payload.get("result_integrity") or {})
    integrity.pop("canonical_payload_sha256", None)
    payload["result_integrity"] = integrity
    return _canonical_sha256(payload)


def _write_summary(out_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Write companion first and canonical JSON last, without overwriting."""

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "anatomical_xai_summary.json"
    markdown_path = out_dir / "anatomical_xai_summary.md"
    if summary_path.exists() or markdown_path.exists():
        raise XAIValidationError("XAI_OUTPUT_ALREADY_EXISTS")
    finalized = _public_payload(dict(summary))
    _validate_completed_summary_schema(finalized)
    _assert_public_summary_safe(finalized)
    markdown = _render_summary_markdown(finalized)
    atomic_write_text(markdown_path, markdown)
    markdown_sha = _sha256_file_local(markdown_path)
    finalized["result_integrity"] = {
        "schema_version": 1,
        "algorithm": "sha256-canonical-json-excluding-canonical_payload_sha256",
        "companion_markdown": "anatomical_xai_summary.md",
        "companion_markdown_sha256": markdown_sha,
    }
    finalized["result_integrity"]["canonical_payload_sha256"] = _summary_integrity_digest(finalized)
    atomic_write_json(summary_path, finalized)
    summary.clear()
    summary.update(finalized)
    return finalized


def _implementation_source_manifest() -> dict[str, str]:
    explicit = {
        Path(__file__).resolve(),
        (ROOT / "scripts/train_3d_mri.py").resolve(),
        (ROOT / "src/evaluation/method_x_xai_forward.py").resolve(),
        (ROOT / "src/evaluation/xai_validation.py").resolve(),
        (ROOT / "src/evaluation/xai_regions.py").resolve(),
        (ROOT / "src/training/method_x_spm_snapshot.py").resolve(),
        (ROOT / "src/utils/reproducibility.py").resolve(),
        (ROOT / "src/method_y_revival/strict_yaml.py").resolve(),
    }
    for directory in (
        ROOT / "src/models/backbones",
        ROOT / "src/models/fusion",
        ROOT / "src/data",
    ):
        explicit.update(path.resolve() for path in directory.rglob("*.py"))
    missing = [path for path in explicit if not path.is_file()]
    if missing:
        raise XAIValidationError("XAI_IMPLEMENTATION_DEPENDENCY_MISSING")
    return {
        path.relative_to(ROOT.resolve()).as_posix(): _sha256_file_local(path)
        for path in sorted(explicit, key=lambda item: item.as_posix())
    }


def _runtime_fingerprint() -> dict[str, Any]:
    distributions: dict[str, str] = {}
    for distribution in ("torch", "numpy", "nibabel", "h5py"):
        try:
            distributions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            distributions[distribution] = "NOT_INSTALLED"
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": distributions,
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        devices = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                devices.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "compute_capability": list(torch.cuda.get_device_capability(index)),
                    }
                )
        runtime["torch_runtime"] = {
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "devices": devices,
        }
    except Exception as exc:
        raise XAIValidationError("XAI_RUNTIME_FINGERPRINT_FAILED") from exc
    return runtime


def build_formal_binding(
    *,
    preflight: Mapping[str, Any],
    atlas_contract: Mapping[str, Any],
    atlas: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a formal XAI result to every mutable scientific input.

    A run-contract digest alone does not cover checkpoint bytes, fold-state
    receipts, selection/config changes, or atlas/reference assets.  This
    canonical payload is the only key accepted for immutable reuse.
    """

    selection = preflight.get("selection") if isinstance(preflight.get("selection"), Mapping) else {}
    folds: list[dict[str, Any]] = []
    raw_folds = preflight.get("folds") if isinstance(preflight.get("folds"), list) else []
    for row in sorted(
        (row for row in raw_folds if isinstance(row, Mapping)),
        key=lambda row: int(row.get("fold", -1)),
    ):
        folds.append(
            {
                "fold": int(row.get("fold", -1)),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "fold_state_sha256": row.get("fold_state_sha256"),
                "split_membership_digest": row.get("split_membership_digest"),
                "run_contract_digest": row.get("run_contract_digest"),
                "cache_sha256": row.get("cache_sha256"),
            }
        )
    inputs = {
        "schema_version": 2,
        "cache_sha256": preflight.get("cache_sha256"),
        "split_digest": preflight.get("split_digest"),
        "split_manifest_sha256": preflight.get("split_manifest_sha256"),
        "spm_snapshot_sha256": preflight.get("spm_snapshot_sha256"),
        "battery_manifest_sha256": preflight.get("battery_manifest_sha256"),
        "run_contract_sha256": preflight.get("run_contract_sha256"),
        "run_contract_digest": preflight.get("run_contract_digest"),
        "backbone": preflight.get("backbone"),
        "selection": {
            "selection_scope": selection.get("selection_scope"),
            "selection_policy": selection.get("selection_policy"),
            "outer_test_used_for_selection": selection.get("outer_test_used_for_selection"),
            "config_sha256": selection.get("config_sha256"),
            "selection_manifest_sha256": selection.get("selection_manifest_sha256"),
            "attribution_method": selection.get("attribution_method"),
            "randomization_seeds": selection.get("randomization_seeds"),
            "bootstrap_resamples": selection.get("bootstrap_resamples"),
            "max_samples_per_fold": selection.get("max_samples_per_fold"),
        },
        "folds": folds,
        "atlas": {
            "registry_sha256": atlas.get("registry_sha256"),
            "labels_sha256": atlas.get("labels_sha256"),
            "atlas_sha256": atlas.get("atlas_sha256"),
            "reference_sha256": atlas.get("reference_sha256"),
            "asset_id": atlas.get("asset_id"),
            "license_id": atlas.get("license_id"),
            "allowed_use": atlas.get("allowed_use"),
            "contract": dict(atlas_contract),
        },
        "implementation": {
            "source_files": _implementation_source_manifest(),
            "runtime": _runtime_fingerprint(),
        },
    }
    return {
        "schema_version": 2,
        "digest": _canonical_sha256(inputs),
        "inputs": inputs,
    }


def _validate_completed_summary_schema(summary: Mapping[str, Any]) -> None:
    if str(summary.get("status", "")).upper() != "COMPLETED":
        raise XAIValidationError("XAI_OUTPUT_EXISTS_NONTERMINAL")
    if type(summary.get("schema_version")) is not int or summary.get("schema_version") != FINAL_SUMMARY_SCHEMA_VERSION:
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_SCHEMA_INVALID")
    required_flags = {
        "anatomical_attribution_available": True,
        "aggregate_only": True,
        "contains_raw_identifiers": False,
        "contains_subject_level_outputs": False,
        "region_claim_allowed": False,
    }
    if any(summary.get(key) is not expected for key, expected in required_flags.items()):
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_SAFETY_FLAGS_INVALID")
    folds = summary.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_FOLD_COVERAGE_INVALID")
    observed_folds: set[int] = set()
    for row in folds:
        if not isinstance(row, Mapping) or type(row.get("fold")) is not int:
            raise XAIValidationError("XAI_COMPLETED_OUTPUT_FOLD_COVERAGE_INVALID")
        observed_folds.add(int(row["fold"]))
        regions = row.get("region_attribution")
        gate = row.get("gate")
        if not isinstance(regions, list) or not regions or not isinstance(gate, Mapping):
            raise XAIValidationError("XAI_COMPLETED_OUTPUT_FOLD_SCHEMA_INVALID")
        occlusion = gate.get("occlusion_fidelity")
        sanity = gate.get("randomization_sanity")
        if not isinstance(occlusion, Mapping) or str(occlusion.get("status")) != "PASS":
            raise XAIValidationError("XAI_COMPLETED_OUTPUT_FOLD_GATE_INVALID")
        if (
            not isinstance(sanity, list)
            or [item.get("seed") for item in sanity if isinstance(item, Mapping)] != [13, 29, 47]
            or any(str(item.get("status")) != "PASS" for item in sanity if isinstance(item, Mapping))
            or any(not isinstance(item, Mapping) for item in sanity)
        ):
            raise XAIValidationError("XAI_COMPLETED_OUTPUT_RANDOMIZATION_GATE_INVALID")
    if observed_folds != set(range(5)):
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_FOLD_COVERAGE_INVALID")
    fold_gates = summary.get("fold_gates")
    if not isinstance(fold_gates, Mapping) or str(fold_gates.get("status")) != "COMPLETED":
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_GATE_NOT_COMPLETED")
    aggregate = summary.get("aggregate_region_attribution")
    if not isinstance(aggregate, list) or not aggregate:
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_AGGREGATE_INVALID")
    for row in aggregate:
        if not isinstance(row, Mapping) or type(row.get("n_folds")) is not int or row.get("n_folds") != 5:
            raise XAIValidationError("XAI_COMPLETED_OUTPUT_AGGREGATE_INVALID")
        for key in ("mean", "median", "ci_low", "ci_high"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise XAIValidationError("XAI_COMPLETED_OUTPUT_AGGREGATE_INVALID")
    _assert_public_summary_safe(summary)


def _validate_stored_binding(
    binding: Mapping[str, Any] | None, *, expected_binding: Mapping[str, Any]
) -> None:
    if not isinstance(binding, Mapping) or not isinstance(binding.get("inputs"), Mapping):
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_BINDING_INVALID")
    if binding.get("schema_version") != 2 or expected_binding.get("schema_version") != 2:
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_BINDING_INVALID")
    stored_digest = str(binding.get("digest", ""))
    expected_digest = str(expected_binding.get("digest", ""))
    if _canonical_sha256(binding["inputs"]) != stored_digest:
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_BINDING_TAMPERED")
    if not isinstance(expected_binding.get("inputs"), Mapping) or _canonical_sha256(
        expected_binding["inputs"]
    ) != expected_digest:
        raise XAIValidationError("XAI_EXPECTED_BINDING_INVALID")
    if stored_digest != expected_digest:
        raise XAIValidationError("XAI_OUTPUT_EXISTS_WITH_DIFFERENT_PROVENANCE")


def _validate_result_integrity(summary: Mapping[str, Any], *, summary_path: Path) -> None:
    integrity = summary.get("result_integrity")
    if not isinstance(integrity, Mapping) or integrity.get("schema_version") != 1:
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_INTEGRITY_INVALID")
    if integrity.get("algorithm") != "sha256-canonical-json-excluding-canonical_payload_sha256":
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_INTEGRITY_INVALID")
    if str(integrity.get("canonical_payload_sha256", "")) != _summary_integrity_digest(summary):
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_CONTENT_TAMPERED")
    if integrity.get("companion_markdown") != "anatomical_xai_summary.md":
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_COMPANION_INVALID")
    companion = summary_path.with_name("anatomical_xai_summary.md")
    if not companion.is_file() or _sha256_file_local(companion) != str(
        integrity.get("companion_markdown_sha256", "")
    ):
        raise XAIValidationError("XAI_COMPLETED_OUTPUT_COMPANION_INVALID")


def classify_existing_summary(summary_path: Path, *, expected_binding: Mapping[str, Any]) -> str:
    """Reuse only a complete, self-authenticating result with current inputs."""

    companion = summary_path.with_name("anatomical_xai_summary.md")
    if not summary_path.exists():
        if companion.exists():
            raise XAIValidationError("XAI_OUTPUT_PARTIAL_ARTIFACT_PRESENT")
        return "RUN_NEW"
    if not summary_path.is_file():
        raise XAIValidationError("XAI_OUTPUT_PATH_INVALID")
    prior = _read_json(summary_path)
    _validate_completed_summary_schema(prior)
    _validate_stored_binding(
        prior.get("formal_binding") if isinstance(prior.get("formal_binding"), Mapping) else None,
        expected_binding=expected_binding,
    )
    _validate_result_integrity(prior, summary_path=summary_path)
    return "REUSE_COMPLETED"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corrected_root = _project_path(args.corrected_root).resolve()
    supervision: dict[str, Any] | None = None
    out_dir = DEFAULT_OUTPUT
    formal_action = "NONE"
    existing_summary: Path | None = None
    try:
        try:
            config_payload = load_unique_yaml(DEFAULT_CONFIG) or {}
        except (OSError, UnicodeError, ValueError) as exc:
            raise XAIValidationError("XAI_SELECTION_CONFIG_INVALID") from exc
        if not isinstance(config_payload, dict):
            raise XAIValidationError("XAI_SELECTION_CONFIG_INVALID")
        out_dir = _xai_output_from_config(config_payload).resolve()
        battery_manifest = _read_json(corrected_root / "battery_manifest.json")
        protocol = battery_manifest.get("locked_protocol", {}) if isinstance(battery_manifest, dict) else {}
        supervision = evaluate_autonomous_supervision(
            root=ROOT,
            output_root=corrected_root,
            phase="xai",
            config={
                "resolution": protocol.get("resolution", 128),
                "folds": protocol.get("folds", 5),
                "seed": protocol.get("seed", 13),
                "hardware": {"hard_stop_vram_gb": 7.5},
            },
        )
        if supervision.get("decision") == "BLOCK":
            raise XAIValidationError("AUTOSUPERVISOR_BLOCKED")
        upstream_stages = assert_xai_upstream_stages_completed(corrected_root)
        selection = resolve_xai_selection(
            config_path=DEFAULT_CONFIG,
            corrected_root=corrected_root,
            require_completed_manifest=True,
        )
        model_root = None
        if selection.get("selection_manifest"):
            selected_root = selection.get("selected_model_root")
            if selected_root:
                model_root = Path(str(selected_root))
        preflight = preflight_corrected_v1(
            corrected_root=corrected_root,
            cache_path=DEFAULT_CACHE,
            split_dir=DEFAULT_SPLITS,
            backbone=str(selection["backbone"]),
            selection=selection,
            model_root=model_root,
        )
        preflight["upstream_stages"] = upstream_stages
        atlas = resolve_locked_atlas()
        names = load_registered_region_names(
            registry_path=Path(atlas["registry_path"]), label_table_path=Path(atlas["labels_path"])
        )
        atlas_contract = validate_registered_atlas(
            registry_path=Path(atlas["registry_path"]),
            atlas_path=Path(atlas["atlas_path"]),
            reference_path=Path(atlas["reference_path"]),
        )
        formal_binding = build_formal_binding(
            preflight=preflight,
            atlas_contract=atlas_contract,
            atlas=atlas,
        )
        if args.mode == "audit":
            # Audit is a truthful readiness packet; it never executes a model
            # forward and therefore remains safe while MX-02 is unfinished.
            summary = {
                **blocked_summary("XAI_FORWARD_DEFERRED_UNTIL_FORMAL_RUN"),
                "status": "READY_FOR_FORMAL_RUN",
                "anatomical_attribution_available": False,
                "preflight": _public_payload(preflight),
                "atlas_contract": _public_payload(atlas_contract),
                "atlas_provenance": _public_payload(atlas),
                "atlas_label_count": len(names),
                "selection": selection,
                "formal_binding": formal_binding,
            }
        else:
            existing = out_dir / "anatomical_xai_summary.json"
            existing_summary = existing
            existing_decision = classify_existing_summary(
                existing,
                expected_binding=formal_binding,
            )
            if existing_decision == "REUSE_COMPLETED":
                summary = _read_json(existing)
                formal_action = "REUSED"
            else:
                # Only an actual new formal forward may append supervision
                # evidence. Audit and immutable reuse remain byte-for-byte and
                # registry read-only.
                enforce_autonomous_supervision(root=ROOT, output_root=corrected_root, report=supervision)
                forward = run_forward_xai(
                    corrected_root=corrected_root,
                    cache_path=DEFAULT_CACHE,
                    split_dir=DEFAULT_SPLITS,
                    atlas_path=Path(atlas["atlas_path"]),
                    region_names=names,
                    preflight=preflight,
                    max_samples_per_fold=int(selection.get("max_samples_per_fold", 0)),
                    device_name="auto",
                    randomization_seeds=tuple(selection.get("randomization_seeds", (13, 29, 47))),
                    bootstrap_resamples=max(1000, int(selection.get("bootstrap_resamples", 1000))),
                )
                summary = {
                    "schema_version": FINAL_SUMMARY_SCHEMA_VERSION,
                    "status": forward.get("status", "BLOCKED_XAI_GATE"),
                    "preflight": _public_payload(preflight),
                    "atlas_contract": _public_payload(atlas_contract),
                    "atlas_provenance": _public_payload(atlas),
                    "atlas_label_count": len(names),
                    "anatomical_attribution_available": forward.get("status") == "COMPLETED",
                    "formal_binding": formal_binding,
                    **forward,
                }
                formal_action = "WRITE" if summary.get("status") == "COMPLETED" else "ATTEMPT_BLOCKED"
    except XAIValidationError as exc:
        summary = blocked_summary(str(exc))
        formal_action = "ATTEMPT_BLOCKED" if args.mode == "run" else "NONE"
    except Exception as exc:  # formal XAI must fail closed on any runtime fault
        summary = blocked_summary(f"XAI_FORWARD_FAILED:{type(exc).__name__}")
        formal_action = "ATTEMPT_BLOCKED" if args.mode == "run" else "NONE"
    if formal_action != "REUSED":
        summary["autonomous_supervision"] = {
            "decision": supervision.get("decision") if supervision else "NOT_EVALUATED",
            "human_approval_required": False,
            "manual_override_allowed": False,
            "fail_closed": True,
        }
        summary = _public_payload(summary)
    report_dir = corrected_root / "reports"
    if args.mode == "audit":
        # Audit is intentionally stdout-only. It must not create readiness
        # files, append registry events, or mutate a formal output directory.
        pass
    elif formal_action == "WRITE":
        try:
            _write_summary(out_dir, summary)
        except XAIValidationError as exc:
            summary = blocked_summary(str(exc))
            summary["autonomous_supervision"] = {
                "decision": supervision.get("decision") if supervision else "NOT_EVALUATED",
                "human_approval_required": False,
                "manual_override_allowed": False,
                "fail_closed": True,
            }
            formal_action = "ATTEMPT_BLOCKED"
            atomic_write_json(report_dir / "BLOCKED_MX06_XAI.json", summary)
    elif formal_action == "ATTEMPT_BLOCKED":
        atomic_write_json(report_dir / "BLOCKED_MX06_XAI.json", summary)
    # REUSED deliberately performs no write: the completed artifact remains
    # byte-for-byte immutable; the reuse marker is stdout-only.
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if args.mode == "audit":
        return 0 if summary.get("status") == "READY_FOR_FORMAL_RUN" else 2
    return 0 if summary.get("status") == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
