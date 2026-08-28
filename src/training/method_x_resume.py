from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.training.method_x_outer_compatibility import (
    HistoricalCompatibilityAuthority,
)
from src.utils.io_helpers import atomic_write_json
from src.utils.reproducibility import canonical_digest


RUN_CONTRACT_SCHEMA_VERSION = 3
RUNTIME_RESUME_SCHEMA_VERSION = 1

# Loader keys that change data order or exact-resume semantics. These are
# scientific even though they live under "loader": a batch or worker change
# alters the sequence the model sees, so it can never be waved through as
# benign runtime plumbing.
SCIENTIFIC_LOADER_KEYS = frozenset({"batch_size", "num_workers", "persistent_workers"})

# Parameter keys that describe the toolchain rather than the experiment.
NON_SCIENTIFIC_PARAMETER_KEYS = frozenset({"source_contract_digest", "runtime_versions"})

_ALLOWED_H_MIRROR_POLICIES = frozenset({"epoch", "fold", "epoch_and_fold"})
_ALLOWED_REPLAY_POLICIES = frozenset({"deterministic"})
_ALLOWED_STAGING_ROOT_CLASSES = frozenset(
    {"C_ALLOWLISTED_CHECKPOINT_STAGING", "H_ONLY"}
)

FOLD_STAGES = (
    "TRAINING",
    "CHECKPOINT_LOCKED",
    "OUTER_EVAL_IN_PROGRESS",
    "OUTER_EVAL_DONE",
    "FINALIZED",
)

# Cadence-1 staging writes a weight-only sidecar next to progress. The source-
# only migrator used to glob ``method_x_3d_fold_*.pt`` and demand a run
# contract inside that sidecar, which is exactly the live ResNet18 resume
# failure after the first best-val write.
_FOLD_CHECKPOINT_NAME = re.compile(
    r"^method_x_3d_fold_(\d+)(?:_(progress|locked|best_state))?\.pt$"
)


def parse_fold_checkpoint_filename(filename: str) -> tuple[int, str]:
    """Return ``(fold, kind)`` for a trainer checkpoint basename.

    ``kind`` is ``final``, ``progress``, ``locked``, or ``best_state``. Any
    other ``method_x_3d_fold_*.pt`` name is unknown and must stay blocked.
    """

    match = _FOLD_CHECKPOINT_NAME.fullmatch(Path(str(filename)).name)
    if match is None:
        raise RuntimeError(f"RUN_CONTRACT_MIGRATION_INVALID_CHECKPOINT_NAME:{filename}")
    suffix = match.group(2)
    return int(match.group(1)), "final" if suffix is None else suffix


def assert_offloaded_best_state_sidecar(payload: Any, *, filename: str) -> None:
    """Offloaded best weights are not a run-contract checkpoint.

    ``CheckpointStager.record_best_state`` writes only weight tensors plus
    metric metadata. Rewriting that file with a run contract would break the
    provenance sidecar hash and ``load_best_state``.
    """

    if not isinstance(payload, dict):
        raise RuntimeError(f"RUN_CONTRACT_MIGRATION_BEST_STATE_INVALID:{filename}")
    if "run_contract" in payload or "run_contract_digest" in payload:
        raise RuntimeError(
            f"RUN_CONTRACT_MIGRATION_BEST_STATE_CONTRACT_PRESENT:{filename}"
        )
    if "best_model_state_dict" not in payload and "best_averaged_state_dict" not in payload:
        raise RuntimeError(f"RUN_CONTRACT_MIGRATION_BEST_STATE_WEIGHTS_MISSING:{filename}")


def classify_source_only_progress_bind(
    *,
    filename: str,
    fold: int,
    action: str,
    finalized_folds: set[int] | frozenset[int],
) -> str:
    """Classify a globbed ``*.pt`` during source-only contract migration.

    Live batteries keep leftover ``method_x_3d_fold_{N}_progress.pt`` beside the
    immutable finalized ``method_x_3d_fold_{N}.pt``. Those leftovers are not
    Task 2 resume sources. Binding or rewriting them is a scientific-adjacent
    mutation of historical bytes. Only the current TRAINING fold's progress
    file may be rewritten and bound.
    """

    if not str(filename).endswith("_progress.pt"):
        return "not_progress"
    if fold in finalized_folds:
        if action != "keep_finalized":
            raise RuntimeError(
                f"RUN_CONTRACT_MIGRATION_FINALIZED_PROGRESS_NOT_FROZEN:{filename}"
            )
        return "ignore_leftover_finalized"
    if action == "keep_finalized":
        raise RuntimeError("RUN_CONTRACT_MIGRATION_PROGRESS_MARKED_FINALIZED")
    if action not in {"rewrite", "keep"}:
        raise RuntimeError(f"RUN_CONTRACT_MIGRATION_PROGRESS_ACTION_INVALID:{filename}")
    return "bind_current"
_ALLOWED_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"TRAINING"},
    "TRAINING": {"TRAINING", "CHECKPOINT_LOCKED"},
    "CHECKPOINT_LOCKED": {"CHECKPOINT_LOCKED", "OUTER_EVAL_IN_PROGRESS", "FINALIZED"},
    "OUTER_EVAL_IN_PROGRESS": {"OUTER_EVAL_DONE"},
    "OUTER_EVAL_DONE": {"OUTER_EVAL_DONE", "FINALIZED"},
    "FINALIZED": {"FINALIZED"},
}


def _contract_body(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in contract.items() if key != "digest"}


def _flatten_contract(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_contract(value[key], child))
        return flattened
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def _is_scientific_key(key: str) -> bool:
    if key.startswith("loader."):
        return key.split(".", 1)[1] in SCIENTIFIC_LOADER_KEYS
    if key.startswith("parameters."):
        field = key.split(".", 1)[1]
        return not any(
            field == name or field.startswith(f"{name}.")
            for name in NON_SCIENTIFIC_PARAMETER_KEYS
        )
    return key in {
        "cache_sha256",
        "split_digest",
        "spm_snapshot_sha256",
        "folds",
        "epochs",
        "seed",
    }


def contract_diff(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Explain a resume mismatch without exposing identifiers or raw paths."""

    found = payload.get("run_contract") if isinstance(payload, Mapping) else None
    found_body = _contract_body(found) if isinstance(found, Mapping) else {}
    expected_body = _contract_body(expected) if isinstance(expected, Mapping) else {}
    found_flat = _flatten_contract(found_body)
    expected_flat = _flatten_contract(expected_body)
    changed = sorted({*found_flat, *expected_flat})
    scientific: list[str] = []
    source: list[str] = []
    runtime: list[str] = []
    other: list[str] = []
    for key in changed:
        if found_flat.get(key) == expected_flat.get(key):
            continue
        if "source_contract_digest" in key:
            source.append(key)
        elif _is_scientific_key(key):
            scientific.append(key)
        elif "runtime_versions" in key or key.startswith("loader."):
            runtime.append(key)
        else:
            other.append(key)
    return {
        "schema_version": 1,
        "scientific_changes": scientific,
        "source_changes": source,
        "runtime_changes": runtime,
        "other_changes": other,
        "found_digest": canonical_digest(found_body) if found_body else None,
        "expected_digest": canonical_digest(expected_body) if expected_body else None,
        "resume_allowed": not scientific and not other,
        "source_only_migration_allowed": bool(source) and not scientific and not other,
    }


def build_run_contract(
    *,
    cache_sha256: str,
    split_digest: str,
    folds: int,
    epochs: int,
    seed: int,
    parameters: Mapping[str, Any],
    loader: Mapping[str, Any],
    spm_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    normalized_spm_digest = str(spm_snapshot_sha256 or "").strip().lower() or None
    if normalized_spm_digest is not None and (
        len(normalized_spm_digest) != 64
        or any(char not in "0123456789abcdef" for char in normalized_spm_digest)
    ):
        raise ValueError("INVALID_SPM_SNAPSHOT_SHA256")
    body = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "cache_sha256": str(cache_sha256),
        "split_digest": str(split_digest),
        "spm_snapshot_sha256": normalized_spm_digest,
        "folds": int(folds),
        "epochs": int(epochs),
        "seed": int(seed),
        "parameters": dict(parameters),
        "loader": dict(loader),
    }
    return {**body, "digest": canonical_digest(body)}


def scientific_contract_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Project a run contract onto the fields that define the experiment.

    Everything the toolchain owns (source digest, library versions, prefetch and
    pinning plumbing) is dropped so that a pure runtime change leaves this
    projection - and therefore the scientific digest - untouched.
    """

    body = _contract_body(contract) if isinstance(contract, Mapping) else {}
    parameters = {
        key: value
        for key, value in (body.get("parameters") or {}).items()
        if key not in NON_SCIENTIFIC_PARAMETER_KEYS
    }
    loader = {
        key: value
        for key, value in (body.get("loader") or {}).items()
        if key in SCIENTIFIC_LOADER_KEYS
    }
    return {
        "schema_version": body.get("schema_version"),
        "cache_sha256": body.get("cache_sha256"),
        "split_digest": body.get("split_digest"),
        "spm_snapshot_sha256": body.get("spm_snapshot_sha256"),
        "folds": body.get("folds"),
        "epochs": body.get("epochs"),
        "seed": body.get("seed"),
        "parameters": parameters,
        "loader": loader,
    }


def scientific_contract_digest(contract: Mapping[str, Any]) -> str:
    """Digest of the experiment definition alone.

    A change here is never migrated in place; it requires a new versioned run.
    """

    return canonical_digest(scientific_contract_projection(contract))


def build_runtime_resume_contract(
    *,
    staging_root_class: str,
    full_cadence: int,
    h_mirror_policy: str,
    journal_enabled: bool,
    replay_policy: str,
) -> dict[str, Any]:
    """Describe where and how often resume state is written.

    Only a policy class is recorded, never a concrete staging path, so no local
    user path is embedded in checkpoints or mismatch reports.
    """

    if str(staging_root_class) not in _ALLOWED_STAGING_ROOT_CLASSES:
        raise ValueError(f"INVALID_STAGING_ROOT_CLASS:{staging_root_class}")
    if int(full_cadence) < 1:
        raise ValueError(f"INVALID_CHECKPOINT_FULL_CADENCE:{full_cadence}")
    if str(h_mirror_policy) not in _ALLOWED_H_MIRROR_POLICIES:
        raise ValueError(f"INVALID_H_MIRROR_POLICY:{h_mirror_policy}")
    if str(replay_policy) not in _ALLOWED_REPLAY_POLICIES:
        raise ValueError(f"INVALID_RESUME_REPLAY_POLICY:{replay_policy}")
    body = {
        "schema_version": RUNTIME_RESUME_SCHEMA_VERSION,
        "staging_root_class": str(staging_root_class),
        "full_cadence": int(full_cadence),
        "h_mirror_policy": str(h_mirror_policy),
        "journal_enabled": bool(journal_enabled),
        "replay_policy": str(replay_policy),
    }
    return {**body, "digest": canonical_digest(body)}


def runtime_resume_digest(contract: Mapping[str, Any]) -> str:
    return canonical_digest(_contract_body(contract))


def storage_policy_digest(path: str | Path) -> str:
    """Bind the active storage policy file into the resume contract."""

    policy_path = Path(path)
    if not policy_path.is_file():
        raise RuntimeError(f"STORAGE_POLICY_MISSING:{policy_path.name}")
    return hashlib.sha256(policy_path.read_bytes()).hexdigest()


def classify_contract_change(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    expected_runtime: Mapping[str, Any] | None = None,
    expected_storage_digest: str | None = None,
) -> dict[str, Any]:
    """Decide between RESUME, AUDITED_MIGRATION, and NEW_VERSIONED_RUN.

    A scientific change always outranks runtime, source, and storage-policy
    changes, so a mixed diff can never be downgraded to a migration.
    """

    diff = contract_diff(payload, expected)
    found_contract = payload.get("run_contract") if isinstance(payload, Mapping) else None
    found_runtime = payload.get("runtime_resume_contract") if isinstance(payload, Mapping) else None
    legacy_runtime_absent = expected_runtime is not None and not isinstance(
        found_runtime, Mapping
    )
    runtime_changes = list(diff["runtime_changes"])
    if isinstance(found_runtime, Mapping) and expected_runtime is not None:
        if runtime_resume_digest(found_runtime) != runtime_resume_digest(expected_runtime):
            runtime_changes.append("runtime_resume_contract")
    storage_changed = bool(
        expected_storage_digest is not None
        and payload.get("storage_policy_digest") != expected_storage_digest
    )
    blocking = list(diff["scientific_changes"]) + list(diff["other_changes"])
    if blocking:
        decision = "NEW_VERSIONED_RUN"
    elif diff["source_changes"] or runtime_changes or storage_changed or legacy_runtime_absent:
        decision = "AUDITED_MIGRATION"
    else:
        decision = "RESUME"
    return {
        "schema_version": 1,
        "decision": decision,
        "scientific_changes": list(diff["scientific_changes"]),
        "source_changes": list(diff["source_changes"]),
        "runtime_changes": sorted(runtime_changes),
        "other_changes": list(diff["other_changes"]),
        "storage_policy_changed": storage_changed,
        "legacy_runtime_contract_absent": legacy_runtime_absent,
        "scientific_digest_expected": scientific_contract_digest(expected),
        "scientific_digest_found": scientific_contract_digest(found_contract)
        if isinstance(found_contract, Mapping)
        else None,
    }


def append_source_history(
    path: str | Path,
    *,
    old_digest: str,
    new_digest: str,
    decision: str,
    equality_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one audited migration record. Never rewrites earlier entries."""

    if str(decision) != "AUDITED_MIGRATION":
        raise RuntimeError(f"SCIENTIFIC_MIGRATION_FORBIDDEN:{decision}")
    amendment_path = Path(path)
    history: list[Any] = []
    if amendment_path.is_file():
        existing = json.loads(amendment_path.read_text(encoding="utf-8"))
        if isinstance(existing, Mapping):
            history = list(existing.get("source_history") or [])
    history.append(
        {
            "old_digest": str(old_digest),
            "new_digest": str(new_digest),
            "decision": str(decision),
            "equality_report": dict(equality_report),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    amendment = {
        "schema_version": 1,
        "append_only": True,
        "source_history": history,
    }
    atomic_write_json(amendment_path, amendment)
    return amendment


def source_history_binds_digest(
    *,
    history: Any,
    from_digest: str,
    to_digest: str,
) -> bool:
    """True iff a unique successor chain walks ``from_digest`` to ``to_digest``.

    Forks, cycles, missing successors, and non-sha256 digests are False.
    Equal digests bind with no history. This is the same unique-path rule the
    autosupervisor uses; it does not waive a scientific contract mismatch.
    """

    if not _is_sha256(from_digest) or not _is_sha256(to_digest):
        return False
    if from_digest == to_digest:
        return True
    if not isinstance(history, list):
        return False
    current = from_digest
    seen = {current}
    while current != to_digest:
        successors = [
            item
            for item in history
            if isinstance(item, Mapping)
            and item.get("from_run_contract_digest") == current
        ]
        if len(successors) != 1:
            return False
        nxt = str(successors[0].get("to_run_contract_digest") or "")
        if not _is_sha256(nxt) or nxt in seen:
            return False
        seen.add(nxt)
        current = nxt
    return True


def assert_resume_contract(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    source_history: Any = None,
) -> None:
    found = payload.get("run_contract")
    found_digest = payload.get("run_contract_digest")
    expected_body = _contract_body(expected)
    expected_digest = canonical_digest(expected_body)
    if not isinstance(found, Mapping):
        raise RuntimeError("RUN_CONTRACT_MISSING")
    found_body = _contract_body(found)
    actual_found_digest = canonical_digest(found_body)
    internally_consistent = (
        found_digest == actual_found_digest
        and found.get("digest") == actual_found_digest
        and expected.get("digest") == expected_digest
    )
    exact_match = internally_consistent and actual_found_digest == expected_digest
    source_only_ancestor = (
        internally_consistent
        and source_history is not None
        and source_contract_migration_allowed(payload, expected)
        and source_history_binds_digest(
            history=source_history,
            from_digest=actual_found_digest,
            to_digest=expected_digest,
        )
    )
    if exact_match or source_only_ancestor:
        return
    diff = contract_diff(payload, expected)
    raise RuntimeError(
        "RUN_CONTRACT_MISMATCH:"
        + json.dumps(
            {"expected": expected_digest, "found": actual_found_digest, "contract_diff": diff},
            sort_keys=True,
        )
    )


def source_contract_migration_allowed(
    payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Allow only a source-code digest refresh when scientific settings match.

    Resume mechanics and non-semantic loader plumbing can change after a
    checkpoint is written. Cache, split, seed, folds, epochs, model and every
    training parameter must remain byte-for-byte contract-compatible.
    """

    found = payload.get("run_contract") if isinstance(payload, Mapping) else None
    if not isinstance(found, Mapping) or not isinstance(expected, Mapping):
        return False
    found_body = dict(_contract_body(found))
    expected_body = dict(_contract_body(expected))
    found_parameters = dict(found_body.get("parameters") or {})
    expected_parameters = dict(expected_body.get("parameters") or {})
    found_source = found_parameters.pop("source_contract_digest", None)
    expected_source = expected_parameters.pop("source_contract_digest", None)
    if not found_source or not expected_source or found_source == expected_source:
        return False
    found_body["parameters"] = found_parameters
    expected_body["parameters"] = expected_parameters
    return found_body == expected_body


def read_fold_state(path: str | Path) -> dict[str, Any] | None:
    state_path = Path(path)
    if not state_path.exists():
        return None
    value = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("stage") not in FOLD_STAGES:
        raise RuntimeError(f"INVALID_FOLD_STATE:{state_path}")
    return value


def transition_fold_state(
    path: str | Path,
    *,
    fold: int,
    target: str,
    contract_digest: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if target not in FOLD_STAGES:
        raise RuntimeError(f"UNKNOWN_FOLD_STATE:{target}")
    current = read_fold_state(path)
    previous = current.get("stage") if current else None
    if current and int(current.get("fold", -1)) != int(fold):
        raise RuntimeError("FOLD_STATE_FOLD_MISMATCH")
    if target not in _ALLOWED_TRANSITIONS[previous]:
        raise RuntimeError(f"ILLEGAL_FOLD_STATE_TRANSITION:{previous}->{target}")
    if current and current.get("contract_digest") != contract_digest:
        raise RuntimeError("FOLD_STATE_RUN_CONTRACT_MISMATCH")
    state = {
        "schema_version": 1,
        "fold": int(fold),
        "stage": target,
        "previous_stage": previous,
        "contract_digest": str(contract_digest),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "details": dict(details or {}),
    }
    atomic_write_json(path, state)
    return state


def outer_evaluation_resume_action(stage: str | None, *, locked_result_exists: bool) -> str:
    if stage in {None, "TRAINING"}:
        raise RuntimeError(f"OUTER_EVALUATION_NOT_CHECKPOINT_LOCKED:{stage}")
    if stage == "CHECKPOINT_LOCKED" and not locked_result_exists:
        return "RUN"
    if stage in {"OUTER_EVAL_DONE", "FINALIZED"} and locked_result_exists:
        return "REUSE"
    if stage == "OUTER_EVAL_IN_PROGRESS" and locked_result_exists:
        return "REUSE_COMMITTED_RECOVER_STATE"
    if stage == "OUTER_EVAL_IN_PROGRESS" and not locked_result_exists:
        raise RuntimeError("AMBIGUOUS_OUTER_EVALUATION_STATE")
    raise RuntimeError(f"OUTER_EVALUATION_STATE_INCONSISTENT:{stage}")


def build_locked_outer_result(
    *,
    fold: int,
    run_contract_digest: str,
    split_membership_digest: str,
    y_true: list[int],
    y_prob: list[float],
    test_logits: list[float],
) -> dict[str, Any]:
    body = {
        "fold": int(fold),
        "evaluation_count": 1,
        "run_contract_digest": str(run_contract_digest),
        "split_membership_digest": str(split_membership_digest),
        "y_true": [int(value) for value in y_true],
        "y_prob": [float(value) for value in y_prob],
        "test_logits": [float(value) for value in test_logits],
        "raw_identifiers_stored": False,
    }
    return {**body, "result_digest": canonical_digest(body)}


def validate_locked_outer_result(
    locked_result: Mapping[str, Any],
    *,
    expected_fold: int,
    expected_contract_digest: str,
    expected_split_membership_digest: str | None = None,
    expected_count: int | None = None,
) -> dict[str, Any]:
    body = {key: value for key, value in locked_result.items() if key != "result_digest"}
    labels = locked_result.get("y_true")
    probabilities = locked_result.get("y_prob")
    logits = locked_result.get("test_logits")
    valid = (
        int(locked_result.get("fold", -1)) == int(expected_fold)
        and locked_result.get("run_contract_digest") == expected_contract_digest
        and int(locked_result.get("evaluation_count", 0)) == 1
        and isinstance(locked_result.get("split_membership_digest"), str)
        and len(locked_result["split_membership_digest"]) == 64
        and (expected_split_membership_digest is None or locked_result["split_membership_digest"] == expected_split_membership_digest)
        and isinstance(labels, list)
        and isinstance(probabilities, list)
        and isinstance(logits, list)
        and len(labels) > 0
        and len(labels) == len(probabilities) == len(logits)
        and (expected_count is None or len(labels) == int(expected_count))
        and all(int(value) in {0, 1} for value in labels)
        and all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in probabilities)
        and all(math.isfinite(float(value)) for value in logits)
        and locked_result.get("raw_identifiers_stored") is False
        and locked_result.get("result_digest") == canonical_digest(body)
    )
    if not valid:
        raise RuntimeError("COMMITTED_OUTER_EVALUATION_INVALID")
    return {
        "y_true": [int(value) for value in locked_result["y_true"]],
        "y_prob": [float(value) for value in locked_result["y_prob"]],
        "test_logits": [float(value) for value in locked_result["test_logits"]],
        "split_membership_digest": str(locked_result["split_membership_digest"]),
        "result_digest": str(locked_result["result_digest"]),
    }


def outer_prediction_identity_digest(locked_result: Mapping[str, Any]) -> str:
    """Hash the immutable, subject-aligned prediction identity of one outer fold.

    The digest deliberately excludes migration metadata and metrics.  It proves
    that a source-only metadata migration did not recompute or reorder the
    once-opened outer-test predictions.
    """

    return canonical_digest(
        {
            "fold": locked_result.get("fold"),
            "evaluation_count": locked_result.get("evaluation_count"),
            "split_membership_digest": locked_result.get("split_membership_digest"),
            "y_true": locked_result.get("y_true"),
            "y_prob": locked_result.get("y_prob"),
            "test_logits": locked_result.get("test_logits"),
        }
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _validate_historical_outer_compatibility(
    *,
    checkpoint: Mapping[str, Any],
    committed: Mapping[str, Any],
    locked_result: Mapping[str, Any],
    locked: Mapping[str, Any],
    expected_fold: int,
    expected_contract_digest: str,
    checkpoint_sha256: str | None,
    outer_file_sha256: str | None,
    evidence: HistoricalCompatibilityAuthority | None,
) -> bool:
    """Validate the narrow receipt escape hatch for the five frozen artifacts.

    This is intentionally not a general mismatch waiver.  Every byte-level and
    prediction-level identity in the externally pinned receipt must match, and
    the outer loader invocation count must remain zero.
    """

    required_keys = {
        "schema_version",
        "artifact",
        "scope",
        "backbone",
        "fold",
        "checkpoint_sha256",
        "outer_file_sha256",
        "run_contract_digest",
        "split_membership_digest",
        "evaluation_count",
        "embedded_result_digest",
        "external_result_digest",
        "prediction_identity_digest",
        "outer_loader_invocation_count",
    }
    if not isinstance(evidence, HistoricalCompatibilityAuthority):
        return False
    evidence_payload = evidence.evidence
    if set(evidence_payload) != required_keys:
        return False
    embedded_digest = committed.get("result_digest")
    return bool(
        evidence_payload.get("schema_version") == 1
        and evidence_payload.get("artifact")
        == "method_x_historical_outer_result_compatibility_entry"
        and evidence_payload.get("scope") == "EXACT_FIVE_FROZEN_ARTIFACTS_ONLY"
        and evidence_payload.get("backbone") == checkpoint.get("backbone")
        and type(evidence_payload.get("fold")) is int
        and evidence_payload.get("fold") == int(expected_fold)
        and type(evidence_payload.get("evaluation_count")) is int
        and evidence_payload.get("evaluation_count") == 1
        and type(evidence_payload.get("outer_loader_invocation_count")) is int
        and evidence_payload.get("outer_loader_invocation_count") == 0
        and _is_sha256(checkpoint_sha256)
        and _is_sha256(outer_file_sha256)
        and evidence_payload.get("checkpoint_sha256") == checkpoint_sha256
        and evidence_payload.get("outer_file_sha256") == outer_file_sha256
        and evidence_payload.get("run_contract_digest") == expected_contract_digest
        and evidence_payload.get("split_membership_digest")
        == locked.get("split_membership_digest")
        and evidence_payload.get("embedded_result_digest") == embedded_digest
        and evidence_payload.get("external_result_digest") == locked.get("result_digest")
        and _is_sha256(embedded_digest)
        and embedded_digest != locked.get("result_digest")
        and evidence_payload.get("prediction_identity_digest")
        == outer_prediction_identity_digest(locked_result)
        and committed.get("y_true") == locked.get("y_true")
        and committed.get("y_prob") == locked.get("y_prob")
        and committed.get("test_logits") == locked.get("test_logits")
        and committed.get("split_membership_digest")
        == locked.get("split_membership_digest")
    )


def reuse_committed_outer_result(
    checkpoint: Mapping[str, Any],
    locked_result: Mapping[str, Any],
    *,
    expected_fold: int,
    expected_contract_digest: str,
    checkpoint_sha256: str | None = None,
    outer_file_sha256: str | None = None,
    compatibility_evidence: HistoricalCompatibilityAuthority | None = None,
) -> dict[str, Any]:
    """Validate and return a committed outer result without touching the test loader."""

    committed = checkpoint.get("locked_outer_test_evaluation")
    locked = validate_locked_outer_result(
        locked_result,
        expected_fold=expected_fold,
        expected_contract_digest=expected_contract_digest,
        expected_split_membership_digest=str(checkpoint.get("split_membership_digest", "")),
    )
    common_valid = (
        isinstance(committed, Mapping)
        and int(checkpoint.get("fold", -1)) == int(expected_fold)
        and checkpoint.get("run_contract_digest") == expected_contract_digest
        and int(committed.get("evaluation_count", 0)) == 1
        and isinstance(committed.get("metrics"), Mapping)
        and committed.get("y_true") == locked["y_true"]
        and committed.get("y_prob") == locked["y_prob"]
        and committed.get("test_logits") == locked["test_logits"]
        and committed.get("split_membership_digest") == locked["split_membership_digest"]
    )
    exact_digest_match = bool(
        common_valid and committed.get("result_digest") == locked["result_digest"]
    )
    compatibility_match = bool(
        common_valid
        and _validate_historical_outer_compatibility(
            checkpoint=checkpoint,
            committed=committed,
            locked_result=locked_result,
            locked=locked,
            expected_fold=expected_fold,
            expected_contract_digest=expected_contract_digest,
            checkpoint_sha256=checkpoint_sha256,
            outer_file_sha256=outer_file_sha256,
            evidence=compatibility_evidence,
        )
    )
    valid = exact_digest_match or compatibility_match
    if not valid:
        raise RuntimeError("COMMITTED_OUTER_EVALUATION_INVALID")
    calibrated = committed.get("calibrated_prob")
    if calibrated is not None and (
        not isinstance(calibrated, list) or len(calibrated) != len(locked["y_true"])
    ):
        raise RuntimeError("COMMITTED_OUTER_EVALUATION_INVALID")
    return {
        "metrics": dict(committed["metrics"]),
        **locked,
        "calibrated_prob": [float(value) for value in calibrated] if calibrated is not None else [],
        "compatibility_status": (
            "EXACT_FROZEN_HISTORICAL_RECEIPT"
            if compatibility_match
            else "EXACT_EMBEDDED_EXTERNAL_MATCH"
        ),
    }


def validate_exact_resume_loader_contract(
    *,
    required: bool,
    monai_augment: bool,
    num_workers: int,
    persistent_workers: bool,
) -> None:
    """Reject loader modes whose process-local augmentation state cannot be restored exactly."""

    if not required:
        return
    if persistent_workers:
        raise RuntimeError("EXACT_RESUME_PERSISTENT_WORKERS_UNSUPPORTED")
    if monai_augment and int(num_workers) != 0:
        raise RuntimeError("EXACT_RESUME_MONAI_REQUIRES_NUM_WORKERS_ZERO")
