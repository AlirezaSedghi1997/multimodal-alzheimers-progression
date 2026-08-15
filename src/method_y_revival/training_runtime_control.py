"""Hash-bound, resumable dispatch for a future Method Y training executor.

This module does not implement or start model training.  It provides the
atomic boundary through which a Task-6-approved executor can later run.  A
successful result is committed once and is returned on resume without a
second executor call.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Protocol

from .atomic_io import atomic_write_json, read_json
from .contracts import RunContractMismatch, sha256_mapping


class TrainingRuntimeControlError(ValueError):
    """Raised before an unbound or non-resumable executor can run."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ResumableTrainingExecutor(Protocol):
    executor_id: str
    implementation_sha256: str
    resume_safe: bool
    test_only: bool

    def run(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value.lower()) is not None


def executor_contract(executor: ResumableTrainingExecutor) -> dict[str, Any]:
    descriptor = {
        "executor_id": getattr(executor, "executor_id", None),
        "implementation_sha256": getattr(executor, "implementation_sha256", None),
        "resume_safe": getattr(executor, "resume_safe", None),
        "test_only": getattr(executor, "test_only", None),
    }
    if (
        not isinstance(descriptor["executor_id"], str)
        or not descriptor["executor_id"].strip()
        or not _is_sha256(descriptor["implementation_sha256"])
        or descriptor["resume_safe"] is not True
        or not isinstance(descriptor["test_only"], bool)
    ):
        raise TrainingRuntimeControlError("RESUMABLE_EXECUTOR_CONTRACT_INVALID")
    return descriptor


def _read_self_hashed(path: Path, field: str) -> dict[str, Any]:
    payload = read_json(path)
    claimed = payload.get(field)
    body = {key: value for key, value in payload.items() if key != field}
    if claimed != sha256_mapping(body):
        raise TrainingRuntimeControlError(f"{field.upper()}_MISMATCH")
    return payload


def _write_self_hashed(path: Path, payload: dict[str, Any], field: str) -> None:
    wrapped = dict(payload)
    wrapped[field] = sha256_mapping(payload)
    atomic_write_json(path, wrapped)


def _validated_executor_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "status",
        "stage",
        "completed_units",
        "failed_units",
        "terminal_reason",
        "artifact_sha256",
    }
    if set(payload).difference(allowed):
        raise TrainingRuntimeControlError("TRAINING_EXECUTOR_RESULT_SCHEMA_INVALID")
    status = payload.get("status")
    if not isinstance(status, str) or not status.strip():
        raise TrainingRuntimeControlError("TRAINING_EXECUTOR_RESULT_STATUS_REQUIRED")
    for name in ("completed_units", "failed_units"):
        value = payload.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise TrainingRuntimeControlError("TRAINING_EXECUTOR_RESULT_SCHEMA_INVALID")
    if "stage" in payload and not isinstance(payload["stage"], str):
        raise TrainingRuntimeControlError("TRAINING_EXECUTOR_RESULT_SCHEMA_INVALID")
    if (
        "terminal_reason" in payload
        and payload["terminal_reason"] is not None
        and not isinstance(payload["terminal_reason"], str)
    ):
        raise TrainingRuntimeControlError("TRAINING_EXECUTOR_RESULT_SCHEMA_INVALID")
    artifacts = payload.get("artifact_sha256", {})
    if not isinstance(artifacts, Mapping) or any(
        Path(str(name)).name != str(name) or not _is_sha256(digest)
        for name, digest in artifacts.items()
    ):
        raise TrainingRuntimeControlError("TRAINING_EXECUTOR_RESULT_SCHEMA_INVALID")
    return dict(payload)


def dispatch_training_runtime(
    *,
    output_root: str | Path,
    training_control_contract_sha256: str,
    receipt_sha256: str,
    receipt_executor_contract: Mapping[str, Any],
    executor: ResumableTrainingExecutor | None,
) -> dict[str, Any]:
    """Dispatch exactly one hash-bound executor result, recovering atomically."""

    if not _is_sha256(training_control_contract_sha256) or not _is_sha256(
        receipt_sha256
    ):
        raise TrainingRuntimeControlError("RUNTIME_PARENT_DIGEST_INVALID")
    if executor is None:
        raise TrainingRuntimeControlError("HASH_BOUND_TRAINING_EXECUTOR_REQUIRED")
    actual_executor = executor_contract(executor)
    if actual_executor["test_only"] is not True:
        raise TrainingRuntimeControlError("SYNTHETIC_SMOKE_ONLY_EXECUTOR_REQUIRED")
    if dict(receipt_executor_contract) != actual_executor:
        raise TrainingRuntimeControlError("TRAINING_EXECUTOR_CONTRACT_MISMATCH")

    root = Path(output_root)
    state_path = root / "training_runtime_state.json"
    result_path = root / "training_runtime_result.json"
    expected_parent = {
        "training_control_contract_sha256": training_control_contract_sha256,
        "pretraining_readiness_receipt_sha256": receipt_sha256,
        "training_executor": actual_executor,
    }

    state: dict[str, Any] | None = None
    if state_path.is_file():
        state_payload = _read_self_hashed(state_path, "state_sha256")
        state = {
            key: value for key, value in state_payload.items() if key != "state_sha256"
        }
        if any(state.get(key) != value for key, value in expected_parent.items()):
            raise RunContractMismatch("RUN_CONTRACT_MISMATCH:training_runtime_state")

    if result_path.is_file():
        result_payload = _read_self_hashed(result_path, "result_sha256")
        result = {
            key: value for key, value in result_payload.items() if key != "result_sha256"
        }
        if any(result.get(key) != value for key, value in expected_parent.items()):
            raise RunContractMismatch("RUN_CONTRACT_MISMATCH:training_runtime_result")
        if state is None or state.get("status") not in {
            "TRAINING_RUNTIME_IN_PROGRESS",
            "TRAINING_RUNTIME_COMMITTED",
        }:
            raise TrainingRuntimeControlError("RUNTIME_RESULT_WITHOUT_VALID_STATE")
        if state.get("status") == "TRAINING_RUNTIME_IN_PROGRESS":
            committed_state = {
                "schema_version": "2.0",
                "status": "TRAINING_RUNTIME_COMMITTED",
                **expected_parent,
                "result_sha256": result_payload["result_sha256"],
            }
            _write_self_hashed(state_path, committed_state, "state_sha256")
        elif state.get("result_sha256") != result_payload["result_sha256"]:
            raise TrainingRuntimeControlError("RUNTIME_STATE_RESULT_HASH_MISMATCH")
        return result

    if state is not None and state.get("status") == "TRAINING_RUNTIME_COMMITTED":
        raise TrainingRuntimeControlError("COMMITTED_RUNTIME_RESULT_MISSING")
    in_progress = {
        "schema_version": "2.0",
        "status": "TRAINING_RUNTIME_IN_PROGRESS",
        **expected_parent,
        "resume_policy": "same_hash_bound_resume_safe_executor",
    }
    _write_self_hashed(state_path, in_progress, "state_sha256")
    context = {
        **expected_parent,
        "output_root": str(root.resolve()),
        "resume": True,
    }
    executor_result = executor.run(context)
    if not isinstance(executor_result, Mapping):
        raise TrainingRuntimeControlError("TRAINING_EXECUTOR_RESULT_MUST_BE_MAPPING")
    validated_executor_result = _validated_executor_result(executor_result)
    result = {
        "schema_version": "2.0",
        **expected_parent,
        "executor_result": validated_executor_result,
    }
    _write_self_hashed(result_path, result, "result_sha256")
    committed_result = _read_self_hashed(result_path, "result_sha256")
    committed_state = {
        "schema_version": "2.0",
        "status": "TRAINING_RUNTIME_COMMITTED",
        **expected_parent,
        "result_sha256": committed_result["result_sha256"],
    }
    _write_self_hashed(state_path, committed_state, "state_sha256")
    return result
