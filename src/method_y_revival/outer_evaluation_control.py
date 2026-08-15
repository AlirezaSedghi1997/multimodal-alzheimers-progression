from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable

from .atomic_io import atomic_write_json, read_json
from .contracts import RunContractMismatch, sha256_mapping


class OuterEvaluationControlError(ValueError):
    """Raised before any repeated or contract-inconsistent outer-test access."""


class OuterEvaluationInterruption(RuntimeError):
    """Synthetic crash boundary after an atomic result commit."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha(value: str, label: str) -> str:
    digest = str(value)
    if not _SHA256.fullmatch(digest):
        raise OuterEvaluationControlError(f"INVALID_{label.upper()}_SHA256")
    return digest


def _self_hashed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = sha256_mapping(payload)
    return result


def _read_self_hashed(path: Path, field: str) -> dict[str, Any]:
    payload = read_json(path)
    claimed = payload.get(field)
    body = {key: value for key, value in payload.items() if key != field}
    if claimed != sha256_mapping(body):
        raise OuterEvaluationControlError(f"{field.upper()}_MISMATCH")
    return body


class OuterFoldController:
    """Exactly-once state machine for one already-locked outer fold."""

    def __init__(
        self,
        root: str | Path,
        *,
        fold: int,
        run_contract_sha256: str,
    ) -> None:
        self.fold = int(fold)
        if self.fold < 0:
            raise OuterEvaluationControlError("NONNEGATIVE_OUTER_FOLD_REQUIRED")
        self.run_contract_sha256 = _require_sha(run_contract_sha256, "run_contract")
        self.fold_dir = Path(root) / f"fold_{self.fold}"
        self.state_path = self.fold_dir / "state.json"
        self.result_path = self.fold_dir / "outer_result.json"
        if self.state_path.exists():
            self._load_state()

    def _load_state(self) -> dict[str, Any]:
        state = _read_self_hashed(self.state_path, "state_sha256")
        if state.get("run_contract_sha256") != self.run_contract_sha256:
            raise RunContractMismatch("RUN_CONTRACT_MISMATCH:outer_fold_state")
        if state.get("fold") != self.fold:
            raise OuterEvaluationControlError("OUTER_STATE_FOLD_MISMATCH")
        return state

    def _commit_state(self, state: str, **fields: Any) -> None:
        payload = {
            "schema_version": "2.0",
            "fold": self.fold,
            "run_contract_sha256": self.run_contract_sha256,
            "state": state,
            **fields,
        }
        atomic_write_json(self.state_path, _self_hashed(payload, "state_sha256"))

    def lock_checkpoint(self, checkpoint_sha256: str) -> None:
        checkpoint = _require_sha(checkpoint_sha256, "checkpoint")
        if self.state_path.exists():
            state = self._load_state()
            if state.get("checkpoint_sha256") != checkpoint:
                raise OuterEvaluationControlError("LOCKED_CHECKPOINT_MISMATCH")
            if state.get("state") not in {
                "CHECKPOINT_LOCKED",
                "OUTER_EVAL_IN_PROGRESS",
                "OUTER_EVAL_COMMITTED",
                "FINALIZED",
            }:
                raise OuterEvaluationControlError("OUTER_FOLD_NOT_RELOCKABLE")
            return
        self._commit_state("CHECKPOINT_LOCKED", checkpoint_sha256=checkpoint)

    def _load_result(self, checkpoint_sha256: str) -> dict[str, Any] | None:
        if not self.result_path.exists():
            return None
        payload = _read_self_hashed(self.result_path, "result_record_sha256")
        if payload.get("run_contract_sha256") != self.run_contract_sha256:
            raise RunContractMismatch("RUN_CONTRACT_MISMATCH:outer_result")
        if payload.get("checkpoint_sha256") != checkpoint_sha256:
            raise OuterEvaluationControlError("OUTER_RESULT_CHECKPOINT_MISMATCH")
        if payload.get("fold") != self.fold:
            raise OuterEvaluationControlError("OUTER_RESULT_FOLD_MISMATCH")
        if payload.get("metrics_sha256") != sha256_mapping(payload.get("metrics", {})):
            raise OuterEvaluationControlError("OUTER_METRICS_HASH_MISMATCH")
        if not isinstance(payload.get("metrics"), dict):
            raise OuterEvaluationControlError("OUTER_METRICS_MAPPING_REQUIRED")
        return payload

    def evaluate_once(
        self,
        evaluator: Callable[[], dict[str, Any]],
        *,
        interrupt_after_result_commit: bool = False,
    ) -> dict[str, Any]:
        if not self.state_path.exists():
            raise OuterEvaluationControlError("CHECKPOINT_LOCK_REQUIRED")
        state = self._load_state()
        checkpoint = state.get("checkpoint_sha256")
        if not isinstance(checkpoint, str):
            raise OuterEvaluationControlError("CHECKPOINT_LOCK_REQUIRED")
        committed = self._load_result(checkpoint)
        if committed is not None:
            if state.get("state") != "FINALIZED":
                self._commit_state(
                    "OUTER_EVAL_COMMITTED",
                    checkpoint_sha256=checkpoint,
                    result_record_sha256=sha256_mapping(
                        {
                            key: value
                            for key, value in committed.items()
                            if key != "result_record_sha256"
                        }
                    ),
                )
            return committed["metrics"]
        if state.get("state") == "FAILED":
            raise OuterEvaluationControlError("FAILED_OUTER_FOLD_IS_TERMINAL")
        if state.get("state") == "OUTER_EVAL_IN_PROGRESS":
            self._commit_state(
                "FAILED",
                checkpoint_sha256=checkpoint,
                terminal_reason="AMBIGUOUS_OUTER_ACCESS_WITHOUT_COMMITTED_RESULT",
            )
            raise OuterEvaluationControlError(
                "AMBIGUOUS_OUTER_ACCESS_WITHOUT_COMMITTED_RESULT"
            )
        if state.get("state") != "CHECKPOINT_LOCKED":
            raise OuterEvaluationControlError("INVALID_OUTER_EVALUATION_STATE")
        self._commit_state("OUTER_EVAL_IN_PROGRESS", checkpoint_sha256=checkpoint)
        try:
            metrics = evaluator()
        except Exception as exc:
            self._commit_state(
                "FAILED",
                checkpoint_sha256=checkpoint,
                terminal_reason=f"EVALUATOR_EXCEPTION:{type(exc).__name__}",
            )
            raise
        if not isinstance(metrics, dict):
            self._commit_state(
                "FAILED",
                checkpoint_sha256=checkpoint,
                terminal_reason="OUTER_METRICS_MAPPING_REQUIRED",
            )
            raise OuterEvaluationControlError("OUTER_METRICS_MAPPING_REQUIRED")
        body = {
            "schema_version": "2.0",
            "fold": self.fold,
            "run_contract_sha256": self.run_contract_sha256,
            "checkpoint_sha256": checkpoint,
            "metrics": metrics,
            "metrics_sha256": sha256_mapping(metrics),
        }
        atomic_write_json(
            self.result_path,
            _self_hashed(body, "result_record_sha256"),
        )
        if interrupt_after_result_commit:
            raise OuterEvaluationInterruption("AFTER_OUTER_RESULT_COMMIT")
        self._commit_state(
            "OUTER_EVAL_COMMITTED",
            checkpoint_sha256=checkpoint,
            result_record_sha256=sha256_mapping(body),
        )
        return metrics

    def finalize(self) -> None:
        if not self.state_path.exists():
            raise OuterEvaluationControlError("CHECKPOINT_LOCK_REQUIRED")
        state = self._load_state()
        checkpoint = state.get("checkpoint_sha256")
        if not isinstance(checkpoint, str) or self._load_result(checkpoint) is None:
            raise OuterEvaluationControlError("COMMITTED_OUTER_RESULT_REQUIRED")
        self._commit_state(
            "FINALIZED",
            checkpoint_sha256=checkpoint,
            result_record_sha256=sha256_mapping(
                {
                    key: value
                    for key, value in read_json(self.result_path).items()
                    if key != "result_record_sha256"
                }
            ),
        )


def summarize_outer_evaluation(
    root: str | Path,
    *,
    expected_folds: int,
    run_contract_sha256: str,
) -> dict[str, Any]:
    """Aggregate only when every required fold has one verified result."""

    digest = _require_sha(run_contract_sha256, "run_contract")
    completed: list[dict[str, Any]] = []
    failed: list[int] = []
    missing: list[int] = []
    for fold in range(int(expected_folds)):
        state_path = Path(root) / f"fold_{fold}" / "state.json"
        if not state_path.exists():
            missing.append(fold)
            continue
        try:
            controller = OuterFoldController(root, fold=fold, run_contract_sha256=digest)
            state = controller._load_state()
            checkpoint = state.get("checkpoint_sha256")
            result = (
                controller._load_result(checkpoint)
                if isinstance(checkpoint, str)
                else None
            )
            if state.get("state") in {"OUTER_EVAL_COMMITTED", "FINALIZED"} and result:
                completed.append(result["metrics"])
            elif state.get("state") == "FAILED":
                failed.append(fold)
            else:
                missing.append(fold)
        except (OuterEvaluationControlError, RunContractMismatch):
            failed.append(fold)
    if len(completed) != int(expected_folds):
        return {
            "status": "FAILED_INCOMPLETE_OUTER_CV",
            "expected_outer_folds": int(expected_folds),
            "outer_folds_completed": len(completed),
            "failed_folds": failed,
            "missing_folds": missing,
            "metrics_aggregated": False,
        }
    numeric_keys = sorted(
        set.intersection(
            *[
                {
                    key
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                }
                for metrics in completed
            ]
        )
    )
    summary: dict[str, Any] = {
        "status": "COMPLETED",
        "expected_outer_folds": int(expected_folds),
        "outer_folds_completed": len(completed),
        "metrics_aggregated": True,
    }
    for key in numeric_keys:
        summary[f"mean_{key}"] = sum(float(row[key]) for row in completed) / len(
            completed
        )
    return summary
