from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .atomic_io import atomic_write_json, read_json
from .contracts import sha256_mapping


class EvaluationContractError(ValueError):
    """Raised instead of silently publishing partial or repeated outer evaluation."""


def valid_survival_horizons(
    *,
    train_times: np.ndarray,
    test_times: np.ndarray,
    requested_horizons: Iterable[float],
    min_at_risk: int,
) -> dict[float, dict[str, Any]]:
    train = np.asarray(train_times, dtype=float)
    test = np.asarray(test_times, dtype=float)
    train = train[np.isfinite(train) & (train > 0)]
    test = test[np.isfinite(test) & (test > 0)]
    if train.size == 0 or test.size == 0:
        raise EvaluationContractError("NO_POSITIVE_FINITE_FOLLOWUP")
    lower = max(float(train.min()), float(test.min()))
    upper = min(float(train.max()), float(test.max()))
    if lower >= upper:
        raise EvaluationContractError("NO_COMMON_SURVIVAL_SUPPORT")
    result: dict[float, dict[str, Any]] = {}
    for requested in requested_horizons:
        horizon = float(requested)
        at_risk = int(np.sum(test >= horizon))
        estimable = lower < horizon < upper and at_risk >= int(min_at_risk)
        result[horizon] = {
            "status": "ESTIMABLE" if estimable else "NOT_ESTIMABLE",
            "n_at_risk": at_risk,
            "common_support": [lower, upper],
        }
    return result


def require_complete_outer_cv(
    fold_results: Iterable[dict[str, Any]], *, expected_folds: int
) -> list[dict[str, Any]]:
    rows = list(fold_results)
    observed = {int(row["fold"]) for row in rows if row.get("status") == "COMPLETED"}
    if len(rows) != expected_folds or observed != set(range(expected_folds)):
        raise EvaluationContractError(
            f"FAILED_INCOMPLETE_OUTER_CV:expected={expected_folds}:completed={sorted(observed)}"
        )
    return rows


class OuterEvaluationStore:
    def __init__(self, root: str | Path, *, fold: int, run_contract_digest: str):
        self.fold = int(fold)
        self.run_contract_digest = str(run_contract_digest)
        self.fold_dir = Path(root) / f"fold_{self.fold}"
        self.state_path = self.fold_dir / "state.json"
        self.result_path = self.fold_dir / "outer_result.json"

    def _load_committed(self, checkpoint_digest: str) -> dict[str, Any] | None:
        if not self.result_path.exists():
            return None
        payload = read_json(self.result_path)
        if payload.get("run_contract_digest") != self.run_contract_digest:
            raise EvaluationContractError("OUTER_RESULT_RUN_CONTRACT_MISMATCH")
        if payload.get("checkpoint_digest") != checkpoint_digest:
            raise EvaluationContractError("OUTER_RESULT_CHECKPOINT_MISMATCH")
        if payload.get("result_sha256") != sha256_mapping(payload["metrics"]):
            raise EvaluationContractError("OUTER_RESULT_HASH_MISMATCH")
        atomic_write_json(
            self.state_path,
            {"state": "OUTER_EVAL_COMMITTED", "result_sha256": payload["result_sha256"]},
        )
        return payload["metrics"]

    def evaluate_once(
        self, *, checkpoint_digest: str, evaluator: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        committed = self._load_committed(checkpoint_digest)
        if committed is not None:
            return committed
        atomic_write_json(
            self.state_path,
            {"state": "OUTER_EVAL_IN_PROGRESS", "checkpoint_digest": checkpoint_digest},
        )
        metrics = evaluator()
        result_digest = sha256_mapping(metrics)
        atomic_write_json(
            self.result_path,
            {
                "fold": self.fold,
                "run_contract_digest": self.run_contract_digest,
                "checkpoint_digest": checkpoint_digest,
                "metrics": metrics,
                "result_sha256": result_digest,
            },
        )
        atomic_write_json(
            self.state_path,
            {"state": "OUTER_EVAL_COMMITTED", "result_sha256": result_digest},
        )
        return metrics
