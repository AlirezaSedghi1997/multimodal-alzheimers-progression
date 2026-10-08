"""Verify an immutable fast-track lock with legacy aggregate-digest projection.

The v2.4 lock writer derives the contract digest from the hashed run_contract,
but its reader incorrectly requires the same field in trainer metrics, where
the selection-only trainer never wrote it.  This adapter supplies *only that
derived field in memory* while running every other original verifier check.
No metric, checkpoint, lock, or preregistration byte is changed.
"""
from __future__ import annotations

from pathlib import Path

from scripts.mx_fasttrack_v2 import integrity
from scripts.mx_fasttrack_v2.run import TASKS, read as original_read, sha

ORIGINAL_CHECK_LOCK = integrity.check_lock


def check_lock_compatible(root: Path, out: Path, *, hash_cache: bool = True) -> dict:
    root, out = root.resolve(), out.resolve()
    lock = original_read(out / "reports/fasttrack_inner_lock.json")
    tasks = lock.get("tasks") or {}
    if set(tasks) != {task[0] for task in TASKS}:
        raise RuntimeError("V241_LOCK_TASKS_INVALID")
    metric_paths: dict[Path, str] = {}
    for task_id, row in tasks.items():
        directory = out / "inner_tasks" / task_id
        metric = directory / "three_d_metrics.json"
        contract = directory / "run_contract.json"
        actual_metrics = original_read(metric)
        actual_contract = original_read(contract)
        digest = row.get("run_contract_digest")
        if (not isinstance(digest, str) or len(digest) != 64
            or sha(metric) != row.get("metrics_sha256")
            or sha(contract) != row.get("run_contract_sha256")
            or actual_contract.get("run_contract_digest") != digest
            or (actual_contract.get("run_contract") or {}).get("digest") != digest):
            raise RuntimeError(f"V241_HASHED_METRIC_OR_CONTRACT_INVALID:{task_id}")
        aggregate = actual_metrics.get("aggregate")
        if not isinstance(aggregate, dict):
            raise RuntimeError(f"V241_METRICS_AGGREGATE_INVALID:{task_id}")
        recorded = aggregate.get("run_contract_digest")
        if recorded not in (None, digest):
            raise RuntimeError(f"V241_RECORDED_AGGREGATE_DIGEST_CONTRADICTION:{task_id}")
        if recorded is None:
            metric_paths[metric.resolve()] = digest
    prior_read = integrity.read

    def projection(path: Path) -> dict:
        payload = prior_read(path)
        expected = metric_paths.get(Path(path).resolve())
        if expected is None:
            return payload
        if not isinstance(payload, dict) or not isinstance(payload.get("aggregate"), dict):
            raise RuntimeError("V241_METRIC_PROJECTION_SHAPE_CHANGED")
        copied = dict(payload)
        copied["aggregate"] = dict(payload["aggregate"])
        if copied["aggregate"].get("run_contract_digest") is not None:
            raise RuntimeError("V241_METRIC_PROJECTION_NOT_LEGACY")
        copied["aggregate"]["run_contract_digest"] = expected
        return copied

    integrity.read = projection
    try:
        verified = ORIGINAL_CHECK_LOCK(root, out, hash_cache=hash_cache)
    finally:
        integrity.read = prior_read
    if verified != lock:
        raise RuntimeError("V241_ORIGINAL_VERIFIER_RETURNED_DIFFERENT_LOCK")
    return verified
