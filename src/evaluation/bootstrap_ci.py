from __future__ import annotations

from collections.abc import Callable, Sequence
from statistics import mean
from typing import Any

import numpy as np


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y_true = np.asarray(y_true)
    idx_parts = []
    for cls in np.unique(y_true):
        cls_idx = np.where(y_true == cls)[0]
        if len(cls_idx) > 0:
            idx_parts.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
    if not idx_parts:
        raise ValueError("No indices available for bootstrap")
    out = np.concatenate(idx_parts)
    rng.shuffle(out)
    return out


def guarded_bootstrap_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metric_fn: Callable[[Sequence[int], Sequence[float]], float],
    *,
    n_boot: int | None = 1000,
    n_bootstrap: int | None = None,
    ci_level: float = 0.95,
    seed: int = 13,
    stratified: bool = True,
    min_successful: int | None = None,
) -> dict[str, Any]:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score length mismatch")

    requested = int(n_bootstrap if n_bootstrap is not None else (n_boot if n_boot is not None else 1000))
    base = {
        "n_bootstrap_requested": requested,
        "n_successful": 0,
        "stratified": bool(stratified),
        "ci_level": float(ci_level),
    }
    if len(y_true) == 0:
        return {**base, "status": "EMPTY", "mean": None, "ci_low": None, "ci_high": None, "point": None, "ci_lower": None, "ci_upper": None}

    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(y_score, dtype=float)
    if len(np.unique(y_arr)) < 2:
        return {**base, "status": "SINGLE_CLASS", "mean": None, "ci_low": None, "ci_high": None, "point": None, "ci_lower": None, "ci_upper": None, "guard_status": "SINGLE_CLASS_NO_DISCRIMINATION_METRIC"}

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(y_arr)
    point = float(metric_fn(y_arr, s_arr))
    for _ in range(requested):
        idx = stratified_bootstrap_indices(y_arr, rng) if stratified else rng.integers(0, n, size=n)
        yt = y_arr[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            vals.append(float(metric_fn(yt, s_arr[idx])))
        except Exception:
            continue
    if not vals:
        return {**base, "status": "NO_VALID_RESAMPLES", "mean": None, "ci_low": None, "ci_high": None, "point": point, "ci_lower": None, "ci_upper": None}

    vals.sort()
    alpha = 1.0 - float(ci_level)
    ci_low = float(np.percentile(vals, 100 * alpha / 2))
    ci_high = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    min_ok = min_successful if min_successful is not None else int(0.8 * requested)
    status = "OK" if len(vals) >= min_ok else "WARN_LOW_SUCCESSFUL_BOOTSTRAPS"
    return {
        **base,
        "status": status,
        "mean": mean(vals),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_valid": len(vals),
        "point": point,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "n_successful": len(vals),
    }


def metric_point_with_single_class_guard(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metric_fn: Callable[[Sequence[int], Sequence[float]], float],
    metric_name: str,
) -> dict[str, Any]:
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(y_score, dtype=float)
    if metric_name.lower() in {"roc_auc", "pr_auc", "average_precision"} and len(np.unique(y_arr)) < 2:
        return {"point": None, "status": "SINGLE_CLASS_NO_DISCRIMINATION_METRIC"}
    try:
        return {"point": float(metric_fn(y_arr, s_arr)), "status": "OK"}
    except Exception as exc:
        return {"point": None, "status": "METRIC_FAILED", "error": str(exc)}