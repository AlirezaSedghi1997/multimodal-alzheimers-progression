from __future__ import annotations

from collections.abc import Callable, Sequence
from random import Random
from statistics import mean
from typing import Any


def guarded_bootstrap_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metric_fn: Callable[[list[int], list[float]], float],
    *,
    n_boot: int = 1000,
    seed: int = 13,
) -> dict[str, Any]:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score length mismatch")
    if len(y_true) == 0:
        return {"status": "EMPTY", "mean": None, "ci_low": None, "ci_high": None}
    if len(set(y_true)) < 2:
        return {"status": "SINGLE_CLASS", "mean": None, "ci_low": None, "ci_high": None}
    rng = Random(seed)
    vals: list[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yt = [int(y_true[i]) for i in idx]
        if len(set(yt)) < 2:
            continue
        ys = [float(y_score[i]) for i in idx]
        vals.append(float(metric_fn(yt, ys)))
    if not vals:
        return {"status": "NO_VALID_RESAMPLES", "mean": None, "ci_low": None, "ci_high": None}
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return {"status": "OK", "mean": mean(vals), "ci_low": lo, "ci_high": hi, "n_valid": len(vals)}

