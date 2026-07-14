from __future__ import annotations

from typing import Any, Iterable


class ModelSelectionError(ValueError):
    """Raised when inner-validation evidence is incomplete or malformed."""


def select_inner_winner(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(candidates)
    if not rows:
        raise ModelSelectionError("NO_COMPLETED_INNER_CANDIDATES")
    required = {"candidate_id", "mean_ibs", "standard_error", "complexity_rank"}
    if any(not required.issubset(row) for row in rows):
        raise ModelSelectionError("INCOMPLETE_INNER_CANDIDATE")
    best = min(rows, key=lambda row: float(row["mean_ibs"]))
    threshold = float(best["mean_ibs"]) + float(best["standard_error"])
    eligible = [row for row in rows if float(row["mean_ibs"]) <= threshold]
    return min(
        eligible,
        key=lambda row: (
            int(row["complexity_rank"]),
            float(row["mean_ibs"]),
            str(row["candidate_id"]),
        ),
    )
