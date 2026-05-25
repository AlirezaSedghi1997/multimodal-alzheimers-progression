from __future__ import annotations

from typing import Any, Sequence


def delong_roc_auc_guarded(
    y_true: Sequence[int],
    y_score_a: Sequence[float],
    y_score_b: Sequence[float],
    context: dict[str, Any],
) -> dict[str, Any]:
    split_type = context.get("split_type")
    name = str(context.get("name", "")).lower()
    if split_type != "external_independent_binary":
        raise ValueError(f"DeLong allowed only for external_independent_binary. Got {split_type}")
    if not bool(context.get("external_is_truly_independent", False)):
        raise ValueError("DeLong requires external_is_truly_independent=True.")
    if "fold" in name or "cv" in name:
        raise ValueError("DeLong is forbidden on CV folds.")
    if len(set(y_true)) < 2:
        raise ValueError("DeLong requires binary external set with both classes.")
    if len(y_true) != len(y_score_a) or len(y_true) != len(y_score_b):
        raise ValueError("Input length mismatch.")
    return {"status": "ALLOWED", "method": "delong_guard_placeholder"}

