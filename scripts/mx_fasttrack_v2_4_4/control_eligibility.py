"""Deterministic, atlas-only eligibility for three 1.5x size-matched controls."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def eligible_atlas_ids(atlas: np.ndarray) -> tuple[int, ...]:
    labels, counts = np.unique(np.asarray(atlas), return_counts=True)
    volumes = {int(label): int(count) for label, count in zip(labels, counts, strict=True)
               if int(label) > 0}
    if not volumes:
        raise ValueError("V244_ATLAS_NO_POSITIVE_LABELS")
    return tuple(sorted(label for label, volume in volumes.items()
                        if sum(other != label and 2 / 3 <= count / volume <= 1.5
                               for other, count in volumes.items()) >= 3))


def eligible_ranked_regions(regions: Sequence[Mapping], eligible: frozenset[int]) -> list[dict]:
    """Preserve original attribution order, excluding only infeasible ROIs."""
    if not regions or not eligible:
        raise ValueError("V244_NO_ASSESSABLE_ATLAS_REGIONS")
    if len({int(row["atlas_id"]) for row in regions}) != len(regions):
        raise ValueError("V244_DUPLICATE_ATLAS_REGION")
    selected = [dict(row) for row in regions if int(row["atlas_id"]) in eligible]
    if not selected:
        raise ValueError("V244_NO_ASSESSABLE_ATTRIBUTION_REGION")
    return selected


def run_with_eligibility(*, original_run, atlas_data: np.ndarray,
                         amendment_sha256: str, **kwargs) -> dict:
    """Apply the atlas-only filter only within the uncommitted fifth fold."""
    from src.evaluation import method_x_xai_forward as forward

    eligible = frozenset(eligible_atlas_ids(atlas_data))
    if len(eligible) < 4:
        raise ValueError("V244_TOO_FEW_ASSESSABLE_REGIONS")
    original_aggregate = forward._aggregate_regions
    choices: list[dict] = []

    def filtered_aggregate(attribution_map, atlas, region_names):
        regions = original_aggregate(attribution_map, atlas, region_names)
        chosen = eligible_ranked_regions(regions, eligible)
        choices.append({"unrestricted_top_id": int(regions[0]["atlas_id"]),
                        "assessable_top_id": int(chosen[0]["atlas_id"])})
        return chosen

    forward._aggregate_regions = filtered_aggregate
    try:
        result = original_run(atlas_data=atlas_data, **kwargs)
    finally:
        forward._aggregate_regions = original_aggregate
    if (len(choices) != 2 or len(result.get("crossfit_directions", [])) != 2
        or any(direction.get("top_region_id") != choice["assessable_top_id"]
               for direction, choice in zip(result["crossfit_directions"], choices, strict=True))):
        raise ValueError("V244_ELIGIBILITY_SELECTION_TRACE_INVALID")
    return {**result,
            "post_hoc_control_eligibility_amendment_sha256": amendment_sha256,
            "control_eligibility_policy": "ATLAS_ONLY_THREE_CONTROLS_WITHIN_1_5X",
            "selection_trace": choices,
            "original_four_fold_receipts_preserved": True}
