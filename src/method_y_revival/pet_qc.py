from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def automated_image_qc(image, reference_image) -> dict[str, Any]:
    data = np.asarray(image.dataobj, dtype=np.float32)
    finite_fraction = float(np.isfinite(data).mean())
    shape_matches = tuple(image.shape) == tuple(reference_image.shape)
    affine_matches = bool(
        np.allclose(np.asarray(image.affine), np.asarray(reference_image.affine), atol=1e-6, rtol=0.0)
    )
    spacing = np.sqrt(np.sum(np.asarray(image.affine)[:3, :3] ** 2, axis=0))
    reference_spacing = np.sqrt(
        np.sum(np.asarray(reference_image.affine)[:3, :3] ** 2, axis=0)
    )
    failure_reasons: list[str] = []
    if finite_fraction != 1.0:
        failure_reasons.append("NONFINITE_VOXELS")
    if not shape_matches:
        failure_reasons.append("SHAPE_MISMATCH")
    if not affine_matches:
        failure_reasons.append("AFFINE_MISMATCH")
    finite_values = data[np.isfinite(data)]
    quantiles = (
        np.percentile(finite_values, [0, 1, 5, 50, 95, 99, 100]).astype(float).tolist()
        if finite_values.size
        else [None] * 7
    )
    return {
        "readable": True,
        "finite_fraction": finite_fraction,
        "shape_matches_template": shape_matches,
        "affine_matches_template": affine_matches,
        "voxel_spacing_matches": bool(np.allclose(spacing, reference_spacing, atol=1e-6)),
        "voxel_spacing_mm": spacing.astype(float).tolist(),
        "intensity_quantiles": quantiles,
        "qc_status": "PASS" if not failure_reasons else "FAIL",
        "failure_reasons": failure_reasons,
    }


def _selection_key(row: dict[str, Any], seed: int) -> str:
    payload = "|".join(
        [
            str(seed),
            str(row.get("tracer", "UNKNOWN")),
            str(row.get("cohort", "UNKNOWN")),
            str(row.get("manufacturer", "UNKNOWN")),
            str(row["scan_key"]),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_visual_qc(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    fraction: float,
    minimum: int,
    maximum: int,
) -> list[str]:
    materialized = sorted(list(rows), key=lambda row: str(row["scan_key"]))
    if not materialized:
        return []
    target = min(
        len(materialized),
        min(int(maximum), max(int(minimum), int(math.ceil(fraction * len(materialized))))),
    )
    # Failures and robust outliers are mandatory review cases.  The requested
    # maximum constrains only the stratified random sample; it must never cause
    # a scientifically important failure/outlier to be silently omitted.
    selected = {
        str(row["scan_key"])
        for row in materialized
        if str(row.get("qc_status")) == "FAIL"
        or str(row.get("outlier_status")) == "OUTLIER"
    }
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        if str(row["scan_key"]) not in selected:
            groups[
                (
                    str(row.get("tracer", "UNKNOWN")),
                    str(row.get("cohort", "UNKNOWN")),
                    str(row.get("manufacturer", "UNKNOWN")),
                )
            ].append(row)
    for group in sorted(groups):
        if len(selected) >= target:
            break
        winner = min(groups[group], key=lambda row: _selection_key(row, seed))
        selected.add(str(winner["scan_key"]))
    remaining = [row for row in materialized if str(row["scan_key"]) not in selected]
    for row in sorted(remaining, key=lambda item: _selection_key(item, seed)):
        if len(selected) >= target:
            break
        selected.add(str(row["scan_key"]))
    return sorted(selected)
