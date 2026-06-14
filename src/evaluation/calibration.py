from __future__ import annotations

import numpy as np


def calibration_analysis(
    y_true: list[int],
    y_prob: list[float],
    *,
    n_bins: int = 10,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(y_prob, bin_edges[1:-1])
    bin_data = []
    for i in range(n_bins):
        mask = bins == i
        if mask.sum() == 0:
            bin_data.append({
                "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "count": 0,
                "mean_predicted": None,
                "fraction_positive": None,
            })
            continue
        bin_data.append({
            "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
            "count": int(mask.sum()),
            "mean_predicted": float(y_prob[mask].mean()),
            "fraction_positive": float(y_true[mask].mean()),
        })
    valid_bins = [b for b in bin_data if b["count"] > 0]
    if valid_bins:
        ece = sum(
            b["count"] / len(y_true) * abs(b["fraction_positive"] - b["mean_predicted"])
            for b in valid_bins
        )
    else:
        ece = 0.0
    return {
        "n_bins": n_bins,
        "n_samples": len(y_true),
        "ece": float(ece),
        "bins": bin_data,
    }


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_diagram_data(
    y_true: list[int],
    y_prob: list[float],
    *,
    n_bins: int = 10,
) -> dict:
    analysis = calibration_analysis(y_true, y_prob, n_bins=n_bins)
    perfect_calibration = [
        {"predicted": b["bin_center"], "actual": b["bin_center"]}
        for b in analysis["bins"]
        if b["count"] > 0
    ]
    return {
        "calibration": analysis,
        "perfect_calibration": perfect_calibration,
    }
