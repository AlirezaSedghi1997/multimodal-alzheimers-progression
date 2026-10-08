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


def sigmoid_logits(logits) -> list[float]:
    import numpy as _np

    arr = _np.asarray(logits, dtype=float)
    probs = 1.0 / (1.0 + _np.exp(-arr))
    return probs.tolist() if isinstance(logits, list) else probs


def apply_temperature_to_logits(logits, temperature: float):
    import numpy as _np

    arr = _np.asarray(logits, dtype=float)
    scaled = arr / float(temperature) if float(temperature) != 0 else arr
    return scaled.tolist() if isinstance(logits, list) else scaled


def fit_temperature_scaling(logits, labels) -> dict:
    import numpy as _np

    logit_arr = _np.asarray(logits, dtype=float)
    label_arr = _np.asarray(labels, dtype=float)
    # simple grid search for temperature that minimizes NLL
    best_temp = 1.0
    best_nll = float("inf")
    for temp in _np.linspace(0.05, 5.0, 100):
        scaled = logit_arr / float(temp)
        probs = 1.0 / (1.0 + _np.exp(-scaled))
        eps = 1e-12
        probs = _np.clip(probs, eps, 1 - eps)
        nll = float(-_np.mean(label_arr * _np.log(probs) + (1 - label_arr) * _np.log(1 - probs)))
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)
    # compute NLL before/after
    probs_before = 1.0 / (1.0 + _np.exp(-logit_arr))
    probs_before = _np.clip(probs_before, 1e-12, 1 - 1e-12)
    before_nll = float(-_np.mean(label_arr * _np.log(probs_before) + (1 - label_arr) * _np.log(1 - probs_before)))
    return {
        "temperature": best_temp,
        "before_nll": before_nll,
        "after_nll": best_nll,
        "n_validation": int(len(labels)),
    }


def calibration_metric_summary(y_true, y_prob) -> dict:
    import numpy as _np

    y_true_arr = _np.asarray(y_true, dtype=float)
    y_prob_arr = _np.asarray(y_prob, dtype=float)
    brier = float(_np.mean((y_prob_arr - y_true_arr) ** 2))
    # simple ECE
    ece_dict = calibration_analysis(list(map(int, y_true)), list(map(float, y_prob)))
    return {"brier": brier, "ece": float(ece_dict.get("ece", 0.0)), "n": int(len(y_true))}


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
