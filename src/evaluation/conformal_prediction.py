from __future__ import annotations

import numpy as np


def _as_ints(values: list[int] | tuple[int, ...] | np.ndarray) -> list[int]:
    return [int(v) for v in values]


def _as_floats(values: list[float] | tuple[float, ...] | np.ndarray) -> list[float]:
    return [float(v) for v in values]


def compute_nonconformity_scores(
    y_true: list[int],
    y_prob: list[float],
    *,
    score_type: str = "margin",
) -> list[float]:
    if score_type == "margin":
        scores = []
        for y, p in zip(y_true, y_prob):
            if y == 1:
                scores.append(1.0 - p)
            else:
                scores.append(p)
        return scores
    elif score_type == "entropy":
        scores = []
        for y, p in zip(y_true, y_prob):
            p_clamp = max(min(p, 1.0 - 1e-7), 1e-7)
            if y == 1:
                scores.append(-np.log(p_clamp))
            else:
                scores.append(-np.log(1.0 - p_clamp))
        return scores
    else:
        raise ValueError(f"Unsupported score_type: {score_type}")


def conformal_calibration(
    cal_scores: list[float],
    *,
    alpha: float = 0.1,
) -> dict:
    n = len(cal_scores)
    if n == 0:
        return {
            "quantile": float("inf"),
            "coverage_level": 1.0 - alpha,
            "calibration_size": 0,
        }
    sorted_scores = sorted(cal_scores)
    quantile_index = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    quantile_index = max(0, min(quantile_index, n - 1))
    quantile = sorted_scores[quantile_index]
    return {
        "quantile": float(quantile),
        "coverage_level": 1.0 - alpha,
        "calibration_size": n,
    }


def conformal_prediction_sets(
    y_prob: list[float],
    quantile: float,
    *,
    score_type: str = "margin",
) -> list[list[int]]:
    prediction_sets = []
    for p in y_prob:
        if score_type == "margin":
            set_ = []
            if (1.0 - p) <= quantile:
                set_.append(1)
            if p <= quantile:
                set_.append(0)
            if not set_:
                if p >= 0.5:
                    set_ = [1]
                else:
                    set_ = [0]
            prediction_sets.append(sorted(set_))
        elif score_type == "entropy":
            set_ = []
            p_clamp = max(min(p, 1.0 - 1e-7), 1e-7)
            if -np.log(p_clamp) <= quantile:
                set_.append(1)
            if -np.log(1.0 - p_clamp) <= quantile:
                set_.append(0)
            if not set_:
                if p >= 0.5:
                    set_ = [1]
                else:
                    set_ = [0]
            prediction_sets.append(sorted(set_))
        else:
            raise ValueError(f"Unsupported score_type: {score_type}")
    return prediction_sets


def evaluate_conformal_prediction(
    y_true_cal: list[int],
    y_prob_cal: list[float],
    y_true_eval: list[int] | None = None,
    y_prob_eval: list[float] | None = None,
    *,
    alpha: float = 0.1,
    score_type: str = "margin",
) -> dict:
    if y_true_eval is None or y_prob_eval is None:
        y_true_eval = y_true_cal
        y_prob_eval = y_prob_cal
    cal_scores = compute_nonconformity_scores(y_true_cal, y_prob_cal, score_type=score_type)
    cal_result = conformal_calibration(cal_scores, alpha=alpha)
    prediction_sets = conformal_prediction_sets(y_prob_eval, cal_result["quantile"], score_type=score_type)
    n = len(y_true_eval)
    correct = sum(1 for y, s in zip(y_true_eval, prediction_sets) if y in s)
    coverage = correct / n if n else 0.0
    avg_set_size = np.mean([len(s) for s in prediction_sets]) if prediction_sets else 0.0
    empty_sets = sum(1 for s in prediction_sets if len(s) == 0)
    return {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": float(coverage),
        "average_prediction_set_size": float(avg_set_size),
        "empty_sets": empty_sets,
        "calibration_quantile": cal_result["quantile"],
        "calibration_size": cal_result["calibration_size"],
        "score_type": score_type,
        "n_samples": n,
    }


def conformal_risk_control(
    y_true: list[int],
    y_prob: list[float],
    *,
    target_error_rate: float = 0.1,
    score_type: str = "margin",
) -> dict:
    cal_scores = compute_nonconformity_scores(y_true, y_prob, score_type=score_type)
    n = len(cal_scores)
    if n == 0:
        return {
            "threshold": 0.5,
            "target_error_rate": target_error_rate,
            "empirical_error_rate": 0.0,
        }
    sorted_scores = sorted(cal_scores)
    quantile_index = int(np.ceil(n * (1.0 - target_error_rate))) - 1
    quantile_index = max(0, min(quantile_index, n - 1))
    threshold = sorted_scores[quantile_index]
    predictions = [1 if p >= threshold else 0 for p in y_prob]
    errors = sum(1 for y, pred in zip(y_true, predictions) if y != pred)
    empirical_error = errors / n if n else 0.0
    return {
        "threshold": float(threshold),
        "target_error_rate": target_error_rate,
        "empirical_error_rate": float(empirical_error),
        "n_samples": n,
    }
