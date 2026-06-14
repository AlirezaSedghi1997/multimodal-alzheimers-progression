from __future__ import annotations


def _as_ints(values: list[int] | tuple[int, ...]) -> list[int]:
    return [int(v) for v in values]


def _as_floats(values: list[float] | tuple[float, ...]) -> list[float]:
    return [float(v) for v in values]


def confusion_counts(y_true: list[int], y_prob: list[float], threshold: float) -> dict[str, int]:
    y = _as_ints(y_true)
    p = _as_floats(y_prob)
    if len(y) != len(p):
        raise ValueError("y_true and y_prob must have identical length")
    pred = [1 if score >= float(threshold) else 0 for score in p]
    tp = sum(1 for actual, guess in zip(y, pred) if actual == 1 and guess == 1)
    tn = sum(1 for actual, guess in zip(y, pred) if actual == 0 and guess == 0)
    fp = sum(1 for actual, guess in zip(y, pred) if actual == 0 and guess == 1)
    fn = sum(1 for actual, guess in zip(y, pred) if actual == 1 and guess == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_at_threshold(y_true: list[int], y_prob: list[float], threshold: float) -> dict[str, float]:
    counts = confusion_counts(y_true, y_prob, threshold)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    n = tp + tn + fp + fn
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / n if n else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        **counts,
    }


def candidate_thresholds(y_prob: list[float] | tuple[float, ...]) -> list[float]:
    values = {0.5}
    values.update(round(i / 20, 2) for i in range(1, 20))
    values.update(round(float(v), 6) for v in y_prob)
    return sorted(v for v in values if 0.0 < v < 1.0)


def select_threshold(
    validation_y: list[int],
    validation_prob: list[float],
    *,
    objective: str = "f1",
    candidates: list[float] | None = None,
    min_validation_size: int = 4,
) -> dict:
    if objective not in {"f1", "balanced_accuracy", "accuracy"}:
        raise ValueError(f"Unsupported threshold objective: {objective}")
    n_classes = len(set(validation_y))
    if n_classes < 2:
        return {
            "threshold_source": "validation",
            "threshold_objective": objective,
            "selected_threshold": 0.5,
            "candidate_thresholds": [],
            "validation_metrics_at_selected_threshold": {},
            "threshold_skipped_reason": "single_class_in_validation",
        }
    if len(validation_y) < min_validation_size:
        return {
            "threshold_source": "validation",
            "threshold_objective": objective,
            "selected_threshold": 0.5,
            "candidate_thresholds": [],
            "validation_metrics_at_selected_threshold": {},
            "threshold_skipped_reason": f"validation_size_{len(validation_y)}_below_min_{min_validation_size}",
        }
    thresholds = candidates or candidate_thresholds(validation_prob)
    rows = []
    for threshold in thresholds:
        row = metrics_at_threshold(validation_y, validation_prob, threshold)
        row["balanced_accuracy"] = (row["sensitivity"] + row["specificity"]) / 2.0
        rows.append(row)
    best = max(rows, key=lambda row: (float(row[objective]), float(row["specificity"]), -float(row["threshold"])))
    return {
        "threshold_source": "validation",
        "threshold_objective": objective,
        "selected_threshold": best["threshold"],
        "candidate_thresholds": thresholds,
        "validation_metrics_at_selected_threshold": best,
    }


def evaluate_with_train_selected_threshold(
    *,
    validation_y: list[int],
    validation_prob: list[float],
    test_y: list[int],
    test_prob: list[float],
    objective: str = "f1",
) -> dict:
    selection = select_threshold(validation_y, validation_prob, objective=objective)
    if "threshold_skipped_reason" in selection:
        return {
            **selection,
            "test_metrics": metrics_at_threshold(test_y, test_prob, 0.5),
            "test_labels_used_for_threshold_selection": False,
        }
    test_metrics = metrics_at_threshold(test_y, test_prob, float(selection["selected_threshold"]))
    return {
        **selection,
        "test_metrics": test_metrics,
        "test_labels_used_for_threshold_selection": False,
    }
