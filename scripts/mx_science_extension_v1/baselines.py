"""Fit-only prevalence and tabular controls for one locked outer fold.

The C grid is selected on that fold's inner validation only.  No outer label,
cohort, PET date/image, scanner, raw identifier, or path enters either model.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PHYSICAL = ("age", "education_years", "apoe4_carrier", "mmse_base", "sex")
C_GRID = (0.01, 0.1, 1.0, 10.0)


def _number(value: object) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("TABULAR_NONNUMERIC_VALUE") from exc
    if not math.isfinite(result):
        raise ValueError("TABULAR_NONFINITE_VALUE")
    return result


def _sex(value: object) -> float:
    text = str(value or "").strip().lower()
    if text in {"1", "m", "male"}:
        return 1.0
    # The locked split encodes male=1, female=2; the cache audit maps every
    # non-1 code to zero.  Preserve that exact binary meaning for this arm.
    if text in {"0", "2", "f", "female"}:
        return 0.0
    if not text:
        return float("nan")
    raise ValueError("TABULAR_SEX_ENCODING_UNRECOGNIZED")


def feature_matrix(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    if not rows:
        raise ValueError("TABULAR_EMPTY_PARTITION")
    values = []
    for row in rows:
        if not all(name in row for name in PHYSICAL):
            raise ValueError("TABULAR_ALLOWED_FEATURE_MISSING")
        values.append([*(_number(row[name]) for name in PHYSICAL[:-1]), _sex(row["sex"])])
    return np.asarray(values, dtype=np.float64)


def _labels(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    result = []
    for row in rows:
        value = str(row.get("ab_label", ""))
        if value not in {"0", "1"}:
            raise ValueError("TABULAR_LABEL_INVALID")
        result.append(int(value))
    return np.asarray(result, dtype=np.int64)


def fit_controls(fit: Sequence[Mapping[str, object]],
                 validation: Sequence[Mapping[str, object]],
                 outer: Sequence[Mapping[str, object]],
                 *, c_grid: Sequence[float] = C_GRID) -> dict:
    """Return probabilities, never using outer labels for fit or selection."""
    x_fit, x_val, x_outer = map(feature_matrix, (fit, validation, outer))
    y_fit, y_val = _labels(fit), _labels(validation)
    if (set(y_fit) != {0, 1} or set(y_val) != {0, 1}
        or not c_grid or any(not math.isfinite(c) or c <= 0 for c in c_grid)
        or len(set(c_grid)) != len(c_grid)):
        raise ValueError("TABULAR_INNER_CLASS_OR_C_GRID_INVALID")
    if any(np.all(np.isnan(x_fit[:, column])) for column in range(x_fit.shape[1])):
        raise ValueError("TABULAR_FIT_FEATURE_ALL_MISSING")
    prevalence = float(y_fit.mean())
    if not 0 < prevalence < 1:
        raise ValueError("PREVALENCE_FIT_ONE_CLASS")
    losses = []
    models = []
    for c in sorted(float(item) for item in c_grid):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=False)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=c, solver="lbfgs", max_iter=2000,
                                                class_weight=None, random_state=13)),
        ])
        model.fit(x_fit, y_fit)
        probability = model.predict_proba(x_val)[:, 1]
        loss = float(log_loss(y_val, probability, labels=[0, 1]))
        if not math.isfinite(loss):
            raise ValueError("TABULAR_INNER_LOSS_NONFINITE")
        losses.append((loss, c))
        models.append(model)
    choice = min(range(len(losses)), key=lambda i: losses[i])
    tabular = models[choice].predict_proba(x_outer)[:, 1]
    if not np.all(np.isfinite(tabular)) or np.any(tabular < 0) or np.any(tabular > 1):
        raise ValueError("TABULAR_OUTER_PROBABILITY_INVALID")
    return {
        "fit_n": len(fit), "inner_validation_n": len(validation), "outer_n": len(outer),
        "prevalence_probability": prevalence,
        "prevalence_outer_probabilities": [prevalence] * len(outer),
        "tabular_outer_probabilities": [float(value) for value in tabular],
        "selected_c": losses[choice][1],
        "inner_validation_bce_by_c": {str(c): loss for loss, c in losses},
        "selection_scope": "fit_to_inner_only_no_outer_label_access",
        "preprocessing_fit_scope": "fit_partition_only",
        "physical_feature_order": list(PHYSICAL),
    }
