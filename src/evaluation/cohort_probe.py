from __future__ import annotations

from typing import Any

import numpy as np


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC-AUC requires both classes")
    return float(roc_auc_score(y_true, y_score))


def cohort_label_from_substudy(substudy: np.ndarray) -> np.ndarray:
    """Map substudy to binary cohort target: A4=1, LEARN=0."""
    out = np.zeros(len(substudy), dtype=int)
    for i, value in enumerate(substudy):
        out[i] = 1 if str(value).upper() == "A4" else 0
    return out


def run_linear_cohort_probe(
    x: np.ndarray,
    y_cohort: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int = 5,
    seed: int = 13,
) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=float)
    y_cohort = np.asarray(y_cohort, dtype=int)
    splitter = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if groups is not None
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    split_args = (x, y_cohort, groups) if groups is not None else (x, y_cohort)
    fold_rows = []
    y_all: list[int] = []
    p_all: list[float] = []
    for fold, (tr, te) in enumerate(splitter.split(*split_args)):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
            ]
        )
        model.fit(x[tr], y_cohort[tr])
        prob = model.predict_proba(x[te])[:, 1]
        auc = roc_auc_binary(y_cohort[te], prob)
        fold_rows.append({"fold": fold, "n_test": int(len(te)), "roc_auc": auc})
        y_all.extend(y_cohort[te].tolist())
        p_all.extend(prob.tolist())
    agg_auc = roc_auc_binary(np.asarray(y_all), np.asarray(p_all))
    return {
        "probe_type": "linear",
        "model": "logistic_regression",
        "folds": fold_rows,
        "roc_auc": agg_auc,
        "n": int(len(y_all)),
    }


def run_nonlinear_cohort_probe(
    x: np.ndarray,
    y_cohort: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int = 5,
    seed: int = 17,
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import Pipeline

    x = np.asarray(x, dtype=float)
    y_cohort = np.asarray(y_cohort, dtype=int)
    splitter = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if groups is not None
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    split_args = (x, y_cohort, groups) if groups is not None else (x, y_cohort)
    fold_rows = []
    y_all: list[int] = []
    p_all: list[float] = []
    for fold, (tr, te) in enumerate(splitter.split(*split_args)):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        max_iter=200,
                        learning_rate=0.05,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(x[tr], y_cohort[tr])
        prob = model.predict_proba(x[te])[:, 1]
        auc = roc_auc_binary(y_cohort[te], prob)
        fold_rows.append({"fold": fold, "n_test": int(len(te)), "roc_auc": auc})
        y_all.extend(y_cohort[te].tolist())
        p_all.extend(prob.tolist())
    agg_auc = roc_auc_binary(np.asarray(y_all), np.asarray(p_all))
    return {
        "probe_type": "nonlinear",
        "model": "hist_gradient_boosting_classifier",
        "folds": fold_rows,
        "roc_auc": agg_auc,
        "n": int(len(y_all)),
    }


def summarize_cohort_probe(linear: dict[str, Any], nonlinear: dict[str, Any]) -> dict[str, Any]:
    linear_auc = float(linear["roc_auc"])
    nonlinear_auc = float(nonlinear["roc_auc"])
    gap = nonlinear_auc - linear_auc
    if linear_auc >= 0.9:
        interpretation = "Cohort is linearly encoded in the input representation."
    elif linear_auc < 0.75 and nonlinear_auc >= 0.9:
        interpretation = "Cohort is encoded nonlinearly; report as covert/structured cohort signal."
    elif nonlinear_auc >= 0.9:
        interpretation = "Strong cohort signal in both linear and nonlinear probes."
    else:
        interpretation = "Cohort encoding in the input representation is limited or ambiguous."
    return {
        "linear_probe_auc": linear_auc,
        "nonlinear_probe_auc": nonlinear_auc,
        "linear_vs_nonlinear_gap": gap,
        "interpretation": interpretation,
        "deconfounding_success_rule": "Never claim deconfounding succeeded based on the linear probe alone.",
    }
