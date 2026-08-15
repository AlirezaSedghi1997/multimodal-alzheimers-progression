from __future__ import annotations

from copy import deepcopy
from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis


class ModelFamilyContractError(ValueError):
    """Raised when model families are not scientifically matched."""


_PREPROCESSING_POLICY = {
    "imputation": "median_fit_on_inner_train_only",
    "scaling": "standard_scaler_fit_on_inner_train_only",
    "feature_selection": "fit_on_inner_train_only_if_prespecified",
    "harmonization": "fit_on_inner_train_only_if_prespecified",
    "post_winner_refit": "fit_on_entire_outer_train_only",
}

_FAMILY_GRIDS: dict[str, list[dict[str, Any]]] = {
    "coxnet": [
        {"alpha": 0.1, "l1_ratio": 0.9},
        {"alpha": 0.01, "l1_ratio": 0.5},
        {"alpha": 0.001, "l1_ratio": 0.1},
    ],
    "rsf": [
        {"n_estimators": 300, "min_samples_leaf": 20, "max_features": 0.5},
        {"n_estimators": 300, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"n_estimators": 300, "min_samples_leaf": 5, "max_features": "sqrt"},
    ],
    "gbsa": [
        {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 1},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    ],
}


def build_matched_candidate_registry(
    *, seed: int, budget_per_family: int = 3
) -> dict[str, Any]:
    """Return equal-budget candidates under one fold-local preprocessing policy."""

    budget = int(budget_per_family)
    if budget <= 0 or budget > min(len(rows) for rows in _FAMILY_GRIDS.values()):
        raise ModelFamilyContractError("UNSUPPORTED_EQUAL_HPO_BUDGET")
    families: dict[str, list[dict[str, Any]]] = {}
    family_rank = {"coxnet": 0, "gbsa": 1, "rsf": 2}
    for family in ("coxnet", "rsf", "gbsa"):
        rows: list[dict[str, Any]] = []
        for index, parameters in enumerate(_FAMILY_GRIDS[family][:budget]):
            rows.append(
                {
                    "candidate_id": f"{family}_{index:02d}",
                    "family": family,
                    "hyperparameters": deepcopy(parameters),
                    "seed": int(seed),
                    "complexity_rank": family_rank[family] * 100 + index,
                    "hpo_budget_units": 1,
                }
            )
        families[family] = rows
    return {
        "families": families,
        "budget_per_family": budget,
        "selection_metric": "mean_inner_integrated_brier_score",
        "selection_rule": "one_standard_error_simplest_candidate",
        "preprocessing_policy": deepcopy(_PREPROCESSING_POLICY),
    }


def build_estimator_pipeline(candidate: dict[str, Any]) -> Pipeline:
    """Build a fresh deterministic estimator; fitting is always fold-local."""

    family = str(candidate.get("family", ""))
    parameters = deepcopy(candidate.get("hyperparameters"))
    if not isinstance(parameters, dict):
        raise ModelFamilyContractError("CANDIDATE_HYPERPARAMETERS_REQUIRED")
    seed = int(candidate.get("seed"))
    if family == "coxnet":
        alpha = float(parameters.pop("alpha"))
        model = CoxnetSurvivalAnalysis(
            alphas=[alpha],
            l1_ratio=float(parameters.pop("l1_ratio")),
            fit_baseline_model=True,
            max_iter=100000,
            **parameters,
        )
    elif family == "rsf":
        model = RandomSurvivalForest(
            random_state=seed,
            n_jobs=1,
            **parameters,
        )
    elif family == "gbsa":
        model = GradientBoostingSurvivalAnalysis(
            loss="coxph",
            random_state=seed,
            **parameters,
        )
    else:
        raise ModelFamilyContractError(f"UNSUPPORTED_MODEL_FAMILY:{family}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
