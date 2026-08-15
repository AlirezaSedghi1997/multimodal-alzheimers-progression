from __future__ import annotations

import numpy as np
from sksurv.util import Surv


def test_matched_family_registry_has_equal_budgets_and_common_fold_local_policy():
    from src.method_y_revival.model_family_adapters import (
        build_matched_candidate_registry,
    )

    registry = build_matched_candidate_registry(seed=20260714, budget_per_family=3)

    assert set(registry["families"]) == {"coxnet", "rsf", "gbsa"}
    assert {len(rows) for rows in registry["families"].values()} == {3}
    assert registry["budget_per_family"] == 3
    assert registry["selection_metric"] == "mean_inner_integrated_brier_score"
    assert registry["preprocessing_policy"] == {
        "imputation": "median_fit_on_inner_train_only",
        "scaling": "standard_scaler_fit_on_inner_train_only",
        "feature_selection": "fit_on_inner_train_only_if_prespecified",
        "harmonization": "fit_on_inner_train_only_if_prespecified",
        "post_winner_refit": "fit_on_entire_outer_train_only",
    }
    candidates = [candidate for rows in registry["families"].values() for candidate in rows]
    assert len({candidate["candidate_id"] for candidate in candidates}) == 9
    assert all(candidate["seed"] == 20260714 for candidate in candidates)
    assert all(candidate["hpo_budget_units"] == 1 for candidate in candidates)


def test_every_family_uses_a_fresh_fold_local_pipeline_and_fixed_seed():
    from src.method_y_revival.model_family_adapters import (
        build_estimator_pipeline,
        build_matched_candidate_registry,
    )

    registry = build_matched_candidate_registry(seed=29, budget_per_family=1)
    x_train = np.asarray(
        [
            [0.0, 1.0],
            [2.0, np.nan],
            [4.0, 5.0],
            [6.0, 7.0],
            [8.0, 9.0],
            [10.0, 11.0],
            [12.0, 13.0],
            [14.0, 15.0],
            [16.0, 17.0],
            [18.0, 19.0],
            [20.0, 21.0],
            [22.0, 23.0],
        ]
    )
    y_train = Surv.from_arrays(
        event=np.asarray([True, False] * 6),
        time=np.asarray([100.0 + index * 20.0 for index in range(12)]),
    )

    for family, candidates in registry["families"].items():
        first = build_estimator_pipeline(candidates[0])
        second = build_estimator_pipeline(candidates[0])
        assert first is not second
        first.fit(x_train, y_train)
        second.fit(x_train, y_train)
        expected_imputed_mean = np.asarray([11.0, 12.833333333333334])
        np.testing.assert_allclose(first.named_steps["scaler"].mean_, expected_imputed_mean)
        np.testing.assert_allclose(first.predict(x_train), second.predict(x_train))
        assert first.named_steps["model"].__class__.__name__ == {
            "coxnet": "CoxnetSurvivalAnalysis",
            "rsf": "RandomSurvivalForest",
            "gbsa": "GradientBoostingSurvivalAnalysis",
        }[family]


def test_complexity_rank_orders_regularized_and_low_capacity_candidates_first():
    from src.method_y_revival.model_family_adapters import (
        build_matched_candidate_registry,
    )

    families = build_matched_candidate_registry(
        seed=20260714, budget_per_family=3
    )["families"]

    assert families["coxnet"][0]["hyperparameters"] == {
        "alpha": 0.1,
        "l1_ratio": 0.9,
    }
    assert families["rsf"][0]["hyperparameters"]["min_samples_leaf"] == 20
    assert families["gbsa"][0]["hyperparameters"]["max_depth"] == 1
    for candidates in families.values():
        ranks = [candidate["complexity_rank"] for candidate in candidates]
        assert ranks == sorted(ranks)
