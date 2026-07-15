from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _synthetic_survival_frame(n: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    pet = rng.normal(size=n)
    event = np.asarray([index % 3 != 0 for index in range(n)], dtype=int)
    time = 200.0 + np.arange(n) * 20.0 + (pet * 10.0)
    return pd.DataFrame(
        {
            "patient_hash": [f"pt-{index:03d}" for index in range(n)],
            "event": event,
            "time_days": time,
            "treatment_arm": ["placebo" if index % 2 else "treatment" for index in range(n)],
            "age": 65.0 + rng.normal(size=n),
            "baseline_pacc_raw": rng.normal(size=n),
            "tau_meta_roi": pet,
        }
    )


def test_survival_evaluation_grid_stays_inside_train_and_test_support():
    from src.method_y_revival.regional_training import survival_evaluation_grid

    grid = survival_evaluation_grid(
        np.array([100.0, 500.0, 1000.0]),
        np.array([200.0, 700.0, 900.0]),
        points=8,
    )

    assert len(grid) == 8
    assert grid.min() > 200.0
    assert grid.max() < 900.0


def test_nested_regional_training_completes_every_fold_and_locks_winners(tmp_path, monkeypatch):
    import src.method_y_revival.regional_training as training_module
    from src.method_y_revival.regional_training import run_regional_nested_cv

    report = run_regional_nested_cv(
        frame=_synthetic_survival_frame(),
        feature_sets={
            "clinical_cognition": ["age", "baseline_pacc_raw"],
            "clinical_cognition_pet": ["age", "baseline_pacc_raw", "tau_meta_roi"],
        },
        output_root=tmp_path,
        outer_folds=3,
        inner_folds=2,
        seed=19,
        alpha_grid=[0.1, 1.0],
        requested_horizons=[300.0, 600.0],
        min_at_risk=2,
    )

    assert report["status"] == "OUTER_EVALUATION_DONE"
    assert set(report["variants"]) == {"clinical_cognition", "clinical_cognition_pet"}
    for variant in report["variants"].values():
        assert variant["outer_folds_completed"] == 3
        assert all(fold["status"] == "COMPLETED" for fold in variant["folds"])
    locked = json.loads((tmp_path / "locked" / "tau_locked_config.json").read_text(encoding="utf-8"))
    assert locked["outer_folds"] == 3
    assert locked["selection_source"] == "inner_validation_ibs_only"

    def fail_if_hpo_repeated(*args, **kwargs):
        raise AssertionError("inner HPO must be reused after checkpoint lock")

    monkeypatch.setattr(training_module, "_select_alpha", fail_if_hpo_repeated)
    repeated = run_regional_nested_cv(
        frame=_synthetic_survival_frame(),
        feature_sets={
            "clinical_cognition": ["age", "baseline_pacc_raw"],
            "clinical_cognition_pet": ["age", "baseline_pacc_raw", "tau_meta_roi"],
        },
        output_root=tmp_path,
        outer_folds=3,
        inner_folds=2,
        seed=19,
        alpha_grid=[0.1, 1.0],
        requested_horizons=[300.0, 600.0],
        min_at_risk=2,
    )
    assert repeated["status"] == "OUTER_EVALUATION_DONE"


def test_regional_training_rejects_outcome_columns_as_features(tmp_path):
    import pytest

    from src.method_y_revival.regional_training import RegionalTrainingError, run_regional_nested_cv

    with pytest.raises(RegionalTrainingError, match="FORBIDDEN_MODEL_FEATURE"):
        run_regional_nested_cv(
            frame=_synthetic_survival_frame(),
            feature_sets={"invalid": ["age", "event"]},
            output_root=tmp_path,
            outer_folds=3,
            inner_folds=2,
            seed=19,
            alpha_grid=[0.1],
            requested_horizons=[300.0],
            min_at_risk=2,
        )
