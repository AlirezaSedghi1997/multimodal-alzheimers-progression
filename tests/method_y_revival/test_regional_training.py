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
            "treatment_active": [0.0 if index % 2 else 1.0 for index in range(n)],
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
        assert sum(len(fold["oof_predictions"]) for fold in variant["folds"]) == len(
            _synthetic_survival_frame()
        )
        assert all(
            set(fold["oof_predictions"][0])
            >= {"patient_hash", "event", "time_days", "risk_score", "survival_probabilities"}
            for fold in variant["folds"]
        )
    locked = json.loads((tmp_path / "locked" / "tau_locked_config.json").read_text(encoding="utf-8"))
    assert locked["outer_folds"] == 3
    assert locked["selection_source"] == "inner_validation_ibs_only"
    assert (tmp_path / "splits" / "tau_split_contract.json").is_file()
    assert (tmp_path / "outer_evaluation" / "tau" / "clinical_cognition").is_dir()

    from src.method_y_revival.science_gates import paired_oof_bootstrap

    bootstrap = paired_oof_bootstrap(
        report["variants"]["clinical_cognition"],
        report["variants"]["clinical_cognition_pet"],
        iterations=20,
        seed=23,
    )
    assert bootstrap["status"] == "COMPLETED"
    assert bootstrap["successful_iterations"] == 20
    assert len(bootstrap["delta_ibs_95ci"]) == 2
    assert len(bootstrap["delta_cindex_95ci"]) == 2

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


def test_inner_hpo_rejects_numerically_unstable_candidate(monkeypatch):
    import src.method_y_revival.regional_training as module

    frame = _synthetic_survival_frame(12)
    by_subject = frame.set_index("patient_hash", drop=False)
    inner = [
        {
            "train_subjects": frame.patient_hash.iloc[:8].tolist(),
            "validation_subjects": frame.patient_hash.iloc[8:].tolist(),
        }
    ]

    class FakeModel:
        def __init__(self, alpha):
            self.alpha = alpha

        def fit(self, x, y):
            return self

    monkeypatch.setattr(module, "_pipeline", lambda alpha: FakeModel(alpha))

    def fake_evaluate(model, *args, **kwargs):
        if model.alpha < 1.0:
            raise ValueError("Input estimate contains infinity")
        return {"integrated_brier_score": 0.2}

    monkeypatch.setattr(module, "_evaluate_model", fake_evaluate)
    winner, candidates = module._select_alpha(
        by_subject.reset_index(drop=True),
        features=["age"],
        inner_folds=inner,
        alpha_grid=[0.001, 1.0],
    )

    assert winner["alpha"] == 1.0
    assert candidates[0]["status"] == "REJECTED_NUMERICAL_INSTABILITY"
    assert candidates[1]["status"] == "COMPLETED"


def test_amyloid_training_uses_a4_only_and_adds_only_continuous_centiloid(tmp_path):
    from src.method_y_revival.endpoint_builder import write_locked_endpoint
    from src.method_y_revival.regional_training import (
        run_amyloid_regional_development_from_locked_endpoint,
    )

    frame = _synthetic_survival_frame(54).rename(columns={"tau_meta_roi": "baseline_amyloid_centiloid"})
    frame["cohort"] = ["A4"] * 48 + ["LEARN"] * 6
    frame["sex_binary"] = 0.0
    frame["education_years"] = 16.0
    rows = frame.to_dict(orient="records")
    write_locked_endpoint(
        rows,
        output_path=tmp_path / "endpoints" / "amyloid_progression_endpoint.parquet",
        contract_path=tmp_path / "endpoints" / "amyloid_endpoint_contract.json",
        source_hashes={"amyloid": "a" * 64},
        endpoint_definition={"tracer": "FBP", "target_source": "future_progression"},
    )
    config = {
        "seed": 19,
        "validation": {
            "outer_folds": 3,
            "inner_folds": 2,
            "requested_horizons_days": [300.0, 600.0],
            "minimum_at_risk_per_horizon": 2,
            "bootstrap_iterations": 20,
        },
        "hpo": {"cox": {"alpha": [0.1]}},
    }

    report = run_amyloid_regional_development_from_locked_endpoint(
        config=config,
        output_root=tmp_path,
    )

    assert report["development_cohort"] == "A4"
    assert report["development_membership_count"] == 48
    assert report["internal_transport_membership_count"] == 6
    features = report["locked_config"]["feature_sets"]
    added = set(features["clinical_cognition_amyloid"]) - set(features["clinical_cognition"])
    assert added == {"baseline_amyloid_centiloid"}
    assert "treatment_active" in features["clinical_cognition"]
    assert report["claim_status"] == "EXPLORATORY"
    assert report["paired_bootstrap"]["status"] == "COMPLETED"
    assert "SCANNER_SHORTCUT_NOT_CONTROLLED" in report["science_gate"]["blockers"]
    assert (tmp_path / "locked" / "amyloid_locked_config.json").is_file()
