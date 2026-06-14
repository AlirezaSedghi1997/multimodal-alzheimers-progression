import pytest


def test_calibration_analysis_basic():
    from src.evaluation.calibration import calibration_analysis

    y_true = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.85, 0.15, 0.75]
    result = calibration_analysis(y_true, y_prob, n_bins=5)
    assert result["n_bins"] == 5
    assert result["n_samples"] == 10
    assert 0.0 <= result["ece"] <= 1.0
    assert len(result["bins"]) == 5


def test_calibration_analysis_perfect():
    from src.evaluation.calibration import calibration_analysis

    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    result = calibration_analysis(y_true, y_prob, n_bins=2)
    assert result["ece"] < 0.01


def test_brier_score():
    from src.evaluation.calibration import brier_score

    y_true = [0, 1]
    y_prob = [0.0, 1.0]
    assert brier_score(y_true, y_prob) == 0.0
    y_prob_wrong = [1.0, 0.0]
    assert brier_score(y_true, y_prob_wrong) == 1.0


def test_reliability_diagram_data():
    from src.evaluation.calibration import reliability_diagram_data

    y_true = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.85, 0.15, 0.75]
    result = reliability_diagram_data(y_true, y_prob, n_bins=5)
    assert "calibration" in result
    assert "perfect_calibration" in result
    assert len(result["perfect_calibration"]) > 0
