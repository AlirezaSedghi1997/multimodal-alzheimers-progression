import pytest


def test_conformal_prediction_basic():
    from src.evaluation.conformal_prediction import evaluate_conformal_prediction

    y_true = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.85, 0.15, 0.75]
    result = evaluate_conformal_prediction(y_true, y_prob, alpha=0.1)
    assert result["alpha"] == 0.1
    assert result["target_coverage"] == 0.9
    assert 0.0 <= result["empirical_coverage"] <= 1.0
    assert result["average_prediction_set_size"] >= 1.0
    assert result["n_samples"] == 10


def test_conformal_prediction_high_coverage():
    from src.evaluation.conformal_prediction import evaluate_conformal_prediction

    y_true = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.85, 0.15, 0.75]
    result = evaluate_conformal_prediction(y_true, y_prob, alpha=0.05)
    assert result["target_coverage"] == 0.95
    assert result["empirical_coverage"] >= 0.9


def test_conformal_prediction_sets():
    from src.evaluation.conformal_prediction import conformal_prediction_sets

    y_prob = [0.1, 0.5, 0.9]
    sets = conformal_prediction_sets(y_prob, 0.5, score_type="margin")
    assert len(sets) == 3
    for s in sets:
        assert len(s) >= 1
        assert all(label in [0, 1] for label in s)


def test_conformal_risk_control():
    from src.evaluation.conformal_prediction import conformal_risk_control

    y_true = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.85, 0.15, 0.75]
    result = conformal_risk_control(y_true, y_prob, target_error_rate=0.1)
    assert result["target_error_rate"] == 0.1
    assert 0.0 <= result["empirical_error_rate"] <= 1.0
    assert result["n_samples"] == 10


def test_nonconformity_scores_margin():
    from src.evaluation.conformal_prediction import compute_nonconformity_scores

    y_true = [0, 1]
    y_prob = [0.3, 0.7]
    scores = compute_nonconformity_scores(y_true, y_prob, score_type="margin")
    assert abs(scores[0] - 0.3) < 1e-6
    assert abs(scores[1] - 0.3) < 1e-6


def test_nonconformity_scores_entropy():
    import math

    from src.evaluation.conformal_prediction import compute_nonconformity_scores

    y_true = [0, 1]
    y_prob = [0.3, 0.7]
    scores = compute_nonconformity_scores(y_true, y_prob, score_type="entropy")
    assert abs(scores[0] - (-math.log(0.7))) < 1e-6
    assert abs(scores[1] - (-math.log(0.7))) < 1e-6
