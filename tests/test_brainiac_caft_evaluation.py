"""Outer exactly-once evaluation, calibration and promotion tests (Plan 006)."""

from __future__ import annotations

import numpy as np
import pytest

from src.research import brainiac_caft_evaluation as ce


def _scores(n: int, seed: int = 0, positive_rate: float = 0.4):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < positive_rate).astype("int64")
    logits = rng.normal(size=n).astype("float64") + labels * 1.5
    return [f"s_{seed}_{i:04d}" for i in range(n)], labels, logits


# ---------------------------------------------------------------------------
# outer state machine
# ---------------------------------------------------------------------------


def test_outer_loader_raises_and_counts_zero_before_authorization():
    machine = ce.OuterStateMachine(fold=0)
    assert machine.state() == "HEAD_LOCKED"
    with pytest.raises(RuntimeError, match="OUTER_NOT_AUTHORIZED"):
        machine.load_outer()
    assert machine.loader_calls() == 0
    machine.transition("CALIBRATION_LOCKED")
    with pytest.raises(RuntimeError, match="OUTER_NOT_AUTHORIZED"):
        machine.load_outer()
    assert machine.loader_calls() == 0


def test_outer_snapshot_exactly_once_and_crash_reuse(tmp_path):
    machine = ce.OuterStateMachine(fold=0)
    machine.transition("CALIBRATION_LOCKED")
    machine.transition("OUTER_AUTHORIZED")
    machine.transition("OUTER_SNAPSHOT_PREPARED")
    snapshot = machine.snapshot_path()
    assert machine.loader_calls() == 1
    # crash after snapshot: re-entry must reuse, not reload
    machine2 = ce.OuterStateMachine(fold=0, resume_state="OUTER_SNAPSHOT_PREPARED")
    assert machine2.snapshot_path() == snapshot
    assert machine2.loader_calls() == 0
    machine.transition("OUTER_PREDICTIONS_COMMITTED")
    machine.transition("FOLD_FINALIZED")
    assert machine.state() == "FOLD_FINALIZED"


def test_state_machine_rejects_invalid_transitions():
    machine = ce.OuterStateMachine(fold=1)
    with pytest.raises(RuntimeError, match="INVALID_TRANSITION"):
        machine.transition("FOLD_FINALIZED")


# ---------------------------------------------------------------------------
# calibration role separation
# ---------------------------------------------------------------------------


def test_temperature_uses_only_probability_calibration():
    subjects, labels, logits = _scores(60, seed=3)
    result = ce.fit_temperature(subjects, labels, logits, role="probability_calibration")
    assert np.isfinite(result["temperature"]) and result["temperature"] > 0
    with pytest.raises(RuntimeError, match="CALIBRATION_ROLE_FORBIDDEN"):
        ce.fit_temperature(subjects, labels, logits, role="conformal_calibration")


def test_conformal_uses_only_conformal_calibration():
    subjects, labels, logits = _scores(60, seed=4)
    result = ce.fit_conformal(subjects, labels, logits, role="conformal_calibration")
    assert result["alpha"] == 0.10
    with pytest.raises(RuntimeError, match="CALIBRATION_ROLE_FORBIDDEN"):
        ce.fit_conformal(subjects, labels, logits, role="probability_calibration")


def test_mondrian_fallback_below_thirty_minority():
    subjects, labels, logits = _scores(50, seed=5, positive_rate=0.2)  # ~10 pos
    result = ce.fit_conformal(subjects, labels, logits, role="conformal_calibration")
    assert result["mondrian_status"] == "NOT_MEASURABLE"
    assert result["method"] == "split_conformal_marginal"


def test_calibration_label_is_empirical_post_hoc():
    subjects, labels, logits = _scores(60, seed=6)
    result = ce.fit_conformal(subjects, labels, logits, role="conformal_calibration")
    assert result["evidence_label"] == "EMPIRICAL_POST_HOC_CALIBRATION"
    assert "finite_sample" not in str(result).lower()


def test_no_refit_after_seeing_calibration_metrics():
    subjects, labels, logits = _scores(60, seed=7)
    first = ce.fit_temperature(subjects, labels, logits, role="probability_calibration")
    with pytest.raises(RuntimeError, match="REFIT_FORBIDDEN"):
        ce.fit_temperature(
            subjects, labels, logits, role="probability_calibration",
            previous_state=first,
        )


# ---------------------------------------------------------------------------
# metrics / negative controls / promotion
# ---------------------------------------------------------------------------


def test_metrics_are_finite_and_ranged():
    subjects, labels, logits = _scores(80, seed=8)
    report = ce.compute_metrics(labels, logits)
    for field in ("roc_auc", "pr_auc", "brier", "ece"):
        assert np.isfinite(report[field]), field
    assert 0.0 <= report["roc_auc"] <= 1.0
    assert 0.0 <= report["brier"] <= 1.0


def test_negative_controls_block_promotion_when_signal_is_destroyed():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=60)
    logits = rng.normal(size=60) + labels * 2.5  # strong real signal
    controls = ce.run_negative_controls(labels, logits, seed=13)
    assert controls["base_auc"] > 0.85
    # destroying the pairing must collapse to chance => controls fail
    assert controls["pass_all"] is False
    # single permutation draws carry ~0.07 sd noise; a 0.3 band is the
    # statistically sane chance bound for one draw
    assert abs(controls["shuffled_embedding_auc"] - 0.5) < 0.30


def test_promotion_blocked_by_unknown_or_not_measurable():
    promotion = ce.evaluate_technical_promotion(
        delta_auc=0.05, bootstrap_lower_ci=0.01, delta_brier=0.001,
        delta_ece=0.001, confound_worsened=False, sanity_passed=True,
        measurable=True,
    )
    assert promotion["decision"] == "TECHNICAL_PROMOTION"
    blocked = ce.evaluate_technical_promotion(
        delta_auc=0.05, bootstrap_lower_ci=0.01, delta_brier=0.001,
        delta_ece=0.001, confound_worsened=False, sanity_passed=True,
        measurable=False,
    )
    assert blocked["decision"] == "BLOCKED_NOT_MEASURABLE"
    too_small = ce.evaluate_technical_promotion(
        delta_auc=0.01, bootstrap_lower_ci=0.01, delta_brier=0.001,
        delta_ece=0.001, confound_worsened=False, sanity_passed=True,
        measurable=True,
    )
    assert too_small["decision"] == "BLOCKED_DELTA_AUC"
    assert promotion["lora_unlocked"] is False


def test_external_absence_is_nonblocking_and_labelled():
    status = ce.external_evaluation_status(dua_present=False)
    assert status["external_status"] == "EXTERNAL_UNAVAILABLE_NONBLOCKING"
    assert status["evidence_label"] == "ROBUSTNESS_EVIDENCE_NOT_EXTERNAL_VALIDATION"
