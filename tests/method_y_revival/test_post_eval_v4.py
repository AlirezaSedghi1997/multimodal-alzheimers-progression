"""RED-first tests for post-training evaluation program (03-combined-plan + A1).

CPU-only, synthetic data, no training contact, no live dirs. Must FAIL before
`post_eval_v4.py` exists and PASS after.
"""
from __future__ import annotations

import numpy as np
import pytest

SEED = 13


def _synth(n=120, seed=SEED, censor_rate=0.7):
    rng = np.random.RandomState(seed)
    t = rng.exponential(scale=800.0, size=n)
    c = rng.exponential(scale=800.0 * (1 - censor_rate) / max(censor_rate, 1e-6), size=n)
    time = np.minimum(t, c)
    event = (t <= c)
    risk = 1.0 - (t / t.max())
    return time, event, risk


def test_rng_streams_deterministic():
    from src.method_y_revival.post_eval_v4 import spawned_rng
    a = spawned_rng(13, "bootstrap").random(5)
    b = spawned_rng(13, "bootstrap").random(5)
    assert (a == b).all()
    c = spawned_rng(13, "permutation").random(5)
    assert not (a == c).all()


def test_ipcw_weights_truncation_flag():
    from src.method_y_revival.post_eval_v4 import ipcw_weights
    time, event, _ = _synth(200)
    w, diag = ipcw_weights(time, event, time, event, cap=10.0)
    assert (w > 0).all() and w.max() <= 10.0 + 1e-9
    assert 0.0 <= diag["frac_truncated"] <= 1.0 and diag["min_G"] > 0


def test_tau_star_rule_no_outer_clause():
    from src.method_y_revival.post_eval_v4 import tau_star
    g_times = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0])
    g_vals = np.array([1.0, 0.8, 0.5, 0.2, 0.05])
    t = tau_star(g_times, g_vals, max_horizon=1640.0, gmin=0.15)
    assert t == 1500.0


def test_uno_perfect_ordering_and_harrell_delta():
    from src.method_y_revival.post_eval_v4 import uno_c, harrell_c
    time, event, risk = _synth(150)
    u = uno_c(time, event, time, event, risk, tau=1000.0)
    h = harrell_c(time, event, risk)
    assert 0.9 < u["c"] <= 1.0 and 0.9 < h["c"] <= 1.0
    assert u["tau"] == 1000.0


def test_ibs_finite_and_scaled():
    from src.method_y_revival.post_eval_v4 import ibs_summary
    time, event, risk = _synth(150)
    surv = np.clip(1.0 - risk[:, None] * np.array([0.2, 0.4, 0.6]), 0.01, 0.99)
    out = ibs_summary(time, event, time, event, surv, times=[300.0, 600.0, 900.0])
    assert np.isfinite(out["ibs"]) and 0.0 <= out["ibs"] <= 1.0
    assert "brier_curve" in out and len(out["brier_curve"]) == 3


def test_calibration_descriptive_slope_ci():
    from src.method_y_revival.post_eval_v4 import calibration_at_horizon
    rng = np.random.RandomState(3)
    n = 300
    p = rng.uniform(0.2, 0.9, size=n)
    time = rng.exponential(scale=1000.0, size=n)
    event = rng.rand(n) < p * 0.8
    out = calibration_at_horizon(time, event, p, horizon=800.0,
                                 train_time=time, train_event=event, n_bins=5)
    assert "slope" in out and "slope_ci" in out and "ici" in out
    assert out["n_bins"] == 5


def test_paired_bootstrap_covers_zero_for_identical():
    from src.method_y_revival.post_eval_v4 import paired_bootstrap_delta
    rng = np.random.RandomState(5)
    a = rng.rand(80)
    out = paired_bootstrap_delta(a, a, n_boot=200, seed=13, fold_id=np.zeros(80, int))
    assert out["ci_low"] <= 0.0 <= out["ci_high"]
    assert out["method"] == "percentile-fold-stratified"
    assert "bca" in out


def test_paired_bootstrap_detects_shift():
    from src.method_y_revival.post_eval_v4 import paired_bootstrap_delta
    rng = np.random.RandomState(6)
    a = rng.rand(120)
    b = a + 0.2
    out = paired_bootstrap_delta(a, b, n_boot=200, seed=13, fold_id=np.zeros(120, int))
    assert out["ci_high"] < 0


def test_permutation_p_range():
    from src.method_y_revival.post_eval_v4 import permutation_paired_p
    rng = np.random.RandomState(7)
    a = rng.rand(60)
    p = permutation_paired_p(a, a + 0.001, n_perm=100, seed=13)
    assert 0.0 <= p <= 1.0


def test_ni_template_decisions():
    from src.method_y_revival.post_eval_v4 import ni_retention
    assert ni_retention(-0.004, (-0.009, 0.001), margin=0.01, higher_better=False)["decision"] == "RETAIN_NEGATIVE"
    assert ni_retention(0.005, (-0.004, 0.014), margin=0.01, higher_better=False)["decision"] == "INCONCLUSIVE_UNDERPOWERED"
    assert ni_retention(0.03, (0.01, 0.05), margin=0.02, higher_better=True)["decision"] == "SUPERIOR"


def test_risk_table_counts():
    from src.method_y_revival.post_eval_v4 import risk_table
    time, event, _ = _synth(100)
    out = risk_table(time, event, horizons=[500.0, 1000.0])
    for r, h in zip(out, (500.0, 1000.0)):
        cens_before = int(((~event) & (time < h)).sum())
        assert r["at_risk"] + r["events_by"] + cens_before == 100
    assert all(r["events_by"] >= 0 for r in out)


def test_subgroup_min_n_rules():
    from src.method_y_revival.post_eval_v4 import subgroup_decision
    assert subgroup_decision(n_events=30) == "CI_ALLOWED"
    assert subgroup_decision(n_events=15) == "DESCRIPTIVE_ONLY"
    assert subgroup_decision(n_events=9) == "NOT_ESTIMABLE"


def test_stratified_manufacturer_descriptive():
    from src.method_y_revival.post_eval_v4 import stratified_manufacturer_report
    out = stratified_manufacturer_report(
        {"Siemens": (0.70, 60), "GE": (0.65, 9)}, min_events=25)
    assert out["manufacturers"]["Siemens"]["status"] == "ESTIMABLE"
    assert out["manufacturers"]["GE"]["status"] == "NOT_MEASURABLE"
    assert out["gate_input"] is False


def test_tripod_machine_checks():
    from src.method_y_revival.post_eval_v4 import tripod_machine_checks
    good = {"fold_hashes": ["a"], "n_events": 10, "metric_def": "Uno-C(tau=1095)",
            "B": 2000, "seed": 13, "code_hash": "x" * 64, "pairing": "paired-within-contrast"}
    out = tripod_machine_checks(good)
    assert out["status"] == "PASS"
    bad = dict(good, B=0)
    out = tripod_machine_checks(bad)
    assert out["status"] == "FAIL"


def test_attestation_schema_two_slots():
    from src.method_y_revival.post_eval_v4 import validate_attestation
    att = {"item": "PROBAST-bias-1",
           "attesters": [{"name": "A", "quote": "q1", "digest": "d" * 64},
                         {"name": "B", "quote": "q2", "digest": "d" * 64}],
           "dissent": ""}
    assert validate_attestation(att)["status"] == "ATTESTED_ONLY"
    att1 = dict(att, attesters=att["attesters"][:1])
    assert validate_attestation(att1)["status"] == "FAIL"


def test_thesis_writer_allowlist_and_ceiling():
    from src.method_y_revival.post_eval_v4 import thesis_write
    rec = {"pointer": "$.e2.uno", "text": "Uno-C 0.70 [0.62, 0.77]",
           "tokens": ["Uno-C", "0.70"]}
    out = thesis_write([rec], ceiling="POST_HOC_INTERNAL_EXPLORATORY",
                       section="results")
    assert out["status"] == "PASS" and "0.70" in out["markdown"]
    bad = thesis_write([dict(rec, text="Proves early diagnosis")],
                       ceiling="POST_HOC_INTERNAL_EXPLORATORY", section="results")
    assert bad["status"] == "BLOCKED"


def test_thesis_not_measurable_boilerplate():
    from src.method_y_revival.post_eval_v4 import thesis_write
    out = thesis_write([], ceiling="POST_HOC_INTERNAL_EXPLORATORY", section="results",
                       not_measurable=["LEG-EVAL-06-tau-GE"])
    assert out["status"] == "PASS" and "NOT_MEASURABLE" in out["markdown"]


def test_privacy_k_suppression():
    from src.method_y_revival.post_eval_v4 import privacy_cell_check, scan_text
    assert privacy_cell_check(n_subjects=30, n_events=12)["status"] == "PASS"
    assert privacy_cell_check(n_subjects=9, n_events=9)["status"] == "SUPPRESSED"
    assert scan_text("aggregate Uno-C 0.70")["status"] == "PASS"
    assert scan_text("subject pt_abc123 path H:/x")["status"] == "BLOCKED"


def test_legs_runner_halts_on_blocked():
    from src.method_y_revival.post_eval_v4 import run_legs
    legs = [{"id": "LEG-EVAL-00", "status": "PASS"},
            {"id": "LEG-EVAL-02", "status": "BLOCKED", "reason": "x"},
            {"id": "LEG-EVAL-03", "status": "PASS"}]
    out = run_legs(legs)
    assert out["status"] == "BLOCKED" and out["halted_at"] == "LEG-EVAL-02"
    legs2 = [{"id": "LEG-EVAL-00", "status": "PASS"},
             {"id": "LEG-EVAL-06", "status": "NOT_MEASURABLE", "evidence": "e.json"}]
    out = run_legs(legs2)
    assert out["status"] == "PASS"


def test_entry_gate_requires_verifier(tmp_path):
    from src.method_y_revival.post_eval_v4 import entry_gate
    with pytest.raises(RuntimeError, match="BLOCKED"):
        entry_gate(run_root=tmp_path / "nope")
