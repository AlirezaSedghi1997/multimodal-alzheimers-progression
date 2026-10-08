"""RED-first tests for post-training evaluation program (03-combined-plan + A1).

CPU-only, synthetic data, no training contact, no live dirs. Must FAIL before
`post_eval_v4.py` exists and PASS after.
"""
from __future__ import annotations

import numpy as np
import pytest

SEED = 13


def _synth(n=120, seed=SEED, censor_rate=0.3):
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
    keep = time < np.quantile(time, .95)
    w, diag = ipcw_weights(time, event, time[keep], event[keep], cap=10.0)
    assert (w > 0).all() and w.max() <= 10.0 + 1e-9
    assert 0.0 <= diag["frac_truncated"] <= 1.0 and diag["min_G"] > 0


def test_ipcw_uses_censoring_indicator_not_outcome(monkeypatch):
    import src.method_y_revival.post_eval_v4 as pe

    captured = {}

    def fake_km(times, events):
        captured["events"] = np.asarray(events, dtype=bool).copy()
        return np.asarray([0.0, 1.0]), np.asarray([1.0, 1.0])

    monkeypatch.setattr(pe, "_censoring_km", fake_km)
    train_event = np.asarray([True, False, True, False])
    pe.ipcw_weights(np.arange(4.0), train_event, np.asarray([1.0]),
                    np.asarray([True]))
    # Reverse KM accepts the original outcome event indicator, not its inverse.
    assert captured["events"].tolist() == train_event.tolist()


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


def test_ibs_uses_train_km_null_and_drops_equal_support_horizon():
    from src.method_y_revival.post_eval_v4 import ibs_summary

    train_time = np.asarray([100., 200., 300., 400., 500., 1000.])
    train_event = np.asarray([True, False, True, False, True, False])
    test_time = np.asarray([50., 150., 250., 350., 450., 550.])
    test_event = np.asarray([False, True, False, True, False, False])
    surv = np.asarray([[.95, .85, .75], [.9, .8, .7], [.85, .75, .65],
                       [.8, .7, .6], [.75, .65, .55], [.7, .6, .5]])
    out = ibs_summary(train_time, train_event, test_time, test_event, surv,
                      times=[200., 400., 600.])
    assert out["dropped_times"] == [600.0]
    assert out["null_source"] == "train_only_kaplan_meier"
    assert len(out["null_survival"]) == 2


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


def test_calibration_uses_event_risk_not_survival_probability():
    from src.method_y_revival.post_eval_v4 import calibration_at_horizon

    rng = np.random.RandomState(312)
    event_risk = rng.uniform(0.1, 0.9, size=4000)
    event = rng.rand(event_risk.size) < event_risk
    time = np.where(event, 200.0, 800.0)
    out = calibration_at_horizon(
        time, event, 1.0 - event_risk, horizon=500.0,
        train_time=time, train_event=event,
    )
    assert out["status"] == "PASS"
    assert out["prediction_scale"] == "event_risk_by_horizon"
    assert 0.8 < out["slope"] < 1.2
    assert out["ici"] < 0.05
    assert out["slope_ci"][0] <= out["slope"] <= out["slope_ci"][1]
    assert out["intercept_ci"][0] <= out["intercept"] <= out["intercept_ci"][1]


def test_kaplan_meier_values_are_steps_not_linearly_interpolated():
    from src.method_y_revival.post_eval_v4 import _km_step_values, ipcw_weights

    assert _km_step_values([200.0, 800.0], [1.0, 0.0], [500.0]).tolist() == [1.0]
    weights, _ = ipcw_weights(
        np.array([200.0, 800.0]), np.array([True, False]),
        np.array([500.0]), np.array([False]),
    )
    assert weights.tolist() == [1.0]


def test_calibration_perfect_separation_is_not_estimable():
    from src.method_y_revival.post_eval_v4 import calibration_at_horizon

    event = np.array([True] * 30 + [False] * 30)
    time = np.where(event, 200.0, 800.0)
    pred_surv = np.where(event, 0.1, 0.9)
    out = calibration_at_horizon(
        time, event, pred_surv, horizon=500.0,
        train_time=time, train_event=event,
    )
    assert out["status"] == "NOT_ESTIMABLE"
    assert out["reason"] == "PERFECT_SEPARATION"


def test_calibration_excludes_censoring_before_horizon():
    from src.method_y_revival.post_eval_v4 import calibration_at_horizon

    time = np.asarray([100., 200., 300., 400., 500., 600., 700., 800.])
    event = np.asarray([False, False, True, False, True, False, True, False])
    pred = np.asarray([.2, .25, .3, .35, .45, .5, .65, .7])
    out = calibration_at_horizon(time, event, pred, horizon=500.,
                                 train_time=time, train_event=event, n_bins=4)
    assert out["n_excluded_censored_before_horizon"] == 3
    assert out["n_used"] == 5


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


def test_science_gate_separates_deferred_legs_from_execution_status():
    from src.method_y_revival.post_eval_v4 import science_gate_from_legs

    out = science_gate_from_legs([
        {"id": "LEG-EVAL-00", "status": "PASS"},
        {"id": "LEG-EVAL-12", "status": "NOT_MEASURABLE",
         "evidence": "external_pending.json"},
    ], external_pending=True)
    assert out["science_gate_status"] == "BLOCKED"
    assert out["positive_claim_allowed"] is False
    assert len(out["blockers"]) == 2


def test_entry_gate_requires_verifier(tmp_path):
    from src.method_y_revival.post_eval_v4 import entry_gate
    with pytest.raises(RuntimeError, match="BLOCKED"):
        entry_gate(run_root=tmp_path / "nope")


def test_journal_keys_branch_unique(tmp_path):
    import json as _js
    import sys as _sys
    _sys.path.insert(0, r"H:\New project")
    from scripts.run_method_y_post_eval_v4 import _read_access_counts
    base = tmp_path / "private" / "outer_evaluation"
    for branch in ("amyloid", "tau"):
        for arm in ("A0",):
            for fold in (0, 1):
                d = base / branch / arm / f"fold_{fold}"
                d.mkdir(parents=True)
                (d / "outer_access_journal.json").write_text(
                    _js.dumps({"access_count": 1}), encoding="utf-8")
    counts = _read_access_counts(tmp_path)
    assert len(counts) == 4
    assert "tau/A0/fold_1" in counts


def test_sanitize_absolute_paths_roundtrip(tmp_path):
    from src.method_y_revival.post_eval_v4 import sanitize_absolute_paths
    src = tmp_path / "orig.json"
    src.write_text('{"a": "H:\\\\New project\\\\x\\\\y.json", "b": 1}', encoding="utf-8")
    before = src.read_bytes()
    out = sanitize_absolute_paths(source=src, dest=tmp_path / "mirror.json",
                                  project_root=r"H:\New project")
    assert out["status"] == "APPLIED" and out["replacements"] == 1
    assert src.read_bytes() == before
    mirror = (tmp_path / "mirror.json").read_text(encoding="utf-8")
    assert "PROJECT_ROOT/x" in mirror and "H:\\" not in mirror
    import json as _js
    assert _js.loads(mirror)["b"] == 1
    out2 = sanitize_absolute_paths(source=tmp_path / "mirror.json",
                                   dest=tmp_path / "m2.json",
                                   project_root=r"H:\New project")
    assert out2["status"] == "NOTHING_TO_SANITIZE"


def _locked_tree(root, digest="d" * 64):
    import json as _js
    (root / "orchestration_state.json").write_text(_js.dumps(
        {"stage": "FINALIZED", "outer_cells_finalized": 50}), encoding="utf-8")
    (root / "run_contract.json").write_text(
        _js.dumps({"digest": digest}), encoding="utf-8")
    pairs = {}
    for branch in ("amyloid", "tau"):
        for arm in ("A0", "A1", "A2", "A3", "A4"):
            for fold in range(5):
                rep = {"schema_version": 1, "status": "COMPLETED",
                       "run_contract_sha256": digest,
                       "aggregate": {
                           "uno_c_index": 0.68, "integrated_brier_score": 0.09,
                           "n_test": 100, "test_events": 20, "n_train": 400,
                           "horizon_metrics": {
                               "1095.75": {"dynamic_auc": 0.70,
                                           "status": "ESTIMABLE"}}}}
                p = root / "reports" / f"{branch}_{arm}_fold_{fold}_public.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(_js.dumps(rep), encoding="utf-8")
    for pair in ("amyloid:A0_vs_A1", "tau:A0_vs_A1"):
        pairs[pair] = {"acceptance_gate": {
            "delta_ibs_95ci": [-0.002, 0.001],
            "delta_uno_cindex_95ci": [-0.01, 0.02],
            "fold_delta_ibs": [0.001] * 5,
            "exploratory_operational_acceptance": False}}
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "factorial_benchmark_summary.json").write_text(_js.dumps(
        {"status": "COMPLETED", "pair_reports": pairs}), encoding="utf-8")


def test_compile_locked_reports_end_to_end(tmp_path):
    from src.method_y_revival.post_eval_v4 import compile_locked_reports
    _locked_tree(tmp_path)
    out = compile_locked_reports(root=tmp_path)
    assert out["status"] == "PASS"
    assert len(out["per_arm"]) == 10
    assert out["contrasts"][0]["ni_ibs"]["decision"] == "RETAIN_NEGATIVE"
    assert out["legs"]["status"] == "PASS"
    assert out["omnibus"] is not None


def test_compile_locked_reports_missing_core(tmp_path):
    from src.method_y_revival.post_eval_v4 import compile_locked_reports
    with pytest.raises(RuntimeError, match="BLOCKED"):
        compile_locked_reports(root=tmp_path)


def test_support_truncate_tail_leg():
    from src.method_y_revival.post_eval_v4 import support_truncate
    rows = [{"time_days": float(t)} for t in (100.0, 500.0, 900.0)]
    out = support_truncate(rows, [500.0, 1640.0])
    assert out["horizons"] == [500.0]
    assert out["legs"][0]["status"] == "NOT_MEASURABLE"
    out = support_truncate(rows, [500.0])
    assert out["horizons"] == [500.0] and out["legs"] == []
    with pytest.raises(RuntimeError, match="BLOCKED"):
        support_truncate(rows, [5000.0])


def test_support_truncate_records_horizon_equal_to_tmax():
    from src.method_y_revival.post_eval_v4 import support_truncate

    out = support_truncate([{"time_days": 100.}, {"time_days": 500.}],
                           [300., 500.])
    assert out["horizons"] == [300.]
    assert out["legs"][0]["status"] == "NOT_MEASURABLE"
    assert out["legs"][0]["dropped_horizons"] == [500.]


def _oof_fixture(seed=13):
    from src.method_y_revival.post_eval_v4 import evaluate_oof_bundle
    rng = np.random.RandomState(seed)
    rows, train = [], {}
    for branch, (n, censor) in (("amyloid", (60, 0.7)),):
        for outer in (0, 1):
            tt = rng.exponential(800.0, size=120)
            cc = rng.exponential(1200.0, size=120)
            train[(branch, outer)] = (np.minimum(tt, cc), tt <= cc)
            for i in range(n):
                t = float(rng.exponential(800.0))
                c = float(rng.exponential(1200.0))
                ev = bool(t <= c)
                to = float(min(t, c))
                base = 1.0 - to / 2000.0
                for arm, shift in (("A0", 0.0), ("A1", 0.05), ("A2", -0.03)):
                    risk = float(np.clip(base + shift, 0.01, 0.99))
                    rows.append({
                        "patient_hash": f"p{branch}{outer}_{i}", "arm": arm,
                        "branch": branch, "outer_fold": outer,
                        "split_digest": "d" * 64, "time_days": to,
                        "event": ev, "risk": risk,
                        "surv_at_horizons": {"500.0": 1.0 - risk * 0.5,
                                             "1000.0": 1.0 - risk * 0.8}})
    return rows, train


def test_evaluate_oof_bundle_end_to_end():
    from src.method_y_revival.post_eval_v4 import evaluate_oof_bundle
    rows, train = _oof_fixture()
    out = evaluate_oof_bundle(rows, train_by_partition=train,
                              horizons=[500.0, 1000.0], tau=1000.0,
                              B=100, seed=13)
    assert out["status"] == "PASS" and out["arms"] == ["A0", "A1", "A2"]
    assert len(out["contrasts"]) == 2 and all(
        c["status"] == "INVALID_FOR_PERFORMANCE_INFERENCE" and c["ni"] is None
        for c in out["contrasts"])
    assert out["per_arm"]["A0"]["events"] > 0
    for arm in out["arms"]:
        item = out["per_arm"][arm]
        assert item["calibration_folds_estimable"] <= item["calibration_folds_total"]
        if item["calibration_status"] != "ESTIMABLE_ALL_FOLDS":
            assert item["calibration_slope_mean"] is None
    assert out["omnibus"] is not None


def test_evaluate_oof_bundle_missing_train_ipcw():
    from src.method_y_revival.post_eval_v4 import evaluate_oof_bundle
    rows, _ = _oof_fixture()
    with pytest.raises(RuntimeError, match="BLOCKED"):
        evaluate_oof_bundle(rows, train_by_partition={},
                            horizons=[500.0], tau=500.0, B=50, seed=1)


def test_audit_probe_weights():
    from src.method_y_revival.post_eval_v4 import audit_probe_weights
    good = {"weights_source": "inner-train", "solver": "lbfgs", "seed": 13,
            "C": 1.0, "digest": "a" * 64}
    assert audit_probe_weights(good)["status"] == "PASS"
    bad = dict(good, weights_source="outer-fit")
    assert audit_probe_weights(bad)["status"] == "PROBE_REUSE_VIOLATION"
    assert audit_probe_weights({})["status"] == "PROBE_REUSE_VIOLATION"


def test_bh_fdr_and_omnibus():
    from src.method_y_revival.post_eval_v4 import bh_fdr, omnibus_friedman
    rej = bh_fdr([0.001, 0.01, 0.03, 0.5, 0.9], alpha=0.05)
    assert rej == [True, True, True, False, False]
    arms = [[0.70, 0.72, 0.69, 0.71, 0.70],
            [0.71, 0.73, 0.70, 0.72, 0.71],
            [0.55, 0.56, 0.54, 0.55, 0.56]]
    out = omnibus_friedman(arms)
    assert 0.0 <= out["p"] <= 1.0 and out["stat"] >= 0


def test_mixedlm_descriptive_or_not_measurable():
    import pandas as pd
    from src.method_y_revival.post_eval_v4 import mixedlm_pacc, missingness_table
    rng = np.random.RandomState(11)
    rows = []
    for s in range(40):
        for t in (0, 365, 730):
            rows.append({"subject": f"s{s}", "time": t,
                         "treatment": s % 2,
                         "pacc": 50 - 0.01 * t + rng.randn() * 2.0})
    df = pd.DataFrame(rows)
    out = mixedlm_pacc(df)
    assert out["status"] in ("DESCRIPTIVE_FIT", "NOT_MEASURABLE")
    if out["status"] == "DESCRIPTIVE_FIT":
        assert "treatment" in str(out["params"])
    mt = missingness_table(df)
    assert mt["n_subjects"] == 40 and "dropout_note" in mt


def test_intake_lint_static_only():
    from src.method_y_revival.post_eval_v4 import intake_lint
    good = {"schema_version": 1, "tracer": "FBP", "endpoint": "CDR",
            "censoring": "administrative", "quarantine": "private-only",
            "mapping_lock": False, "data_rows": []}
    assert intake_lint(good)["status"] == "PASS"
    bad = dict(good, data_rows=[{"x": 1}])
    assert intake_lint(bad)["status"] == "BLOCKED"
    bad2 = dict(good, mapping_lock=True)
    assert intake_lint(bad2)["status"] == "BLOCKED"


def test_cam_gates_read():
    from src.method_y_revival.post_eval_v4 import cam_gates_read
    rep = {"ceiling": "POST_HOC_INTERNAL_EXPLORATORY",
           "gates": {"determinism_rerun": "bit-identical", "fidelity": "PASS"},
           "legs": {"status": "PASS"}}
    assert cam_gates_read(rep)["status"] == "PASS"
    rep2 = dict(rep, ceiling="CONFIRMED_DIAGNOSTIC")
    assert cam_gates_read(rep2)["status"] == "BLOCKED"
    blocked = dict(rep, science_gate_status="FAIL",
                   claim_status="NO_CLAIM", claim_blockers=["agreement"])
    assert cam_gates_read(blocked)["status"] == "BLOCKED"


def test_assert_fusion_keys_dedup():
    from src.method_y_revival.post_eval_v4 import assert_fusion_keys
    rows = [{"patient_hash": f"p{i}", "arm": "A0", "branch": "amyloid",
             "outer_fold": i % 5, "split_digest": "d" * 64} for i in range(10)]
    assert assert_fusion_keys(rows)["status"] == "PASS"
    dup = rows + [dict(rows[0])]
    assert assert_fusion_keys(dup)["status"] == "BLOCKED"
    assert assert_fusion_keys([{"patient_hash": "p1"}])["status"] == "BLOCKED"


def test_firewall_assertion():
    from src.method_y_revival.post_eval_v4 import (
        assert_single_access, firewall_assertion)
    evts = [{"type": "outer_result_read", "t_view": 25, "t_lock": 20}]
    assert firewall_assertion(evts)["status"] == "PASS"
    evts2 = [{"type": "outer_result_read", "t_view": 5, "t_lock": 20}]
    assert firewall_assertion(evts2)["status"] == "EXACTLY_ONCE_AT_RISK"
    assert firewall_assertion([])["status"] == "PASS"
    assert assert_single_access({"a/0": 1, "a/1": 1})["status"] == "PASS"
    with pytest.raises(RuntimeError, match="BLOCKED"):
        assert_single_access({"a/0": 1, "a/1": 2})
    with pytest.raises(RuntimeError, match="BLOCKED"):
        assert_single_access({})


def test_selection_stability():
    from src.method_y_revival.post_eval_v4 import selection_stability
    recs = [{"outer": k, "winner_candidate_id": "lr1e4_wd1e3"} for k in range(5)]
    out = selection_stability(recs)
    assert out["stable"] is True and out["n_partitions"] == 5
    recs2 = recs + [{"outer": 5, "winner_candidate_id": "other"}]
    assert selection_stability(recs2)["stable"] is False


def test_d_calibration_never_gate():
    from src.method_y_revival.post_eval_v4 import d_calibration
    rng = np.random.RandomState(21)
    s = rng.uniform(0.05, 0.95, size=100)
    out = d_calibration(s, n_bins=10)
    assert out["gate_effect"] == "NONE" and 0.0 <= out["p"] <= 1.0
    assert out["n_bins"] == 10 and out["n"] == 100


def test_interaction_did():
    from src.method_y_revival.post_eval_v4 import interaction_did
    rng = np.random.RandomState(31)
    base = rng.rand(120)
    sub = np.array([0] * 60 + [1] * 60)
    out = interaction_did(base, base + 0.001, sub, n_boot=200, seed=13)
    assert "ci_low" in out and "ci_high" in out
    assert out["n_subgroups"] == 2


def test_antolini_ctd_sane():
    from src.method_y_revival.post_eval_v4 import antolini_ctd
    time = np.array([100.0, 200.0, 300.0, 400.0])
    event = np.array([True, True, False, True])
    perfect = np.array([[0.2, 0.1, 0.05], [0.9, 0.3, 0.15],
                        [0.95, 0.9, 0.85], [0.9, 0.8, 0.2]])
    out = antolini_ctd(time, event, perfect, times=[100.0, 200.0, 300.0])
    assert 0.5 < out["ctd"] <= 1.0
    assert out["role"] == "EXPLORATORY"


def test_run_legs_unknown_propagates():
    from src.method_y_revival.post_eval_v4 import run_legs
    out = run_legs([{"id": "LEG-EVAL-01", "status": "PASS"},
                    {"id": "LEG-EVAL-06", "status": "UNKNOWN",
                     "evidence": "pending_provenance.json"}])
    assert out["status"] == "PASS"
    assert out["completed"] == ["LEG-EVAL-01", "LEG-EVAL-06"]


def test_ban_check_prose_gate():
    from src.method_y_revival.post_eval_v4 import ban_check

    assert ban_check("regions of higher model attribution") == {
        "status": "PASS", "hits": []}
    bad = ban_check("Alzheimer-involved regions show amyloid markers on Centiloid scale")
    assert bad["status"] == "BLOCKED" and len(bad["hits"]) >= 3
