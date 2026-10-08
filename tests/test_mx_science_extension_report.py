"""MX-07 extension reporting must not hide negative results or XAI gaps."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.mx_science_extension_v1 import report


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _analysis() -> dict:
    metrics = {
        metric: {"selected_minus_comparator": value,
                 "patient_bootstrap_ci95": [value - 0.1, value + 0.1],
                 "status": "ESTIMATED"}
        for metric, value in (("auroc", -0.02), ("auprc", 0.04),
                              ("brier", 0.01), ("calibration_intercept", 0.03),
                              ("calibration_slope", -0.05))}
    comparisons = {"hybrid_selected_minus_tabular_only": {"metrics": metrics}}
    arms = {"hybrid_selected": {"seed_set": [13, 29, 47],
                                "metrics": {"auroc": 0.7, "auprc": 0.8,
                                            "brier": 0.2,
                                            "calibration_intercept": 0.0,
                                            "calibration_slope": 1.0}}}
    return {"all_patient_count": 1336, "outer_lock_sha256": "o" * 64,
            "subsets": {window: {"n": 100, "positive_n": 50,
                                 "negative_n": 50, "status": "DESCRIPTIVE_POST_HOC_INTERNAL",
                                 "comparisons": comparisons, "arm_metrics": arms}
                        for window in ("all", "90", "180", "365")}}


def test_addendum_renders_negative_effects_and_limits(tmp_path: Path,
                                                      monkeypatch) -> None:
    monkeypatch.setattr(report, "OUTPUT", Path("out"))
    _write(tmp_path / "out/reports/design_lock.json", {"status": "LOCKED"})
    _write(tmp_path / "out/reports/day1_patient_provenance_audit.json", {
        "pet_image_provenance": "UNVERIFIED", "scanner_overlap": "UNKNOWN",
        "pseudo_holdout_prior_access": "UNKNOWN"})
    result = report.render_addendum(tmp_path, _analysis(), {"status": "BLOCKED_EXPLORATORY_GATE"})
    assert "جهت اثر به نفع مدل منتخب نیست" in result
    assert "A4 فقط کلاس ۱، LEARN فقط کلاس ۰" in result
    assert "تأیید ناحیهٔ بیماری نیستند" in result
    assert "BLOCKED_EXPLORATORY_GATE" in result
    assert "TRIPOD+AI" in result


def test_absent_five_fold_xai_cannot_complete_all(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report, "PARENT", Path("parent"))
    monkeypatch.setattr(report, "require_parent_mx7_terminal",
                        lambda root: {"freeze_sha256": "f" * 64,
                                      "v242_amendment_sha256": "a" * 64})
    _write(tmp_path / "parent/reports/fasttrack_v242_supervisor_state.json",
           {"steps": {"mx6_v22_xai_crossfit": {"status": "PAUSED_OR_FAILED_RESUMABLE"}}})
    with pytest.raises(RuntimeError, match="XAI_FIVE_FOLD_SUMMARY_NOT_DONE"):
        report.xai_audit_payload(tmp_path)

