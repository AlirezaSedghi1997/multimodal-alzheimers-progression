"""Patient-level paired analysis and fail-closed panel schema tests."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scripts.mx_science_extension_v1.analysis import (
    ARMS, _metrics, _seed_mean_metrics, analyze_panel, validate_panel,
)


def _panel(n: int = 20):
    rows = []
    for i in range(n):
        label = i % 2
        base = 0.75 if label else 0.25
        predictions = {}
        for arm, seeds in ARMS.items():
            predictions[arm] = {}
            for seed in seeds:
                offset = {"hybrid_selected": 0.0, "hybrid_baseline": -0.03,
                          "hybrid_dense": 0.02, "tabular_only": -0.08,
                          "prevalence": -0.1, "mri_only": -0.12,
                          "random_label": -0.18}[arm]
                predictions[arm][str(seed)] = max(0.01, min(0.99, base + offset
                                                                + (seed - 13) * 0.001))
        rows.append({"patient_hash": f"patient-{i}", "label": label,
                     "fold": i % 5, "gap_days": float(i * 10),
                     "predictions": predictions})
    return rows


def test_panel_requires_exact_patient_seed_arm_grid():
    panel = _panel()
    assert len(validate_panel(panel, expected_n=20)) == 20
    duplicate = copy.deepcopy(panel)
    duplicate[1]["patient_hash"] = duplicate[0]["patient_hash"]
    with pytest.raises(ValueError, match="ID_LABEL"):
        validate_panel(duplicate, expected_n=20)
    missing_seed = copy.deepcopy(panel)
    del missing_seed[0]["predictions"]["hybrid_selected"]["47"]
    with pytest.raises(ValueError, match="SEEDS_INCOMPLETE"):
        validate_panel(missing_seed, expected_n=20)


def test_seed_metrics_are_averaged_not_pseudoreplicated():
    panel = validate_panel(_panel(), expected_n=20)
    indices = np.arange(len(panel))
    labels = np.asarray([row["label"] for row in panel])
    observed = _seed_mean_metrics(labels, panel, indices, "hybrid_selected", (13, 29, 47))
    expected = np.mean([
        roc_auc_score(labels, [row["predictions"]["hybrid_selected"][seed]
                               for row in panel]) for seed in (13, 29, 47)
    ])
    assert observed["auroc"] == pytest.approx(expected)


def test_fast_metrics_match_reference_with_tied_scores(monkeypatch):
    from scripts.mx_science_extension_v1 import analysis

    monkeypatch.setattr(analysis, "calibration", lambda y, p: None)
    y = np.asarray([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int64)
    p = np.asarray([0.2, 0.7, 0.2, 0.7, 0.9, 0.1, 0.6, 0.6])
    observed = _metrics(y, p)
    assert observed["auroc"] == pytest.approx(roc_auc_score(y, p))
    assert observed["auprc"] == pytest.approx(average_precision_score(y, p))
    assert observed["brier"] == pytest.approx(brier_score_loss(y, p))


def test_temporal_subsets_report_counts_without_subject_export(monkeypatch):
    # Keep this fixture fast while exercising the real patient-resampling path.
    from scripts.mx_science_extension_v1 import analysis

    monkeypatch.setattr(analysis, "calibration", lambda y, p: None)
    report = analyze_panel(_panel(), expected_n=20, resamples=1000, bootstrap_seed=29)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["all_patient_count"] == 20
    assert report["subsets"]["90"]["n"] == 10
    assert report["subsets"]["180"]["n"] == 19
    assert report["seed_repetitions_independent"] is False
    assert "patient-" not in str(report)
