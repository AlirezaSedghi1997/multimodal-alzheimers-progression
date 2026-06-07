from __future__ import annotations

import json
from pathlib import Path


def test_cohort_probe_summary_exists():
    path = Path("results/tables/cohort_probe/cohort_probe_summary.json")
    assert path.exists(), "Run extract-3d-embeddings-oof then run_cohort_probe.py"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["status"] == "COMPLETED"
    assert len(obj["blocks"]) >= 2
    for block in obj["blocks"]:
        assert "linear_probe_auc" in block
        assert "nonlinear_probe_auc" in block
        assert "linear_vs_nonlinear_gap" in block


def test_centiloid_regression_exists():
    path = Path("results/final_results/centiloid_regression/centiloid_regression_metrics.json")
    assert path.exists(), "Run scripts/run_centiloid_regression.py"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["status"] == "COMPLETED"
    assert "within_a4" in obj["subsets"]
    assert "within_learn" in obj["subsets"]
    assert obj["centiloid_as_predictor"] is False


def test_embedding_extraction_summary_private():
    path = Path("results/tables/cohort_probe/method_x_3d_embedding_extraction_summary.json")
    assert path.exists()
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["public_export_allowed"] is False
    assert obj["n"] >= 1359
