from __future__ import annotations

import json
from pathlib import Path


def test_phase1_baselines_summary_exists():
    path = Path("results/final_results/phase1_baselines/phase1_baselines_summary.json")
    assert path.exists(), "Run scripts/run_branch_baselines.py first"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["status"] == "COMPLETED"
    assert obj["contains_leaky_mri_surrogates"] is False
    assert "fixed_tabular" in obj["results"]
    assert "roi_wmh_reads" in obj["results"]


def test_incremental_value_table_exists():
    path = Path("results/tables/phase1_incremental_value_table.json")
    assert path.exists(), "Run scripts/build_incremental_value_table.py first"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["status"] == "COMPLETED"
    assert len(obj["rows"]) >= 3


def test_gpu_preflight_passes():
    path = Path("results/governance/gpu_preflight_check.json")
    assert path.exists(), "Run scripts/gpu_preflight_check.py first"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["status"] == "PASS"
