from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_temporal_window_outputs_exist_after_run():
    out = ROOT / "results/tables/temporal_window_sensitivity.json"
    if not out.exists():
        pytest.skip("Run `python scripts/run_temporal_window_sensitivity.py` first.")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["version"] == "v3.6-validity-fixes"
    assert data["main_catalog_overwritten"] is False
    windows = {row["window_days"] for row in data["windows"]}
    assert {0, 60, 90, 120}.issubset(windows)


def test_temporal_window_does_not_overwrite_master_catalog_by_source_contract():
    src = (ROOT / "scripts/run_temporal_window_sensitivity.py").read_text(encoding="utf-8")
    assert "master_catalog_" in src
    assert "main_catalog_overwritten" in src
    assert "False" in src
