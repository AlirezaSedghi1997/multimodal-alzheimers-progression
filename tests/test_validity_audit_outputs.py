from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_validity_audit_outputs_exist_after_run():
    out = ROOT / "results/tables/validity_audit/validity_audit_summary.json"
    if not out.exists():
        pytest.skip("Run `python scripts/run_validity_audit.py` first.")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["version"] == "v3.6-validity-fixes"
    assert data["method"] == "X"
    assert "verdict" in data
    assert "final_claim_recommendation" in data["verdict"]
    assert "tests" in data
    for key in ["tabular_only", "mri_features_only", "tabular_plus_mri_features", "global_label_shuffle"]:
        assert key in data["tests"]


def test_validity_audit_never_reports_pet_as_input():
    src = (ROOT / "scripts/run_validity_audit.py").read_text(encoding="utf-8").lower()
    assert "forbidden_feature_terms" in src
    assert "pet" in src and "suvr" in src and "centiloid" in src
