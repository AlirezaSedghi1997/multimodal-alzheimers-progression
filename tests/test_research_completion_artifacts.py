import json
from pathlib import Path


def test_power_analysis_is_not_placeholder_if_generated():
    p = Path("results/tables/power_analysis.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["status"] == "DESCRIPTIVE_POWER_ANALYSIS"
    assert obj["catalog_n"] > 0
    assert obj["uses_pet_as_input"] is False
    assert obj["uses_raw_identifiers"] is False


def test_gap_matrix_contract_if_generated():
    p = Path("results/tables/implementation_gap_matrix.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    statuses = {row["status"] for row in obj["rows"]}
    assert statuses.issubset({"DONE", "PARTIAL", "MISSING", "SKIP_WITH_JUSTIFICATION"})
    assert any(row["requirement"] == "Real 3D MRI training" for row in obj["rows"])
