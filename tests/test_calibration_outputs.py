import json
from pathlib import Path


def test_calibration_fit_log_never_uses_external_if_generated():
    p = Path("results/final_results/calibration_fit_log.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    for rec in obj["records"]:
        assert rec["external_used_for_fit"] is False
        assert "external_holdout" not in rec["fit_data_source"]


def test_calibration_report_contract_if_generated():
    p = Path("results/final_results/calibration/calibration_report.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["external_used_for_fit"] is False
    assert obj["uses_pet_as_input"] is False
    assert obj["uses_raw_identifiers"] is False

