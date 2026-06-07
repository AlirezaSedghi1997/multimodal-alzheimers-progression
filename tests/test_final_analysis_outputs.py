import json
from pathlib import Path


def test_final_analysis_contract_if_generated():
    p = Path("results/final_results/final_analysis/final_analysis_summary.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["method"] == "X"
    assert obj["method_x_compliant"] is True
    assert obj["pet_as_model_input"] is False
    assert obj["raw_identifiers_used"] is False
    assert obj["splits"]["external_holdout_type"] == "locked_pseudo_external_random_patient_holdout"
    assert obj["bias"]["risk_level"] == "HIGH"
    if obj.get("tau_regression") is not None:
        assert obj["gates"]["gate_tau"] == obj["tau_regression"]["mode"]
    if obj.get("three_d") is not None:
        assert obj["gates"]["gate_3d_training"] in {"PASSED", "PARTIAL", "DEBUG_ONLY"}
        if obj["gates"]["gate_3d_training"] == "PARTIAL":
            aggregate = obj["three_d"]["aggregate"]
            assert aggregate["mode"] == "FULL_3D_TRAINING"
            assert aggregate["device"] == "cuda"
            assert aggregate["final_claim_allowed"] is False
            assert aggregate["claim_policy"]["claim_blockers"]


def test_final_report_mentions_caveat_if_generated():
    p = Path("results/final_results/final_analysis/final_analysis_report.md")
    if not p.exists():
        return
    text = p.read_text(encoding="utf-8").lower()
    assert "pseudo-external" in text
    assert "cohort-label" in text
