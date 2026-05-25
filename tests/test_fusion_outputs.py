import json
from pathlib import Path


def test_fusion_summary_contract_if_generated():
    p = Path("results/final_results/fusion/fusion_summary.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    best = obj["best"]
    assert obj["method_x_compliant"] is True
    assert obj["cohort_label_confound_warning"] is True
    assert best["uses_pet_as_input"] is False
    assert best["uses_raw_identifiers"] is False
    assert obj["gate_3_status"] in {"PASSED", "PARTIAL"}

