import json
from pathlib import Path


def test_external_eval_contract_if_generated():
    p = Path("results/final_results/external/external_metrics.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["external_holdout_type"] == "locked_pseudo_external_random_patient_holdout"
    assert obj["is_truly_independent_external"] is False
    assert obj["uses_pet_as_input"] is False
    assert obj["uses_raw_identifiers"] is False
    assert obj["gate_4_status"] == "DONE"

