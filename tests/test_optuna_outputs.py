import json
from pathlib import Path


def test_optuna_summary_contract_if_generated():
    p = Path("results/final_results/optuna/optuna_summary.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    best = obj["best"]
    assert obj["method_x_compliant"] is True
    assert obj["cohort_label_confound_warning"] is True
    assert best["uses_pet_as_input"] is False
    assert best["uses_raw_identifiers"] is False
    assert 0.0 <= best["roc_auc"] <= 1.0

