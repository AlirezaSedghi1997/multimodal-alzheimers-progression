import json
from pathlib import Path


def test_baseline_metrics_contract_if_generated():
    p = Path("results/final_results/baseline/baseline_metrics.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    agg = obj["aggregate"]
    assert 0.0 <= agg["roc_auc"] <= 1.0
    assert 0.0 <= agg["pr_auc"] <= 1.0
    assert agg["uses_pet_as_input"] is False
    assert agg["uses_raw_identifiers"] is False

