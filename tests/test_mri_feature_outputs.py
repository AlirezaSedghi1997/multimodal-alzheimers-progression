import json
from pathlib import Path


def test_mri_feature_summary_contract_if_generated():
    p = Path("results/tables/mri_feature_extraction_summary.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["contains_raw_images"] is False
    assert obj["contains_pet_input"] is False
    assert obj["contains_raw_identifiers"] is False


def test_mri_feature_baseline_contract_if_generated():
    p = Path("results/final_results/mri_feature_baseline/mri_feature_baseline_metrics.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    agg = obj["aggregate"]
    assert 0.0 <= agg["roc_auc"] <= 1.0
    assert agg["uses_pet_as_input"] is False
    assert agg["uses_raw_identifiers"] is False

