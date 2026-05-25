import json
from pathlib import Path

from scripts.subgroup_performance_audit import age_group, education_group, mmse_group, summarize_group


def test_subgroup_bins_are_stable():
    assert age_group({"age": "69"}) == "age_lt_70"
    assert age_group({"age": "75"}) == "age_75_79"
    assert education_group({"education_years": "16"}) == "education_13_16"
    assert mmse_group({"mmse_base": "30"}) == "mmse_30"


def test_summarize_group_handles_single_class():
    out = summarize_group([1, 1], [0.8, 0.9])
    assert out["roc_auc"] is None
    assert out["single_class_caveat"] is True


def test_subgroup_audit_contract_if_generated():
    p = Path("results/final_results/subgroup_performance_audit.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["hard_fairness_gate"] is False
    assert obj["uses_pet_as_input"] is False
    assert obj["uses_raw_identifiers"] is False
    assert "sex" in obj["splits"]["cv_oof"]["subgroups"]
