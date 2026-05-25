import json
from pathlib import Path


def test_cross_cohort_bias_report_contract_if_generated():
    p = Path("results/tables/cross_cohort_bias_report.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["raw_identifiers_used"] is False
    assert obj["risk_level"] in {"MODERATE", "HIGH"}
    assert 0.0 <= obj["audit"]["cohort_label_cramers_v"] <= 1.0


def test_cross_cohort_bias_markdown_mentions_pseudo_external_if_generated():
    p = Path("results/tables/cross_cohort_bias_report.md")
    if not p.exists():
        return
    text = p.read_text(encoding="utf-8").lower()
    assert "pseudo-external" in text
    assert "cohort-label" in text

