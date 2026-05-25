import json
from pathlib import Path

from scripts.build_cognitive_features import COGNITIVE_FEATURES


def test_cognitive_feature_names_do_not_include_raw_identifiers():
    forbidden = ("bid", "prevbid", "patient_id", "sample_id", "raw")
    bad = [feature for feature in COGNITIVE_FEATURES if any(term in feature.lower() for term in forbidden)]
    assert not bad


def test_cognitive_feature_summary_if_generated_is_privacy_safe():
    path = Path("results/tables/cognitive_feature_summary.json")
    if not path.exists():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["contains_pet_input"] is False
    assert obj["contains_raw_identifiers"] is False
    assert obj["output_granularity"] == "hashed_sample_level"
    assert obj["method_x_claim_unchanged"] is True
