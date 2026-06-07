from __future__ import annotations

import json
from pathlib import Path


def test_sf_stress_tests_summary_exists():
    path = Path("results/final_results/method_x_cog_sf/sf_stress_tests_summary.json")
    assert path.exists(), "Run scripts/run_method_x_cog_sf_stress_tests.py"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["status"] == "COMPLETED"
    assert obj["branch"] == "method_x_cog_sf"
    assert obj["uses_pet_as_input"] is False
    assert obj["uses_raw_identifiers"] is False
    assert "scientific_interpretation" in obj
    assert "forbidden_claims" in obj["scientific_interpretation"]
    arm_ids = {arm["arm"] for arm in obj["arms"]}
    assert "sf_only_5fold_cv" in arm_ids
    assert "sf_inclusive_confound_controlled_5fold_cv" in arm_ids
    assert "train_sf_test_enrolled_internal_decoupling" in arm_ids
    assert "train_enrolled_test_sf_internal_decoupling" in arm_ids
    assert "external validation" not in obj["disclaimer"].lower() or "not external" in obj["disclaimer"].lower()


def test_sf_inclusive_cognitive_features_cover_all_substudies():
    path = Path("data/processed/cognitive_features_method_x_cog_sf.csv")
    assert path.exists(), "Run build_cognitive_features.py with method_x_cog_sf catalog"
    rows = list(path.open(encoding="utf-8"))
    assert len(rows) >= 4485  # header + 4484 samples


def test_sf_label_discrepancy_documented():
    path = Path("results/final_results/method_x_cog_sf/sf_stress_tests_summary.json")
    if not path.exists():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    disc = obj["sf_label_discrepancy"]
    assert disc["sf_rows"] > 0
    assert disc["primary_vs_sensitivity_disagree"] >= 0
