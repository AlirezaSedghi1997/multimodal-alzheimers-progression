import json
from pathlib import Path

from scripts.run_tau_branch import nearest_t1, subject_features_from_catalog, t1_by_patient_hash_from_catalog


def test_tau_regression_contract_if_generated():
    p = Path("results/final_results/tau/tau_regression_metrics.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["mode"] in {"FULL", "STRONG_BOOTSTRAP", "DESCRIPTIVE_ONLY", "SKIP_WITH_JUSTIFICATION"}
    assert obj["uses_tau_as_input"] is False
    assert obj["uses_pet_as_input"] is False
    assert obj["uses_raw_identifiers"] is False
    if obj["mode"] == "SKIP_WITH_JUSTIFICATION":
        assert obj["rows_used"] == 0
        assert obj["skip_reason"]
        assert obj["missing_required_files"]
        return
    assert obj["mode"] == "FULL"
    assert obj["rows_used"] >= 200
    if obj.get("schema_version") == 2:
        assert obj["cv_strategy"].startswith("GroupKFold")
        assert obj["unique_patients_used"] <= obj["rows_used"]
        assert 0.0 <= obj["mri_feature_join_rate"] <= 1.0
        assert obj["rows_aligned"] == obj["rows_used"] + obj["rows_missing_mri_features"]
        assert obj["target_summary"]["n"] == obj["rows_used"]
        assert "mean_baseline" in obj
        assert "delta_vs_mean_baseline" in obj
        assert obj["t1_candidate_source"] in {"sv_visit_timing", "locked_master_catalog"}
        assert obj["subject_feature_source"] in {"subjinfo", "locked_master_catalog"}


def test_tau_catalog_t1_fallback_uses_locked_hashed_catalog(tmp_path):
    catalog = tmp_path / "master_catalog.csv"
    catalog.write_text(
        "\n".join(
            [
                "patient_hash,t1_file_hash,t1_day_t0",
                "pt_a,t1_1,10",
                "pt_a,t1_1,10",
                "pt_a,t1_2,30",
                "pt_b,t1_3,",
            ]
        ),
        encoding="utf-8",
    )
    candidates = t1_by_patient_hash_from_catalog(catalog)
    assert list(candidates) == ["pt_a"]
    assert [row["t1_file_hash"] for row in candidates["pt_a"]] == ["t1_1", "t1_2"]
    assert nearest_t1(24.0, candidates["pt_a"], max_gap=10.0)["t1_file_hash"] == "t1_2"


def test_tau_catalog_subject_fallback_uses_locked_hashed_catalog(tmp_path):
    catalog = tmp_path / "master_catalog.csv"
    catalog.write_text(
        "\n".join(
            [
                "patient_hash,age,sex,education_years,apoe4_carrier,mmse_base",
                "pt_a,70,1,16,1,29",
                "pt_b,,,,,",
            ]
        ),
        encoding="utf-8",
    )
    features = subject_features_from_catalog(catalog)
    assert features["pt_a"]["age"] == 70.0
    assert features["pt_a"]["sex"] == "1"
    assert features["pt_a"]["education_years"] == 16.0
    assert features["pt_a"]["apoe4_carrier"] == 1.0
    assert features["pt_a"]["mmse_base"] == 29.0
