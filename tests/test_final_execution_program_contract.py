from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "policy" / "final_execution_program.json"
DOC = ROOT / "docs" / "final_execution_program.md"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_method_x_allowed_inputs_are_fixed() -> None:
    config = load_config()
    assert config["method_x_allowed_inputs"] == [
        "t1_structural_mri",
        "age",
        "sex",
        "education_years",
        "apoe4_carrier",
        "mmse_base",
    ]
    assert "centiloid_as_predictor" in config["method_x_forbidden_inputs"]
    assert "blood_biomarkers" in config["method_x_forbidden_inputs"]


def test_label_freeze_cutoff_is_frozen_in_governance_artifact() -> None:
    config = load_config()
    label = config["label_freeze"]
    freeze = json.loads((ROOT / "results/governance/label_freeze.json").read_text(encoding="utf-8"))
    assert freeze["status"] == "FROZEN_BEFORE_MODELING"
    assert freeze["centiloid_threshold"] == label["centiloid_threshold"]
    assert "No change after modeling starts" in label["change_policy"]


def test_method_x_cog_does_not_replace_thesis_anchor() -> None:
    config = load_config()
    purpose = config["branches"]["method_x_cog_sf"]["purpose"]
    assert "does not replace Method X-Strict" in purpose
    assert "exploratory/extended branch" in purpose


def test_minimum_phase2_mvp_is_complete() -> None:
    config = load_config()
    mvp = config["phase2_minimum_mvp"]
    assert len(mvp) == 10
    for item in [
        "5_fold_subject_grouped_cv",
        "locked_pseudo_external_evaluation",
        "linear_and_nonlinear_cohort_probe",
        "continuous_centiloid_within_cohort_analysis",
    ]:
        assert item in mvp


def test_optional_items_are_not_mvp_blockers() -> None:
    config = load_config()
    optional = config["optional_supplementary"]
    assert optional["leakage_inflation_negative_control"]["enabled"] == "if_time_permits_after_mvp"
    assert "Supplementary Materials" in optional["leakage_inflation_negative_control"]["preferred_location"]
    assert optional["ssl"]["mvp_participation"] is False
    assert optional["ssl"]["never_enters_method_x_strict"] is True


def test_written_program_contains_claim_control_language() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "locked pseudo-external evaluation" in text
    assert "not Alzheimer disease diagnosis" in text
    assert "Method X-Cog/SF is the main deconfounding-oriented exploratory/extended branch" in text
    assert "Supplementary Materials unless reviewers request otherwise" in text
