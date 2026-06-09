import json
from pathlib import Path


def test_method_y_multimodal_design_if_generated_preserves_method_x_boundary():
    path = Path("results/tables/method_y_multimodal_design.json")
    if not path.exists():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["status"] == "DESIGN_READY_NOT_TRAINED"
    assert obj["training_started"] is False
    # Method Y PET spatial preprocessing may run before training; Method X boundary must stay intact.
    assert isinstance(obj["pet_preprocessing_started"], bool)
    if obj["pet_preprocessing_started"]:
        progress = obj.get("pet_preprocess_progress") or {}
        assert progress.get("preprocessing_started") is True
    assert obj["method_x_claim_unchanged"] is True
    assert obj["method_x_pet_input_allowed"] is False
    assert obj["method_y_pet_input_allowed"] is True


def test_method_y_multimodal_target_policy_blocks_clinical_ad_without_label():
    path = Path("results/tables/method_y_multimodal_design.json")
    if not path.exists():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    policy = obj["target_policy"]
    assert policy["current_primary_target"] == "AB-positive vs AB-negative biomarker status"
    assert policy["current_target_is_clinical_alzheimer_diagnosis"] is False
    assert policy["clinical_alzheimer_output_allowed_now"] is False
    assert policy["clinical_alzheimer_requires_independent_label"] is True


def test_method_y_multimodal_scenarios_are_reported_separately():
    path = Path("results/tables/method_y_multimodal_design.json")
    if not path.exists():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    scenarios = {item["name"]: item for item in obj["modality_scenarios"]}
    assert {"mri_only", "pet_only", "mri_pet", "mri_pet_cognitive"}.issubset(scenarios)
    assert scenarios["mri_only"]["method_x_compliant"] is True
    assert scenarios["pet_only"]["method_x_compliant"] is False
    assert scenarios["mri_pet"]["method_x_compliant"] is False
    assert scenarios["mri_pet_cognitive"]["method_x_compliant"] is False
