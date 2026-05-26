import json
import re
from pathlib import Path


def test_clinical_ad_feasibility_if_generated_is_policy_safe():
    path = Path("results/tables/clinical_ad_target_feasibility.json")
    if not path.exists():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["status"] == "FEASIBILITY_ONLY_NOT_TRAINED"
    assert obj["training_started"] is False
    assert obj["pet_preprocessing_started"] is False
    assert obj["subject_level_outputs_exported"] is False
    assert obj["raw_identifiers_exported"] is False
    assert obj["raw_paths_exported"] is False
    assert obj["method_x_claim_unchanged"] is True
    assert obj["pet_as_method_x_input"] is False
    assert obj["pet_as_method_x_cog_input"] is False
    assert obj["current_target_is_clinical_alzheimer_diagnosis"] is False
    assert obj["clinical_alzheimer_output_allowed_now"] is False
    assert obj["clinical_alzheimer_requires_independent_label"] is True


def test_clinical_ad_feasibility_blocks_unapproved_clinical_targets():
    path = Path("results/tables/clinical_ad_target_feasibility.json")
    if not path.exists():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    labels = obj["label_availability"]
    assert labels["cn_mci_ad_dementia_availability"]["supported_now"] is False
    assert labels["diagnosis_at_visit_availability"]["supported_now"] is False
    assert labels["conversion_followup_availability"]["supported_now"] is False
    assert labels["adjudicated_diagnosis_availability"]["supported_now"] is False
    assert labels["diagnosis_at_visit_availability"]["status"] == "CDR_SCALE_AVAILABLE_NOT_DIAGNOSIS"


def test_clinical_ad_public_markdown_has_no_subject_level_tokens():
    path = Path("results/tables/clinical_ad_target_feasibility.md")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"\bB\d{4,}\b", text)
    assert "H:\\" not in text
    assert "/mnt/h/" not in text
    assert "patient_id" not in text
    assert "sample_id" not in text
    assert "PREVBID" not in text
