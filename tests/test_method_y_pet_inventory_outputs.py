import json
import re
from pathlib import Path


def test_method_y_pet_inventory_if_generated_is_aggregate_safe():
    path = Path("results/tables/method_y_pet_inventory.json")
    if not path.exists():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["pet_as_method_x_input"] is False
    assert obj["pet_as_method_x_cog_input"] is False
    assert obj["raw_identifiers_exported"] is False
    assert obj["raw_paths_exported"] is False
    assert obj["subject_level_outputs_exported"] is False
    assert obj["aggregate_sensitivity"]["pet_derived_values_used_for_model_training"] is False
    assert obj["aggregate_sensitivity"]["pet_images_used_for_model_training"] is False


def test_method_y_public_markdown_has_no_subject_level_tokens():
    path = Path("results/tables/method_y_pet_inventory.md")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"\bB\d{4,}\b", text)
    assert "H:\\" not in text
    assert "/mnt/h/" not in text
    assert "patient_id" not in text
    assert "sample_id" not in text
