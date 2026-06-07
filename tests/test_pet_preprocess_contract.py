from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "preprocess_pet.py"
POLICY = PROJECT_ROOT / "configs" / "model" / "method_y_gap_cohort_eval_policy.yaml"


def load_module():
    spec = importlib.util.spec_from_file_location("preprocess_pet", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["preprocess_pet"] = module
    spec.loader.exec_module(module)
    return module


def test_policy_forbids_pet_in_method_x():
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    assert policy["method"] == "Method_Y_PET_MM"
    assert policy["method_x_pet_input_allowed"] is False
    assert policy["method_y_pet_input_allowed"] is True
    assert policy["temporal_alignment"]["gap_is_model_feature"] is False


def test_output_root_guard_rejects_method_x_cache():
    module = load_module()
    with pytest.raises(RuntimeError):
        module.assert_output_root_is_method_y_only(Path("data/cached/method_x/pet"))
    with pytest.raises(RuntimeError):
        module.assert_output_root_is_method_y_only(Path("data/processed"))
    module.assert_output_root_is_method_y_only(Path("data/preprocessed_pet"))


def test_gap_tier_policy_ordering():
    module = load_module()
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    assert module.tier_for_gap(60, policy) == "primary_strict"
    assert module.tier_for_gap(90, policy) == "primary_strict"
    assert module.tier_for_gap(91, policy) == "secondary_window"
    assert module.tier_for_gap(120, policy) == "secondary_window"
    assert module.tier_for_gap(121, policy) == "relaxed_exploratory"
    assert module.tier_for_gap(365, policy) == "relaxed_exploratory"
    assert module.tier_for_gap(366, policy) == "excluded_from_training"
    assert module.tier_for_gap(None, policy) == "missing"


def test_filename_parser_accepts_common_patterns():
    module = load_module()
    assert module.parse_pet_image_filename("A4_B12345_FBP.nii.gz") == ("A4", "B12345")
    assert module.parse_pet_image_filename("B12345_LEARN_FBP.nii.gz") == ("LEARN", "B12345")
    assert module.parse_pet_image_filename("SF_12345_FTP.nii") == ("SF", "B12345")


def test_script_text_contains_privacy_and_method_x_guards():
    txt = SCRIPT.read_text(encoding="utf-8")
    required = [
        "method_x_pet_input_allowed",
        "raw_identifiers_exported",
        "raw_paths_exported_public",
        "pet_t1_gap_used_as_model_feature",
        "assert_output_root_is_method_y_only",
    ]
    for token in required:
        assert token in txt
