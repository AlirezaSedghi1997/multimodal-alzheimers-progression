from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def test_spatial_policy_has_literature_aligned_temporal_tiers():
    p = load("configs/model/method_y_spatial_preprocess_policy.yaml")
    tiers = p["temporal_pairing"]["eligibility_tiers"]
    assert tiers["gold_standard_30"]["max_abs_gap_days"] == 30
    assert tiers["primary_strict"]["max_abs_gap_days"] == 90
    assert tiers["secondary_window"]["max_abs_gap_days"] == 120
    assert p["temporal_pairing"]["exclude_from_primary_training_if_gap_gt_days"] == 90
    assert p["temporal_pairing"]["gap_is_model_feature"] is False


def test_spatial_policy_requires_two_step_registration_and_blocks_identity():
    p = load("configs/model/method_y_spatial_preprocess_policy.yaml")
    assert p["spatial_preprocessing"]["required_for_training_ready_outputs"] is True
    assert p["spatial_preprocessing"]["execute_without_spatial_alignment"] == "BLOCK"
    assert "pet_to_subject_t1_rigid" in p["spatial_preprocessing"]["required_pipeline"]
    assert "subject_t1_to_mni_nonlinear" in p["spatial_preprocessing"]["required_pipeline"]
    assert "apply_combined_transform_to_pet" in p["spatial_preprocessing"]["required_pipeline"]


def test_suvr_policy_has_tracer_specific_equations_and_boundaries():
    p = load("configs/model/method_y_suvr_reference_policy.yaml")
    assert p["global_rules"]["scalar_suvr_as_method_x_input"] is False
    assert p["global_rules"]["scalar_centiloid_as_method_y_primary_training_feature"] is False
    fbp = p["amyloid_pet"]["tracers"]["fbp"]
    fbb = p["amyloid_pet"]["tracers"]["fbb"]
    assert fbp["suvr_threshold_primary_reference"] == 1.11
    assert fbp["centiloid_slope"] == 188.22
    assert fbb["suvr_threshold_primary_reference"] == 1.08
    assert fbb["centiloid_intercept"] == -151.87
    assert p["amyloid_pet"]["primary_reference"]["name"] == "whole_cerebellum"
    assert p["amyloid_pet"]["sensitivity_references"][0]["name"] == "cerebellar_grey"


def test_final_policy_blocks_naive_primary_and_requires_baselines():
    p = load("configs/model/final_method_x_y_pet_phase2_policy.yaml") if (ROOT / "configs/model/final_method_x_y_pet_phase2_policy.yaml").exists() else load("configs/model/final_method_x_y_v36_pet_phase2_policy.yaml")
    updates = p["literature_aligned_updates"]
    assert updates["modeling"]["pet_only_baseline_required"] is True
    assert updates["modeling"]["mri_only_baseline_on_method_y_split_required"] is True
    assert updates["modeling"]["early_channel_concat_primary_forbidden"] is True
    assert updates["evaluation"]["patient_level_split_required"] is True
    assert updates["evaluation"]["bootstrap_ci_default"] is True
