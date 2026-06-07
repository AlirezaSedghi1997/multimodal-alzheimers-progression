from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_final_policy_coordinates_method_x_and_method_y():
    p = ROOT / "configs/model/final_method_x_y_v36_pet_phase2_policy.yaml"
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert obj["scope"]["method_x"]["pet_image_input_allowed"] is False
    assert obj["scope"]["method_y"]["pet_image_input_allowed"] is True
    assert obj["method_y_pet_phase2_spatial_qc"]["spatial_preprocessing"]["execute_without_spatial_alignment"] == "BLOCK"
    assert obj["method_y_pet_phase2_spatial_qc"]["suvr_reference_policy"]["scalar_suvr_or_centiloid_as_model_input"] is False


def test_final_package_contains_old_validity_and_new_pet_phase2():
    required = [
        "scripts/run_validity_audit.py",
        "scripts/run_temporal_window_sensitivity.py",
        "scripts/run_external_eval.py",
        "scripts/build_thesis_grade_report.py",
        "scripts/build_pet_private_manifest.py",
        "scripts/preprocess_pet_spatial.py",
        "scripts/build_pet_volume_cache.py",
        "scripts/check_method_y_training_readiness.py",
        "configs/model/v3_6_validity_policy.yaml",
        "configs/model/method_y_spatial_preprocess_policy.yaml",
        "configs/model/method_y_suvr_reference_policy.yaml",
        "docs/method_y_phase2_modeling_design.md",
        "AGENT_EXECUTION_PLAN_METHOD_XY.md",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), rel


def test_final_makefile_targets_include_both_paths():
    txt = (ROOT / "patches/MAKEFILE_FINAL_METHOD_XY_V36_PET_TARGETS.mk").read_text(encoding="utf-8")
    for token in ["final-method-x-sequence", "final-method-y-pet-sequence", "validity-audit", "method-y-pet-execute"]:
        assert token in txt


def test_agent_plan_keeps_method_x_and_method_y_separate():
    txt = (ROOT / "AGENT_EXECUTION_PLAN_METHOD_XY.md").read_text(encoding="utf-8").lower()
    assert "do not modify method y during this phase" in txt
    assert "method y pet dry-run" in txt
    assert "do not edit" in txt
