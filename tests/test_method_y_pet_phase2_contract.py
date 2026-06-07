from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_spatial_policy_contains_mandatory_gates():
    p = ROOT / "configs/model/method_y_spatial_preprocess_policy.yaml"
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert obj["spatial_preprocessing"]["required_for_training_ready_outputs"] is True
    assert obj["spatial_preprocessing"]["execute_without_spatial_alignment"] == "BLOCK"
    assert obj["sample_selection"]["one_pet_per_patient_for_training"] is True
    assert "suvr" in obj["method_x_boundary"]["forbidden_model_input_terms"]
    assert "centiloid" in obj["method_x_boundary"]["forbidden_model_input_terms"]


def test_suvr_policy_blocks_scalar_features():
    p = ROOT / "configs/model/method_y_suvr_reference_policy.yaml"
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert obj["global_rules"]["scalar_suvr_as_method_x_input"] is False
    assert obj["global_rules"]["scalar_centiloid_as_method_y_training_feature"] is False
    assert obj["roi_suvr"]["allowed_model_input"] is False


def test_key_files_exist():
    required = [
        "scripts/build_pet_private_manifest.py",
        "scripts/preprocess_pet_spatial.py",
        "scripts/build_pet_volume_cache.py",
        "scripts/check_method_y_training_readiness.py",
        "scripts/train_pet_method_y.py",
        "src/utils/pet_manifest.py",
        "src/utils/pet_spatial_qc.py",
        "patches/MAKEFILE_METHOD_Y_PHASE2_TARGETS.mk",
        "docs/agent_method_y_pet_phase2_playbook.md",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), rel
