"""RED test for dependency closure - P015."""
import pytest


def test_from_import_submodule_closure():
    from src.research.brainiac_dependency_closure import build_dependency_closure
    from pathlib import Path

    closure = build_dependency_closure(Path("scripts/research/prepare_brainiac_raw_t1_v2.py"))
    # Check that from-import submodule is resolved (e.g., from src.research.brainiac_adapter import ...)
    assert any("brainiac_adapter" in k for k in closure), "Submodule import not resolved"



def test_dynamic_vendor_inputs_are_explicitly_bound_outside_ast_closure():
    from scripts.research import prepare_brainiac_raw_t1_v2 as runner

    manifest = runner._dependency_manifest()
    dynamic = manifest["dynamic_code_inputs"]
    assert dynamic
    assert all("vendor/BrainIAC/" in row["path"] for row in dynamic.values())
    assert all(len(row["sha256"]) == 64 and row["bytes"] >= 0 for row in dynamic.values())
    assert dynamic["official_model"]["bytes"] > 0
    assert dynamic["official_dataset"]["bytes"] > 0


def test_static_policy_config_drift():
    from pathlib import Path
    import hashlib

    config_path = Path("configs/research/brainiac_preprocessing_r1_primary_v2.yaml")
    policy_path = Path("governance/brainiac_preprocessing_r1_primary_v2_policy.json")

    # These should exist and have hashes
    assert config_path.is_file(), "Config not found"
    assert policy_path.is_file(), "Policy not found"

    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()

    assert len(config_hash) == 64
    assert len(policy_hash) == 64
