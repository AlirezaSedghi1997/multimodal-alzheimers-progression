from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_integrated_policy_present_and_separates_method_x_y():
    p = ROOT / "configs/model/integrated_method_x_y_patch_policy.yaml"
    assert p.exists(), "Missing integrated Method X/Y policy"
    policy = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert policy["method_x"]["pet_role"] == "label_target_only_never_input"
    assert policy["method_y"]["pet_input_allowed"] is True
    assert policy["method_y"]["method_x_pet_input_allowed"] is False
    assert policy["reporting_policy"]["method_x_and_method_y_tables_must_be_separate"] is True


def test_integrated_package_contains_both_patch_families():
    required = [
        "scripts/run_validity_audit.py",
        "scripts/run_temporal_window_sensitivity.py",
        "scripts/run_optuna_nested.py",
        "scripts/run_external_eval.py",
        "scripts/run_cognitive_family_ablation.py",
        "scripts/train_3d_mri.py",
        "scripts/build_thesis_grade_report.py",
        "scripts/preprocess_pet.py",
        "scripts/method_y_gap_filter.py",
        "docs/method_y_preregistration.md",
        "docs/thesis_defense_talking_points.md",
        "configs/model/v3_6_validity_policy.yaml",
        "configs/model/method_y_gap_cohort_eval_policy.yaml",
        "configs/model/integrated_method_x_y_patch_policy.yaml",
        "governance/v3_6_validity_fixes_policy.md",
        "governance/method_y_pet_phase2_gate_policy.md",
        "governance/integrated_method_x_y_patch_policy.md",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), f"Missing integrated patch file: {rel}"


def test_integrated_makefile_targets_exist():
    p = ROOT / "patches/MAKEFILE_INTEGRATED_METHOD_XY_TARGETS.mk"
    assert p.exists(), "Missing integrated Makefile targets"
    txt = p.read_text(encoding="utf-8")
    for target in ["validity-audit", "method-y-pet-dry-run", "integrated-contract-tests", "thesis-report"]:
        assert target in txt
