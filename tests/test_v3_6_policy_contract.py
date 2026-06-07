from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_v36_policy_exists_and_keeps_method_x_pet_free():
    p = ROOT / "configs/model/v3_6_validity_policy.yaml"
    assert p.exists(), "Missing v3.6 policy YAML"
    txt = p.read_text(encoding="utf-8")
    assert 'method: "X"' in txt
    assert "pet_role: \"label_target_only_never_input\"" in txt
    assert "pet_t1_gap_as_feature" in txt
    assert "locked_pseudo_external_holdout" in txt
    assert "delong_policy: \"blocked_until_true_external_binary_set\"" in txt


def test_required_v36_scripts_present():
    required = [
        "scripts/run_validity_audit.py",
        "scripts/run_temporal_window_sensitivity.py",
        "scripts/run_optuna_nested.py",
        "scripts/run_external_eval.py",
        "scripts/run_cognitive_family_ablation.py",
        "scripts/train_3d_mri.py",
        "scripts/build_thesis_grade_report.py",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), f"Missing {rel}"


def test_scripts_do_not_allow_pet_as_method_x_feature_by_default():
    combined = ""
    for rel in ["scripts/run_validity_audit.py", "scripts/run_optuna_nested.py", "scripts/run_external_eval.py"]:
        combined += (ROOT / rel).read_text(encoding="utf-8") + "\n"
    forbidden_terms_guarded = ["pet", "suvr", "centiloid", "fbp", "ftp", "tracer", "gap"]
    for term in forbidden_terms_guarded:
        assert term in combined.lower(), f"Expected forbidden-term guard to mention {term}"
