from pathlib import Path

from scripts.preprocess_mri_full import check_toolchain
from scripts.check_preprocess_toolchain import main


def test_toolchain_report_lists_unblock_instructions(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "")
    report = check_toolchain(tmp_path / "missing_template.nii.gz")
    assert report["status"] == "BLOCKED"
    assert "how_to_unblock" in report
    assert "antsRegistrationSyNQuick.sh" in report["required"]["registration"]
    assert report["template_exists"] is False


def test_registration_requires_synquick_not_apply_transforms():
    report = check_toolchain(Path("missing_template.nii.gz"))
    assert report["required"]["registration"] == ["antsRegistrationSyNQuick.sh"]


def test_diagnostic_only_returns_success_when_blocked(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        "sys.argv",
        ["check_preprocess_toolchain.py", "--template", "missing_template.nii.gz", "--diagnostic_only"],
    )
    assert main() == 0
