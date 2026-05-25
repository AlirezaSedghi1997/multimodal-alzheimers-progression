from pathlib import Path


def test_external_drive_policy_exists():
    assert Path("governance/external_drive_policy.md").exists()

