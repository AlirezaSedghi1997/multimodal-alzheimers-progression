from pathlib import Path


def test_device_policy_exists():
    text = Path("configs/compute/device_policy.yaml").read_text(encoding="utf-8")
    assert "hard_stop_gb: 7.5" in text
    assert "cpu_fallback_allowed: true" in text

