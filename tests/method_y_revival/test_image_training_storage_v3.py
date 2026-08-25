from __future__ import annotations

from pathlib import Path
import pytest


def test_storage_roots_on_h_and_no_c_temp():
    ps1 = Path("scripts/run_method_y_image_training_v3.ps1").read_text(encoding="utf-8")
    # All env must point to H runtime via Join-Path
    assert "$RuntimeRoot = Join-Path $ProjectRoot '.runtime\\method_y_image_v3'" in ps1
    assert "$env:TMP = Join-Path $RuntimeRoot 'tmp'" in ps1
    assert "$env:TEMP = $env:TMP" in ps1
    assert "$env:TMPDIR = $env:TMP" in ps1
    assert "H:\\New project" in ps1
    # Ensure no C:\ temp redirection
    lines = [l for l in ps1.splitlines() if "env:" in l.lower()]
    for line in lines:
        assert "C:\\" not in line


def test_ps1_no_background_job_or_redirect_to_c():
    ps1 = Path("scripts/run_method_y_image_training_v3.ps1").read_text(encoding="utf-8")
    assert "Start-Process" not in ps1
    assert "Start-Job" not in ps1
    # No redirect to C
    assert "C:\\" not in ps1.replace("H:\\New project", "")


def test_output_and_cache_on_h():
    import yaml

    config = yaml.safe_load(Path("configs/method_y_image_training_v3_1.yaml").read_text(encoding="utf-8"))
    assert config["output_root"].startswith("results/method_y_revival_v3_image")
    assert config["project_root"] == "H:/New project"
    # Derived caches on H via relative path that resolves to H
    for branch in ["amyloid", "tau"]:
        derived = config["branches"][branch]["derived_cache"]
        assert derived.startswith("caches/method_y/revival_v3_image")
