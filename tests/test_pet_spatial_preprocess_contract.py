from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_preprocess_spatial_blocks_identity_execute_without_flag():
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "preprocess_pet_spatial.py"),
        "--execute",
        "--spatial_mode", "identity_for_tests",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode != 0
    assert "identity_for_tests" in (proc.stderr + proc.stdout)


def test_preprocess_spatial_has_required_cli_text():
    text = (ROOT / "scripts" / "preprocess_pet_spatial.py").read_text(encoding="utf-8")
    assert "--n_jobs" in text
    assert "PetResumeManifest" in text
    assert "assert_method_y_output_root" in text
    assert "assess_qc_gates" in text
