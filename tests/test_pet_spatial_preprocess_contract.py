from pathlib import Path
import os
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
    assert "spm_pet_t1_mni" in text
    assert "preprocess_pet_spm_t1_mni" in text
    assert "--auto_batches" in text


def test_preprocess_spm_execute_blocks_without_spm_dir():
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "preprocess_pet_spatial.py"),
        "--execute",
        "--spatial_mode",
        "spm_pet_t1_mni",
        "--data_root",
        "H:/A4 Learn",
        "--limit",
        "1",
    ]
    env = os.environ.copy()
    env.pop("SPM_DIR", None)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    assert proc.returncode != 0
    assert "SPM toolchain" in (proc.stderr + proc.stdout) or "spm_dir" in (proc.stderr + proc.stdout).lower()
