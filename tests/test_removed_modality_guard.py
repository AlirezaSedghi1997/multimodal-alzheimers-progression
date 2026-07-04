from __future__ import annotations

from pathlib import Path
import subprocess


def test_removed_imaging_branch_leaves_no_text_traces():
    root = Path(__file__).resolve().parents[1]
    removed = "f" + "mri"
    forbidden_terms = [
        removed,
        "rs-" + removed,
        "rs_" + removed,
        "mri_" + removed,
        "functional " + "connectivity",
        "connectivity_" + "features",
        "extract_" + removed,
        "bo" + "ld",
    ]
    pattern = "|".join(forbidden_terms)
    result = subprocess.run(
        [
            "rg",
            "-n",
            "-i",
            pattern,
            str(root),
            "--glob",
            "!**/.git/**",
            "--glob",
            "!**/.venv/**",
            "--glob",
            "!**/.cache/**",
            "--glob",
            "!**/__pycache__/**",
            "--glob",
            "!**/.pytest_cache/**",
            "--glob",
            "!**/.pytest_tmp*/**",
            "--glob",
            f"!{Path(__file__).name}",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
