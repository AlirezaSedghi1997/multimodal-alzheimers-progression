from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.spm_preprocess_backend import resolve_command
from src.utils.io_helpers import atomic_write_json, atomic_write_text


def check_spm_toolchain(matlab: str | None, spm_dir: str | None) -> dict[str, object]:
    matlab_cmd = resolve_command(matlab, ["matlab", "matlab.exe"])
    spm_path = Path(spm_dir) if spm_dir else None
    missing = []
    if matlab_cmd is None:
        missing.append("matlab")
    if spm_path is None or not spm_path.exists():
        missing.append("spm_dir")
    if spm_path is None or not (spm_path / "spm.m").exists():
        missing.append("spm_m_file")
    if spm_path is None or not (spm_path / "tpm" / "TPM.nii").exists():
        missing.append("spm_tpm")
    status = "READY" if not missing else "BLOCKED"
    return {
        "status": status,
        "backend": "spm",
        "matlab": matlab_cmd,
        "spm_dir": str(spm_path) if spm_path else None,
        "spm_dir_exists": bool(spm_path and spm_path.exists()),
        "spm_m_exists": bool(spm_path and (spm_path / "spm.m").exists()),
        "spm_tpm_exists": bool(spm_path and (spm_path / "tpm" / "TPM.nii").exists()),
        "missing": sorted(set(missing)),
        "how_to_unblock": [
            "Set MATLAB_BIN to the MATLAB executable, or pass --matlab.",
            "Set SPM_DIR to the folder containing spm.m, or pass --spm_dir.",
            "For Windows MATLAB called from WSL, pass the /mnt/c/.../matlab.exe path.",
        ],
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
    }


def write_spm_toolchain_report(report: dict[str, object]) -> None:
    atomic_write_json("results/tables/spm_preprocess_toolchain_report.json", report)
    lines = [
        "# SPM Preprocessing Toolchain Report\n\n",
        f"- status: `{report['status']}`\n",
        f"- backend: `{report['backend']}`\n",
        f"- missing: `{report['missing']}`\n",
        f"- matlab: `{report['matlab']}`\n",
        f"- spm_dir: `{report['spm_dir']}`\n",
        "- contains_pet_input: `False`\n",
        "- contains_raw_identifiers: `False`\n",
    ]
    atomic_write_text("results/tables/spm_preprocess_toolchain_report.md", "".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matlab", default=os.environ.get("MATLAB_BIN"))
    ap.add_argument("--spm_dir", default=os.environ.get("SPM_DIR"))
    ap.add_argument("--diagnostic_only", action="store_true")
    args = ap.parse_args()

    report = check_spm_toolchain(args.matlab, args.spm_dir)
    write_spm_toolchain_report(report)
    print(json.dumps(report, indent=2))
    if args.diagnostic_only:
        return 0
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
