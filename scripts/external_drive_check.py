from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import REQUIRED_RELATIVE_FILES, resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import redact_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    args = ap.parse_args()
    root = resolve_data_root(args.data_root)
    checks = {name: (root / rel).exists() for name, rel in REQUIRED_RELATIVE_FILES.items()}
    t1_dir = root / "T1"
    t1_count = sum(1 for p in t1_dir.rglob("*") if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))) if t1_dir.exists() else 0
    report = {
        "data_root_redacted": redact_path(root),
        "root_exists": root.exists(),
        "read_only_policy": True,
        "required_files": checks,
        "t1_nifti_count": t1_count,
        "status": "READY" if root.exists() and all(checks.values()) and t1_count > 0 else "BLOCKED",
    }
    atomic_write_json("results/tables/external_drive_check.json", report)
    md = [
        "# External Drive Check\n\n",
        f"- data_root: `{redact_path(root)}`\n",
        f"- root_exists: `{report['root_exists']}`\n",
        f"- t1_nifti_count: `{t1_count}`\n",
        f"- status: `{report['status']}`\n",
    ]
    atomic_write_text("results/tables/external_drive_check.md", "".join(md))
    print(f"status={report['status']} t1_nifti_count={t1_count}")
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

