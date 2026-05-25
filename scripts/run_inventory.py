from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import REQUIRED_RELATIVE_FILES, T1_NAME_PATTERN, resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import redact_path
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def csv_header_and_rows(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return header, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    args = ap.parse_args()
    root = resolve_data_root(args.data_root)
    pattern = re.compile(T1_NAME_PATTERN, re.IGNORECASE)
    files: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for key, rel in REQUIRED_RELATIVE_FILES.items():
        path = root / rel
        if not path.exists():
            missing.append(key)
            files[key] = {"exists": False}
            continue
        header, rows = csv_header_and_rows(path)
        files[key] = {"exists": True, "rows": rows, "columns": header}
    t1_dir = root / "T1"
    t1_total = 0
    t1_matching = 0
    substudy_counts: dict[str, int] = {}
    if t1_dir.exists():
        for p in t1_dir.rglob("*"):
            if not p.is_file() or not (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
                continue
            t1_total += 1
            m = pattern.match(p.name)
            if m:
                t1_matching += 1
                substudy = m.group("substudy").upper()
                substudy_counts[substudy] = substudy_counts.get(substudy, 0) + 1
    report = {
        "data_root_redacted": redact_path(root),
        "required_files": files,
        "missing_required": missing,
        "t1": {
            "nifti_count": t1_total,
            "matching_expected_pattern": t1_matching,
            "nonmatching_expected_pattern": t1_total - t1_matching,
            "substudy_counts": substudy_counts,
        },
        "status": "READY" if root.exists() and not missing and t1_total > 0 else "BLOCKED",
    }
    atomic_write_json("results/tables/dataset_inventory.json", report)
    md = [
        "# Dataset Inventory\n\n",
        f"- data_root: `{redact_path(root)}`\n",
        f"- status: `{report['status']}`\n",
        f"- T1 NIfTI count: `{t1_total}`\n",
        f"- T1 matching canonical pattern: `{t1_matching}`\n",
        f"- T1 nonmatching canonical pattern: `{t1_total - t1_matching}`\n\n",
        "## Required CSV Sources\n\n",
        "| source | exists | rows | columns |\n",
        "|---|---:|---:|---:|\n",
    ]
    for key, info in files.items():
        md.append(f"| {key} | {info.get('exists')} | {info.get('rows', '')} | {len(info.get('columns', []))} |\n")
    atomic_write_text("results/tables/dataset_inventory.md", "".join(md))
    write_run_manifest("results/logs/inventory_manifest.json", stage="inventory", data_root=str(root), extra={"status": report["status"]})
    StateManager().append_event("inventory", str(report["status"]), "Aggregate inventory completed.")
    print(f"status={report['status']} t1={t1_total} canonical={t1_matching}")
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

