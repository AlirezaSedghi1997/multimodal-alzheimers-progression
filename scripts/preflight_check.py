from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import REQUIRED_RELATIVE_FILES, resolve_data_root
from src.utils.io_helpers import atomic_write_json
from src.utils.privacy import redact_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    args = ap.parse_args()
    root = resolve_data_root(args.data_root)
    report = {
        "data_root_redacted": redact_path(root),
        "exists": root.exists(),
        "required_files": {},
        "python": sys.version.split()[0],
    }
    for key, rel in REQUIRED_RELATIVE_FILES.items():
        report["required_files"][key] = (root / rel).exists()
    out = Path("results/tables/preflight_report.json")
    atomic_write_json(out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["exists"] and all(report["required_files"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

