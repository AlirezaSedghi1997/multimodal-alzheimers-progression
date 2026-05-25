from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import stable_hash
from src.utils.state_manager import StateManager


SITE_LIKE_TERMS = ("site", "scanner", "manufacturer", "field", "center", "protocol", "vendor", "magnet", "model")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def site_like_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if any(term in col.lower() for term in SITE_LIKE_TERMS)]


def hashed_counts(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        value = (row.get(column) or "").strip()
        if not value:
            counts["<missing>"] += 1
        else:
            counts[stable_hash(value, prefix="scanner")] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    args = ap.parse_args()

    rows = read_rows(Path(args.catalog))
    columns = list(rows[0].keys()) if rows else []
    candidates = site_like_columns(columns)
    if not candidates:
        report = {
            "schema_version": 1,
            "status": "SKIP_WITH_JUSTIFICATION",
            "reason": "No site/scanner metadata columns are present in the locked master catalog.",
            "candidate_columns": [],
            "fold_wise_sensitivity_run": False,
            "site_scanner_used_as_model_input": False,
            "uses_raw_identifiers": False,
        }
    else:
        report = {
            "schema_version": 1,
            "status": "DESCRIPTIVE_ONLY",
            "reason": "Site/scanner-like metadata columns were found; values are reported only as hashed aggregate counts.",
            "candidate_columns": candidates,
            "hashed_category_counts": {col: hashed_counts(rows, col) for col in candidates},
            "fold_wise_sensitivity_run": False,
            "site_scanner_used_as_model_input": False,
            "uses_raw_identifiers": False,
        }
    atomic_write_json("results/tables/site_scanner_drift_sensitivity_report.json", report)
    lines = [
        "# Site/Scanner Drift Sensitivity Report\n\n",
        f"- status: `{report['status']}`\n",
        f"- reason: `{report['reason']}`\n",
        f"- candidate_columns: `{report['candidate_columns']}`\n",
        "- site_scanner_used_as_model_input: `False`\n",
        "- uses_raw_identifiers: `False`\n",
    ]
    atomic_write_text("results/tables/site_scanner_drift_sensitivity_report.md", "".join(lines))
    StateManager().append_event("site_scanner_drift", str(report["status"]), str(report["reason"]))
    print(json.dumps({"status": report["status"], "candidate_columns": report["candidate_columns"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
