from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Clinical endpoint branch feasibility audit (CPU-only).")
    ap.add_argument("--catalog", default="data/processed/catalogs/clinical_endpoint_catalog.csv")
    ap.add_argument("--out_json", default="results/tables/clinical_endpoint_readiness.json")
    ap.add_argument("--out_md", default="docs/clinical_endpoint_readiness.md")
    args = ap.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise RuntimeError(f"Missing catalog: {catalog_path}")
    rows = read_rows(catalog_path)
    enrolled = [row for row in rows if row.get("substudy") in {"A4", "LEARN"}]

    cdr_feasible = sum(1 for row in rows if truthy(row.get("endpoint_feasible_cdr_worsening")))
    pacc_feasible = sum(1 for row in rows if truthy(row.get("endpoint_feasible_pacc_decline")))
    enrolled_cdr = sum(1 for row in enrolled if truthy(row.get("endpoint_feasible_cdr_worsening")))
    enrolled_pacc = sum(1 for row in enrolled if truthy(row.get("endpoint_feasible_pacc_decline")))

    report: dict[str, Any] = {
        "schema_version": 1,
        "branch": "clinical_endpoint",
        "status": "EXPLORATORY_FEASIBILITY_ONLY",
        "phase": 3,
        "claim_boundary": (
            "Exploratory clinical/cognitive progression proxy only. "
            "Do not call Alzheimer diagnosis or conversion-to-AD without independent validated label."
        ),
        "forbidden_claims": [
            "Alzheimer disease diagnosis",
            "conversion-to-AD",
            "clinical AD classification",
        ],
        "rows": len(rows),
        "enrolled_rows": len(enrolled),
        "substudy_counts": dict(Counter(row.get("substudy", "") for row in rows)),
        "endpoint_feasibility": {
            "cdr_worsening_any": cdr_feasible,
            "pacc_decline_any": pacc_feasible,
            "cdr_worsening_enrolled": enrolled_cdr,
            "pacc_decline_enrolled": enrolled_pacc,
        },
        "blockers": [
            "LONGITUDINAL_ENDPOINT_LABELS_NOT_FROZEN",
            "PROGRESSION_MODELING_NOT_STARTED",
            "SF_LONGITUDINAL_COVERAGE_LIMITED",
        ],
        "required_before_modeling": [
            "Freeze CDR-worsening and PACC-decline endpoint definitions",
            "Baseline-only predictors with non-future visit discipline",
            "Separate claim boundary from Method X-Strict amyloid status",
        ],
        "catalog_artifact": str(catalog_path),
    }

    atomic_write_json(Path(args.out_json), report)
    lines = [
        "# Clinical Endpoint Branch Readiness\n\n",
        f"- status: `{report['status']}`\n",
        f"- rows: `{report['rows']}`\n",
        f"- enrolled rows (A4/LEARN): `{report['enrolled_rows']}`\n",
        f"- CDR worsening feasible (enrolled): `{enrolled_cdr}`\n",
        f"- PACC decline feasible (enrolled): `{enrolled_pacc}`\n\n",
        "## Claim boundary\n\n",
        f"{report['claim_boundary']}\n\n",
        "## Blockers\n\n",
    ]
    for blocker in report["blockers"]:
        lines.append(f"- `{blocker}`\n")
    atomic_write_text(Path(args.out_md), "".join(lines))

    write_run_manifest(
        "results/logs/clinical_endpoint_readiness_manifest.json",
        stage="clinical_endpoint_readiness",
        data_root=None,
        extra={"status": report["status"], "rows": len(rows)},
    )
    StateManager().append_event(
        "clinical_endpoint_readiness",
        report["status"],
        "Clinical endpoint branch feasibility audit completed.",
    )
    print(json.dumps({"status": report["status"], "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
