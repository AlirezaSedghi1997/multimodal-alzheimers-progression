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


BLOCKER_GLOSSARY: dict[str, str] = {
    "LONGITUDINAL_ENDPOINT_LABELS_NOT_FROZEN": (
        "CDR-worsening and PACC-decline endpoint definitions are not yet frozen in governance."
    ),
    "PROGRESSION_MODELING_NOT_STARTED": (
        "No progression models trained; branch remains feasibility-only."
    ),
    "SF_LONGITUDINAL_COVERAGE_LIMITED": (
        "SF rows lack the same longitudinal endpoint coverage as enrolled A4/LEARN MRI sample."
    ),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def substudy_audit(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for substudy in sorted({row.get("substudy", "") for row in rows}):
        sub = [row for row in rows if row.get("substudy") == substudy]
        out[substudy] = {
            "rows": len(sub),
            "cdr_worsening_feasible": sum(1 for row in sub if truthy(row.get("endpoint_feasible_cdr_worsening"))),
            "pacc_decline_feasible": sum(1 for row in sub if truthy(row.get("endpoint_feasible_pacc_decline"))),
            "has_cdr_series": sum(1 for row in sub if truthy(row.get("has_cdr_series"))),
            "has_pacc_series": sum(1 for row in sub if truthy(row.get("has_pacc_series"))),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Clinical endpoint branch feasibility audit (CPU-only).")
    ap.add_argument("--catalog", default="data/processed/catalogs/clinical_endpoint_catalog.csv")
    ap.add_argument("--manifest", default="data/processed/catalogs/branch_catalog_manifest.json")
    ap.add_argument("--out_json", default="results/tables/clinical_endpoint_readiness.json")
    ap.add_argument("--out_md", default="docs/clinical_endpoint_readiness.md")
    args = ap.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise RuntimeError(f"Missing catalog: {catalog_path}")
    rows = read_rows(catalog_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8")) if Path(args.manifest).exists() else {}
    branch_summary = (manifest.get("branch_summaries") or {}).get("clinical_endpoint", {})

    enrolled = [row for row in rows if row.get("substudy") in {"A4", "LEARN"}]
    sf_rows = [row for row in rows if row.get("substudy") == "SF"]
    substudy_stats = substudy_audit(rows)

    report: dict[str, Any] = {
        "schema_version": 2,
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
        "sf_rows": len(sf_rows),
        "substudy_counts": dict(Counter(row.get("substudy", "") for row in rows)),
        "cohort_label_cramers_v": branch_summary.get("cohort_label_cramers_v"),
        "substudy_audit": substudy_stats,
        "endpoint_feasibility": {
            "cdr_worsening_enrolled": substudy_stats.get("A4", {}).get("cdr_worsening_feasible", 0)
            + substudy_stats.get("LEARN", {}).get("cdr_worsening_feasible", 0),
            "pacc_decline_enrolled": substudy_stats.get("A4", {}).get("pacc_decline_feasible", 0)
            + substudy_stats.get("LEARN", {}).get("pacc_decline_feasible", 0),
            "cdr_worsening_sf": substudy_stats.get("SF", {}).get("cdr_worsening_feasible", 0),
            "pacc_decline_sf": substudy_stats.get("SF", {}).get("pacc_decline_feasible", 0),
        },
        "blockers": list(BLOCKER_GLOSSARY.keys()),
        "blocker_glossary": BLOCKER_GLOSSARY,
        "required_before_modeling": [
            "Freeze CDR-worsening and PACC-decline endpoint definitions in governance",
            "Use baseline-only predictors with non-future visit discipline",
            "Keep separate claim boundary from Method X-Strict amyloid biomarker status",
            "Report SF longitudinal limitation explicitly if SF rows are included",
        ],
        "interpretation": (
            "Enrolled A4/LEARN rows show structural feasibility flags for progression endpoints, "
            "but endpoint labels are not frozen and no models are trained. "
            "This branch must not be used to support Method X amyloid-status claims."
        ),
        "catalog_artifact": str(catalog_path),
    }

    atomic_write_json(Path(args.out_json), report)

    lines = [
        "# Clinical Endpoint Branch — Readiness Audit\n\n",
        f"- status: **`{report['status']}`**\n",
        f"- total rows: `{report['rows']}`\n",
        f"- enrolled (A4/LEARN): `{report['enrolled_rows']}`\n",
        f"- SF rows: `{report['sf_rows']}`\n",
        f"- cohort-label Cramér's V (catalog): `{report.get('cohort_label_cramers_v')}`\n\n",
        "## Claim boundary\n\n",
        f"{report['claim_boundary']}\n\n",
        f"{report['interpretation']}\n\n",
        "## Substudy feasibility\n\n",
        "| substudy | rows | CDR worsening feasible | PACC decline feasible | has CDR series | has PACC series |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for substudy, stats in sorted(substudy_stats.items()):
        lines.append(
            f"| {substudy} | {stats['rows']} | {stats['cdr_worsening_feasible']} | {stats['pacc_decline_feasible']} | "
            f"{stats['has_cdr_series']} | {stats['has_pacc_series']} |\n"
        )
    lines.extend(["\n", "## Blockers\n\n"])
    for blocker, gloss in BLOCKER_GLOSSARY.items():
        lines.append(f"- **`{blocker}`** — {gloss}\n")
    lines.extend(["\n", "## Required before modeling\n\n"])
    for item in report["required_before_modeling"]:
        lines.append(f"- {item}\n")
    lines.extend(["\n", "## Forbidden claims\n\n"])
    for claim in report["forbidden_claims"]:
        lines.append(f"- {claim}\n")
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
        "Clinical endpoint branch audit refreshed with substudy breakdown.",
    )
    print(json.dumps({"status": report["status"], "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
