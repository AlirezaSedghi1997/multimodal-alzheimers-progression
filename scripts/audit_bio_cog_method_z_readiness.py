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


BLOOD_FEATURES = ["pTau217", "GFAP", "NfL", "pTau181", "Ab42", "Ab40", "Ab42_40_ratio"]
FORBIDDEN_AS_METHOD_X = True


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def present(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text not in {"", "NA", "NAN", "NULL"}


def coverage(rows: list[dict[str, str]], column: str) -> dict[str, Any]:
    n = len(rows)
    count = sum(1 for row in rows if present(row.get(column)))
    return {
        "present": count,
        "missing": n - count,
        "coverage_rate": round(count / n, 6) if n else 0.0,
    }


def subset_coverage(rows: list[dict[str, str]], substudy: str) -> dict[str, Any]:
    sub = [row for row in rows if row.get("substudy") == substudy]
    return {
        "rows": len(sub),
        "features": {feature: coverage(sub, feature) for feature in BLOOD_FEATURES},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 Bio-Cog/Method Z feasibility audit (CPU-only).")
    ap.add_argument("--catalog", default="data/processed/catalogs/bio_cog_method_z_catalog.csv")
    ap.add_argument("--out_json", default="results/tables/bio_cog_method_z_readiness.json")
    ap.add_argument("--out_md", default="results/tables/bio_cog_method_z_readiness.md")
    args = ap.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise RuntimeError(f"Missing catalog: {catalog_path}")
    rows = read_rows(catalog_path)
    if not rows:
        raise RuntimeError("Bio-Cog/Method Z catalog is empty.")

    feature_cov = {feature: coverage(rows, feature) for feature in BLOOD_FEATURES}
    usable_any = sum(
        1
        for row in rows
        if any(present(row.get(feature)) for feature in BLOOD_FEATURES)
    )
    usable_core = sum(
        1
        for row in rows
        if present(row.get("pTau217")) or present(row.get("GFAP"))
    )

    blockers: list[str] = []
    if feature_cov["NfL"]["present"] == 0:
        blockers.append("NFL_COLUMN_EMPTY_IN_LOCKED_CATALOG")
    if feature_cov["pTau181"]["present"] == 0:
        blockers.append("PTAU181_COLUMN_EMPTY_IN_LOCKED_CATALOG")
    if feature_cov["Ab42_40_ratio"]["present"] == 0:
        blockers.append("AB42_40_RATIO_NOT_POPULATED_IN_LOCKED_CATALOG")
    if usable_core < 500:
        blockers.append("INSUFFICIENT_CORE_BLOOD_FEATURE_COVERAGE")

    status = "FEASIBILITY_ONLY_NOT_TRAINED" if blockers else "DESIGN_READY_AWAITING_PHASE3_TRAINING"

    report: dict[str, Any] = {
        "schema_version": 1,
        "branch": "bio_cog_method_z",
        "status": status,
        "phase": 3,
        "claim_boundary": (
            "Minimally invasive biomarker-assisted branch; never supports the non-invasive Method X claim."
        ),
        "pet_as_method_x_input": False,
        "supports_method_x_strict_claim": False,
        "rows": len(rows),
        "substudy_counts": dict(Counter(row.get("substudy", "") for row in rows)),
        "label_counts": dict(Counter(row.get("ab_label_primary", "") for row in rows)),
        "feature_coverage": feature_cov,
        "usable_rows_any_blood_feature": usable_any,
        "usable_rows_core_blood_feature": usable_core,
        "subset_coverage": {
            "A4": subset_coverage(rows, "A4"),
            "LEARN": subset_coverage(rows, "LEARN"),
            "SF": subset_coverage(rows, "SF"),
        },
        "blockers": blockers,
        "required_before_training": [
            "Frozen blood-feature allowlist and no-PET-input test",
            "SF-aware internal stress-test protocol for blood branch",
            "Locked pseudo-external evaluation plan separate from Method X",
            "Permutation/ablation for blood biomarker families",
        ],
        "catalog_artifact": str(catalog_path),
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    atomic_write_json(out_json, report)

    lines = [
        "# Bio-Cog / Method Z Readiness (Phase 3)\n\n",
        f"- status: `{report['status']}`\n",
        f"- rows: `{report['rows']}`\n",
        f"- claim boundary: {report['claim_boundary']}\n\n",
        "## Feature coverage\n\n",
        "| feature | present | coverage_rate |\n",
        "|---|---:|---:|\n",
    ]
    for feature, stats in feature_cov.items():
        lines.append(f"| {feature} | {stats['present']} | {stats['coverage_rate']} |\n")
    lines.extend(
        [
            "\n",
            "## Blockers\n\n",
        ]
    )
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker}`\n")
    else:
        lines.append("- none\n")
    lines.extend(
        [
            "\n",
            "## Required before training\n\n",
        ]
    )
    for item in report["required_before_training"]:
        lines.append(f"- {item}\n")
    atomic_write_text(out_md, "".join(lines))

    write_run_manifest(
        "results/logs/bio_cog_method_z_readiness_manifest.json",
        stage="bio_cog_method_z_readiness",
        data_root=None,
        extra={"status": status, "rows": len(rows)},
    )
    StateManager().append_event(
        "bio_cog_method_z_readiness",
        status,
        "Phase 3 Bio-Cog/Method Z feasibility audit completed (CPU-only).",
    )
    print(json.dumps({"status": status, "rows": len(rows), "blockers": blockers}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
