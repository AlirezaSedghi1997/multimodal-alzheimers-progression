from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cohort_audit import audit_cohort_label_coupling
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def relabel_rows(rows: list[dict[str, str]], label_key: str = "ab_label_primary") -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        if label_key in item:
            item["ab_label"] = item[label_key]
        elif "ab_label" in item:
            item["ab_label"] = item["ab_label"]
        out.append(item)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Cramér's V cohort-label coupling audit before/after SF inclusion.")
    ap.add_argument("--master_catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--sf_catalog", default="data/processed/catalogs/method_x_cog_sf_catalog.csv")
    ap.add_argument("--out_json", default="results/governance/cohort_label_coupling_audit.json")
    ap.add_argument("--out_md", default="results/governance/cohort_label_coupling_audit.md")
    args = ap.parse_args()

    master_path = Path(args.master_catalog)
    sf_path = Path(args.sf_catalog)
    if not master_path.exists():
        raise RuntimeError(f"Missing master catalog: {master_path}")
    if not sf_path.exists():
        raise RuntimeError(f"Missing SF-inclusive catalog: {sf_path}. Run build_branch_catalogs.py first.")

    before_rows = relabel_rows(read_csv_rows(master_path))
    after_rows = relabel_rows(read_csv_rows(sf_path))
    before = audit_cohort_label_coupling(before_rows)
    after = audit_cohort_label_coupling(after_rows)

    sf_only = [r for r in after_rows if str(r.get("substudy")) == "SF"]
    sf_audit = audit_cohort_label_coupling(sf_only)
    va_mismatch = 0
    for row in sf_only:
        primary = str(row.get("ab_label_primary", ""))
        sensitivity = str(row.get("ab_label_sensitivity", ""))
        if primary != sensitivity:
            va_mismatch += 1

    report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "before_sf_inclusion": {
            "catalog": str(master_path),
            "description": "A4 + LEARN enrolled MRI catalog only",
            **before,
        },
        "after_sf_inclusion": {
            "catalog": str(sf_path),
            "description": "A4 + LEARN + SF internal decoupling catalog",
            **after,
        },
        "sf_only_subset": {
            "rows": len(sf_only),
            **sf_audit,
            "primary_vs_sensitivity_label_mismatch_count": va_mismatch,
        },
        "interpretation": {
            "before_sf": "Near-perfect cohort-label coupling expected (Cramér's V ≈ 1.0).",
            "after_sf": "Coupling should decrease because SF contains both biomarker classes within one screening context.",
            "sf_role": "Internal decoupling/stress-test only; not external validation.",
        },
        "raw_identifiers_written": False,
    }
    atomic_write_json(args.out_json, report)
    lines = [
        "# Cohort-Label Coupling Audit (Before/After SF)\n\n",
        "## Before SF inclusion (A4 + LEARN enrolled MRI)\n\n",
        f"- rows: `{before['rows']}`\n",
        f"- Cramér's V: `{before['cohort_label_cramers_v']:.4f}`\n",
        f"- label_by_substudy: `{before['label_by_substudy']}`\n\n",
        "## After SF inclusion (Method X-Cog/SF catalog)\n\n",
        f"- rows: `{after['rows']}`\n",
        f"- Cramér's V: `{after['cohort_label_cramers_v']:.4f}`\n",
        f"- label_by_substudy: `{after['label_by_substudy']}`\n\n",
        "## SF-only subset\n\n",
        f"- rows: `{len(sf_only)}`\n",
        f"- Cramér's V: `{sf_audit['cohort_label_cramers_v']:.4f}`\n",
        f"- primary vs sensitivity label mismatches: `{va_mismatch}`\n\n",
        "## Interpretation\n\n",
        "Before SF inclusion, binary A4/LEARN classification is mathematically close to a cohort classifier. "
        "After SF inclusion, label and cohort decouple within the SF screening context. "
        "SF remains an internal decoupling instrument, not external validation.\n",
    ]
    atomic_write_text(args.out_md, "".join(lines))
    write_run_manifest(
        "results/logs/cohort_label_coupling_audit_manifest.json",
        stage="cohort_label_coupling_audit",
        data_root=None,
        extra={
            "before_cramers_v": before["cohort_label_cramers_v"],
            "after_cramers_v": after["cohort_label_cramers_v"],
        },
    )
    StateManager().append_event(
        "cohort_label_coupling_audit",
        "COMPLETED",
        f"Cramér's V before={before['cohort_label_cramers_v']:.4f} after={after['cohort_label_cramers_v']:.4f}.",
    )
    print(
        json.dumps(
            {
                "before_cramers_v": before["cohort_label_cramers_v"],
                "after_cramers_v": after["cohort_label_cramers_v"],
                "sf_rows": len(sf_only),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
