from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cramers_v_2xk(table: dict[str, dict[str, int]]) -> float:
    labels = sorted({label for counts in table.values() for label in counts})
    cohorts = sorted(table)
    if len(labels) < 2 or len(cohorts) < 2:
        return 1.0
    total = sum(table[c].get(l, 0) for c in cohorts for l in labels)
    if total == 0:
        return 0.0
    row_totals = {c: sum(table[c].get(l, 0) for l in labels) for c in cohorts}
    col_totals = {l: sum(table[c].get(l, 0) for c in cohorts) for l in labels}
    chi2 = 0.0
    for c in cohorts:
        for l in labels:
            expected = row_totals[c] * col_totals[l] / total
            if expected > 0:
                observed = table[c].get(l, 0)
                chi2 += (observed - expected) ** 2 / expected
    denom = total * min(len(cohorts) - 1, len(labels) - 1)
    return math.sqrt(chi2 / denom) if denom > 0 else 0.0


def audit_catalog(rows: list[dict[str, str]]) -> dict[str, object]:
    by_cohort_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        by_cohort_label[row["substudy"]][row["ab_label"]] += 1
    table = {cohort: dict(counts) for cohort, counts in by_cohort_label.items()}
    pure_cohorts = {
        cohort: (len([label for label, n in counts.items() if n > 0]) == 1)
        for cohort, counts in table.items()
    }
    return {
        "rows": len(rows),
        "label_counts": dict(Counter(r["ab_label"] for r in rows)),
        "substudy_counts": dict(Counter(r["substudy"] for r in rows)),
        "label_by_substudy": table,
        "cohort_label_cramers_v": cramers_v_2xk(table),
        "all_cohorts_single_label": all(pure_cohorts.values()),
        "single_label_by_cohort": pure_cohorts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--baseline", default="results/final_results/baseline/baseline_metrics.json")
    args = ap.parse_args()

    catalog = Path(args.catalog)
    if not catalog.exists():
        raise RuntimeError("Missing master catalog.")
    rows = read_csv_rows(catalog)
    audit = audit_catalog(rows)
    baseline_obj = None
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        baseline_obj = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_auc = None
    if baseline_obj:
        baseline_auc = baseline_obj.get("aggregate", {}).get("roc_auc")
    risk_level = "HIGH" if audit["all_cohorts_single_label"] else "MODERATE"
    recommendation = (
        "Do not interpret current AUC as pathology-only generalization. "
        "Report cohort-label confounding, keep external holdout labeled pseudo-external, "
        "and require sensitivity analyses before final claims."
    )
    report = {
        "risk_level": risk_level,
        "baseline_roc_auc": baseline_auc,
        "audit": audit,
        "recommendation": recommendation,
        "raw_identifiers_used": False,
    }
    atomic_write_json("results/tables/cross_cohort_bias_report.json", report)
    lines = [
        "# Cross-Cohort Bias Report\n\n",
        f"- risk_level: `{risk_level}`\n",
        f"- baseline_roc_auc: `{baseline_auc}`\n",
        f"- cohort_label_cramers_v: `{audit['cohort_label_cramers_v']:.4f}`\n",
        f"- all_cohorts_single_label: `{audit['all_cohorts_single_label']}`\n",
        f"- label_by_substudy: `{audit['label_by_substudy']}`\n",
        "- raw_identifiers_used: `False`\n\n",
        "## Interpretation\n\n",
        recommendation,
        "\n\n## Required Reporting Language\n\n",
        "The local A4/LEARN catalog has strong cohort-label coupling; the locked holdout is pseudo-external, not an independent external validation cohort.\n",
    ]
    atomic_write_text("results/tables/cross_cohort_bias_report.md", "".join(lines))
    write_run_manifest("results/logs/cross_cohort_bias_manifest.json", stage="cross_cohort_bias", data_root=None, extra={"risk_level": risk_level})
    StateManager().append_event("cross_cohort_bias", risk_level, "Cross-cohort label confounding audit completed.")
    print(json.dumps({"risk_level": risk_level, "cramers_v": audit["cohort_label_cramers_v"], "baseline_roc_auc": baseline_auc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

