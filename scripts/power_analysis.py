from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.state_manager import StateManager


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(str(int(float(r["ab_label"]))) for r in rows).items()}


def wilson_ci(k: int, n: int, z: float = 1.96) -> dict[str, float | int | None]:
    if n <= 0:
        return {"n": n, "count": k, "proportion": None, "ci_low": None, "ci_high": None}
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return {"n": n, "count": k, "proportion": p, "ci_low": max(0.0, center - margin), "ci_high": min(1.0, center + margin)}


def metric_ci(obj: dict, *path: str) -> dict | None:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if not isinstance(cur, dict):
        return None
    return cur.get("roc_auc_ci")


def main() -> int:
    catalog = Path("data/processed/master_catalog.csv")
    if not catalog.exists():
        raise SystemExit("master_catalog.csv missing; run catalog stage first")
    rows = read_rows(catalog)
    splits = load_json(Path("splits/split_summary.json"))
    fusion = load_json(Path("results/final_results/fusion/fusion_summary.json"))
    external = load_json(Path("results/final_results/external/external_metrics.json"))
    three_d = load_json(Path("results/final_results/three_d/three_d_metrics.json"))
    total_counts = label_counts(rows)
    total_n = len(rows)
    pos_n = total_counts.get("1", 0)
    neg_n = total_counts.get("0", 0)
    external_counts = splits.get("external_label_counts", {})
    dev_counts = splits.get("dev_label_counts", {})
    report = {
        "schema_version": 1,
        "status": "DESCRIPTIVE_POWER_ANALYSIS",
        "catalog_n": total_n,
        "label_counts": total_counts,
        "positive_prevalence_ci": wilson_ci(pos_n, total_n),
        "negative_prevalence_ci": wilson_ci(neg_n, total_n),
        "splits": {
            "dev_rows": splits.get("dev_rows"),
            "external_rows": splits.get("external_rows"),
            "dev_label_counts": dev_counts,
            "external_label_counts": external_counts,
            "external_holdout_type": splits.get("external_holdout_type"),
            "folds": splits.get("folds"),
        },
        "metric_precision": {
            "fusion_roc_auc_ci": metric_ci(fusion, "best"),
            "external_like_roc_auc_ci": external.get("roc_auc_ci") if external else None,
            "three_d_roc_auc": three_d.get("aggregate", {}).get("roc_auc") if three_d else None,
            "three_d_external_like_roc_auc": (
                three_d.get("aggregate", {}).get("external_like", {}).get("roc_auc") if three_d else None
            ),
        },
        "interpretation": (
            "Descriptive power and precision report based on realized N, class balance, and bootstrap CI widths. "
            "Engineering gates are not clinical proof because the holdout is pseudo-external and cohort-label confounding is high."
        ),
        "caveats": [
            "locked_pseudo_external_holdout_not_true_external_validation",
            "complete_cohort_label_confounding_A4_positive_LEARN_negative",
            "AUC gates are engineering evidence, not diagnostic clinical validation",
        ],
        "uses_pet_as_input": False,
        "uses_raw_identifiers": False,
    }
    atomic_write_json("results/tables/power_analysis.json", report)
    lines = [
        "# Power Analysis\n\n",
        f"- status: `{report['status']}`\n",
        f"- catalog_n: `{total_n}`\n",
        f"- label_counts: `{total_counts}`\n",
        f"- positive_prevalence_95ci: `{report['positive_prevalence_ci']['ci_low']:.4f}` to `{report['positive_prevalence_ci']['ci_high']:.4f}`\n",
        f"- dev_rows: `{splits.get('dev_rows')}`\n",
        f"- external_like_rows: `{splits.get('external_rows')}`\n",
        f"- external_holdout_type: `{splits.get('external_holdout_type')}`\n",
        "- caveat: `locked pseudo-external holdout; severe cohort-label confounding remains`\n",
        "- uses_pet_as_input: `False`\n",
        "- uses_raw_identifiers: `False`\n",
    ]
    atomic_write_text("results/tables/power_analysis.md", "".join(lines))
    StateManager().append_event("power_analysis", "COMPLETED", "Power analysis generated from realized catalog and split counts.")
    print(json.dumps({"catalog_n": total_n, "status": report["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
