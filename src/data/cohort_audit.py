from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable


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
    for cohort in cohorts:
        for label in labels:
            expected = row_totals[cohort] * col_totals[label] / total
            if expected > 0:
                observed = table[cohort].get(label, 0)
                chi2 += (observed - expected) ** 2 / expected
    denom = total * min(len(cohorts) - 1, len(labels) - 1)
    return math.sqrt(chi2 / denom) if denom > 0 else 0.0


def cohort_label_table(rows: Iterable[dict[str, object]], cohort_key: str = "substudy", label_key: str = "ab_label") -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        cohort = str(row.get(cohort_key, ""))
        label = str(row.get(label_key, ""))
        if cohort and label not in {"", "None"}:
            table[cohort][label] += 1
    return {cohort: dict(counts) for cohort, counts in table.items()}


def audit_cohort_label_coupling(rows: list[dict[str, object]], cohort_key: str = "substudy", label_key: str = "ab_label") -> dict[str, object]:
    table = cohort_label_table(rows, cohort_key=cohort_key, label_key=label_key)
    pure_cohorts = {
        cohort: (len([label for label, n in counts.items() if n > 0]) == 1)
        for cohort, counts in table.items()
    }
    return {
        "rows": len(rows),
        "label_counts": dict(Counter(str(r.get(label_key, "")) for r in rows)),
        "substudy_counts": dict(Counter(str(r.get(cohort_key, "")) for r in rows)),
        "label_by_substudy": table,
        "cohort_label_cramers_v": cramers_v_2xk(table),
        "all_cohorts_single_label": all(pure_cohorts.values()) if pure_cohorts else False,
        "single_label_by_cohort": pure_cohorts,
    }
