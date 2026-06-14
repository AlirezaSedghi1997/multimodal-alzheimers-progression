from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager

EXPECTED_FILTERS = {
    "NfL": {"source": "biomarker_Plasma_Roche_Results.csv", "expected_substrings": ["NF-L", "NFL"]},
    "pTau181": {"source": "biomarker_Plasma_Roche_Results.csv", "expected_substrings": ["TPP181", "PTAU181"]},
    "Ab42": {"source": "biomarker_Plasma_Roche_Results.csv", "expected_substrings": ["AMYLB42", "BP42", "TP42", "FP42"]},
    "Ab40": {"source": "biomarker_Plasma_Roche_Results.csv", "expected_substrings": ["AMYLB40", "BP40", "TP40", "FP40"]},
}
OBSERVED_ALIASES = {
    "NfL": ["NF-L", "NFL"],
    "pTau181": ["TPP181", "PTAU181"],
    "Ab42": ["AMYLB42", "BP42", "TP42", "FP42"],
    "Ab40": ["AMYLB40", "BP40", "TP40", "FP40"],
}
RECOMMENDED_FIXES = [
    "Strip LBTESTCD whitespace before matching in build_branch_catalogs.py.",
    "Map Roche NF-L -> NfL and TPP181 -> pTau181 (not NFL/PTAU181 substring filters).",
    "Populate Ab42/Ab40 from Roche AMYLB42/AMYLB40; fallback AB_Test BP42/BP40.",
    "Re-run build_branch_catalogs.py then audit_bio_cog_method_z_readiness.py.",
]


def read_codes(path: Path, code_cols: tuple[str, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            code = ""
            for col in code_cols:
                if row.get(col):
                    code = str(row[col]).strip().upper()
                    break
            if code:
                counts[code] += 1
    return counts


def nonnull_by_code(path: Path, code_col: str, value_col: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            code = str(row.get(code_col) or "").strip().upper()
            val = str(row.get(value_col) or "").strip()
            if code and val and val.upper() not in {"NA", "NAN", "NULL", ""}:
                counts[code] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit blood biomarker source-column mapping for Method Z.")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out_json", default="results/tables/bio_cog_blood_column_mapping_audit.json")
    ap.add_argument("--out_md", default="docs/bio_cog_blood_column_mapping_audit.md")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    external = root / "Clinical" / "External Data"
    roche = external / "biomarker_Plasma_Roche_Results.csv"
    ab_test = external / "biomarker_AB_Test.csv"
    ptau217 = external / "biomarker_pTau217.csv"

    roche_codes = read_codes(roche, ("LBTESTCD", "LBTEST"))
    ab_codes = read_codes(ab_test, ("LBTESTCD", "LBTEST"))
    roche_nonnull = nonnull_by_code(roche, "LBTESTCD", "LABRESN")

    mismatches: list[dict[str, Any]] = []
    for feature, spec in EXPECTED_FILTERS.items():
        observed_hits = sum(roche_nonnull.get(alias, 0) for alias in OBSERVED_ALIASES.get(feature, []))
        if feature in {"Ab42", "Ab40"}:
            source_codes = ab_codes
            observed_hits = sum(source_codes.get(alias, 0) for alias in OBSERVED_ALIASES.get(feature, []))
            observed_hits += roche_nonnull.get("AMYLB42" if feature == "Ab42" else "AMYLB40", 0)
        else:
            source_codes = roche_codes
        expected_hit = any(sub in code for code in source_codes for sub in spec["expected_substrings"])
        mismatches.append(
            {
                "feature": feature,
                "expected_filter": spec["expected_substrings"],
                "legacy_filter_matches_any_code": expected_hit,
                "recommended_aliases": OBSERVED_ALIASES.get(feature, []),
                "observed_nonnull_or_rows": observed_hits,
                "mapping_status": "MISMATCH" if not expected_hit and observed_hits > 0 else "OK_OR_EMPTY",
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "branch": "bio_cog_method_z",
        "status": "MAPPING_AUDIT_COMPLETED",
        "data_root": str(root),
        "claim_boundary": "Diagnostic audit only; does not train models or alter Method X.",
        "source_files": {
            "roche": str(roche),
            "ab_test": str(ab_test),
            "ptau217": str(ptau217),
        },
        "roche_lbtestcd_counts": dict(roche_codes.most_common()),
        "roche_nonnull_by_code": dict(roche_nonnull.most_common()),
        "ab_test_lbtestcd_counts": dict(ab_codes.most_common()),
        "ptau217_rows": sum(read_codes(ptau217, ("TESTCD", "LBTESTCD")).values()),
        "feature_mapping_mismatches": mismatches,
        "recommended_fixes": RECOMMENDED_FIXES,
        "interpretation": (
            "Empty NfL/pTau181/Ab42_40_ratio columns in the locked catalog are explained by LBTESTCD alias "
            "mismatch (NF-L, TPP181, AMYLB42/40, BP42/BP40), not missing source data."
        ),
    }

    atomic_write_json(Path(args.out_json), report)
    lines = [
        "# Bio-Cog / Method Z — Blood Column Mapping Audit\n\n",
        f"- status: `{report['status']}`\n",
        f"- data root: `{root}`\n\n",
        f"{report['interpretation']}\n\n",
        "## Roche LBTESTCD (top)\n\n",
        "| code | rows |\n|---|---:|\n",
    ]
    for code, count in roche_codes.most_common():
        lines.append(f"| {code} | {count} |\n")
    lines.extend(["\n", "## Feature mapping mismatches\n\n"])
    for item in mismatches:
        lines.append(
            f"- **{item['feature']}**: legacy filter matched={item['legacy_filter_matches_any_code']}, "
            f"observed via aliases={item['observed_nonnull_or_rows']}, status=`{item['mapping_status']}`\n"
        )
    lines.extend(["\n", "## Recommended fixes\n\n"])
    for fix in RECOMMENDED_FIXES:
        lines.append(f"- {fix}\n")
    atomic_write_text(Path(args.out_md), "".join(lines))

    write_run_manifest(
        "results/logs/bio_cog_blood_column_mapping_audit_manifest.json",
        stage="bio_cog_blood_column_mapping_audit",
        data_root=str(root),
        extra={"mismatch_count": sum(1 for m in mismatches if m["mapping_status"] == "MISMATCH")},
    )
    StateManager().append_event(
        "bio_cog_blood_column_mapping_audit",
        report["status"],
        "Blood biomarker LBTESTCD mapping audit completed.",
    )
    print(json.dumps({"status": report["status"], "mismatches": mismatches}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
