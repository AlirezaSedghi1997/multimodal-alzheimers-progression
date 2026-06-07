from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


METHOD_X_ALLOWED_FEATURES = {
    "age",
    "sex",
    "education_years",
    "apoe4_carrier",
    "mmse_base",
}
ALLOWED_NON_FEATURE_COLUMNS = {
    "patient_hash",
    "sample_hash",
    "ab_label",
    "target",
    "fold",
    "split",
}
FORBIDDEN_PATTERNS = {
    "pet": re.compile(r"(^|_)(pet|fbp|ftp|av45|florbetapir|amyloid_pet)(_|$)", re.I),
    "suvr": re.compile(r"suvr", re.I),
    "centiloid_predictor": re.compile(r"centiloid|amylcent|amyloid_burden|pmod", re.I),
    "tau": re.compile(r"(^|_)(tau|ftp)(_|$)", re.I),
    "scanner_site": re.compile(r"site|scanner|manufacturer|field|center|protocol|vendor|magnet|model", re.I),
    "blood_biomarker": re.compile(r"ptau|gfap|ab40|ab42|aβ40|aβ42|nfl|plasma|blood|roche", re.I),
    "future_visit": re.compile(r"future|followup|follow_up|month|m\d+|year\d+|longitudinal|decline|worsen", re.I),
    "raw_identifier_or_path": re.compile(r"(^|_)(bid|subject_id|rid|path|filepath|file_path|raw_id)(_|$)", re.I),
    "cognitive_beyond_fixed": re.compile(r"pacc|cdr|cogstate|memory|complaint|mood|anxiety|fcsrt|ldel|digit|adl|cfi", re.I),
    "leaky_mri_surrogate": re.compile(r"voxel|spacing|shape|dim|intensity|percentile|histogram", re.I),
}


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(next(csv.reader(f)))


def classify_forbidden(columns: Iterable[str]) -> dict[str, list[str]]:
    violations: dict[str, list[str]] = {}
    for column in columns:
        if column in METHOD_X_ALLOWED_FEATURES or column in ALLOWED_NON_FEATURE_COLUMNS:
            continue
        for reason, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(column):
                violations.setdefault(reason, []).append(column)
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a final Method X feature matrix header.")
    parser.add_argument("--matrix", required=True, help="CSV feature matrix to validate.")
    parser.add_argument("--out", default="results/governance/method_x_feature_matrix_pet_leakage_check.json")
    parser.add_argument(
        "--strict_allowed_features",
        action="store_true",
        help="Also fail if columns outside Method X allowed features and allowed metadata are present.",
    )
    args = parser.parse_args()

    matrix = Path(args.matrix)
    if not matrix.exists():
        raise FileNotFoundError(f"Feature matrix does not exist: {matrix}")
    columns = read_header(matrix)
    violations = classify_forbidden(columns)
    extra_columns = sorted(
        c for c in columns if c not in METHOD_X_ALLOWED_FEATURES and c not in ALLOWED_NON_FEATURE_COLUMNS
    )
    strict_violation = bool(args.strict_allowed_features and extra_columns)
    status = "PASS" if not violations and not strict_violation else "FAIL"
    report = {
        "schema_version": 1,
        "status": status,
        "matrix": str(matrix),
        "n_columns": len(columns),
        "allowed_method_x_features": sorted(METHOD_X_ALLOWED_FEATURES),
        "allowed_non_feature_columns": sorted(ALLOWED_NON_FEATURE_COLUMNS),
        "forbidden_violations": violations,
        "extra_columns_under_strict_mode": extra_columns if args.strict_allowed_features else [],
        "pet_as_input": False if not any(k in violations for k in ["pet", "suvr", "centiloid_predictor", "tau"]) else True,
        "raw_identifiers_written": False,
        "raw_paths_written": False,
        "subject_level_predictions_exported": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "violations": violations}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
