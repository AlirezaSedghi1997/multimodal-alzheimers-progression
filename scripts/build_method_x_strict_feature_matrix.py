from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ALLOWED_OUTPUT_COLUMNS = [
    "patient_hash",
    "sample_hash",
    "ab_label",
    "age",
    "sex",
    "education_years",
    "apoe4_carrier",
    "mmse_base",
]
REQUIRED_SOURCE_COLUMNS = set(ALLOWED_OUTPUT_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build strict Method X feature matrix from master catalog.")
    parser.add_argument("--catalog", default="data/processed/master_catalog.csv")
    parser.add_argument("--out", default="data/processed/method_x_strict_feature_matrix.csv")
    parser.add_argument("--summary", default="results/governance/method_x_strict_feature_matrix_summary.json")
    args = parser.parse_args()

    catalog = Path(args.catalog)
    if not catalog.exists():
        raise FileNotFoundError(f"Master catalog not found: {catalog}")

    rows: list[dict[str, str]] = []
    missing_any = 0
    with catalog.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        source_columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_SOURCE_COLUMNS - source_columns)
        if missing:
            raise RuntimeError(f"Master catalog missing required columns: {missing}")
        for row in reader:
            out_row = {col: row.get(col, "") for col in ALLOWED_OUTPUT_COLUMNS}
            if any(out_row[col] in {"", None} for col in ["age", "sex", "education_years", "apoe4_carrier", "mmse_base", "ab_label"]):
                missing_any += 1
            rows.append(out_row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALLOWED_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "matrix": str(out),
        "rows": len(rows),
        "columns": ALLOWED_OUTPUT_COLUMNS,
        "rows_with_missing_predictor_or_label": missing_any,
        "pet_as_input": False,
        "centiloid_as_predictor": False,
        "suvr_as_predictor": False,
        "tau_pet_as_predictor": False,
        "scanner_site_as_predictor": False,
        "cognitive_beyond_fixed_tabular": False,
        "blood_as_predictor": False,
        "future_visits_as_predictor": False,
        "raw_identifiers_written": False,
        "raw_paths_written": False,
        "subject_level_predictions_exported": False,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "rows": len(rows), "matrix": str(out)}, indent=2))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
