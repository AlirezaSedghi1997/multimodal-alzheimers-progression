from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import stable_hash
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float | None:
    if value is None or str(value).strip() in {"", "NA", "NAN"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def enrolled_patients(catalog_path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_csv(catalog_path):
        if row.get("substudy") not in {"A4", "LEARN"}:
            continue
        out[row["patient_hash"]] = row
    return out


def build_cdr_events(cdr_rows: list[dict[str, str]], enrolled: set[str]) -> dict[str, dict[str, Any]]:
    visits: dict[str, list[tuple[float, float]]] = defaultdict(list)
    meta: dict[str, tuple[str, str]] = {}
    for row in cdr_rows:
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        if substudy not in {"A4", "LEARN"}:
            continue
        patient_hash = stable_hash(substudy + "|" + bid, prefix="pt")
        if patient_hash not in enrolled:
            continue
        day = to_float(row.get("CDADTC_DAYS_T0"))
        cdr = to_float(row.get("CDGLOBAL"))
        if day is None or cdr is None:
            continue
        visits[patient_hash].append((day, cdr))
        meta[patient_hash] = (substudy, bid)
    out: dict[str, dict[str, Any]] = {}
    for patient_hash, series in visits.items():
        series.sort(key=lambda x: x[0])
        baseline_cdr = series[0][1]
        followups = [cdr for _, cdr in series[1:]]
        event = int(baseline_cdr == 0.0 and any(cdr >= 0.5 for cdr in followups))
        out[patient_hash] = {
            "patient_hash": patient_hash,
            "substudy": meta[patient_hash][0],
            "baseline_global_cdr": baseline_cdr,
            "n_cdr_visits": len(series),
            "cdr_worsening_event": event,
            "endpoint_feasible": len(series) >= 2,
        }
    return out


def build_pacc_events(pacc_rows: list[dict[str, str]], enrolled: set[str]) -> dict[str, dict[str, Any]]:
    visits: dict[str, list[tuple[float, float]]] = defaultdict(list)
    meta: dict[str, tuple[str, str]] = {}
    for row in pacc_rows:
        substudy = (row.get("SUBSTUDY") or "").upper()
        bid = row.get("BID") or ""
        if substudy not in {"A4", "LEARN"}:
            continue
        patient_hash = stable_hash(substudy + "|" + bid, prefix="pt")
        if patient_hash not in enrolled:
            continue
        visit = row.get("VISCODE") or ""
        try:
            day = float(int(float(visit)))
        except (TypeError, ValueError):
            day = float(len(visits[patient_hash]))
        pacc = to_float(row.get("PACC"))
        if pacc is None:
            continue
        visits[patient_hash].append((day, pacc))
        meta[patient_hash] = (substudy, bid)
    out: dict[str, dict[str, Any]] = {}
    for patient_hash, series in visits.items():
        series.sort(key=lambda x: x[0])
        baseline = series[0][1]
        event = int(any(pacc <= baseline - 1.0 for _, pacc in series[1:]))
        out[patient_hash] = {
            "patient_hash": patient_hash,
            "substudy": meta[patient_hash][0],
            "baseline_pacc": baseline,
            "n_pacc_visits": len(series),
            "pacc_decline_event": event,
            "endpoint_feasible": len(series) >= 2,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build frozen clinical endpoint labels for enrolled A4/LEARN.")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--freeze", default="results/governance/clinical_endpoint_freeze.json")
    ap.add_argument("--out_csv", default="results/tables/clinical_endpoint_label_table.csv")
    ap.add_argument("--out_json", default="results/tables/clinical_endpoint_label_summary.json")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_MODELING":
        raise RuntimeError("Clinical endpoint definitions must be frozen before label build.")

    enrolled = enrolled_patients(Path(args.catalog))
    cdr = build_cdr_events(read_csv(root / "Clinical/Raw Data/cdr.csv"), set(enrolled))
    pacc = build_pacc_events(read_csv(root / "Clinical/Derived Data/PACC.csv"), set(enrolled))

    rows: list[dict[str, Any]] = []
    for patient_hash, catalog_row in enrolled.items():
        cdr_row = cdr.get(patient_hash, {})
        pacc_row = pacc.get(patient_hash, {})
        rows.append(
            {
                "patient_hash": patient_hash,
                "sample_hash": catalog_row.get("sample_hash", ""),
                "substudy": catalog_row.get("substudy", ""),
                "age": catalog_row.get("age", ""),
                "sex": catalog_row.get("sex", ""),
                "education_years": catalog_row.get("education_years", ""),
                "apoe4_carrier": catalog_row.get("apoe4_carrier", ""),
                "mmse_base": catalog_row.get("mmse_base", ""),
                "cdr_worsening_event": cdr_row.get("cdr_worsening_event", ""),
                "pacc_decline_event": pacc_row.get("pacc_decline_event", ""),
                "endpoint_feasible_cdr_worsening": cdr_row.get("endpoint_feasible", False),
                "endpoint_feasible_pacc_decline": pacc_row.get("endpoint_feasible", False),
                "n_cdr_visits": cdr_row.get("n_cdr_visits", 0),
                "n_pacc_visits": pacc_row.get("n_pacc_visits", 0),
                "branch": "clinical_endpoint",
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "branch": "clinical_endpoint",
        "freeze_artifact": args.freeze,
        "enrolled_rows": len(rows),
        "cdr_worsening_positive": sum(1 for row in rows if str(row.get("cdr_worsening_event")) == "1"),
        "pacc_decline_positive": sum(1 for row in rows if str(row.get("pacc_decline_event")) == "1"),
        "cdr_feasible": sum(1 for row in rows if row.get("endpoint_feasible_cdr_worsening")),
        "pacc_feasible": sum(1 for row in rows if row.get("endpoint_feasible_pacc_decline")),
        "claim_boundary": freeze.get("claim_boundary"),
        "catalog_artifact": args.out_csv,
    }
    atomic_write_json(Path(args.out_json), summary)
    atomic_write_text(
        out_csv.with_suffix(".md"),
        (
            "# Clinical Endpoint Label Table\n\n"
            f"- enrolled rows: `{summary['enrolled_rows']}`\n"
            f"- CDR worsening events: `{summary['cdr_worsening_positive']}`\n"
            f"- PACC decline events: `{summary['pacc_decline_positive']}`\n"
        ),
    )
    write_run_manifest(
        "results/logs/clinical_endpoint_label_build_manifest.json",
        stage="clinical_endpoint_label_build",
        data_root=str(root),
        extra=summary,
    )
    StateManager().append_event("clinical_endpoint_label_build", "COMPLETED", "Frozen endpoint labels built for enrolled A4/LEARN.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
