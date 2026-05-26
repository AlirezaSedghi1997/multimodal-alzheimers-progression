from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.a4_schema import resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.privacy import stable_hash
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


SOURCE_FILES = {
    "subject_information": Path("Clinical") / "Derived Data" / "SUBJINFO.csv",
    "scheduled_visits": Path("Clinical") / "Derived Data" / "SV.csv",
    "clinical_dementia_rating": Path("Clinical") / "Raw Data" / "cdr.csv",
    "disposition": Path("Clinical") / "Derived Data" / "DS.csv",
}

DICTIONARY_FILES = (
    Path("Clinical") / "Documents" / "Data Dictionaries" / "clinical_datadic.csv",
    Path("Clinical") / "Documents" / "Data Dictionaries" / "derived_datadic.csv",
    Path("Clinical") / "Documents" / "Data Dictionaries" / "external_datadic.csv",
)

DIAGNOSIS_TERMS = (
    "diagnosis",
    "diagnostic",
    "dementia",
    "alzheimer",
    "mci",
    "conversion",
    "adjudicat",
)
CN_MCI_AD_PATTERN = re.compile(
    r"cognitively normal|\bMCI\b|\bAD dementia\b|Alzheimer disease dementia",
    flags=re.IGNORECASE,
)

EXCLUDED_CONTEXT_TERMS = (
    "family history",
    "sibling",
    "parents",
    "concerns",
    "views",
    "hospital admissions",
    "resource use",
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_row_count_and_columns(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return rows, len(header)


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NULL"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_number_label(value: float | None) -> str | None:
    if value is None:
        return None
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def patient_hash_from_raw(row: dict[str, str]) -> str:
    substudy = (row.get("SUBSTUDY") or "").upper()
    raw_subject_key = row.get("BID") or ""
    return stable_hash(substudy + "|" + raw_subject_key, prefix="pt")


def load_locked_patient_hashes(catalog_path: Path) -> set[str]:
    rows = read_csv_rows(catalog_path)
    return {r["patient_hash"] for r in rows if r.get("patient_hash")}


def matched_rows(rows: list[dict[str, str]], locked_patients: set[str]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if patient_hash_from_raw(row) in locked_patients:
            out.append(row)
    return out


def summarize_sources(root: Path) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for name, rel in SOURCE_FILES.items():
        path = root / rel
        rows, columns = csv_row_count_and_columns(path)
        summary[name] = {
            "file_name": rel.name,
            "exists": path.exists(),
            "rows": rows,
            "columns": columns,
            "raw_paths_exported": False,
            "raw_identifiers_exported": False,
        }
    return summary


def field_missingness(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    fields = {
        "global_cdr_at_visit": "CDGLOBAL",
        "cdr_sum_of_boxes": "CDRSB",
        "blinded_phase_cdr_progression_event": "CDEVENT",
        "all_phase_cdr_progression_event": "CDOLEEVENT",
        "visit_day": "CDADTC_DAYS_T0",
    }
    out: dict[str, dict[str, object]] = {}
    denominator = len(rows)
    for public_name, raw_name in fields.items():
        present = sum(1 for row in rows if str(row.get(raw_name, "")).strip() not in {"", "NA"})
        missing = denominator - present
        out[public_name] = {
            "non_missing": present,
            "missing": missing,
            "missing_rate": round(missing / denominator, 4) if denominator else None,
        }
    return out


def row_day(row: dict[str, str]) -> float | None:
    for key in ("CDADTC_DAYS_T0", "CDDY", "VISCODE"):
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def cdr_class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        label = normalize_number_label(safe_float(row.get("CDGLOBAL")))
        if label is None:
            counts["missing"] += 1
        else:
            counts[label] += 1
    return dict(sorted(counts.items(), key=lambda item: (item[0] == "missing", item[0])))


def event_value(row: dict[str, str], key: str) -> int | None:
    value = safe_float(row.get(key))
    if value is None:
        return None
    return 1 if value > 0 else 0


def patient_event_counts(patient_rows: dict[str, list[dict[str, str]]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter({"event": 0, "no_event_or_censored": 0, "missing": 0})
    for rows in patient_rows.values():
        values = [event_value(row, key) for row in rows]
        numeric = [v for v in values if v is not None]
        if not numeric:
            counts["missing"] += 1
        elif any(v == 1 for v in numeric):
            counts["event"] += 1
        else:
            counts["no_event_or_censored"] += 1
    return dict(counts)


def summarize_followup(patient_rows: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    visits_per_patient: Counter[str] = Counter()
    followup_bins: Counter[str] = Counter({"missing_or_single_visit": 0, "<=1y": 0, "1-3y": 0, "3-5y": 0, ">5y": 0})
    with_two_or_more = 0
    for rows in patient_rows.values():
        numeric_days = sorted(day for row in rows if (day := row_day(row)) is not None)
        n_visits = len(numeric_days)
        if n_visits == 0:
            visits_per_patient["0"] += 1
        elif n_visits == 1:
            visits_per_patient["1"] += 1
        elif n_visits <= 4:
            visits_per_patient["2-4"] += 1
        else:
            visits_per_patient["5+"] += 1
        if n_visits >= 2:
            with_two_or_more += 1
            span = numeric_days[-1] - numeric_days[0]
            if span <= 365:
                followup_bins["<=1y"] += 1
            elif span <= 1095:
                followup_bins["1-3y"] += 1
            elif span <= 1825:
                followup_bins["3-5y"] += 1
            else:
                followup_bins[">5y"] += 1
        else:
            followup_bins["missing_or_single_visit"] += 1
    return {
        "patients_with_two_or_more_cdr_timepoints": with_two_or_more,
        "cdr_visits_per_patient_bins": dict(visits_per_patient),
        "cdr_followup_span_bins": dict(followup_bins),
    }


def summarize_conversion(patient_rows: dict[str, list[dict[str, str]]]) -> dict[str, int]:
    counts: Counter[str] = Counter(
        {
            "first_cdr_zero_with_later_positive_global_cdr": 0,
            "first_cdr_zero_without_later_positive_global_cdr": 0,
            "first_cdr_zero_without_later_cdr": 0,
            "first_cdr_nonzero": 0,
            "first_cdr_missing": 0,
        }
    )
    for rows in patient_rows.values():
        ordered = sorted(rows, key=lambda row: (row_day(row) is None, row_day(row) or 0))
        numeric = [(row_day(row), safe_float(row.get("CDGLOBAL"))) for row in ordered]
        numeric = [(day, value) for day, value in numeric if value is not None]
        if not numeric:
            counts["first_cdr_missing"] += 1
            continue
        _, first = numeric[0]
        later = [value for _, value in numeric[1:]]
        if first != 0:
            counts["first_cdr_nonzero"] += 1
        elif not later:
            counts["first_cdr_zero_without_later_cdr"] += 1
        elif any(value > 0 for value in later):
            counts["first_cdr_zero_with_later_positive_global_cdr"] += 1
        else:
            counts["first_cdr_zero_without_later_positive_global_cdr"] += 1
    return dict(counts)


def first_latest_class_counts(patient_rows: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, int]]:
    first: Counter[str] = Counter()
    latest: Counter[str] = Counter()
    for rows in patient_rows.values():
        ordered = sorted(rows, key=lambda row: (row_day(row) is None, row_day(row) or 0))
        numeric = [safe_float(row.get("CDGLOBAL")) for row in ordered]
        numeric = [value for value in numeric if value is not None]
        if not numeric:
            first["missing"] += 1
            latest["missing"] += 1
            continue
        first[normalize_number_label(numeric[0]) or "missing"] += 1
        latest[normalize_number_label(numeric[-1]) or "missing"] += 1
    return {
        "first_observed_global_cdr_patient_counts": dict(first),
        "latest_observed_global_cdr_patient_counts": dict(latest),
    }


def summarize_cdr(cdr_rows: list[dict[str, str]], locked_patients: set[str]) -> dict[str, object]:
    locked_rows = matched_rows(cdr_rows, locked_patients)
    patient_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in locked_rows:
        patient_rows[patient_hash_from_raw(row)].append(row)

    patient_class_balance = first_latest_class_counts(patient_rows)
    return {
        "source_rows": len(cdr_rows),
        "locked_rows_matched_in_memory": len(locked_rows),
        "locked_patients_with_cdr": len(patient_rows),
        "subject_level_outputs_exported": False,
        "field_missingness": field_missingness(locked_rows),
        "row_level_global_cdr_counts": cdr_class_counts(locked_rows),
        **patient_class_balance,
        "patient_level_blinded_phase_cdr_progression_event_counts": patient_event_counts(patient_rows, "CDEVENT"),
        "patient_level_all_phase_cdr_progression_event_counts": patient_event_counts(patient_rows, "CDOLEEVENT"),
        "conversion_candidate_counts": summarize_conversion(patient_rows),
        "followup_availability": summarize_followup(patient_rows),
    }


def summarize_sv(sv_rows: list[dict[str, str]], locked_patients: set[str]) -> dict[str, object]:
    locked_rows = matched_rows(sv_rows, locked_patients)
    patient_rows: dict[str, int] = Counter()
    scheduled_rows = 0
    for row in locked_rows:
        patient_rows[patient_hash_from_raw(row)] += 1
        if (row.get("SVTYPE") or "").strip().upper() == "SCHEDULED":
            scheduled_rows += 1
    visit_bins: Counter[str] = Counter()
    for count in patient_rows.values():
        if count == 1:
            visit_bins["1"] += 1
        elif count <= 4:
            visit_bins["2-4"] += 1
        elif count <= 10:
            visit_bins["5-10"] += 1
        else:
            visit_bins["11+"] += 1
    return {
        "source_rows": len(sv_rows),
        "locked_rows_matched_in_memory": len(locked_rows),
        "locked_patients_with_visit_records": len(patient_rows),
        "scheduled_visit_rows_matched_in_memory": scheduled_rows,
        "visit_records_per_patient_bins": dict(visit_bins),
    }


def scan_dictionaries(root: Path) -> dict[str, object]:
    scanned = 0
    diagnosis_like = 0
    context_excluded = 0
    direct_participant = 0
    adjudicated = 0
    cn_mci_ad = 0
    cdr_progression = 0
    for rel in DICTIONARY_FILES:
        path = root / rel
        if not path.exists():
            continue
        scanned += 1
        for row in read_csv_rows(path):
            text = " ".join(str(v) for v in row.values()).lower()
            if not any(term in text for term in DIAGNOSIS_TERMS):
                continue
            diagnosis_like += 1
            if any(term in text for term in EXCLUDED_CONTEXT_TERMS):
                context_excluded += 1
                continue
            if "adjudicat" in text:
                adjudicated += 1
            if CN_MCI_AD_PATTERN.search(text):
                cn_mci_ad += 1
            if "cdr-global progression event" in text:
                cdr_progression += 1
            if "diagnosis" in text and "relative" not in text and "family" not in text:
                direct_participant += 1
    return {
        "dictionary_files_scanned": scanned,
        "diagnosis_like_entries": diagnosis_like,
        "context_excluded_entries": context_excluded,
        "direct_participant_diagnosis_entries_after_context_exclusion": direct_participant,
        "adjudicated_diagnosis_entries_detected": adjudicated,
        "cn_mci_ad_status_entries_detected": cn_mci_ad,
        "cdr_progression_entries_detected": cdr_progression,
    }


def build_report(root: Path, catalog_path: Path) -> dict[str, Any]:
    locked_patients = load_locked_patient_hashes(catalog_path)
    cdr_rows = read_csv_rows(root / SOURCE_FILES["clinical_dementia_rating"])
    sv_rows = read_csv_rows(root / SOURCE_FILES["scheduled_visits"])

    cdr_summary = summarize_cdr(cdr_rows, locked_patients)
    sv_summary = summarize_sv(sv_rows, locked_patients)
    dictionary_scan = scan_dictionaries(root)

    direct_diagnosis_available = dictionary_scan["direct_participant_diagnosis_entries_after_context_exclusion"] > 0
    adjudicated_available = dictionary_scan["adjudicated_diagnosis_entries_detected"] > 0
    cn_mci_ad_available = dictionary_scan["cn_mci_ad_status_entries_detected"] > 0

    return {
        "schema_version": 1,
        "artifact": "clinical_alzheimer_target_feasibility",
        "status": "FEASIBILITY_ONLY_NOT_TRAINED",
        "training_started": False,
        "pet_preprocessing_started": False,
        "subject_level_outputs_exported": False,
        "raw_identifiers_exported": False,
        "raw_paths_exported": False,
        "method_x_claim_unchanged": True,
        "method_x_allowed_inputs_unchanged": True,
        "pet_as_method_x_input": False,
        "pet_as_method_x_cog_input": False,
        "current_method_x_target": "A-beta positive vs A-beta negative biomarker status",
        "current_target_is_clinical_alzheimer_diagnosis": False,
        "clinical_alzheimer_output_allowed_now": False,
        "clinical_alzheimer_requires_independent_label": True,
        "locked_catalog_patients": len(locked_patients),
        "source_inventory": summarize_sources(root),
        "dictionary_scan": dictionary_scan,
        "label_availability": {
            "cn_mci_ad_dementia_availability": {
                "explicit_label_detected": cn_mci_ad_available,
                "status": "NOT_DETECTED_AS_DIRECT_LABEL" if not cn_mci_ad_available else "CANDIDATE_REVIEW_REQUIRED",
                "supported_now": False,
                "reason": "No direct independent CN/MCI/AD dementia class label was detected in the available source scan.",
            },
            "diagnosis_at_visit_availability": {
                "explicit_diagnosis_at_visit_detected": direct_diagnosis_available,
                "global_cdr_at_visit_available": cdr_summary["locked_patients_with_cdr"] > 0,
                "status": "CDR_SCALE_AVAILABLE_NOT_DIAGNOSIS",
                "supported_now": False,
                "reason": "Visit-level Global CDR is available as a clinical rating scale, but it is not an adjudicated Alzheimer diagnosis label.",
            },
            "conversion_followup_availability": {
                "cdr_progression_event_detected": dictionary_scan["cdr_progression_entries_detected"] > 0,
                "longitudinal_cdr_available": cdr_summary["followup_availability"]["patients_with_two_or_more_cdr_timepoints"] > 0,
                "status": "CANDIDATE_ENDPOINT_REQUIRES_PROTOCOL_LOCK",
                "supported_now": False,
                "reason": "CDR progression/conversion can be studied later only after a prespecified clinical endpoint definition and leakage review.",
            },
            "adjudicated_diagnosis_availability": {
                "adjudicated_source_detected": adjudicated_available,
                "status": "NOT_DETECTED" if not adjudicated_available else "CANDIDATE_REVIEW_REQUIRED",
                "supported_now": False,
            },
        },
        "missingness_and_class_balance": {
            "clinical_dementia_rating": cdr_summary,
            "scheduled_visits": sv_summary,
        },
        "approved_future_work_gates": [
            "Define a clinical label table that is independent of PET-derived biomarker status.",
            "Lock the clinical endpoint before model training.",
            "Separate at-visit diagnosis, baseline status, and follow-up conversion endpoints.",
            "Run leakage review proving PET, SUVR, centiloid, tau, and PET-derived features are not Method X inputs.",
            "Export only aggregate feasibility and performance reports unless a separate privacy review approves lower-level outputs.",
        ],
        "forbidden_claims": [
            "Do not relabel A-beta biomarker status as Alzheimer disease diagnosis.",
            "Do not train a clinical Alzheimer classifier from this feasibility artifact.",
            "Do not use PET images or PET-derived measurements as Method X or Method X-Cog inputs.",
            "Do not publish subject-level clinical labels, predictions, raw identifiers, or raw paths.",
        ],
        "interpretation": (
            "The available files support aggregate feasibility analysis for a future independent clinical target. "
            "They do not authorize a current Alzheimer disease yes/no model claim."
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    cdr = report["missingness_and_class_balance"]["clinical_dementia_rating"]
    labels = report["label_availability"]
    conversion = cdr["conversion_candidate_counts"]
    followup = cdr["followup_availability"]
    md = [
        "# Clinical Alzheimer Target Feasibility\n\n",
        "- status: `FEASIBILITY_ONLY_NOT_TRAINED`\n",
        "- training started: `False`\n",
        "- PET preprocessing started: `False`\n",
        "- Method X target remains: `A-beta positive vs A-beta negative biomarker status`\n",
        "- clinical Alzheimer output allowed now: `False`\n",
        "- independent clinical label required: `True`\n",
        "- subject-level outputs exported: `False`\n\n",
        "## Availability\n\n",
        "| candidate target | availability status | supported now |\n",
        "|---|---|---:|\n",
        f"| CN/MCI/AD dementia classes | {labels['cn_mci_ad_dementia_availability']['status']} | False |\n",
        f"| diagnosis at visit | {labels['diagnosis_at_visit_availability']['status']} | False |\n",
        f"| conversion follow-up | {labels['conversion_followup_availability']['status']} | False |\n",
        f"| adjudicated diagnosis | {labels['adjudicated_diagnosis_availability']['status']} | False |\n\n",
        "## Aggregate CDR Coverage\n\n",
        f"- locked catalog patients: `{report['locked_catalog_patients']}`\n",
        f"- locked patients with CDR: `{cdr['locked_patients_with_cdr']}`\n",
        f"- locked CDR rows matched in memory: `{cdr['locked_rows_matched_in_memory']}`\n",
        f"- patients with two or more CDR timepoints: `{followup['patients_with_two_or_more_cdr_timepoints']}`\n",
        f"- first-observed Global CDR counts: `{json.dumps(cdr['first_observed_global_cdr_patient_counts'], sort_keys=True)}`\n",
        f"- latest-observed Global CDR counts: `{json.dumps(cdr['latest_observed_global_cdr_patient_counts'], sort_keys=True)}`\n",
        f"- conversion candidate counts: `{json.dumps(conversion, sort_keys=True)}`\n\n",
        "## Governance Rule\n\n",
        "A clinical `Alzheimer disease yes/no` output remains blocked until an independent clinical diagnosis or conversion label is defined, locked, and validated. Biomarker status must continue to be reported as `A-beta positive vs A-beta negative`, not as a definitive clinical diagnosis.\n\n",
        "## Leakage Boundary\n\n",
        "PET, SUVR, centiloid, tau, and PET-derived features remain forbidden as Method X or Method X-Cog inputs. This artifact is aggregate feasibility only and does not start PET preprocessing, PET modeling, MRI 3D ablation, fusion, XAI refresh, or final analysis refresh.\n",
    ]
    atomic_write_text(path, "".join(md))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--out", default="results/tables/clinical_ad_target_feasibility.json")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    report = build_report(root, Path(args.catalog))
    out = Path(args.out)
    atomic_write_json(out, report)
    write_markdown(report, out.with_suffix(".md"))
    write_run_manifest(
        "results/logs/clinical_ad_target_feasibility_manifest.json",
        stage="clinical_ad_target_feasibility",
        data_root=str(root),
        extra={"status": report["status"], "training_started": False},
    )
    StateManager().append_event(
        "clinical_ad_target_feasibility",
        report["status"],
        "Aggregate clinical target feasibility report generated.",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "clinical_alzheimer_output_allowed_now": report["clinical_alzheimer_output_allowed_now"],
                "locked_patients_with_cdr": report["missingness_and_class_balance"]["clinical_dementia_rating"][
                    "locked_patients_with_cdr"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
