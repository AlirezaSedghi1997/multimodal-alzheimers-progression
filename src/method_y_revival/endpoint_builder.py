from __future__ import annotations

import hashlib
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .atomic_io import atomic_write_json, read_json
from .contracts import sha256_mapping


class EndpointBuildError(ValueError):
    """Raised when a future endpoint cannot be frozen without privacy or membership defects."""


PRIVATE_ENDPOINT_FIELDS = frozenset(
    {"raw_path", "source_subject_id", "BID", "SUBSTUDY", "sidecar_path"}
)


def endpoint_contract_path(output_root: str | Path, branch: str) -> Path:
    if branch not in {"amyloid", "tau", "paired_at"}:
        raise EndpointBuildError(f"UNSUPPORTED_ENDPOINT_BRANCH:{branch}")
    return Path(output_root) / "endpoints" / f"{branch}_endpoint_contract.json"


def treatment_active_value(value: object) -> float | str:
    text = "" if value is None else str(value).strip().upper()
    if not text or text in {"NA", "N/A", "UNKNOWN", "NONE"}:
        return ""
    if "PLACEBO" in text:
        return 0.0
    if "SOLANEZUMAB" in text or "ACTIVE" in text or "TREATMENT" in text:
        return 1.0
    return ""


def compatibility_endpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    endpoint = config["endpoint"]
    validation = config["validation"]
    return {
        "inputs": {
            "tau_primary_suvr_column": "suvr_persi",
            "tau_sensitivity_suvr_column": "suvr_crus",
            "tau_unavailable_suvr_column": "suvr_cer",
            "tau_scan_analyzed_required": "Yes",
            "tau_ligand_required": "Flortaucipir",
        },
        "endpoint": {
            "primary_name": endpoint["primary_name"],
            "event_definition": "first post-PET CDR increase confirmed by the next visit",
            "index_date_rule": "max(first analyzed FTP scan date, selected baseline CDR date)",
            "forbidden_label_sources": endpoint["forbidden_label_sources"],
            "baseline_tau_to_cdr_window_days": endpoint["pet_to_baseline_window_days"],
            "followup_window_days": endpoint["followup_window_days"],
            "baseline_cdr_required": endpoint["baseline_cdr_required"],
            "event_time_rule": endpoint["event_time_rule"],
        },
        "validation": {
            "min_total_events": validation["minimum_total_events"],
            "min_events_per_heldout_arm": validation[
                "minimum_events_per_treatment_arm"
            ],
        },
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_first_amyloid_scans(
    source_rows: Iterable[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    from scripts.build_method_y_tau_progression import norm, safe_float, subject_key

    groups: dict[tuple[tuple[str, str], str, str, float], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    total = 0
    for source in source_rows:
        total += 1
        ligand = norm(source.get("ligand"))
        analyzed = norm(source.get("scan_analyzed"))
        if ligand.casefold() != "florbetapir":
            counts["rejected_non_fbp_rows"] += 1
            continue
        if analyzed.casefold() != "yes":
            counts["rejected_not_analyzed_rows"] += 1
            continue
        scan_date = safe_float(source.get("scan_date_DAYS_T0"))
        key = subject_key(source)
        if scan_date is None or not key[0] or not key[1]:
            counts["rejected_missing_key_or_date_rows"] += 1
            continue
        viscode = norm(source.get("VISCODE"))
        scan_number = norm(source.get("scan_number"))
        group_key = (key, viscode, scan_number, scan_date)
        group = groups.setdefault(
            group_key,
            {
                "subject_key": key,
                "viscode": viscode,
                "scan_number": scan_number,
                "scan_date": scan_date,
                "regions": {},
            },
        )
        region = norm(source.get("brain_region"))
        if region:
            group["regions"][region] = {
                column: safe_float(source.get(column))
                for column in ("suvr_cer", "suvr_persi", "suvr_crus")
            }

    first: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups.values():
        key = group["subject_key"]
        order = (group["scan_date"], group["viscode"], group["scan_number"])
        if key not in first or order < (
            first[key]["scan_date"],
            first[key]["viscode"],
            first[key]["scan_number"],
        ):
            first[key] = group
    audit = {
        "tracer": "FBP",
        "source_rows": total,
        "eligible_scan_groups": len(groups),
        "first_analyzed_fbp_subjects": len(first),
        **dict(counts),
    }
    for key in (
        "rejected_non_fbp_rows",
        "rejected_not_analyzed_rows",
        "rejected_missing_key_or_date_rows",
    ):
        audit.setdefault(key, 0)
    return first, audit


def build_amyloid_endpoint_rows(
    first_amyloid: dict[tuple[str, str], dict[str, Any]],
    subject_rows: list[dict[str, Any]],
    cdr_rows: list[dict[str, Any]],
    pacc_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from scripts.build_method_y_tau_progression import (
        add_numeric,
        detect_sustained_event,
        index_rows,
        norm,
        patient_hash_for_key,
        safe_float,
        select_baseline_cdr,
        selected_pacc_row,
        sorted_cdr_rows,
        subject_key,
    )

    endpoint = config["endpoint"]
    subject_by_key = {subject_key(row): row for row in subject_rows}
    cdr_by_key = {key: sorted_cdr_rows(rows) for key, rows in index_rows(cdr_rows).items()}
    pacc_by_key = index_rows(pacc_rows)
    allowed_cohorts = {"A4", "LEARN"}
    exclusions: Counter[str] = Counter()
    excluded_cohorts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for key, scan in sorted(first_amyloid.items(), key=lambda item: patient_hash_for_key(item[0])):
        source_cohort = str(key[0]).upper()
        if source_cohort not in allowed_cohorts:
            excluded_cohorts[source_cohort] += 1
            continue
        subject = subject_by_key.get(key)
        if subject is None:
            exclusions["missing_subjinfo"] += 1
            continue
        centiloid = safe_float(subject.get("AMYLCENT"))
        if centiloid is None:
            exclusions["missing_continuous_centiloid"] += 1
            continue
        series = cdr_by_key.get(key, [])
        baseline, gap = select_baseline_cdr(
            series,
            float(scan["scan_date"]),
            float(endpoint["pet_to_baseline_window_days"]),
        )
        if baseline is None:
            exclusions["missing_baseline_cdr_within_window"] += 1
            continue
        if float(baseline["_cdglobal"]) != float(endpoint["baseline_cdr_required"]):
            exclusions["baseline_cdr_not_zero"] += 1
            continue
        index_date = max(float(scan["scan_date"]), float(baseline["_date"]))
        followups = [
            row
            for row in series
            if index_date < float(row["_date"]) <= index_date + float(endpoint["followup_window_days"])
        ]
        if not followups:
            exclusions["no_post_index_cdr_followup"] += 1
            continue
        has_event, event_date = detect_sustained_event(
            followups,
            event_time_rule=str(endpoint["event_time_rule"]),
        )
        observed_time = (
            float(event_date) - index_date
            if has_event and event_date is not None
            else min(
                float(endpoint["followup_window_days"]),
                float(followups[-1]["_date"]) - index_date,
            )
        )
        if observed_time <= 0:
            exclusions["nonpositive_observed_time"] += 1
            continue
        pacc = selected_pacc_row(pacc_by_key.get(key, []), str(scan["viscode"]))
        composite = scan["regions"].get("Composite_Summary", {})
        row: dict[str, Any] = {
            "patient_hash": patient_hash_for_key(key),
            "cohort": source_cohort,
            "treatment_arm": norm(subject.get("TX")) or "NA",
            "treatment_active": treatment_active_value(subject.get("TX")),
            "baseline_gap_days_abs": round(float(gap or 0.0), 6),
            "index_source": "max_fbp_scan_and_baseline_cdr_dates",
            "event": int(has_event),
            "time_days": round(float(observed_time), 6),
            "censoring_rule": "event_time" if has_event else "right_censored",
            "followup_cdr_visits": len(followups),
            "baseline_cdglobal": float(baseline["_cdglobal"]),
            "baseline_cdrsb": "" if baseline.get("_cdrsb") is None else float(baseline["_cdrsb"]),
            "baseline_amyloid_centiloid": centiloid,
            "global_amyloid_suvr_persi": "" if composite.get("suvr_persi") is None else float(composite["suvr_persi"]),
            "uses_raw_identifier": False,
            "uses_raw_path": False,
        }
        add_numeric(row, "age", subject.get("AGEYR"))
        sex = norm(subject.get("SEX"))
        row["sex_binary"] = 1.0 if sex in {"2", "M", "Male", "male"} else 0.0 if sex else ""
        add_numeric(row, "education_years", subject.get("EDCCNTU"))
        carrier = safe_float(subject.get("AAPOEGNPRSNFLG"))
        if carrier is None:
            carrier = safe_float(subject.get("APOEGNPRSNFLG"))
        row["apoe4_carrier"] = "" if carrier is None else carrier
        for output_name, source_name in {
            "baseline_pacc_raw": "PACC.raw",
            "baseline_pacc_change": "PACC",
            "baseline_fctotal96": "FCTOTAL96",
            "baseline_ldeltotal": "LDELTOTAL",
            "baseline_digittotal": "DIGITTOTAL",
            "baseline_mmscore": "MMSCORE",
        }.items():
            add_numeric(row, output_name, pacc.get(source_name) if pacc else None)
        output.append(row)
    output.sort(key=lambda row: str(row["patient_hash"]))
    audit = {
        "status": "COMPLETED",
        "tracer": "FBP",
        "development_cohort": "A4",
        "internal_transport_cohort": "LEARN",
        "eligible_rows": len(output),
        "events": sum(int(row["event"]) for row in output),
        "cohort_counts": dict(Counter(str(row["cohort"]) for row in output)),
        "event_counts_by_cohort": dict(
            Counter(str(row["cohort"]) for row in output if int(row["event"]) == 1)
        ),
        "exclusions": dict(exclusions),
        "excluded_cohorts": dict(excluded_cohorts),
        "forbidden_label_sources_not_used": endpoint["forbidden_label_sources"],
        "same_pet_derived_target_used": False,
        "treatment_structure": "active_vs_placebo_covariate",
    }
    return output, audit


def build_amyloid_endpoint_from_sources(
    *, config: dict[str, Any], output_root: str | Path
) -> dict[str, Any]:
    from scripts.build_method_y_tau_progression import read_csv_rows

    raw_root = Path(config["paths"]["raw_root"])
    sources = {
        "amyloid_suvr_csv": raw_root / config["clinical_sources"]["amyloid_suvr"],
        "cdr_csv": raw_root / config["clinical_sources"]["cdr"],
        "pacc_csv": raw_root / config["clinical_sources"]["pacc"],
        "subjinfo_csv": raw_root / config["clinical_sources"]["subject_info"],
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise EndpointBuildError(f"MISSING_ENDPOINT_SOURCES:{missing}")
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    endpoint_definition = {
        "primary_name": config["endpoint"]["primary_name"],
        "tracer": "FBP",
        "predictor": "continuous_centiloid",
        "target_source": "future_sustained_cdr_global_progression",
        "event_time_rule": config["endpoint"]["event_time_rule"],
        "baseline_cdr_required": config["endpoint"]["baseline_cdr_required"],
        "pet_to_baseline_window_days": config["endpoint"]["pet_to_baseline_window_days"],
        "followup_window_days": config["endpoint"]["followup_window_days"],
        "development_cohort": "A4",
        "internal_transport_cohort": "LEARN",
        "forbidden_label_sources": config["endpoint"]["forbidden_label_sources"],
        "treatment_structure": "active_vs_placebo_covariate",
        "same_pet_derived_target_used": False,
    }
    output_directory = Path(output_root) / "endpoints"
    contract_path = endpoint_contract_path(output_root, "amyloid")
    summary_path = output_directory / "amyloid_endpoint_summary.json"
    if contract_path.exists():
        existing = read_json(contract_path)
        if existing.get("source_hashes") != source_hashes:
            raise EndpointBuildError("LOCKED_AMYLOID_ENDPOINT_SOURCE_HASH_MISMATCH")
        if existing.get("endpoint_definition") != endpoint_definition:
            raise EndpointBuildError("LOCKED_AMYLOID_ENDPOINT_DEFINITION_MISMATCH")
        stored = output_directory / str(existing.get("endpoint_path", ""))
        if not stored.is_file() or sha256_file(stored) != existing.get("endpoint_sha256"):
            raise EndpointBuildError("LOCKED_AMYLOID_ENDPOINT_ARTIFACT_HASH_MISMATCH")
        summary = read_json(summary_path) if summary_path.exists() else {"contract": existing}
        summary["status"] = "ENDPOINT_LOCKED_REUSED"
        return summary

    first, source_audit = load_first_amyloid_scans(read_csv_rows(sources["amyloid_suvr_csv"]))
    rows, endpoint_audit = build_amyloid_endpoint_rows(
        first,
        read_csv_rows(sources["subjinfo_csv"]),
        read_csv_rows(sources["cdr_csv"]),
        read_csv_rows(sources["pacc_csv"]),
        config,
    )
    contract = write_locked_endpoint(
        rows,
        output_path=output_directory / "amyloid_progression_endpoint.parquet",
        contract_path=contract_path,
        source_hashes=source_hashes,
        endpoint_definition=endpoint_definition,
    )
    summary = {
        "status": "ENDPOINT_LOCKED",
        "contract": contract,
        "amyloid_source_audit": source_audit,
        "endpoint_audit": endpoint_audit,
        "raw_identifiers_exported": False,
        "raw_paths_exported": False,
    }
    atomic_write_json(summary_path, summary)
    return summary


def build_paired_at_endpoint_rows(
    first_amyloid: dict[tuple[str, str], dict[str, Any]],
    first_tau: dict[tuple[str, str], dict[str, Any]],
    subject_rows: list[dict[str, Any]],
    cdr_rows: list[dict[str, Any]],
    pacc_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    from scripts.build_method_y_tau_progression import build_label_rows, patient_hash_for_key

    window = float(config["branches"]["paired_at"]["scan_window_days"])
    combined: dict[tuple[str, str], dict[str, Any]] = {}
    gap_by_hash: dict[str, float] = {}
    excluded_outside = 0
    for key in sorted(set(first_amyloid).intersection(first_tau)):
        amyloid = first_amyloid[key]
        tau = first_tau[key]
        gap = abs(float(amyloid["scan_date"]) - float(tau["scan_date"]))
        if gap > window:
            excluded_outside += 1
            continue
        later_is_amyloid = float(amyloid["scan_date"]) >= float(tau["scan_date"])
        combined[key] = {
            **tau,
            "scan_date": max(float(amyloid["scan_date"]), float(tau["scan_date"])),
            "viscode": amyloid["viscode"] if later_is_amyloid else tau["viscode"],
        }
        gap_by_hash[patient_hash_for_key(key)] = gap
    if not combined:
        return [], {
            "status": "BLOCKED_NO_PAIRED_AT_WITHIN_WINDOW",
            "paired_within_window": 0,
            "excluded_outside_scan_window": excluded_outside,
            "time_zero_after_both_pet_scans": True,
        }, []
    compatibility = compatibility_endpoint_config(config)
    rows, endpoint_audit, region_features = build_label_rows(
        combined,
        subject_rows,
        cdr_rows,
        pacc_rows,
        compatibility,
    )
    for row in rows:
        row["paired_fbp_ftp_gap_days"] = float(gap_by_hash[str(row["patient_hash"])])
        row["index_source"] = "max_fbp_ftp_and_baseline_cdr_dates"
        row["treatment_active"] = treatment_active_value(row.get("treatment_arm"))
    audit = {
        **endpoint_audit,
        "paired_within_window": len(combined),
        "paired_endpoint_rows": len(rows),
        "excluded_outside_scan_window": excluded_outside,
        "scan_window_days": window,
        "time_zero_after_both_pet_scans": True,
    }
    return rows, audit, region_features


def build_paired_at_endpoint_from_sources(
    *, config: dict[str, Any], output_root: str | Path
) -> dict[str, Any]:
    from scripts.build_method_y_tau_progression import (
        load_first_tau_scans,
        read_csv_rows,
    )

    raw_root = Path(config["paths"]["raw_root"])
    sources = {
        "amyloid_suvr_csv": raw_root / config["clinical_sources"]["amyloid_suvr"],
        "tau_suvr_csv": raw_root / config["clinical_sources"]["tau_suvr"],
        "cdr_csv": raw_root / config["clinical_sources"]["cdr"],
        "pacc_csv": raw_root / config["clinical_sources"]["pacc"],
        "subjinfo_csv": raw_root / config["clinical_sources"]["subject_info"],
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise EndpointBuildError(f"MISSING_PAIRED_AT_SOURCES:{missing}")
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    endpoint_definition = {
        "branch": "paired_at",
        "tracers": ["FBP", "FTP"],
        "scan_window_days": config["branches"]["paired_at"]["scan_window_days"],
        "time_zero_rule": "max_fbp_ftp_and_baseline_cdr_dates",
        "event_time_rule": config["endpoint"]["event_time_rule"],
        "followup_window_days": config["endpoint"]["followup_window_days"],
        "treatment_structure": "active_vs_placebo_covariate",
        "same_pet_derived_target_used": False,
    }
    output_directory = Path(output_root) / "endpoints"
    contract_path = endpoint_contract_path(output_root, "paired_at")
    summary_path = output_directory / "paired_at_endpoint_summary.json"
    if contract_path.exists():
        existing = read_json(contract_path)
        if existing.get("source_hashes") != source_hashes:
            raise EndpointBuildError("LOCKED_PAIRED_AT_SOURCE_HASH_MISMATCH")
        if existing.get("endpoint_definition") != endpoint_definition:
            raise EndpointBuildError("LOCKED_PAIRED_AT_DEFINITION_MISMATCH")
        stored = output_directory / str(existing.get("endpoint_path", ""))
        if not stored.is_file() or sha256_file(stored) != existing.get("endpoint_sha256"):
            raise EndpointBuildError("LOCKED_PAIRED_AT_ARTIFACT_HASH_MISMATCH")
        summary = read_json(summary_path) if summary_path.exists() else {"contract": existing}
        summary["status"] = "ENDPOINT_LOCKED_REUSED"
        return summary
    first_amyloid, amyloid_audit = load_first_amyloid_scans(
        read_csv_rows(sources["amyloid_suvr_csv"])
    )
    first_tau, tau_audit = load_first_tau_scans(
        sources["tau_suvr_csv"], compatibility_endpoint_config(config)
    )
    rows, endpoint_audit, region_features = build_paired_at_endpoint_rows(
        first_amyloid,
        first_tau,
        read_csv_rows(sources["subjinfo_csv"]),
        read_csv_rows(sources["cdr_csv"]),
        read_csv_rows(sources["pacc_csv"]),
        config,
    )
    contract = write_locked_endpoint(
        rows,
        output_path=output_directory / "paired_at_progression_endpoint.parquet",
        contract_path=contract_path,
        source_hashes=source_hashes,
        endpoint_definition=endpoint_definition,
    )
    summary = {
        "status": "ENDPOINT_LOCKED",
        "contract": contract,
        "endpoint_audit": endpoint_audit,
        "amyloid_source_audit": amyloid_audit,
        "tau_source_audit": tau_audit,
        "regional_feature_count": len(region_features),
        "raw_identifiers_exported": False,
        "raw_paths_exported": False,
    }
    atomic_write_json(summary_path, summary)
    return summary


def write_locked_endpoint(
    rows: Iterable[dict[str, Any]],
    *,
    output_path: str | Path,
    contract_path: str | Path,
    source_hashes: dict[str, str],
    endpoint_definition: dict[str, Any],
) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise EndpointBuildError("EMPTY_ENDPOINT")
    offenders = sorted(
        {field for row in materialized for field in PRIVATE_ENDPOINT_FIELDS if field in row}
    )
    if offenders:
        raise EndpointBuildError(f"PRIVATE_ENDPOINT_FIELD_FORBIDDEN:{offenders}")
    patient_hashes = [str(row.get("patient_hash", "")) for row in materialized]
    if any(not value for value in patient_hashes):
        raise EndpointBuildError("PATIENT_HASH_REQUIRED")
    if len(set(patient_hashes)) != len(patient_hashes):
        raise EndpointBuildError("DUPLICATE_ENDPOINT_SUBJECT")
    for row in materialized:
        if int(row.get("event", -1)) not in {0, 1}:
            raise EndpointBuildError("BINARY_EVENT_REQUIRED")
        if float(row.get("time_days", 0.0)) <= 0.0:
            raise EndpointBuildError("POSITIVE_OBSERVED_TIME_REQUIRED")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(materialized).sort_values("patient_hash")
    storage_format = "parquet"
    parquet_status = "AVAILABLE"
    stored_target = target
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp.parquet", dir=str(target.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        try:
            frame.to_parquet(temporary, index=False)
        except ImportError:
            storage_format = "csv_fallback"
            parquet_status = "BLOCKED_PARQUET_ENGINE_UNAVAILABLE"
            stored_target = target.with_suffix(".csv")
            temporary.unlink(missing_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{stored_target.name}.", suffix=".tmp.csv", dir=str(target.parent)
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(temporary, stored_target)
    finally:
        temporary.unlink(missing_ok=True)
    endpoint_sha256 = sha256_file(stored_target)
    contract = {
        "schema_version": "1.0",
        "locked": True,
        "row_count": len(materialized),
        "event_count": sum(int(row["event"]) for row in materialized),
        "endpoint_sha256": endpoint_sha256,
        "endpoint_path": stored_target.name,
        "storage_format": storage_format,
        "parquet_status": parquet_status,
        "source_hashes": dict(sorted(source_hashes.items())),
        "endpoint_definition": endpoint_definition,
    }
    contract["contract_sha256"] = sha256_mapping(contract)
    atomic_write_json(contract_path, contract)
    return contract


def build_tau_endpoint_from_sources(
    *, config: dict[str, Any], output_root: str | Path
) -> dict[str, Any]:
    from scripts.build_method_y_tau_progression import (
        build_label_rows,
        load_first_tau_scans,
        read_csv_rows,
    )

    raw_root = Path(config["paths"]["raw_root"])
    sources = {
        "tau_suvr_csv": raw_root / config["clinical_sources"]["tau_suvr"],
        "cdr_csv": raw_root / config["clinical_sources"]["cdr"],
        "pacc_csv": raw_root / config["clinical_sources"]["pacc"],
        "subjinfo_csv": raw_root / config["clinical_sources"]["subject_info"],
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise EndpointBuildError(f"MISSING_ENDPOINT_SOURCES:{missing}")
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    endpoint_definition = {
        "primary_name": "sustained_cdr_global_progression",
        "event_time_rule": config["endpoint"]["event_time_rule"],
        "baseline_cdr_required": config["endpoint"]["baseline_cdr_required"],
        "pet_to_baseline_window_days": config["endpoint"][
            "pet_to_baseline_window_days"
        ],
        "followup_window_days": config["endpoint"]["followup_window_days"],
        "forbidden_label_sources": config["endpoint"]["forbidden_label_sources"],
        "treatment_structure": "active_vs_placebo_covariate",
    }
    output_directory = Path(output_root) / "endpoints"
    contract_path = endpoint_contract_path(output_root, "tau")
    summary_path = output_directory / "tau_endpoint_summary.json"
    if contract_path.exists():
        existing = read_json(contract_path)
        if existing.get("source_hashes") != source_hashes:
            raise EndpointBuildError("LOCKED_ENDPOINT_SOURCE_HASH_MISMATCH")
        if existing.get("endpoint_definition") != endpoint_definition:
            raise EndpointBuildError("LOCKED_ENDPOINT_DEFINITION_MISMATCH")
        stored = output_directory / str(existing.get("endpoint_path", ""))
        if not stored.is_file() or sha256_file(stored) != existing.get("endpoint_sha256"):
            raise EndpointBuildError("LOCKED_ENDPOINT_ARTIFACT_HASH_MISMATCH")
        summary = read_json(summary_path) if summary_path.exists() else {"contract": existing}
        summary["status"] = "ENDPOINT_LOCKED_REUSED"
        return summary

    compatibility_config = compatibility_endpoint_config(config)
    first_tau, tau_audit = load_first_tau_scans(
        sources["tau_suvr_csv"], compatibility_config
    )
    rows, endpoint_audit, region_features = build_label_rows(
        first_tau,
        read_csv_rows(sources["subjinfo_csv"]),
        read_csv_rows(sources["cdr_csv"]),
        read_csv_rows(sources["pacc_csv"]),
        compatibility_config,
    )
    for row in rows:
        row["treatment_active"] = treatment_active_value(row.get("treatment_arm"))
    output = output_directory / "tau_progression_endpoint.parquet"
    contract = write_locked_endpoint(
        rows,
        output_path=output,
        contract_path=contract_path,
        source_hashes=source_hashes,
        endpoint_definition=endpoint_definition,
    )
    summary = {
        "status": "ENDPOINT_LOCKED",
        "contract": contract,
        "tau_source_audit": tau_audit,
        "endpoint_audit": endpoint_audit,
        "regional_feature_count": len(region_features),
        "raw_identifiers_exported": False,
        "raw_paths_exported": False,
    }
    atomic_write_json(summary_path, summary)
    return summary
