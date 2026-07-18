"""SQLite registry and deterministic CSV exports."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from core import atomic_write_text, utc_now


TABLES: dict[str, list[str]] = {
    "sources": [
        "source_id", "agent_id", "file_path", "file_role", "canonical_status",
        "full_text_claimed", "actual_evidence_level", "read_complete", "sha256", "notes",
    ],
    "claims": [
        "claim_id", "topic", "claim_text", "claim_type", "source_id", "source_section_or_lines",
        "evidence_label", "primary_reference", "verified_status", "confidence", "contradiction_group",
        "decision_impact", "required_action",
    ],
    "papers": [
        "paper_id", "normalized_title", "authors", "DOI", "arXiv", "year", "online_date", "print_date",
        "status", "full_text_status", "cohort", "N_subjects", "N_scans", "endpoint", "modalities",
        "split", "external_validation", "calibration", "code", "leakage_risk", "nearest_to_candidate", "notes",
    ],
    "datasets": [
        "dataset_id", "access_status", "current_local_availability", "DUA_status", "download_time_estimate",
        "modalities", "functional_variables", "endpoint", "subject_count_total", "usable_subject_count",
        "event_count", "censor_count", "pairing_window", "tracer", "site_scanner", "preprocessing_ready",
        "external_candidate", "blockers", "evidence_source",
    ],
    "ideas": [
        "idea_id", "normalized_family", "source_agents", "exact_hypothesis", "endpoint", "core_innovation",
        "optional_extension", "required_data", "reuse_existing_components", "new_code_cost", "compute_cost",
        "implementation_weeks", "closest_precedents", "novelty_status", "leakage_risk", "clinical_value",
        "publication_story", "hard_gate_status", "score", "confidence", "verdict",
    ],
    "contradictions": [
        "contradiction_id", "topic", "source_A", "claim_A", "source_B", "claim_B", "resolution",
        "evidence_used", "unresolved_risk",
    ],
}


def initialize_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    for table, columns in TABLES.items():
        defs = ", ".join(f'"{column}" TEXT' for column in columns)
        connection.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({defs})')
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_utc TEXT NOT NULL)"
    )
    connection.commit()
    return connection


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def replace_rows(connection: sqlite3.Connection, table: str, rows: Iterable[dict[str, Any]]) -> None:
    if table not in TABLES:
        raise ValueError(table)
    columns = TABLES[table]
    connection.execute(f'DELETE FROM "{table}"')
    placeholders = ", ".join("?" for _ in columns)
    quoted = ", ".join(f'"{column}"' for column in columns)
    values = [[_stringify(row.get(column)) for column in columns] for row in rows]
    if values:
        connection.executemany(f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})', values)


def set_metadata(connection: sqlite3.Connection, values: dict[str, Any]) -> None:
    for key, value in values.items():
        connection.execute(
            "INSERT INTO metadata(key,value,updated_utc) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_utc=excluded.updated_utc",
            (key, _stringify(value), utc_now()),
        )


def commit_registry(
    path: Path,
    rows_by_table: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any] | None = None,
) -> None:
    connection = initialize_db(path)
    try:
        with connection:
            for table, rows in rows_by_table.items():
                replace_rows(connection, table, rows)
            if metadata:
                set_metadata(connection, metadata)
    finally:
        connection.close()


def read_table(path: Path, table: str) -> list[dict[str, Any]]:
    connection = initialize_db(path)
    try:
        cursor = connection.execute(f'SELECT * FROM "{table}"')
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        connection.close()


def export_csv_views(path: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "claims": "claims_master.csv",
        "papers": "papers_master_deduplicated.csv",
        "datasets": "datasets_master.csv",
        "ideas": "ideas_master.csv",
        "contradictions": "contradiction_ledger.csv",
    }
    output: dict[str, str] = {}
    for table, filename in mapping.items():
        rows = read_table(path, table)
        fields = TABLES[table]
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        target = output_dir / filename
        atomic_write_text(target, buffer.getvalue())
        output[filename] = str(target)
    queue = []
    for claim in read_table(path, "claims"):
        if claim.get("verified_status") != "VERIFIED":
            queue.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "source_id": claim.get("source_id"),
                    "source_section_or_lines": claim.get("source_section_or_lines"),
                    "evidence_label": claim.get("evidence_label"),
                    "required_action": claim.get("required_action"),
                }
            )
    target = output_dir / "verification_queue.csv"
    fields = ["claim_id", "source_id", "source_section_or_lines", "evidence_label", "required_action"]
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(queue)
    atomic_write_text(target, buffer.getvalue())
    output["verification_queue.csv"] = str(target)
    return output
