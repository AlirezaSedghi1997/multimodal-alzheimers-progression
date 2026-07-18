"""Input inventory and semantic read bookkeeping for the synthesis workspace."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from core import atomic_write_text, compute_sha256, utc_now


TEXT_SUFFIXES = {
    ".md", ".markdown", ".txt", ".csv", ".json", ".yaml", ".yml", ".py",
    ".ps1", ".toml", ".rst", ".tsv", ".log",
}


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _semantic_normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lower()
    text = re.sub(r"`[^`]+`", "CODE", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _source_role(path: Path, agent_id: str) -> str:
    name = path.name.lower()
    if "final_codex" in name:
        return "supervisory_prompt"
    if agent_id == "prior_codex":
        return "prior_codex_output"
    if agent_id == "brainstorm_prompt":
        return "initial_brainstorm_prompt"
    if agent_id.startswith("agent_"):
        return "agent_package"
    return "supporting_input"


def inventory_file(path: Path, agent_id: str) -> dict[str, Any]:
    path = Path(path)
    raw = path.read_bytes()
    is_text = path.suffix.lower() in TEXT_SUFFIXES or b"\x00" not in raw[:4096]
    text = _read_text(path) if is_text else ""
    lines = text.splitlines()
    source_id = f"src_{compute_sha256(raw)[:16]}"
    return {
        "source_id": source_id,
        "agent_id": agent_id,
        "file_path": str(path),
        "file_role": _source_role(path, agent_id),
        "canonical_status": "CANDIDATE",
        "full_text_claimed": bool(text),
        "actual_evidence_level": "UNVERIFIED",
        "read_complete": False,
        "sha256": compute_sha256(raw),
        "line_count": len(lines),
        "byte_count": len(raw),
        "semantic_sha256": hashlib.sha256(_semantic_normalize(text).encode("utf-8")).hexdigest()
        if text
        else None,
        "notes": "",
    }


def iter_source_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
        if path.is_file() and path.name.lower() not in {"thumbs.db"}:
            yield path


def deduplicate_sources(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_hash: dict[str, dict[str, Any]] = {}
    canonical: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    priority = {
        "supervisory_prompt": 0,
        "initial_brainstorm_prompt": 1,
        "prior_codex_output": 2,
        "agent_package": 3,
        "supporting_input": 4,
    }
    for row in sorted(rows, key=lambda item: (priority.get(item["file_role"], 9), item["file_path"])):
        existing = by_hash.get(row["sha256"])
        if existing is None:
            row = dict(row)
            row["canonical_status"] = "CANONICAL"
            by_hash[row["sha256"]] = row
            canonical.append(row)
        else:
            duplicates.append(
                {
                    "source_id": row["source_id"],
                    "duplicate_sha256": row["sha256"],
                    "duplicate_path": row["file_path"],
                    "canonical_source_id": existing["source_id"],
                    "canonical_path": existing["file_path"],
                    "reason": "EXACT_FILE_CONTENT_DUPLICATE",
                }
            )
    return canonical, duplicates


def chunk_read_log(
    canonical_sources: list[dict[str, Any]],
    log_path: Path,
    chunk_size: int = 300,
) -> list[dict[str, Any]]:
    """Read every canonical text source and produce a deterministic chunk log.

    A chunk hash is used for coverage and duplicate detection. Duplicate chunks
    within Agent 4 are retained in the log but are explicitly marked as
    ``DUPLICATE_COVERAGE_ONLY`` and never become independent evidence rows.
    """
    records: list[dict[str, Any]] = []
    seen_chunks: dict[str, tuple[str, int, int]] = {}
    for source in canonical_sources:
        path = Path(source["file_path"])
        raw = path.read_bytes()
        if not source.get("full_text_claimed"):
            records.append(
                {
                    "source_id": source["source_id"],
                    "file_path": str(path),
                    "line_start": 0,
                    "line_end": 0,
                    "chunk_sha256": compute_sha256(raw),
                    "timestamp_utc": utc_now(),
                    "status": "METADATA_ONLY",
                }
            )
            continue
        text = _read_text(path)
        lines = text.splitlines()
        for start in range(0, len(lines), chunk_size):
            end = min(len(lines), start + chunk_size)
            chunk = "\n".join(lines[start:end])
            normalized = _semantic_normalize(chunk)
            chunk_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            duplicate_of = seen_chunks.get(chunk_hash)
            status = "READ_COMPLETE"
            note = ""
            if duplicate_of is not None and normalized:
                status = "DUPLICATE_COVERAGE_ONLY"
                note = f"{duplicate_of[0]}:{duplicate_of[1]}-{duplicate_of[2]}"
            else:
                seen_chunks[chunk_hash] = (source["source_id"], start + 1, end)
            records.append(
                {
                    "source_id": source["source_id"],
                    "file_path": str(path),
                    "line_start": start + 1,
                    "line_end": end,
                    "chunk_sha256": chunk_hash,
                    "timestamp_utc": utc_now(),
                    "status": status,
                    "duplicate_of": note,
                }
            )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_id", "file_path", "line_start", "line_end", "chunk_sha256", "timestamp_utc", "status", "duplicate_of"]
    output_lines: list[str] = []
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    atomic_write_text(log_path, buffer.getvalue())
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_source.setdefault(record["source_id"], []).append(record)
    for source in canonical_sources:
        source_records = by_source.get(source["source_id"], [])
        source["read_complete"] = bool(source_records) and all(
            record["status"] in {"READ_COMPLETE", "DUPLICATE_COVERAGE_ONLY", "METADATA_ONLY"}
            for record in source_records
        )
        source["notes"] = (
            f"chunks={len(source_records)};"
            f"duplicate_coverage_only={sum(r['status']=='DUPLICATE_COVERAGE_ONLY' for r in source_records)}"
        )
    return records
