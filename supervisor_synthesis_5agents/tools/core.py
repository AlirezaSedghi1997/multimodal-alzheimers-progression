"""Small, dependency-free primitives for the five-agent synthesis workspace."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGES = [
    "INIT",
    "SAFETY_SNAPSHOTTED",
    "INVENTORY_LOCKED",
    "UNIQUE_CONTENT_READ",
    "REGISTRY_LOCKED",
    "PROJECT_AUDITED",
    "DECISIVE_EVIDENCE_VERIFIED",
    "CANDIDATES_SCORED",
    "RED_TEAM_DONE",
    "OUTPUTS_RENDERED",
    "VERIFIED",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_sha256(value: bytes | str | Path) -> str:
    if isinstance(value, Path):
        digest = hashlib.sha256()
        with value.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_init_state(path: Path) -> dict[str, Any]:
    state = read_json(path)
    if state is None:
        state = {
            "schema_version": "1.0",
            "state": "INIT",
            "completed_stages": [],
            "stage_records": {},
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
        }
        atomic_write_json(path, state)
    return state


def transition_state(state: dict[str, Any], next_state: str, record: dict[str, Any] | None = None) -> dict[str, Any]:
    current = state.get("state", "INIT")
    if next_state not in STAGES:
        raise ValueError(f"Unknown synthesis state: {next_state}")
    if STAGES.index(next_state) < STAGES.index(current):
        raise ValueError(f"Non-monotonic state transition {current} -> {next_state}")
    state = dict(state)
    state["state"] = next_state
    state["updated_utc"] = utc_now()
    completed = list(state.get("completed_stages", []))
    if next_state not in completed:
        completed.append(next_state)
    state["completed_stages"] = completed
    if record is not None:
        records = dict(state.get("stage_records", {}))
        records[next_state] = record
        state["stage_records"] = records
    return state


def git_snapshot(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            return result.stdout.strip()
        except OSError as exc:
            return f"ERROR:{exc}"

    status = run("status", "--porcelain=v1")
    diff_names = run("diff", "--name-only")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "git_dir": run("rev-parse", "--git-dir"),
        "git_common_dir": run("rev-parse", "--git-common-dir"),
        "status_porcelain": status,
        "dirty_tree_digest": compute_sha256(status + "\n" + diff_names),
        "dirty_file_count": len(status.splitlines()) if status else 0,
        "captured_utc": utc_now(),
    }


def process_snapshot() -> dict[str, Any]:
    """Return a read-only process snapshot without killing or pausing anything."""
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'python|powershell|matlab|spm' } | "
        "Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | "
        "ConvertTo-Json -Depth 4"
    )
    rows: list[dict[str, Any]] = []
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.stdout.strip():
            parsed = json.loads(result.stdout)
            rows = parsed if isinstance(parsed, list) else [parsed]
    except (OSError, json.JSONDecodeError):
        rows = []
    return {"platform": platform.platform(), "captured_utc": utc_now(), "processes": rows}


def relevant_processes(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    terms = ("method_y", "method x", "transform", "ftp", "spm", "supervisor_synthesis")
    output = []
    for row in snapshot.get("processes", []):
        command = str(row.get("CommandLine") or "").lower()
        if any(term in command for term in terms):
            output.append(row)
    return output


def path_fingerprint(path: Path, max_files: int = 20000, exclude_relative: tuple[str, ...] = ()) -> dict[str, Any]:
    """Fingerprint a tree without copying or mutating it.

    The digest is based on relative names, sizes and mtimes. This is intentional:
    protected project trees can contain large NIfTI/HDF5 data and must not be
    re-read merely to prove that this synthesis workspace did not mutate them.
    """
    if not path.exists():
        return {"exists": False, "path": str(path), "file_count": 0, "digest": compute_sha256("MISSING")}
    if path.is_file():
        stat = path.stat()
        return {
            "exists": True,
            "path": str(path),
            "file_count": 1,
            "digest": compute_sha256(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}"),
        }
    entries: list[str] = []
    considered = 0
    for child in sorted(path.rglob("*"), key=lambda item: str(item).lower()):
        try:
            relative = child.relative_to(path)
        except ValueError:
            relative = child.name
        if any(str(relative).replace("\\", "/").startswith(prefix.replace("\\", "/")) for prefix in exclude_relative):
            continue
        if considered >= max_files:
            entries.append(f"TRUNCATED_AFTER={max_files}")
            break
        considered += 1
        try:
            stat = child.stat()
        except OSError:
            entries.append(f"{relative}|UNREADABLE")
            continue
        if child.is_file():
            entries.append(f"{relative}|{stat.st_size}|{stat.st_mtime_ns}")
    return {
        "exists": True,
        "path": str(path),
        "file_count": len([entry for entry in entries if not entry.startswith("TRUNCATED")]),
        "digest": compute_sha256("\n".join(entries)),
        "truncated": any(entry.startswith("TRUNCATED") for entry in entries),
    }


def disk_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for drive in ("C:\\", "H:\\"):
        try:
            usage = __import__("shutil").disk_usage(drive)
            result[drive] = {
                "total_bytes": usage.total,
                "used_bytes": usage.total - usage.free,
                "free_bytes": usage.free,
            }
        except OSError as exc:
            result[drive] = {"error": str(exc)}
    return result
