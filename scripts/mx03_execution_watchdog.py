#!/usr/bin/env python3
"""Independent strict watchdog for MX-03 execution (ops-only, fail-closed).

This supervisor never trains, never edits science, never passes --no_resume,
never kills processes, and never rewrites manifests, metrics, checkpoints, or
task state. It observes (task state, progress-file freshness, GPU/disk) and
may relaunch ONLY the exact operator-supplied command when the recorded
outcome is a safe pause (exit 75 / PAUSED_RESUMABLE). Every other failure
class yields a BLOCKED receipt for human triage.

Modes:
  --mode once   read-only verdict on stdout, no writes (safe at any time).
  --mode watch  loop: relaunch on safe-pause (bounded), receipts under
                <output-root>/reports/watchdog/.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SAFE_PAUSE_SIGNALS = ("TASK_PAUSED_RESUMABLE", "PAUSED_RESUMABLE")
HARD_SIGNAL_PREFIXES = (
    "TASK_EXECUTION_BLOCKED",
    "RUN_CONTRACT_MISMATCH",
    "CONTRACT_",
    "TASK_ARTIFACT_INVALID",
    "SANITY_GATE_FAILED",
    "CUDA_OUT_OF_MEMORY",
    "OUT_OF_MEMORY",
)
EXIT_PAUSE = 75


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value


def classify_last_outcome(task_state: dict | None, report: dict | None) -> dict:
    """Pure failure classification. Never relaunches on unknown/hard classes."""
    tasks = (task_state or {}).get("tasks", {}) if isinstance(task_state, dict) else {}
    blockers = list((report or {}).get("blockers", []) or [])
    for task_id, entry in tasks.items():
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", ""))
        execution = entry.get("execution", {}) if isinstance(entry.get("execution"), dict) else {}
        code = execution.get("returncode", None)
        if status == "RUNNING":
            return {"class": "RUNNING", "task": task_id, "relaunch": False}
        if status == "PAUSED_RESUMABLE" or code == EXIT_PAUSE:
            return {"class": "SAFE_PAUSE", "task": task_id, "relaunch": True}
    joined = " ".join(str(b) for b in blockers)
    if any(sig in joined for sig in SAFE_PAUSE_SIGNALS) and not any(
        h in joined for h in HARD_SIGNAL_PREFIXES
    ):
        return {"class": "SAFE_PAUSE", "task": None, "relaunch": True}
    if any(h in joined for h in HARD_SIGNAL_PREFIXES):
        return {"class": "HARD_BLOCK", "task": None, "relaunch": False,
                "reason": next(h for h in HARD_SIGNAL_PREFIXES if h in joined)}
    if tasks and all(
        isinstance(e, dict) and e.get("status") == "COMPLETED" for e in tasks.values()
    ):
        return {"class": "ALL_COMPLETED", "task": None, "relaunch": False}
    return {"class": "UNKNOWN", "task": None, "relaunch": False}


def pid_alive(pid: int) -> bool | None:
    """Best-effort liveness; None when the platform gives no answer."""
    try:
        import psutil  # type: ignore

        return psutil.pid_exists(int(pid))
    except Exception:
        pass
    try:
        proc = Path(f"/proc/{int(pid)}")
        return proc.is_dir()
    except Exception:
        return None


def newest_progress_mtime(output_root: Path) -> float | None:
    newest: float | None = None
    runs = output_root / "mx03_runs"
    if not runs.is_dir():
        return None
    for path in runs.rglob("*_progress.pt"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        newest = mtime if newest is None or mtime > newest else newest
    return newest


def disk_free_gb(path: Path) -> float | None:
    try:
        import shutil

        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return None


def gpu_snapshot() -> dict:
    """Best-effort GPU view; absence of nvidia-smi is UNKNOWN, never PASS."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return {"status": "UNKNOWN_NO_SMI"}
    if out.returncode != 0:
        return {"status": "UNKNOWN_SMI_FAILED"}
    return {"status": "OBSERVED", "gpus": [line.strip() for line in out.stdout.splitlines() if line.strip()]}


def build_verdict(output_root: Path, idle_stall_seconds: int) -> dict:
    state = read_json(output_root / "private/mx03_task_state.json")
    report = read_json(output_root / "reports/mx03_validity_orchestrator.json")
    outcome = classify_last_outcome(state, report)
    newest = newest_progress_mtime(output_root)
    now = time.time()
    stalled = newest is not None and (now - newest) > idle_stall_seconds
    running_pids: list[int] = []
    if isinstance(state, dict):
        for entry in (state.get("tasks", {}) or {}).values():
            if isinstance(entry, dict) and entry.get("status") == "RUNNING" and entry.get("pid"):
                try:
                    running_pids.append(int(entry["pid"]))
                except (TypeError, ValueError):
                    continue
    liveness = {pid: pid_alive(pid) for pid in running_pids}
    return {
        "schema_version": 1,
        "checked_utc": utcnow(),
        "outcome": outcome,
        "tasks": {k: (v.get("status") if isinstance(v, dict) else None)
                  for k, v in ((state or {}).get("tasks", {}) or {}).items()},
        "report_status": (report or {}).get("status"),
        "report_blockers": list((report or {}).get("blockers", []) or []),
        "progress_newest_age_seconds": (now - newest) if newest else None,
        "stalled": stalled,
        "running_pids": running_pids,
        "pid_liveness": {str(k): v for k, v in liveness.items()},
        "disk_free_gb": disk_free_gb(output_root),
        "gpu": gpu_snapshot(),
        "human_approval_required": False,
        "manual_override_allowed": False,
    }


def command_is_safe(command: list[str]) -> tuple[bool, str]:
    if "--no_resume" in command:
        return False, "FORBIDDEN_FLAG_--no_resume"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["once", "watch"], default="once")
    parser.add_argument("--command", type=str, default="",
                        help="Exact orchestrator command to relaunch (watch mode, space-separated).")
    parser.add_argument("--max-relaunches", type=int, default=3)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--idle-stall-seconds", type=int, default=3600)
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root

    verdict = build_verdict(output_root, int(args.idle_stall_seconds))
    print(json.dumps(verdict, indent=2, ensure_ascii=False, sort_keys=True))
    if args.mode == "once":
        return 0

    command = args.command.split() if args.command else []
    safe, reason = command_is_safe(command)
    if not safe:
        print(json.dumps({"watchdog": "REFUSED", "reason": reason}))
        return 2
    log_dir = output_root / "reports" / "watchdog"
    log_dir.mkdir(parents=True, exist_ok=True)
    relaunches = 0
    while True:
        verdict = build_verdict(output_root, int(args.idle_stall_seconds))
        outcome = verdict["outcome"]
        receipt = {"schema_version": 1, "utc": utcnow(), "verdict": verdict,
                   "relaunches_used": relaunches}
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        digest = hashlib.sha256(json.dumps(verdict, sort_keys=True).encode()).hexdigest()[:12]
        if outcome["class"] == "ALL_COMPLETED":
            receipt["action"] = "NONE_TERMINAL"
            (log_dir / f"watchdog-{stamp}-{digest}.json").write_text(
                json.dumps(receipt, indent=2), encoding="utf-8")
            print(json.dumps({"watchdog": "TERMINAL", "action": "NONE"}))
            return 0
        if outcome["class"] == "RUNNING":
            receipt["action"] = "NONE_OBSERVED_RUNNING"
            (log_dir / f"watchdog-{stamp}-{digest}.json").write_text(
                json.dumps(receipt, indent=2), encoding="utf-8")
            time.sleep(int(args.interval_seconds))
            continue
        if outcome["relaunch"] and relaunches < int(args.max_relaunches):
            relaunches += 1
            receipt["action"] = f"RELAUNCH_IDENTICAL_COMMAND_{relaunches}"
            (log_dir / f"watchdog-{stamp}-{digest}.json").write_text(
                json.dumps(receipt, indent=2), encoding="utf-8")
            proc = subprocess.run(command, cwd=str(ROOT))
            receipt["child_returncode"] = proc.returncode
            (log_dir / f"watchdog-{stamp}-{digest}-child.json").write_text(
                json.dumps(receipt, indent=2), encoding="utf-8")
            continue
        receipt["action"] = "BLOCKED_FOR_HUMAN_TRIAGE"
        (log_dir / f"watchdog-{stamp}-{digest}.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps({"watchdog": "BLOCKED", "outcome": outcome["class"],
                          "reason": outcome.get("reason")}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
