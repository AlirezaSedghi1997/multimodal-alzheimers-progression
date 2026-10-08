"""Capture the live machine/writer boundary before fast-track preregistration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.run import active_gpu_compute_processes


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def package_verification(package_root: Path) -> dict:
    expected_path = package_root / "SHA256SUMS.txt"
    expected: dict[str, str] = {}
    for line in expected_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line.strip())
        if match:
            expected[match.group(2).replace("\\", "/")] = match.group(1).lower()
    rows = []
    for relative, digest in sorted(expected.items()):
        relative_path = Path(relative)
        safe = not relative_path.is_absolute() and ".." not in relative_path.parts
        path = package_root / relative_path
        actual = sha(path) if safe and path.is_file() else None
        rows.append({
            "relative_path": relative,
            "expected_sha256": digest,
            "actual_sha256": actual,
            "match": actual == digest,
        })
    return {
        "manifest_sha256": sha(expected_path),
        "file_count": len(rows),
        "all_match": bool(rows) and all(row["match"] for row in rows),
        "files": rows,
    }


def windows_python_processes(*, cwd: Path) -> list[dict]:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        raise RuntimeError("POWERSHELL_PROCESS_INVENTORY_UNAVAILABLE")
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
        "Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    result = command([executable, "-NoProfile", "-Command", script], cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError("WINDOWS_PROCESS_INVENTORY_FAILED")
    if not result.stdout.strip():
        return []
    payload = json.loads(result.stdout)
    rows = payload if isinstance(payload, list) else [payload]
    return [row for row in rows if isinstance(row, dict)]


def live_python_writers(processes: list[dict], *, excluded_pids: set[int]) -> list[dict]:
    rows = []
    writer_tokens = (
        "train_3d_mri",
        "run_method_x",
        "evaluate_locked.py",
        "evaluate_hybrid_pair.py",
        "xai_limited.py",
    )
    for info in processes:
        pid = int(info.get("ProcessId", -1))
        if pid in excluded_pids:
            continue
        command_line = str(info.get("CommandLine") or "")
        joined = command_line.lower()
        # The venv launcher and its CPython child both expose the command line.
        # Preflight/supervisor processes are coordinators, not model writers.
        if "capture_preflight.py" in joined or "supervise.py" in joined:
            continue
        if any(token in joined for token in writer_tokens):
            rows.append({
                "pid": pid,
                "parent_pid": int(info.get("ParentProcessId", -1)),
                "name": str(info.get("Name") or ""),
                "created": info.get("CreationDate"),
                "command": command_line,
            })
    return rows


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def original_writer_evidence(root: Path, staging_root: Path, processes: list[dict]) -> dict:
    battery = root / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729"
    blocker = battery / "reports/BLOCKED_MX04_FINETUNE.json"
    task_state = battery / "reports/mx04_task_state.json"
    heartbeats = sorted(
        battery.glob("finetune_v1/**/reports/autosupervisor_heartbeat.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    heartbeat = heartbeats[-1] if heartbeats else None
    resume_files = sorted(
        staging_root.glob("**/resume_current.pt"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    newest_resume = resume_files[-1] if resume_files else None
    checkpoint_metadata = None
    if newest_resume is not None:
        import torch

        payload = torch.load(newest_resume, map_location="cpu", weights_only=True)
        checkpoint_metadata = {
            key: payload.get(key) for key in (
                "fold",
                "epoch",
                "epoch_batch_cursor",
                "optimizer_step_count",
                "last_committed_optimizer_step",
                "checkpoint_generation",
                "updated_utc",
                "fold_terminal",
                "run_contract_digest",
            )
        }
        checkpoint_metadata.update({
            "relative_staging_path": newest_resume.relative_to(staging_root).as_posix(),
            "sha256": sha(newest_resume),
            "size_bytes": newest_resume.stat().st_size,
        })
    heartbeat_body = read_json(heartbeat) if heartbeat is not None else {}
    heartbeat_pid = heartbeat_body.get("pid")
    heartbeat_pid_live = bool(
        isinstance(heartbeat_pid, int)
        and any(
            int(row.get("ProcessId", -1)) == heartbeat_pid
            and any(token in str(row.get("CommandLine") or "").lower()
                    for token in ("train_3d_mri", "run_method_x"))
            for row in processes
        )
    )
    blocker_body = read_json(blocker) if blocker.is_file() else {}
    return {
        "blocker_report": blocker.relative_to(root).as_posix() if blocker.is_file() else None,
        "blocker_report_sha256": sha(blocker) if blocker.is_file() else None,
        "blockers": blocker_body.get("blockers"),
        "status": blocker_body.get("status"),
        "task_state_sha256": sha(task_state) if task_state.is_file() else None,
        "latest_heartbeat": heartbeat.relative_to(root).as_posix() if heartbeat is not None else None,
        "latest_heartbeat_sha256": sha(heartbeat) if heartbeat is not None else None,
        "heartbeat_status": heartbeat_body.get("status"),
        "heartbeat_pid": heartbeat_pid,
        "heartbeat_pid_live": heartbeat_pid_live,
        "newest_resume_checkpoint": checkpoint_metadata,
        "safe_stop_interpretation": "FAIL_CLOSED_STOPPED_NO_LIVE_PROCESS_CHECKPOINT_PRESERVED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-coordinator-pid", type=int)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if root != ROOT.resolve():
        raise RuntimeError("PREFLIGHT_PROJECT_ROOT_MISMATCH")
    package_root = args.package_root.resolve()
    package = package_verification(package_root)
    git_head = command(["git", "rev-parse", "HEAD"], cwd=root)
    git_branch = command(["git", "branch", "--show-current"], cwd=root)
    git_status = command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
    )
    gpu = command([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], cwd=root)
    active_gpu = active_gpu_compute_processes(gpu.stdout) if gpu.returncode == 0 else []
    processes = windows_python_processes(cwd=root)
    excluded_pids = {os.getpid()}
    if args.allow_coordinator_pid is not None:
        excluded_pids.add(int(args.allow_coordinator_pid))
    writers = live_python_writers(processes, excluded_pids=excluded_pids)
    drives = {}
    for name in ("C:\\", "H:\\"):
        usage = shutil.disk_usage(name)
        drives[name[0]] = {
            "total_gib": round(usage.total / 1024**3, 3),
            "used_gib": round(usage.used / 1024**3, 3),
            "free_gib": round(usage.free / 1024**3, 3),
        }
    staging = Path(r"C:\Users\USER\AppData\Local\AMUSE-X\checkpoint_staging")
    staging_files = sorted(path for path in staging.glob("**/*") if path.is_file())
    writer = original_writer_evidence(root, staging, processes)
    blockers = []
    if not package["all_match"]:
        blockers.append("SOURCE_PACKAGE_SHA256_MISMATCH")
    if git_head.returncode or git_branch.returncode or git_status.returncode:
        blockers.append("GIT_STATE_UNREADABLE")
    if gpu.returncode or active_gpu:
        blockers.append("GPU_NOT_IDLE")
    if writers:
        blockers.append("LIVE_METHOD_X_PYTHON_WRITER")
    if drives["C"]["free_gib"] < 30.0:
        blockers.append("C_MINIMUM_FREE_SPACE_NOT_MET")
    if writer.get("heartbeat_pid_live"):
        blockers.append("ORIGINAL_WRITER_HEARTBEAT_PID_STILL_LIVE")
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY" if not blockers else "BLOCKED",
        "blockers": sorted(blockers),
        "git": {
            "head": git_head.stdout.strip(),
            "branch": git_branch.stdout.strip(),
            "dirty": bool(git_status.stdout.strip()),
            "porcelain_entry_count": len(git_status.stdout.splitlines()),
            "porcelain_sha256": hashlib.sha256(git_status.stdout.encode("utf-8")).hexdigest(),
        },
        "drives": drives,
        "gpu": {
            "query_returncode": gpu.returncode,
            "active_conflicting_processes": active_gpu,
            "ignored_codex_ui_signature_only": not active_gpu and bool(gpu.stdout.strip()),
        },
        "live_method_x_python_writers": writers,
        "original_mx04_writer": writer,
        "checkpoint_staging": {
            "root": str(staging),
            "file_count": len(staging_files),
            "total_bytes": sum(path.stat().st_size for path in staging_files),
            "files": [{
                "relative_path": path.relative_to(staging).as_posix(),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            } for path in staging_files],
            "preserved_without_deletion": True,
        },
        "source_package": package,
        "claim": "MACHINE_PREFLIGHT_ONLY_NOT_SCIENTIFIC_COMPLETION",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".new")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, args.out)
    print(json.dumps({
        "status": report["status"],
        "blockers": report["blockers"],
        "output": str(args.out),
    }, ensure_ascii=False))
    return 0 if report["status"] == "READY" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
