from __future__ import annotations

import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json


class ActiveWriterError(RuntimeError):
    """Raised when another live process owns the versioned output root."""


def _process_start_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        if os.name != "nt":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not process:
                return None
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            try:
                ok = ctypes.windll.kernel32.GetProcessTimes(
                    process,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                )
                if not ok:
                    return None
                ticks = (creation.dwHighDateTime << 32) + creation.dwLowDateTime
                return (ticks / 10_000_000.0) - 11_644_473_600.0
            finally:
                ctypes.windll.kernel32.CloseHandle(process)
        except Exception:
            return None


def pid_is_active(pid: int, expected_start_time: float | None = None) -> bool:
    try:
        import psutil

        numeric_pid = int(pid)
        if numeric_pid <= 0 or not psutil.pid_exists(numeric_pid):
            return False
        process = psutil.Process(numeric_pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        if expected_start_time is not None:
            return abs(float(process.create_time()) - float(expected_start_time)) < 1.0
        return True
    except ImportError:
        try:
            numeric_pid = int(pid)
        except (TypeError, ValueError):
            return False
        if numeric_pid <= 0:
            return False
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                process = ctypes.windll.kernel32.OpenProcess(0x1000, False, numeric_pid)
                if not process:
                    return False
                exit_code = wintypes.DWORD()
                try:
                    if not ctypes.windll.kernel32.GetExitCodeProcess(
                        process, ctypes.byref(exit_code)
                    ):
                        return False
                    if int(exit_code.value) != 259:  # STILL_ACTIVE
                        return False
                finally:
                    ctypes.windll.kernel32.CloseHandle(process)
                if expected_start_time is None:
                    return True
                actual = _process_start_time(numeric_pid)
                return actual is not None and abs(actual - float(expected_start_time)) < 1.0
            except Exception:
                return False
        try:
            os.kill(numeric_pid, 0)
            return True
        except OSError:
            return False
    except ValueError:
        return False
    except Exception:
        return False


class WriterLock:
    def __init__(self, path: str | Path, *, output_root: str | Path, run_contract_digest: str = ""):
        self.path = Path(path)
        self.output_root = Path(output_root).resolve()
        self.run_contract_digest = str(run_contract_digest)
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                existing: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ActiveWriterError(f"UNREADABLE_WRITER_LOCK:{self.path}") from exc
            if pid_is_active(existing.get("pid", -1), existing.get("process_start_time")):
                raise ActiveWriterError(
                    f"ACTIVE_WRITER:pid={existing.get('pid')}:output_root={existing.get('output_root')}"
                )
            self.path.unlink()
        payload = {
            "pid": os.getpid(),
            "process_start_time": _process_start_time(os.getpid()),
            "hostname": socket.gethostname(),
            "command_line": " ".join(sys.argv),
            "output_root": str(self.output_root),
            "run_contract_sha256": self.run_contract_digest,
            "acquired_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self.path, payload)
        self.acquired = True

    def release(self) -> None:
        if self.acquired and self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if payload.get("pid") == os.getpid():
                self.path.unlink()
        self.acquired = False

    def __enter__(self) -> "WriterLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
