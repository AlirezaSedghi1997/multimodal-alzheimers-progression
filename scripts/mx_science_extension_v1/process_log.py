"""Stream child output to the console and an append-only per-stage log."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path


@contextmanager
def console_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(json.dumps({"started_at_utc": datetime.now(timezone.utc).isoformat(),
                              "argv": sys.argv}, ensure_ascii=False) + "\n")
        class Tee:
            def __init__(self, stream):
                self.stream = stream
            def write(self, text):
                log.write(text)
                return self.stream.write(text)
            def flush(self):
                log.flush()
                self.stream.flush()
        try:
            with redirect_stdout(Tee(sys.stdout)), redirect_stderr(Tee(sys.stderr)):
                yield
        finally:
            log.flush()
            os.fsync(log.fileno())


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: deque[str] = deque(maxlen=64)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(json.dumps({"started_at_utc": datetime.now(timezone.utc).isoformat(),
                              "command": command}, ensure_ascii=False) + "\n")
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace")
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                tail.append(line)
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                # Kill only this launched tree; a Windows venv has a launcher hop.
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   check=False, capture_output=True)
                else:
                    process.terminate()
                process.wait(timeout=30)
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            log.flush()
            os.fsync(log.fileno())
        log.write(json.dumps({"returncode": code}) + "\n")
    return subprocess.CompletedProcess(command, code, stdout="".join(tail))
