"""Authenticated in-process adapter for only the v2.2 XAI target."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.capture_preflight import windows_python_processes
from scripts.mx_fasttrack_v2_1.supervise_v21 import INTERNAL_PARENT_ENV
from scripts.mx_fasttrack_v2_4_2 import entry_v242
from scripts.mx_fasttrack_v2_4_3.supervise_v243 import OUTPUT, ensure_v243_amendment


def require_v243_coordinator(root: Path) -> None:
    claimed = os.environ.get(INTERNAL_PARENT_ENV)
    if claimed is None or not claimed.isdecimal():
        raise RuntimeError("V243_XAI_ACTIVE_COORDINATOR_REQUIRED")
    coordinator_pid = int(claimed)
    rows = {int(row.get("ProcessId", -1)): row
            for row in windows_python_processes(cwd=root)}
    coordinator = rows.get(coordinator_pid) or {}
    command = str(coordinator.get("CommandLine") or "").lower().replace("/", "\\")
    if "\\scripts\\mx_fasttrack_v2_4_3\\supervise_v243.py" not in command:
        raise RuntimeError("V243_XAI_ACTIVE_COORDINATOR_REQUIRED")
    parent_pid = os.getppid()
    if parent_pid == coordinator_pid:
        return
    launcher = rows.get(parent_pid) or {}
    launcher_command = str(launcher.get("CommandLine") or "").lower().replace("/", "\\")
    if (int(launcher.get("ParentProcessId", -1)) != coordinator_pid
        or "\\scripts\\mx_fasttrack_v2_4_3\\entry_v243.py" not in launcher_command):
        raise RuntimeError("V243_XAI_ACTIVE_COORDINATOR_REQUIRED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("xai-run",))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    if (root != ROOT.resolve() or args.output_root != OUTPUT
        or not out.is_relative_to(root) or out == root):
        raise RuntimeError("V243_PROJECT_OR_OUTPUT_ROOT_INVALID")
    ensure_v243_amendment(root, out, create=False)
    require_v243_coordinator(root)
    from scripts.mx_fasttrack_v2_2 import xai_runner

    # The original guard accepts only its own historical coordinator path.
    # Recheck the same OS ancestry at the actual XAI call site, without
    # relaxing any checkpoint, atlas, split, CUDA, or XAI receipt gate.
    xai_runner.require_internal_coordinator_parent = require_v243_coordinator
    prior = sys.argv
    try:
        sys.argv = [str(root / "scripts/mx_fasttrack_v2_4_2/entry_v242.py"),
                    "xai-run", "--project-root", str(root),
                    "--output-root", str(args.output_root)]
        return entry_v242.main()
    finally:
        sys.argv = prior


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(3)
