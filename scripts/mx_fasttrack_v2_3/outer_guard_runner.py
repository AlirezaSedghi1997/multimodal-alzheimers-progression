"""Run byte-locked v2 outer evaluators with a self-aware GPU occupancy guard.

Only the evaluator's own CUDA context is exempted.  All model, data, split,
checkpoint, receipt and statistical code remains in the hash-bound v2 modules.
The v2.3 amendment binds this wrapper before the first outer-test access.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.run import active_gpu_compute_processes


def gpu_idle_except_current_process() -> None:
    """Fail closed on any compute process except this evaluator's own PID."""
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    active = active_gpu_compute_processes(query.stdout) if query.returncode == 0 else []
    foreign = [row for row in active if row["pid"] != str(os.getpid())]
    if query.returncode != 0 or foreign:
        raise RuntimeError(
            "GPU_BUSY_OR_NVIDIA_SMI_UNAVAILABLE_STOP_ORIGINAL_WRITER_SAFELY:"
            f"returncode={query.returncode}:active="
            f"{json.dumps(foreign, ensure_ascii=False, sort_keys=True)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluator", choices=("hybrid-pair", "locked"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("outer",))
    parser.add_argument("--task", choices=("selected", "mx04_scratch", "mx04_head_last"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    if root != ROOT.resolve() or not out.is_relative_to(root) or out == root:
        raise RuntimeError("V23_PROJECT_OR_OUTPUT_ROOT_INVALID")
    if args.evaluator == "hybrid-pair" and (args.scope is not None or args.task is not None):
        raise RuntimeError("V23_PAIR_REJECTS_LOCKED_OPTIONS")
    if args.evaluator == "locked" and (args.scope != "outer" or args.task is None):
        raise RuntimeError("V23_LOCKED_OUTER_SCOPE_AND_TASK_REQUIRED")

    from scripts.mx_fasttrack_v2_3.supervise_v23 import ensure_outer_guard_amendment

    ensure_outer_guard_amendment(root, out)
    from scripts.mx_fasttrack_v2 import run as locked_run

    locked_run.gpu_idle = gpu_idle_except_current_process
    if args.evaluator == "hybrid-pair":
        from scripts.mx_fasttrack_v2 import evaluate_hybrid_pair as evaluator

        arguments = ["evaluate_hybrid_pair.py", "--project-root", str(root),
                     "--output-root", str(args.output_root)]
    else:
        from scripts.mx_fasttrack_v2 import evaluate_locked as evaluator

        arguments = ["evaluate_locked.py", "--project-root", str(root),
                     "--output-root", str(args.output_root), "--scope", "outer",
                     "--task", args.task]
    previous = sys.argv
    try:
        sys.argv = arguments
        return evaluator.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
