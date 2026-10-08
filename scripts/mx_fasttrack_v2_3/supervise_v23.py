"""Resume v2.2 Method X with a versioned pre-outer GPU scheduling amendment.

The v2/v2.1/v2.2 scientific sources and MX-04 import are immutable.  Only the
v2 outer evaluator launch path differs: its occupancy guard ignores its own
CUDA context, while still refusing another compute process.  No selection,
checkpoint, outcome, metric or statistical rule is changed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.integrity import check_inputs
from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2.supervise import DEFAULT_OUTPUT, DEFAULT_PACKAGE
from scripts.mx_fasttrack_v2_1.supervise_v21 import (
    INTERNAL_PARENT_ENV, coordinator_lock, foreign_method_x_writers, immutable_json,
)
from scripts.mx_fasttrack_v2_2.supervise_v22 import (
    PROTOCOL, Supervisor as V22Supervisor, ensure_protocol,
)

AMENDMENT = "fasttrack_v23_outer_guard_amendment.json"
SOURCE_FILES = (
    "scripts/mx_fasttrack_v2_3/__init__.py",
    "scripts/mx_fasttrack_v2_3/outer_guard_runner.py",
    "scripts/mx_fasttrack_v2_3/supervise_v23.py",
    "docs/method_x_fasttrack_v23_outer_guard_FA.md",
)


def outer_evaluator_command(root: Path, output_relative: Path, evaluator: str,
                            *, scope: str = "outer", task: str | None = None) -> list[str]:
    if scope != "outer":
        raise ValueError("V23_OUTER_ONLY")
    if evaluator not in {"hybrid-pair", "locked"}:
        raise ValueError("V23_OUTER_EVALUATOR_UNKNOWN")
    if (evaluator == "hybrid-pair" and task is not None) or (evaluator == "locked" and task not in {
        "selected", "mx04_scratch", "mx04_head_last"
    }):
        raise ValueError("V23_OUTER_TASK_INVALID")
    command = [sys.executable, str(root / "scripts/mx_fasttrack_v2_3/outer_guard_runner.py"),
               evaluator, "--project-root", str(root), "--output-root", str(output_relative)]
    if evaluator == "locked":
        command.extend(["--scope", "outer", "--task", str(task)])
    return command


def outer_guard_amendment_payload(root: Path, out: Path) -> dict:
    prereg = read(out / "reports/fasttrack_preregistration.json")
    imported = out / "reports/mx04_import_manifest.v1.json"
    if prereg.get("protocol_id") != "method_x_fasttrack_v2_20260926_mx04_import":
        raise RuntimeError("V23_REQUIRES_MX04_IMPORT_PROTOCOL")
    if not imported.is_file():
        raise RuntimeError("V23_MX04_IMPORT_MANIFEST_MISSING")
    manifest = read(imported)
    if (manifest.get("manifest_id") != "method_x_fasttrack_v2_20260926_mx04_import_manifest.v1"
        or manifest.get("status") != "MX04_IMPORTED_UNDER_PROOF"):
        raise RuntimeError("V23_MX04_IMPORT_MANIFEST_INVALID")
    return {
        "schema_version": 1,
        "protocol_id": "METHOD_X_FASTTRACK_V23_PRE_OUTER_SELF_GPU_GUARD",
        "parent_preregistration_sha256": sha(out / "reports/fasttrack_preregistration.json"),
        "mx04_import_manifest_sha256": sha(imported),
        "v21_amendment_sha256": sha(out / "reports/fasttrack_v21_amendment.json"),
        "v22_protocol_sha256": sha(out / "reports" / PROTOCOL),
        "source_sha256": {relative: sha(root / relative) for relative in SOURCE_FILES},
        "scope": "OPERATIONAL_GPU_OCCUPANCY_GUARD_ONLY_BEFORE_OUTER_ACCESS",
        "self_pid_exemption_only": True,
        "foreign_gpu_compute_processes_still_block": True,
        "model_data_selection_metric_or_statistical_change": False,
        "historical_outer_tests_independent": False,
        "mx04_training_rerun_required": False,
    }


def _require_no_outer_access(out: Path) -> None:
    reports = out / "reports"
    if any(reports.glob("evaluation_outer_*")):
        raise RuntimeError("V23_TOO_LATE_OUTER_ARTIFACT_EXISTS")
    private = out / "_private_resume"
    if private.exists() and any(private.glob("evaluation_outer_*")):
        raise RuntimeError("V23_TOO_LATE_OUTER_RECEIPT_EXISTS")
    state_path = reports / "fasttrack_supervisor_state.json"
    if state_path.exists():
        steps = read(state_path).get("steps") or {}
        if any(str(name).startswith(("mx4_outer_", "mx5_outer_"))
               or str(name) == "mx5_selected_outer_receipt" for name in steps):
            raise RuntimeError("V23_TOO_LATE_OUTER_STEP_EXISTS")


def ensure_outer_guard_amendment(root: Path, out: Path) -> dict:
    protocol = out / "reports" / PROTOCOL
    if not protocol.is_file():
        raise RuntimeError("V23_REQUIRES_EXISTING_V22_PROTOCOL")
    ensure_protocol(root, out)
    path = out / "reports" / AMENDMENT
    if not path.exists():
        _require_no_outer_access(out)
    expected = outer_guard_amendment_payload(root, out)
    if not path.exists():
        immutable_json(path, expected)
    elif read(path) != expected:
        raise RuntimeError("V23_AMENDMENT_OR_SOURCE_DRIFT")
    return expected


class Supervisor(V22Supervisor):
    def mx5(self) -> None:
        ensure_outer_guard_amendment(self.root, self.out)
        self.mx4()
        self.run_step("mx5_hybrid_cuda_smoke", self.py(
            "scripts/mx_fasttrack_v2/run.py", "smoke",
            "--project-root", self.root, "--output-root", self.output_relative))
        self.run_step("mx5_hybrid_training_profile", self.py(
            "scripts/mx_fasttrack_v2/run.py", "profile",
            "--project-root", self.root, "--output-root", self.output_relative))
        self.run_step("mx5_two_arm_inner_training", self.py(
            "scripts/mx_fasttrack_v2/run.py", "mx5",
            "--project-root", self.root, "--output-root", self.output_relative))
        self.run_step("mx5_common_inner_lock", self.py(
            "scripts/mx_fasttrack_v2/run.py", "lock",
            "--project-root", self.root, "--output-root", self.output_relative))
        self.run_step("mx5_outer_hybrid_pair", outer_evaluator_command(
            self.root, self.output_relative, "hybrid-pair"))
        for task in ("mx04_scratch", "mx04_head_last"):
            self.run_step(f"mx4_outer_{task}", outer_evaluator_command(
                self.root, self.output_relative, "locked", task=task))
        self.run_step("mx5_selected_outer_receipt", outer_evaluator_command(
            self.root, self.output_relative, "locked", task="selected"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("prepare", "mx4", "mx5", "mx6", "mx7", "all"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    if root != ROOT.resolve() or not out.is_relative_to(root) or out == root:
        raise RuntimeError("V23_PROJECT_OR_OUTPUT_ROOT_INVALID")
    writers = foreign_method_x_writers(root)
    if writers:
        raise RuntimeError(f"V23_FOREIGN_METHOD_X_WRITER_ACTIVE:{[row['pid'] for row in writers]}")
    with coordinator_lock(out / "reports"):
        prior = os.environ.get(INTERNAL_PARENT_ENV)
        os.environ[INTERNAL_PARENT_ENV] = str(os.getpid())
        try:
            ensure_outer_guard_amendment(root, out)
            if args.target == "prepare":
                check_inputs(root, out, hash_cache=True)
                print(json.dumps({"status": "V23_PRE_OUTER_GUARD_PREPARED",
                                  "output_root": str(out)}, ensure_ascii=False))
            else:
                supervisor = Supervisor(root, args.output_root, args.package_root)
                target = "mx7" if args.target == "all" else args.target
                getattr(supervisor, target)()
                print(json.dumps({"status": f"{target.upper()}_V23_SUPERVISOR_TARGET_COMPLETED",
                                  "output_root": str(out)}, ensure_ascii=False))
        finally:
            if prior is None:
                os.environ.pop(INTERNAL_PARENT_ENV, None)
            else:
                os.environ[INTERNAL_PARENT_ENV] = prior
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
