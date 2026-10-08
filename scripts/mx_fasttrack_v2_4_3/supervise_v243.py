"""Resume v2.4.2 after its XAI coordinator false-negative, without rerunning outer inference."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2.supervise import DEFAULT_PACKAGE
from scripts.mx_fasttrack_v2_1.supervise_v21 import (
    INTERNAL_PARENT_ENV, coordinator_lock, foreign_method_x_writers, immutable_json,
)
from scripts.mx_fasttrack_v2_4.supervise_v24 import Supervisor as V24Supervisor
from scripts.mx_fasttrack_v2_4_1.supervise_v241 import _patch_parent_checkers
from scripts.mx_fasttrack_v2_4_2.supervise_v242 import (
    AMENDMENT as V242_AMENDMENT, OUTPUT, Supervisor as V242Supervisor,
    ensure_v242_amendment,
)

AMENDMENT = "fasttrack_v243_postouter_xai_coordinator_repair.json"
V242_STATE = "fasttrack_v242_supervisor_state.json"
V243_STATE = "fasttrack_v243_supervisor_state.json"
SOURCE_FILES = (
    "scripts/mx_fasttrack_v2_4_3/__init__.py",
    "scripts/mx_fasttrack_v2_4_3/entry_v243.py",
    "scripts/mx_fasttrack_v2_4_3/supervise_v243.py",
)
REQUIRED_COMPLETED = (
    "mx4_two_arm_inner_training", "mx5_two_arm_inner_training",
    "mx5_common_inner_lock", "mx5_outer_hybrid_pair",
    "mx4_outer_mx04_scratch", "mx4_outer_mx04_head_last",
    "mx5_selected_outer_receipt", "mx6_v22_atlas_spm_audit",
)
OUTER_REPORTS = (
    "evaluation_outer_hybrid_pair.json",
    "evaluation_outer_mx04_scratch.json",
    "evaluation_outer_mx04_head_last.json",
)


def amendment_payload(root: Path, out: Path) -> dict:
    reports = out / "reports"
    ensure_v242_amendment(root, out, create=False)
    old_path = reports / V242_STATE
    if not old_path.is_file():
        raise RuntimeError("V243_PARENT_JOURNAL_MISSING")
    steps = read(old_path).get("steps") or {}
    if (any((steps.get(name) or {}).get("status") != "COMPLETED"
            or (steps.get(name) or {}).get("returncode") != 0
            for name in REQUIRED_COMPLETED)
        or (steps.get("mx6_v22_xai_crossfit") or {}).get("status")
           != "PAUSED_OR_FAILED_RESUMABLE"
        or (steps.get("mx7_freeze_scoped_report") or {}).get("status") == "COMPLETED"):
        raise RuntimeError("V243_PARENT_JOURNAL_NOT_AT_XAI_FAILURE")
    required = [reports / name for name in OUTER_REPORTS]
    required += [reports / "fasttrack_inner_lock.json", reports / "xai_v22_context.json"]
    if any(not path.is_file() for path in required):
        raise RuntimeError("V243_PARENT_RECEIPT_MISSING")
    return {
        "schema_version": 1,
        "protocol_id": "METHOD_X_FASTTRACK_V243_POSTOUTER_XAI_COORDINATOR_REPAIR",
        "status": "POSTOUTER_TECHNICAL_CONTINUATION_NO_SELECTION_CHANGE",
        "v242_amendment_sha256": sha(reports / V242_AMENDMENT),
        "v242_failed_journal_sha256": sha(old_path),
        "bound_receipt_sha256": {path.name: sha(path) for path in required},
        "source_sha256": {name: sha(root / name) for name in SOURCE_FILES},
        "repair": "authenticate_v243_coordinator_and_venv_launcher_for_xai_only",
        "outer_predictions_recomputed": False,
        "selection_or_training_changed": False,
        "scientific_claim_ceiling": "POST_HOC_INTERNAL_EXPLORATORY",
    }


def ensure_v243_amendment(root: Path, out: Path, *, create: bool) -> dict:
    path = out / "reports" / AMENDMENT
    if not path.is_file() and not create:
        raise RuntimeError("V243_AMENDMENT_MISSING")
    expected = amendment_payload(root, out)
    if not path.is_file():
        if (out / "reports" / V243_STATE).exists():
            raise RuntimeError("V243_JOURNAL_PRECEDES_AMENDMENT")
        immutable_json(path, expected)
    elif read(path) != expected:
        raise RuntimeError("V243_AMENDMENT_OR_BOUND_SOURCE_DRIFT")
    return expected


class Supervisor(V242Supervisor):
    def __init__(self, root: Path, output_relative: Path, package_root: Path) -> None:
        super().__init__(root, output_relative, package_root)
        old_state = self.state
        old_path = self.state_path
        self.state_path = self.reports / V243_STATE
        if self.state_path.is_file():
            self.state = read(self.state_path)
        else:
            self.state = {
                "schema_version": 1,
                "pipeline_order": ["MX4", "MX5", "MX6", "MX7"],
                "migration": {
                    "v242_failed_journal_sha256": sha(old_path),
                    "failed_xai_step": old_state["steps"]["mx6_v22_xai_crossfit"],
                },
                "steps": {key: value for key, value in old_state["steps"].items()
                          if key != "mx6_v22_xai_crossfit"},
            }

    def run_step(self, name: str, command: list[str], *, allowed: set[int] | None = None) -> int:
        if name != "mx6_v22_xai_crossfit":
            return super().run_step(name, command, allowed=allowed)
        command = [sys.executable,
                   str(self.root / "scripts/mx_fasttrack_v2_4_3/entry_v243.py"),
                   "xai-run", "--project-root", str(self.root),
                   "--output-root", str(self.output_relative)]
        return V24Supervisor.run_step(self, name, command, allowed=allowed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("mx6", "mx7", "all"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    if (root != ROOT.resolve() or args.output_root != OUTPUT
        or out == root or not out.is_relative_to(root)):
        raise RuntimeError("V243_PROJECT_OR_OUTPUT_ROOT_INVALID")
    if foreign_method_x_writers(root):
        raise RuntimeError("V243_FOREIGN_METHOD_X_WRITER_ACTIVE")
    with coordinator_lock(out / "reports"):
        prior = os.environ.get(INTERNAL_PARENT_ENV)
        os.environ[INTERNAL_PARENT_ENV] = str(os.getpid())
        try:
            ensure_v243_amendment(root, out, create=True)
            _patch_parent_checkers()
            supervisor = Supervisor(root, args.output_root, DEFAULT_PACKAGE)
            getattr(supervisor, "mx7" if args.target == "all" else args.target)()
            print(json.dumps({"status": f"V243_{args.target.upper()}_COMPLETED",
                              "output_root": str(out)}), flush=True)
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
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(3)
