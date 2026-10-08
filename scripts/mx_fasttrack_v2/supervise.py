"""Single-command, ordered, resumable supervisor for fast-track MX4--MX7.

Each target includes and verifies every earlier target. Re-running the same
command is the supported recovery path after interruption: training delegates
to the exact-resume contract, evaluations resume from committed fold receipts,
MX6 resumes from committed fold summaries, and MX7 uses immutable write-once
artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = Path("results/final_results/method_x_fasttrack_v2_20260926_mx04_import")
DEFAULT_PACKAGE = Path(r"C:\Users\USER\Desktop\Rewrite\method_x_fasttrack_v2")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"EXPECTED_JSON_OBJECT:{path}")
    return value


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def command_sha(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class Supervisor:
    def __init__(self, root: Path, output_relative: Path, package_root: Path) -> None:
        self.root = root
        self.output_relative = output_relative
        self.out = (root / output_relative).resolve()
        self.package_root = package_root.resolve()
        if root != ROOT.resolve() or not self.out.is_relative_to(root) or self.out == root:
            raise RuntimeError("FASTTRACK_SUPERVISOR_ROOT_OR_OUTPUT_INVALID")
        self.reports = self.out / "reports"
        self.state_path = self.reports / "fasttrack_supervisor_state.json"
        self.state = read(self.state_path) if self.state_path.is_file() else {
            "schema_version": 1,
            "pipeline_order": ["MX4", "MX5", "MX6", "MX7"],
            "steps": {},
        }
        if self.state.get("pipeline_order") != ["MX4", "MX5", "MX6", "MX7"]:
            raise RuntimeError("FASTTRACK_SUPERVISOR_STATE_DRIFT")

    def py(self, relative: str, *arguments: object) -> list[str]:
        return [sys.executable, str(self.root / relative), *(str(value) for value in arguments)]

    def run_step(self, name: str, command: list[str], *, allowed: set[int] | None = None) -> int:
        allowed = {0} if allowed is None else allowed
        fingerprint = command_sha(command)
        prior = self.state["steps"].get(name, {})
        if prior and prior.get("command_sha256") not in (None, fingerprint):
            raise RuntimeError(f"SUPERVISOR_COMMAND_DRIFT:{name}")
        self.state["steps"][name] = {
            "status": "RUNNING",
            "attempts": int(prior.get("attempts", 0)) + 1,
            "command_sha256": fingerprint,
            "started_at_utc": utc(),
        }
        atomic(self.state_path, self.state)
        print(json.dumps({"supervisor_step": name, "status": "RUNNING"}), flush=True)
        code = subprocess.run(command, cwd=self.root, check=False).returncode
        current = self.state["steps"][name]
        current["returncode"] = code
        current["finished_at_utc"] = utc()
        current["status"] = "COMPLETED" if code in allowed else "PAUSED_OR_FAILED_RESUMABLE"
        atomic(self.state_path, self.state)
        if code not in allowed:
            raise RuntimeError(f"SUPERVISOR_STEP_NOT_COMPLETE:{name}:EXIT_{code}")
        return code

    def initialize(self) -> None:
        preregistration = self.reports / "fasttrack_preregistration.json"
        if preregistration.is_file():
            from scripts.mx_fasttrack_v2.integrity import check_inputs

            check_inputs(self.root, self.out)
            print(json.dumps({"supervisor_step": "initialize", "status": "REUSED_VERIFIED_PREREGISTRATION"}))
            return
        if not self.package_root.is_dir():
            raise RuntimeError(f"FASTTRACK_SOURCE_PACKAGE_MISSING:{self.package_root}")
        self.run_step("initialize_preflight", self.py(
            "scripts/mx_fasttrack_v2/capture_preflight.py",
            "--project-root", self.root,
            "--package-root", self.package_root,
            "--out", self.reports / "machine_preflight.json",
            "--allow-coordinator-pid", os.getpid(),
        ))
        self.run_step("initialize_data_audit", self.py(
            "scripts/mx_fasttrack_v2/audit_inputs.py",
            "--index", self.root / "docs/fasttrack_private/cache_full_index.csv",
            "--splits", self.root / "splits",
            "--cache", self.root / "data/cached/t1_volume_cache_128.h5",
            "--out", self.reports / "data_semantics_audit.json",
        ))
        self.run_step("initialize_preregister", self.py(
            "scripts/mx_fasttrack_v2/run.py", "plan",
            "--project-root", self.root,
            "--output-root", self.output_relative,
        ))

    def mx4(self) -> None:
        self.initialize()
        self.run_step("mx4_two_arm_inner_training", self.py(
            "scripts/mx_fasttrack_v2/run.py", "mx4",
            "--project-root", self.root,
            "--output-root", self.output_relative,
        ))

    def mx5(self) -> None:
        self.mx4()
        self.run_step("mx5_hybrid_cuda_smoke", self.py(
            "scripts/mx_fasttrack_v2/run.py", "smoke",
            "--project-root", self.root,
            "--output-root", self.output_relative,
        ))
        self.run_step("mx5_hybrid_training_profile", self.py(
            "scripts/mx_fasttrack_v2/run.py", "profile",
            "--project-root", self.root,
            "--output-root", self.output_relative,
        ))
        self.run_step("mx5_two_arm_inner_training", self.py(
            "scripts/mx_fasttrack_v2/run.py", "mx5",
            "--project-root", self.root,
            "--output-root", self.output_relative,
        ))
        self.run_step("mx5_common_inner_lock", self.py(
            "scripts/mx_fasttrack_v2/run.py", "lock",
            "--project-root", self.root,
            "--output-root", self.output_relative,
        ))
        self.run_step("mx5_outer_hybrid_pair", self.py(
            "scripts/mx_fasttrack_v2/evaluate_hybrid_pair.py",
            "--project-root", self.root,
            "--output-root", self.output_relative,
        ))
        for task in ("mx04_scratch", "mx04_head_last"):
            self.run_step(f"mx4_outer_{task}", self.py(
                "scripts/mx_fasttrack_v2/evaluate_locked.py",
                "--project-root", self.root,
                "--output-root", self.output_relative,
                "--scope", "outer", "--task", task,
            ))
        self.run_step("mx5_selected_outer_receipt", self.py(
            "scripts/mx_fasttrack_v2/evaluate_locked.py",
            "--project-root", self.root,
            "--output-root", self.output_relative,
            "--scope", "outer", "--task", "selected",
        ))

    def mx6(self) -> None:
        self.mx5()
        self.run_step("mx6_atlas_spm_audit", self.py(
            "scripts/mx_fasttrack_v2/xai_limited.py",
            "--project-root", self.root,
            "--output-root", self.output_relative,
            "--mode", "audit",
        ))
        code = self.run_step("mx6_limited_xai", self.py(
            "scripts/mx_fasttrack_v2/xai_limited.py",
            "--project-root", self.root,
            "--output-root", self.output_relative,
            "--mode", "run",
        ), allowed={0, 3})
        if code == 3:
            state_path = self.reports / "xai_limited_state.json"
            state = read(state_path) if state_path.is_file() else {}
            if (state.get("status") != "COMPLETED"
                or state.get("gate_status") != "BLOCKED_EXPLORATORY_GATE"):
                self.state["steps"]["mx6_limited_xai"]["status"] = "PAUSED_OR_FAILED_RESUMABLE"
                atomic(self.state_path, self.state)
                raise RuntimeError("MX6_EXIT_3_IS_NOT_A_TERMINAL_SCIENTIFIC_GATE_RESULT")

    def mx7(self) -> None:
        try:
            self.mx6()
        except RuntimeError as exc:
            reason = str(exc)
            if not (
                reason.startswith("SUPERVISOR_STEP_NOT_COMPLETE:mx6_atlas_spm_audit:")
                or reason.startswith("SUPERVISOR_STEP_NOT_COMPLETE:mx6_limited_xai:")
                or reason == "MX6_EXIT_3_IS_NOT_A_TERMINAL_SCIENTIFIC_GATE_RESULT"
            ):
                raise
            # MX6 is optional in the locked fast-track config. Its failure is
            # visible in the supervisor ledger, while the mandatory MX4/MX5
            # evidence can still be frozen with an explicit XAI gap.
            self.state["mx6_optional_blocker"] = {
                "status": "BLOCKED_NOT_COMPLETED",
                "reason": reason,
                "recorded_at_utc": utc(),
            }
            atomic(self.state_path, self.state)
        self.run_step("mx7_freeze_scoped_report", self.py(
            "scripts/mx_fasttrack_v2/freeze_report.py",
            "--project-root", self.root,
            "--output-root", self.output_relative,
        ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("mx4", "mx5", "mx6", "mx7", "all"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    supervisor = Supervisor(args.project_root.resolve(), args.output_root, args.package_root)
    target = "mx7" if args.target == "all" else args.target
    getattr(supervisor, target)()
    print(json.dumps({
        "status": f"{target.upper()}_SUPERVISOR_TARGET_COMPLETED",
        "resumable_reentry_command": [sys.executable, str(Path(__file__).resolve()), args.target],
        "output_root": str(supervisor.out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
