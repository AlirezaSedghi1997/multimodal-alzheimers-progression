"""Single-command, resumable MX-04 through MX-07 continuation under a new source lock.

The historical MX-04 and five MX-05 baseline fold checkpoints are imported
byte-for-byte with an explicit receipt.  No parent preregistration is rewritten,
no outer-test evaluation is replayed, and the imported C staging is preserved.
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
from scripts.mx_fasttrack_v2.run import TASKS, read, sha, task_command
from scripts.mx_fasttrack_v2.supervise import DEFAULT_PACKAGE
from scripts.mx_fasttrack_v2_1.supervise_v21 import (
    INTERNAL_PARENT_ENV, coordinator_lock, ensure_amendment,
    foreign_method_x_writers, immutable_json,
)
from scripts.mx_fasttrack_v2_2.supervise_v22 import (
    PROTOCOL as V22_PROTOCOL, Supervisor as V22Supervisor, ensure_protocol,
)
from scripts.mx_fasttrack_v2_4.import_resume import (
    PROTOCOL, RECEIPT, import_previous, verify_import,
)

OUTPUT = Path("results/final_results/method_x_fasttrack_v2_20260927_mx05_gate_import")
CONFIG = "configs/experiments/method_x_fasttrack_v2_20260927_mx05_gate_import.json"
AMENDMENT = "fasttrack_v24_source_repair_amendment.json"
SOURCE_FILES = (
    "scripts/mx_fasttrack_v2_4/__init__.py",
    "scripts/mx_fasttrack_v2_4/import_resume.py",
    "scripts/mx_fasttrack_v2_4/outer_guard_runner.py",
    "scripts/mx_fasttrack_v2_4/supervise_v24.py",
    "docs/method_x_fasttrack_v24_resume_FA.md",
    CONFIG,
)


def amendment_payload(root: Path, out: Path) -> dict:
    prereg_path = out / "reports/fasttrack_preregistration.json"
    prereg = read(prereg_path)
    if prereg.get("protocol_id") != PROTOCOL:
        raise RuntimeError("V24_WRONG_PREREGISTRATION_PROTOCOL")
    verify_import(root, out)
    return {
        "schema_version": 1,
        "protocol_id": "METHOD_X_FASTTRACK_V24_SOURCE_GATE_REPAIR_IMPORT",
        "parent_preregistration_sha256": sha(prereg_path),
        "v21_amendment_sha256": sha(out / "reports/fasttrack_v21_amendment.json"),
        "v22_protocol_sha256": sha(out / "reports" / V22_PROTOCOL),
        "historical_import_receipt_sha256": sha(out / "reports" / RECEIPT),
        "source_sha256": {relative: sha(root / relative) for relative in SOURCE_FILES},
        "scope": "SOURCE_GATE_REPAIR_AND_HISTORICAL_INNER_FOLD_REUSE_BEFORE_OUTER",
        "scientific_training_parameters_changed": False,
        "historical_checkpoints_retrained": False,
        "external_validation": False,
        "strong_claim_allowed_without_independent_validity": False,
        "self_gpu_context_exempted_only_for_outer_evaluator": True,
    }


def ensure_v24_amendment(root: Path, out: Path) -> dict:
    path = out / "reports" / AMENDMENT
    if not path.exists():
        from scripts.mx_fasttrack_v2_3.supervise_v23 import _require_no_outer_access

        _require_no_outer_access(out)
    expected = amendment_payload(root, out)
    if not path.exists():
        immutable_json(path, expected)
    elif read(path) != expected:
        raise RuntimeError("V24_AMENDMENT_OR_SOURCE_DRIFT")
    return expected


def outer_command(root: Path, output: Path, evaluator: str,
                  *, task: str | None = None) -> list[str]:
    if evaluator not in {"hybrid-pair", "locked"}:
        raise RuntimeError("V24_OUTER_EVALUATOR_INVALID")
    command = [sys.executable, str(root / "scripts/mx_fasttrack_v2_4/outer_guard_runner.py"),
               evaluator, "--project-root", str(root), "--output-root", str(output)]
    if evaluator == "locked":
        if task not in {"selected", "mx04_scratch", "mx04_head_last"}:
            raise RuntimeError("V24_OUTER_TASK_INVALID")
        command += ["--scope", "outer", "--task", task]
    elif task is not None:
        raise RuntimeError("V24_PAIR_TASK_FORBIDDEN")
    return command


class Supervisor(V22Supervisor):
    def py(self, relative: str, *arguments: object) -> list[str]:
        command = super().py(relative, *arguments)
        if relative == "scripts/mx_fasttrack_v2/run.py":
            command += ["--config", str(self.root / CONFIG)]
        return command

    def mx5(self) -> None:
        ensure_v24_amendment(self.root, self.out)
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
        self.run_step("mx5_outer_hybrid_pair", outer_command(
            self.root, self.output_relative, "hybrid-pair"))
        for task in ("mx04_scratch", "mx04_head_last"):
            self.run_step(f"mx4_outer_{task}", outer_command(
                self.root, self.output_relative, "locked", task=task))
        self.run_step("mx5_selected_outer_receipt", outer_command(
            self.root, self.output_relative, "locked", task="selected"))


def prepare(root: Path, output: Path, package: Path) -> Supervisor:
    supervisor = Supervisor(root, output, package)
    supervisor.initialize()
    check_inputs(root, supervisor.out)
    ensure_amendment(root, supervisor.out)
    ensure_protocol(root, supervisor.out)
    import_previous(root, supervisor.out)
    battery = root / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729"
    manifest = read(battery / "battery_manifest.json")
    baseline = next(task for task in TASKS if task[0] == "hybrid_baseline")
    command, _ = task_command(
        baseline, root, battery, supervisor.out, manifest, None
    )
    supervisor.run_step(
        "v24_imported_baseline_contract_preflight",
        [*command, "--contract_preflight_only"],
    )
    ensure_v24_amendment(root, supervisor.out)
    return supervisor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("prepare", "mx4", "mx5", "mx6", "mx7", "all"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    if (root != ROOT.resolve() or not out.is_relative_to(root) or out == root
        or out == (root / "results/final_results/method_x_fasttrack_v2_20260926_mx04_import")):
        raise RuntimeError("V24_PROJECT_OR_OUTPUT_ROOT_INVALID")
    if foreign_method_x_writers(root):
        raise RuntimeError("V24_FOREIGN_METHOD_X_WRITER_ACTIVE")
    with coordinator_lock(out / "reports"):
        previous = os.environ.get(INTERNAL_PARENT_ENV)
        os.environ[INTERNAL_PARENT_ENV] = str(os.getpid())
        try:
            supervisor = prepare(root, args.output_root, args.package_root)
            if args.target != "prepare":
                target = "mx7" if args.target == "all" else args.target
                getattr(supervisor, target)()
            print(json.dumps({"status": f"V24_{args.target.upper()}_COMPLETED",
                              "output_root": str(out)}, ensure_ascii=False))
        finally:
            if previous is None:
                os.environ.pop(INTERNAL_PARENT_ENV, None)
            else:
                os.environ[INTERNAL_PARENT_ENV] = previous
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
