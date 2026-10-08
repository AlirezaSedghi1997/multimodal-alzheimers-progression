"""Versioned MX4-to-MX7 coordinator: v2.1 training, v2.2 XAI and thesis."""
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
from scripts.mx_fasttrack_v2.supervise import DEFAULT_OUTPUT, DEFAULT_PACKAGE
from scripts.mx_fasttrack_v2_1.supervise_v21 import (
    INTERNAL_PARENT_ENV, Supervisor as V21Supervisor, coordinator_lock,
    ensure_amendment, foreign_method_x_writers, immutable_json,
)

PROTOCOL = "fasttrack_v22_protocol.json"
SOURCE_FILES = (
    "scripts/mx_fasttrack_v2_2/__init__.py",
    "scripts/mx_fasttrack_v2_2/supervise_v22.py",
    "scripts/mx_fasttrack_v2_2/xai_crossfit.py",
    "scripts/mx_fasttrack_v2_2/xai_runner.py",
    "scripts/mx_fasttrack_v2_2/independent_gate.py",
    "scripts/mx_fasttrack_v2_2/thesis_report.py",
    "configs/experiments/method_x_xai_v22.json",
    "docs/method_x_xai_v22_protocol_FA.md",
)


def protocol_payload(root: Path, out: Path) -> dict:
    ensure_amendment(root, out)
    prereg = out / "reports/fasttrack_preregistration.json"
    v21 = out / "reports/fasttrack_v21_amendment.json"
    return {"schema_version": 1,
            "protocol_id": "METHOD_X_FASTTRACK_V22_XAI_CROSSFIT_THESIS",
            "parent_preregistration_sha256": sha(prereg),
            "v21_amendment_sha256": sha(v21),
            "source_sha256": {relative: sha(root / relative) for relative in SOURCE_FILES},
            "scope": "POST_LOCK_ADDENDUM_NO_TRAINING_OR_SELECTION_CHANGE",
            "xai_sample_selection": "first_eight_real_inner_validation_per_fold",
            "xai_selection_assessment_separated": True,
            "xai_independent_validation": False,
            "disease_region_confirmation": False,
            "external_validation": False,
            "thesis_policy": "ONLY_HASH_VERIFIED_FROZEN_PARENT_AND_FIVE_FOLD_RECEIPTS",
            "original_full_MX7_completed": False}


def ensure_protocol(root: Path, out: Path) -> dict:
    path = out / "reports" / PROTOCOL
    if not path.exists():
        steps_path = out / "reports/fasttrack_supervisor_state.json"
        steps = (read(steps_path).get("steps") or {}) if steps_path.exists() else {}
        if any(str(name).startswith(("mx5_", "mx6_", "mx7_", "mx4_outer_"))
               for name in steps):
            raise RuntimeError("V22_PROTOCOL_TOO_LATE_CREATE_NEW_RUN")
        if any((out / relative).exists() for relative in (
            "reports/smoke_baseline.json", "reports/smoke_plus_dense_skip.json",
            "inner_tasks/hybrid_baseline", "inner_tasks/hybrid_dense",
            "reports/fasttrack_inner_lock.json",
            "reports/evaluation_outer_hybrid_pair_state.json",
        )):
            raise RuntimeError("V22_PROTOCOL_TOO_LATE_CREATE_NEW_RUN")
    expected = protocol_payload(root, out)
    if not path.exists():
        immutable_json(path, expected)
    elif read(path) != expected:
        raise RuntimeError("V22_PROTOCOL_OR_SOURCE_DRIFT")
    return expected


class Supervisor(V21Supervisor):
    def __init__(self, root: Path, output_relative: Path, package_root: Path) -> None:
        # The v2.1 parent MX7 explicitly omits its superseded XAI layer.
        # v2.2 has a separate five-fold XAI receipt and thesis addendum.
        super().__init__(root, output_relative, package_root,
                         xai_skip_reason="OPERATOR_CHOICE")

    def mx4(self) -> None:
        V21Supervisor.mx4(self)
        ensure_protocol(self.root, self.out)

    def mx5(self) -> None:
        ensure_protocol(self.root, self.out) if (self.reports / PROTOCOL).exists() else None
        V21Supervisor.mx5(self)
        ensure_protocol(self.root, self.out)

    def mx6(self) -> None:
        self.mx5()
        self.run_step("mx6_v22_atlas_spm_audit", self.py(
            "scripts/mx_fasttrack_v2_2/xai_runner.py",
            "--project-root", self.root, "--output-root", self.output_relative,
            "--mode", "audit"))
        code = self.run_step("mx6_v22_xai_crossfit", self.py(
            "scripts/mx_fasttrack_v2_2/xai_runner.py",
            "--project-root", self.root, "--output-root", self.output_relative,
            "--mode", "run"), allowed={0, 3})
        if code == 3:
            from scripts.mx_fasttrack_v2.supervise import atomic
            from scripts.mx_fasttrack_v2_2.xai_runner import verify_xai_summary

            try:
                result = verify_xai_summary(self.root, self.out,
                                            ensure_protocol(self.root, self.out))
                if result["status"] != "BLOCKED_EXPLORATORY_GATE":
                    raise RuntimeError("V22_XAI_EXIT_3_NOT_SCIENTIFIC_GATE")
            except (RuntimeError, ValueError):
                self.state["steps"]["mx6_v22_xai_crossfit"]["status"] = "PAUSED_OR_FAILED_RESUMABLE"
                atomic(self.state_path, self.state)
                raise

    def mx7(self) -> None:
        self.mx6()
        if (self.reports / "xai_limited_summary.json").exists():
            raise RuntimeError("V22_SUPERSEDED_V21_XAI_ALREADY_EXECUTED_REVIEW_REQUIRED")
        V21Supervisor.mx7(self)
        self.run_step("mx7_v22_thesis_addendum", self.py(
            "scripts/mx_fasttrack_v2_2/thesis_report.py",
            "--project-root", self.root, "--output-root", self.output_relative))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("mx4", "mx5", "mx6", "mx7", "all"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    if root != ROOT.resolve() or not out.is_relative_to(root) or out == root:
        raise RuntimeError("V22_PROJECT_OR_OUTPUT_ROOT_INVALID")
    writers = foreign_method_x_writers(root)
    if writers:
        raise RuntimeError(f"V22_FOREIGN_METHOD_X_WRITER_ACTIVE:{[row['pid'] for row in writers]}")
    with coordinator_lock(out / "reports"):
        prior = os.environ.get(INTERNAL_PARENT_ENV)
        os.environ[INTERNAL_PARENT_ENV] = str(os.getpid())
        try:
            supervisor = Supervisor(root, args.output_root, args.package_root)
            target = "mx7" if args.target == "all" else args.target
            getattr(supervisor, target)()
            print(json.dumps({"status": f"{target.upper()}_V22_SUPERVISOR_TARGET_COMPLETED",
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
