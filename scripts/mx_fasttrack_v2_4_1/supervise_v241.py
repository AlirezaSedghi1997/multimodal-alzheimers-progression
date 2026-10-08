"""Resume the v2.4 tree under a versioned pre-outer verifier amendment.

This coordinator has its own step journal but reuses the immutable v2.4 inner
lock, checkpoints, and preregistration.  It never rewrites historical metrics.
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
from scripts.mx_fasttrack_v2_1.supervise_v21 import (
    INTERNAL_PARENT_ENV, coordinator_lock, foreign_method_x_writers, immutable_json,
)
from scripts.mx_fasttrack_v2_4.supervise_v24 import (
    OUTPUT, PROTOCOL, Supervisor as V24Supervisor,
)
from scripts.mx_fasttrack_v2_4_1.compatible_lock import check_lock_compatible

AMENDMENT = "fasttrack_v241_verifier_repair_before_outer.json"
SOURCE_FILES = (
    "scripts/mx_fasttrack_v2_4_1/__init__.py",
    "scripts/mx_fasttrack_v2_4_1/compatible_lock.py",
    "scripts/mx_fasttrack_v2_4_1/entry_v241.py",
    "scripts/mx_fasttrack_v2_4_1/supervise_v241.py",
    "docs/method_x_fasttrack_v241_gate_repair_FA.md",
)


def _no_outer_evidence(out: Path) -> None:
    reports = out / "reports"
    if any(reports.glob("evaluation_outer*")):
        raise RuntimeError("V241_OUTER_EVIDENCE_ALREADY_PRESENT_REVIEW_BEFORE_AMENDMENT")
    resume = out / "_private_resume"
    if resume.exists() and any(resume.rglob("*")):
        raise RuntimeError("V241_PRIVATE_OUTER_RECEIPT_ALREADY_PRESENT")


def amendment_payload(root: Path, out: Path) -> dict:
    reports = out / "reports"
    prereg = read(reports / "fasttrack_preregistration.json")
    lock = read(reports / "fasttrack_inner_lock.json")
    if (prereg.get("protocol_id") != PROTOCOL
        or prereg.get("status") != "LOCKED_BEFORE_GPU"
        or prereg.get("outer_test_access_count") != 0
        or lock.get("status") != "LOCKED_INNER_ONLY_EXPLORATORY"
        or lock.get("outer_test_access_count") != 0):
        raise RuntimeError("V241_PARENT_PREOUTER_SCOPE_INVALID")
    return {
        "schema_version": 1,
        "protocol_id": "METHOD_X_FASTTRACK_V241_IN_MEMORY_LOCK_VERIFIER_REPAIR",
        "status": "LOCKED_BEFORE_FIRST_OUTER_ACCESS",
        "parent_preregistration_sha256": sha(reports / "fasttrack_preregistration.json"),
        "parent_v24_amendment_sha256": sha(reports / "fasttrack_v24_source_repair_amendment.json"),
        "inner_lock_sha256": sha(reports / "fasttrack_inner_lock.json"),
        "source_sha256": {name: sha(root / name) for name in SOURCE_FILES},
        "repair": "derive_absent_metrics_aggregate_contract_digest_in_memory_from_hashed_run_contract",
        "historical_metric_or_checkpoint_bytes_changed": False,
        "selection_rule_changed": False,
        "scientific_training_parameters_changed": False,
        "external_validation": False,
        "strong_claim_allowed": False,
    }


def ensure_v241_amendment(root: Path, out: Path, *, create: bool) -> dict:
    path = out / "reports" / AMENDMENT
    if not path.is_file():
        if not create:
            raise RuntimeError("V241_PREOUTER_AMENDMENT_MISSING")
        _no_outer_evidence(out)
        check_inputs(root, out)
        check_lock_compatible(root, out, hash_cache=False)
    expected = amendment_payload(root, out)
    if not path.is_file():
        immutable_json(path, expected)
    elif read(path) != expected:
        raise RuntimeError("V241_AMENDMENT_OR_SOURCE_DRIFT")
    return expected


class Supervisor(V24Supervisor):
    def __init__(self, root: Path, output_relative: Path, package_root: Path) -> None:
        super().__init__(root, output_relative, package_root)
        self.state_path = self.reports / "fasttrack_v241_supervisor_state.json"
        self.state = (read(self.state_path) if self.state_path.is_file()
                      else {"schema_version": 1, "pipeline_order": ["MX4", "MX5", "MX6", "MX7"],
                            "steps": {}})

    def run_step(self, name: str, command: list[str], *, allowed: set[int] | None = None) -> int:
        modes = {
            "mx5_outer_hybrid_pair": ("hybrid-pair", None),
            "mx4_outer_mx04_scratch": ("locked", "mx04_scratch"),
            "mx4_outer_mx04_head_last": ("locked", "mx04_head_last"),
            "mx5_selected_outer_receipt": ("locked", "selected"),
            "mx6_v22_atlas_spm_audit": ("xai-audit", None),
            "mx6_v22_xai_crossfit": ("xai-run", None),
            "mx7_freeze_scoped_report": ("freeze-guard", None),
            "mx7_v22_thesis_addendum": ("thesis", None),
        }
        if name in modes:
            mode, task = modes[name]
            command = [sys.executable,
                       str(self.root / "scripts/mx_fasttrack_v2_4_1/entry_v241.py"),
                       mode, "--project-root", str(self.root),
                       "--output-root", str(self.output_relative)]
            if task is not None:
                command += ["--task", task]
        return super().run_step(name, command, allowed=allowed)


def _patch_parent_checkers() -> None:
    from scripts.mx_fasttrack_v2 import integrity
    from scripts.mx_fasttrack_v2_1 import supervise_v21
    from scripts.mx_fasttrack_v2_2 import xai_runner

    integrity.check_lock = check_lock_compatible
    supervise_v21.check_lock = check_lock_compatible
    xai_runner.check_lock = check_lock_compatible


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("prepare", "mx4", "mx5", "mx6", "mx7", "all"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--package-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (root / args.output_root).resolve()
    if (root != ROOT.resolve() or not out.is_relative_to(root) or out == root
        or args.output_root != OUTPUT):
        raise RuntimeError("V241_PROJECT_OR_OUTPUT_ROOT_INVALID")
    if foreign_method_x_writers(root):
        raise RuntimeError("V241_FOREIGN_METHOD_X_WRITER_ACTIVE")
    from scripts.mx_fasttrack_v2.supervise import DEFAULT_PACKAGE
    with coordinator_lock(out / "reports"):
        old = os.environ.get(INTERNAL_PARENT_ENV)
        os.environ[INTERNAL_PARENT_ENV] = str(os.getpid())
        try:
            ensure_v241_amendment(root, out, create=True)
            _patch_parent_checkers()
            if args.target != "prepare":
                supervisor = Supervisor(root, args.output_root,
                                        args.package_root or DEFAULT_PACKAGE)
                getattr(supervisor, "mx7" if args.target == "all" else args.target)()
            print(json.dumps({"status": f"V241_{args.target.upper()}_COMPLETED",
                              "output_root": str(out)}, ensure_ascii=False))
        finally:
            if old is None:
                os.environ.pop(INTERNAL_PARENT_ENV, None)
            else:
                os.environ[INTERNAL_PARENT_ENV] = old
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
