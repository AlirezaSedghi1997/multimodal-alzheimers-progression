"""Additive pre-outer v2.4.2 continuation without another 21 GiB tree copy."""
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
from scripts.mx_fasttrack_v2_4.supervise_v24 import OUTPUT, Supervisor as V24Supervisor
from scripts.mx_fasttrack_v2_4_1.compatible_lock import check_lock_compatible
from scripts.mx_fasttrack_v2_4_1.supervise_v241 import (
    Supervisor as V241Supervisor, _no_outer_evidence, _patch_parent_checkers,
    ensure_v241_amendment,
)
from scripts.mx_fasttrack_v2_4_2.checkpoint_projection import baseline_binding

AMENDMENT = "fasttrack_v242_checkpoint_projection_before_outer.json"
SOURCE_FILES = (
    "scripts/mx_fasttrack_v2_4_2/__init__.py",
    "scripts/mx_fasttrack_v2_4_2/checkpoint_projection.py",
    "scripts/mx_fasttrack_v2_4_2/entry_v242.py",
    "scripts/mx_fasttrack_v2_4_2/supervise_v242.py",
    "docs/method_x_fasttrack_v242_checkpoint_projection_FA.md",
)


def amendment_payload(root: Path, out: Path) -> dict:
    reports = out / "reports"
    binding = baseline_binding(root, out)
    return {
        "schema_version": 1,
        "protocol_id": "METHOD_X_FASTTRACK_V242_SOURCE_ONLY_CHECKPOINT_VIEW",
        "status": "LOCKED_BEFORE_FIRST_OUTER_ACCESS",
        "parent_preregistration_sha256": sha(reports / "fasttrack_preregistration.json"),
        "v241_amendment_sha256": sha(reports / "fasttrack_v241_verifier_repair_before_outer.json"),
        "inner_lock_sha256": sha(reports / "fasttrack_inner_lock.json"),
        "historical_migration_receipt_sha256": binding["migration_receipt_sha256"],
        "historical_digest": binding["old_digest"],
        "current_contract_digest": binding["new_digest"],
        "bound_folds": sorted(binding["paths"].values()),
        "source_sha256": {name: sha(root / name) for name in SOURCE_FILES},
        "checkpoint_bytes_changed": False,
        "projection_only_at_locked_evaluator_call_sites": True,
        "selection_or_training_changed": False,
        "external_validation": False,
        "strong_claim_allowed": False,
    }


def ensure_v242_amendment(root: Path, out: Path, *, create: bool) -> dict:
    path = out / "reports" / AMENDMENT
    ensure_v241_amendment(root, out, create=False)
    if not path.is_file():
        if not create:
            raise RuntimeError("V242_PREOUTER_AMENDMENT_MISSING")
        _no_outer_evidence(out)
        check_inputs(root, out)
        check_lock_compatible(root, out, hash_cache=False)
    expected = amendment_payload(root, out)
    if not path.is_file():
        immutable_json(path, expected)
    elif read(path) != expected:
        raise RuntimeError("V242_AMENDMENT_OR_SOURCE_DRIFT")
    return expected


class Supervisor(V241Supervisor):
    def __init__(self, root: Path, output_relative: Path, package_root: Path) -> None:
        super().__init__(root, output_relative, package_root)
        self.state_path = self.reports / "fasttrack_v242_supervisor_state.json"
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
                       str(self.root / "scripts/mx_fasttrack_v2_4_2/entry_v242.py"),
                       mode, "--project-root", str(self.root),
                       "--output-root", str(self.output_relative)]
            if task is not None:
                command += ["--task", task]
        # Bypass v2.4.1's command substitution, retaining v2.1 resume guards.
        return V24Supervisor.run_step(self, name, command, allowed=allowed)


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
        raise RuntimeError("V242_PROJECT_OR_OUTPUT_ROOT_INVALID")
    if foreign_method_x_writers(root):
        raise RuntimeError("V242_FOREIGN_METHOD_X_WRITER_ACTIVE")
    from scripts.mx_fasttrack_v2.supervise import DEFAULT_PACKAGE
    with coordinator_lock(out / "reports"):
        old = os.environ.get(INTERNAL_PARENT_ENV)
        os.environ[INTERNAL_PARENT_ENV] = str(os.getpid())
        try:
            ensure_v242_amendment(root, out, create=True)
            _patch_parent_checkers()
            if args.target != "prepare":
                supervisor = Supervisor(root, args.output_root,
                                        args.package_root or DEFAULT_PACKAGE)
                getattr(supervisor, "mx7" if args.target == "all" else args.target)()
            print(json.dumps({"status": f"V242_{args.target.upper()}_COMPLETED",
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
