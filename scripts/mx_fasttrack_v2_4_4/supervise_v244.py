"""Resume only failed fifth-fold XAI under a disclosed post-hoc atlas amendment."""
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
from scripts.mx_fasttrack_v2.supervise import DEFAULT_PACKAGE
from scripts.mx_fasttrack_v2_1.supervise_v21 import (
    INTERNAL_PARENT_ENV, coordinator_lock, foreign_method_x_writers,
    immutable_json, verify_hybrid_pair_effect,
)
from scripts.mx_fasttrack_v2_4.supervise_v24 import Supervisor as V24Supervisor
from scripts.mx_fasttrack_v2_4_1.compatible_lock import check_lock_compatible
from scripts.mx_fasttrack_v2_4_1.supervise_v241 import _patch_parent_checkers
from scripts.mx_fasttrack_v2_4_2.supervise_v242 import OUTPUT
from scripts.mx_fasttrack_v2_4_3.supervise_v243 import (
    AMENDMENT as V243_AMENDMENT, REQUIRED_COMPLETED, V243_STATE,
    Supervisor as V243Supervisor, ensure_v243_amendment,
)
from scripts.mx_fasttrack_v2_4_4.control_eligibility import eligible_atlas_ids

AMENDMENT = "fasttrack_v244_posthoc_xai_atlas_eligibility.json"
STATE = "fasttrack_v244_supervisor_state.json"
APPENDIX = "fasttrack_v244_xai_eligibility_appendix.json"
SOURCE_FILES = (
    "scripts/mx_fasttrack_v2_4_4/__init__.py",
    "scripts/mx_fasttrack_v2_4_4/control_eligibility.py",
    "scripts/mx_fasttrack_v2_4_4/entry_v244.py",
    "scripts/mx_fasttrack_v2_4_4/supervise_v244.py",
    "docs/method_x_fasttrack_v244_xai_control_eligibility_FA.md",
)


def amendment_payload(root: Path, out: Path) -> dict:
    import nibabel as nib
    import numpy as np
    from scripts.run_method_x_anatomical_xai import DEFAULT_REGISTRY

    ensure_v243_amendment(root, out, create=False)
    reports = out / "reports"
    prior_path = reports / V243_STATE
    prior = read(prior_path)
    steps = prior.get("steps") or {}
    if (any((steps.get(name) or {}).get("status") != "COMPLETED"
            or (steps.get(name) or {}).get("returncode") != 0
            for name in (*REQUIRED_COMPLETED, "mx6_v22_atlas_spm_audit"))
        or (steps.get("mx6_v22_xai_crossfit") or {}).get("status")
           != "PAUSED_OR_FAILED_RESUMABLE"
        or (steps.get("mx7_freeze_scoped_report") or {}).get("status") == "COMPLETED"):
        raise RuntimeError("V244_PARENT_NOT_AT_FIFTH_FOLD_XAI_FAILURE")
    state = read(reports / "xai_v22_state.json")
    context_path = reports / "xai_v22_context.json"
    context = read(context_path)
    receipt_paths = [reports / "xai_v22_folds" / f"fold_{i}.json" for i in range(4)]
    target_path = out / "_private_resume/xai_v22_targets/fold_4.json"
    if any(not path.is_file() for path in (*receipt_paths, target_path)):
        raise RuntimeError("V244_PRIOR_FOLD_OR_FIFTH_TARGET_RECEIPT_MISSING")
    old_hashes = {str(i): sha(path) for i, path in enumerate(receipt_paths)}
    if (state.get("status") not in {"RUNNING", "COMPLETED"}
        or state.get("context_sha256") != sha(context_path)
        or any((state.get("completed_folds") or {}).get(key) != value
               for key, value in old_hashes.items())
        or (read(target_path).get("context_sha256") != sha(context_path))):
        raise RuntimeError("V244_PRIOR_XAI_STATE_OR_TARGET_DRIFT")
    from scripts.mx_fasttrack_v2_2.xai_runner import _validated_receipts
    rows, checked_hashes = _validated_receipts(out, context)
    if [row["fold"] for row in rows[:4]] != list(range(4)) or any(
        checked_hashes.get(key) != value for key, value in old_hashes.items()
    ):
        raise RuntimeError("V244_PRIOR_XAI_FOUR_FOLD_PROOF_INVALID")
    registry = read(DEFAULT_REGISTRY)
    atlas_path = Path(registry["output_atlas_path"]).resolve()
    if (not atlas_path.is_file()
        or sha(atlas_path) != context["atlas_spm_provenance"]["atlas_sha256"]):
        raise RuntimeError("V244_ATLAS_SOURCE_DRIFT")
    atlas = np.asarray(nib.load(atlas_path).dataobj).astype(np.int32)
    eligible = eligible_atlas_ids(atlas)
    if len(eligible) < 4:
        raise RuntimeError("V244_INSUFFICIENT_ATLAS_CONTROL_COVERAGE")
    if any(direction["top_region_id"] not in eligible
           for row in rows[:4] for direction in row["crossfit_directions"]):
        raise RuntimeError("V244_OLD_FOUR_FOLDS_WOULD_CHANGE")
    return {
        "schema_version": 1,
        "protocol_id": "METHOD_X_FASTTRACK_V244_POSTHOC_ATLAS_ONLY_CONTROL_ELIGIBILITY",
        "status": "POST_HOC_EXPLORATORY_REPAIR_BEFORE_FIFTH_FOLD_RECEIPT",
        "v243_amendment_sha256": sha(reports / V243_AMENDMENT),
        "v243_failed_journal_sha256": sha(prior_path),
        "context_sha256": sha(context_path),
        "atlas_sha256": sha(atlas_path),
        "eligible_atlas_ids": list(eligible),
        "excluded_atlas_ids": sorted(set(int(i) for i in np.unique(atlas) if i > 0) - set(eligible)),
        "prior_fold_receipt_sha256": old_hashes,
        "fifth_frozen_target_sha256": sha(target_path),
        "source_sha256": {name: sha(root / name) for name in SOURCE_FILES},
        "selection_rule": "HIGHEST_ATTRIBUTION_AMONG_ATLAS_REGIONS_WITH_THREE_1_5X_CONTROLS",
        "old_four_fold_results_changed": False,
        "outer_or_training_changed": False,
        "post_hoc_after_four_xai_fold_results": True,
        "claim_ceiling": "INTERNAL_EXPLORATORY_NO_DISEASE_REGION_OR_EXTERNAL_VALIDATION",
    }


def ensure_v244_amendment(root: Path, out: Path, *, create: bool) -> dict:
    path = out / "reports" / AMENDMENT
    if not path.is_file() and not create:
        raise RuntimeError("V244_AMENDMENT_MISSING")
    expected = amendment_payload(root, out)
    if not path.is_file():
        if (out / "reports" / STATE).exists() or (out / "reports/xai_v22_folds/fold_4.json").exists():
            raise RuntimeError("V244_AMENDMENT_TOO_LATE")
        immutable_json(path, expected)
    elif read(path) != expected:
        raise RuntimeError("V244_AMENDMENT_OR_SOURCE_DRIFT")
    return expected


class Supervisor(V243Supervisor):
    def __init__(self, root: Path, output_relative: Path, package_root: Path) -> None:
        super().__init__(root, output_relative, package_root)
        old_path, old_state = self.state_path, self.state
        self.state_path = self.reports / STATE
        self.state = (read(self.state_path) if self.state_path.is_file() else {
            "schema_version": 1,
            "pipeline_order": ["MX4", "MX5", "MX6", "MX7"],
            "migration": {"v243_failed_journal_sha256": sha(old_path),
                          "failed_xai_step": old_state["steps"]["mx6_v22_xai_crossfit"]},
            "steps": {key: value for key, value in old_state["steps"].items()
                      if key != "mx6_v22_xai_crossfit"},
        })
        self._mx5_verified = False

    def mx5(self) -> None:
        if self._mx5_verified:
            return
        ensure_v244_amendment(self.root, self.out, create=False)
        check_inputs(self.root, self.out)
        check_lock_compatible(self.root, self.out, hash_cache=False)
        verify_hybrid_pair_effect(self.out)
        self._mx5_verified = True
        print(json.dumps({"supervisor_step": "mx5_bound_reuse",
                          "status": "VERIFIED_PREVIOUSLY_COMPLETED_NO_OUTER_REEVALUATION"}), flush=True)

    def run_step(self, name: str, command: list[str], *, allowed: set[int] | None = None) -> int:
        if name == "mx6_v22_xai_crossfit":
            command = [sys.executable,
                       str(self.root / "scripts/mx_fasttrack_v2_4_4/entry_v244.py"),
                       "xai-run", "--project-root", str(self.root),
                       "--output-root", str(self.output_relative)]
            return V24Supervisor.run_step(self, name, command, allowed=allowed)
        return super().run_step(name, command, allowed=allowed)

    def mx7(self) -> None:
        super().mx7()
        self.run_step("mx7_v244_control_eligibility_appendix", [
            sys.executable, str(self.root / "scripts/mx_fasttrack_v2_4_4/entry_v244.py"),
            "appendix", "--project-root", str(self.root),
            "--output-root", str(self.output_relative),
        ])


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
        raise RuntimeError("V244_PROJECT_OR_OUTPUT_ROOT_INVALID")
    if foreign_method_x_writers(root):
        raise RuntimeError("V244_FOREIGN_METHOD_X_WRITER_ACTIVE")
    with coordinator_lock(out / "reports"):
        prior = os.environ.get(INTERNAL_PARENT_ENV)
        os.environ[INTERNAL_PARENT_ENV] = str(os.getpid())
        try:
            ensure_v244_amendment(root, out, create=True)
            _patch_parent_checkers()
            supervisor = Supervisor(root, args.output_root, DEFAULT_PACKAGE)
            getattr(supervisor, "mx7" if args.target == "all" else args.target)()
            print(json.dumps({"status": f"V244_{args.target.upper()}_COMPLETED",
                              "output_root": str(out)}, flush=True))
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
