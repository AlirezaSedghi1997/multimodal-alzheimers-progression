"""Single-command, terminal-gated MX4--MX7 and scientific add-on supervisor.

Reuse or finish the parent v2.4.4 verified MX-07 freeze, then preregister and execute
the six exact-resume inner arms, lock fold-local choices, evaluate internal
outer folds under private receipts, and write the exploratory MX-07 addendum.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2.integrity import check_inputs
from scripts.mx_fasttrack_v2_1.supervise_v21 import (
    coordinator_lock, foreign_method_x_writers, immutable_json, verify_frozen_scope,
)
from scripts.mx_fasttrack_v2_4_2.supervise_v242 import (
    OUTPUT as PARENT_OUTPUT, ensure_v242_amendment,
)
from scripts.mx_fasttrack_v2_4_3.supervise_v243 import ensure_v243_amendment
from scripts.mx_fasttrack_v2_4_4.supervise_v244 import ensure_v244_amendment
from scripts.mx_science_extension_v1.audit_and_lock import OUTPUT
from scripts.mx_science_extension_v1.execution import (
    LOCK, ensure_execution_lock, execution_payload, preflight_arms, run_arms,
)
from scripts.mx_science_extension_v1.outer import ensure_outer_lock, run_outer
from scripts.mx_science_extension_v1.report import (
    run_analysis, run_report, run_xai_audit, xai_audit_payload,
)
from scripts.mx_science_extension_v1.process_log import console_log, run_logged
from scripts.mx_science_extension_v1.selection import lock_selection
from scripts.mx_science_extension_v1.terminal import require_parent_mx7_terminal

STAGES = ("parent_mx7", "execution_lock", "six_inner_arms", "inner_selection",
          "outer_lock", "outer_predictions", "paired_analysis", "xai_receipt_audit",
          "mx7_extension_report")
V244_CONSOLE_ERROR = "JSONEncoder.__init__() got an unexpected keyword argument 'flush'"


def _atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.new")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


class Supervisor:
    def __init__(self, root: Path) -> None:
        if root != ROOT.resolve():
            raise RuntimeError("EXTENSION_PROJECT_ROOT_INVALID")
        self.root = root
        self.reports = root / OUTPUT / "reports"
        self.journal_path = self.reports / "extension_supervisor_state.v1.json"
        self.journal = read(self.journal_path) if self.journal_path.is_file() else {
            "schema_version": 1, "pipeline_order": list(STAGES), "steps": {}}
        if self.journal.get("pipeline_order") != list(STAGES):
            raise RuntimeError("EXTENSION_SUPERVISOR_JOURNAL_DRIFT")

    def stage(self, name: str, action) -> dict:
        if name not in STAGES:
            raise RuntimeError("EXTENSION_UNKNOWN_STAGE")
        if foreign_method_x_writers(self.root):
            raise RuntimeError("EXTENSION_FOREIGN_METHOD_X_WRITER_ACTIVE")
        prior = self.journal["steps"].get(name) or {}
        now = datetime.now(timezone.utc).isoformat()
        self.journal["steps"][name] = {
            "status": "RUNNING", "attempts": int(prior.get("attempts", 0)) + 1,
            "started_at_utc": now,
        }
        _atomic(self.journal_path, self.journal)
        print(json.dumps({"extension_step": name, "status": "RUNNING"}), flush=True)
        try:
            result = action()
            if not isinstance(result, dict) or not isinstance(result.get("status"), str):
                raise RuntimeError(f"EXTENSION_STAGE_RESULT_INVALID:{name}")
            row = self.journal["steps"][name]
            row.update({"status": "COMPLETED", "result": result,
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()})
            _atomic(self.journal_path, self.journal)
            print(json.dumps({"extension_step": name, **result}, ensure_ascii=False), flush=True)
            return result
        except BaseException as exc:
            self.journal["steps"][name].update({
                "status": "PAUSED_OR_FAILED_RESUMABLE", "error": str(exc),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            _atomic(self.journal_path, self.journal)
            raise

    def parent(self) -> dict:
        out = self.root / PARENT_OUTPUT
        try:
            ensure_v242_amendment(self.root, out, create=False)
            if (out / "reports/fasttrack_v243_postouter_xai_coordinator_repair.json").is_file():
                ensure_v243_amendment(self.root, out, create=False)
            if (out / "reports/fasttrack_v244_posthoc_xai_atlas_eligibility.json").is_file():
                ensure_v244_amendment(self.root, out, create=False)
        except RuntimeError as exc:
            if (out / "reports/fasttrack_thesis_traceability.json").is_file():
                raise RuntimeError("EXTENSION_PARENT_FROZEN_BUT_NOT_VERIFIABLE") from exc
            raise
        try:
            return require_parent_mx7_terminal(self.root)
        except RuntimeError as exc:
            # Only a genuine nonterminal parent may be resumed.  An existing
            # freeze with drift is never silently converted into a new run.
            if (out / "reports/fasttrack_thesis_traceability.json").is_file():
                try:
                    verify_frozen_scope(self.root, out, xai_skipped=True)
                except (RuntimeError, ValueError, KeyError) as verification_error:
                    raise RuntimeError("EXTENSION_PARENT_FROZEN_BUT_NOT_VERIFIABLE") from verification_error
        command = [sys.executable,
                   str(self.root / "scripts/mx_fasttrack_v2_4_4/supervise_v244.py"),
                   "all", "--project-root", str(self.root)]
        completed = run_logged(command, cwd=self.root,
                               log_path=self.reports / "logs" / "parent_v244.log")
        if completed.returncode:
            # v244 is source-locked. Its final print passes flush to json.dumps.
            # Recover ONLY that exact console failure, after all artifacts verify.
            errors = []
            for line in (completed.stdout or "").splitlines():
                try:
                    errors.append(json.loads(line))
                except ValueError:
                    pass
            if (completed.returncode != 3 or not any(
                isinstance(row, dict) and row == {"status": "BLOCKED", "reason": V244_CONSOLE_ERROR}
                for row in errors
            )):
                raise RuntimeError(f"EXTENSION_PARENT_V244_NOT_COMPLETE:EXIT_{completed.returncode}")
            terminal = require_parent_mx7_terminal(self.root)
            verify_frozen_scope(self.root, out, xai_skipped=True)
            immutable_json(self.reports / "parent_v244_console_recovery.v1.json", {
                "status": "VERIFIED_TERMINAL_AFTER_CONSOLE_ONLY_FAILURE",
                "error": V244_CONSOLE_ERROR, "parent": terminal,
                "locked_parent_source_modified": False,
            })
        ensure_v244_amendment(self.root, out, create=False)
        return require_parent_mx7_terminal(self.root)

    def all(self, *, stop_after_lock: bool = False) -> dict:
        self.stage("parent_mx7", self.parent)
        def _lock() -> dict:
            path = self.reports / LOCK
            # Rehash the 21 GiB cache once per `all` invocation, not once per
            # arm/fold.  The parent source and split contract are checked too.
            check_inputs(self.root, self.root / PARENT_OUTPUT)
            verify_frozen_scope(self.root, self.root / PARENT_OUTPUT, xai_skipped=True)
            # Exercise the late XAI consumer before spending days on new training.
            xai_audit_payload(self.root)
            if not path.is_file():
                # These six read-only trainer contract checks run before source
                # preregistration and before any new GPU optimization.
                preflight_arms(self.root, require_lock=False)
            ensure_execution_lock(self.root, create=True)
            return {"status": "EXECUTION_LOCK_VERIFIED", "sha256": sha(path)}
        self.stage("execution_lock", _lock)
        if stop_after_lock:
            return {"status": "READY_TO_RESUME_SIX_INNER_ARMS_NO_TRAINING_STARTED"}
        self.stage("six_inner_arms", lambda: run_arms(self.root))
        self.stage("inner_selection", lambda: {
            "status": lock_selection(self.root)["status"],
            "sha256": sha(self.reports / "fold_local_hybrid_selection.v1.json")})
        self.stage("outer_lock", lambda: {
            "status": ensure_outer_lock(self.root, create=True)["status"],
            "sha256": sha(self.reports / "extension_outer_lock.v1.json")})
        self.stage("outer_predictions", lambda: run_outer(self.root))
        self.stage("paired_analysis", lambda: run_analysis(self.root))
        self.stage("xai_receipt_audit", lambda: run_xai_audit(self.root))
        return self.stage("mx7_extension_report", lambda: run_report(self.root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("all", "status", "preflight"))
    parser.add_argument("--stop-after-lock", action="store_true",
                        help="Run all through the execution lock, without starting training.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.stop_after_lock and args.target != "all":
        parser.error("--stop-after-lock requires all")
    root = args.project_root.resolve()
    if root != ROOT.resolve():
        raise RuntimeError("EXTENSION_PROJECT_ROOT_INVALID")
    supervisor = Supervisor(root)
    if args.target == "status":
        print(json.dumps(supervisor.journal, ensure_ascii=False, indent=2))
        return 0
    if foreign_method_x_writers(root):
        raise RuntimeError("EXTENSION_FOREIGN_METHOD_X_WRITER_ACTIVE")
    with coordinator_lock(supervisor.reports):
        if args.target == "preflight":
            check_inputs(root, root / PARENT_OUTPUT)
            execution_payload(root)
            verify_frozen_scope(root, root / PARENT_OUTPUT, xai_skipped=True)
            xai_audit_payload(root)
            preflight_arms(root, require_lock=(supervisor.reports / LOCK).is_file())
            print(json.dumps({"status": "EXTENSION_PREFLIGHT_PASS_NO_TRAINING"}), flush=True)
            return 0
        result = supervisor.all(stop_after_lock=args.stop_after_lock)
    if args.stop_after_lock:
        print(json.dumps(result), flush=True)
        return 0
    print(json.dumps({"status": "ALL_COMPLETED_POST_HOC_INTERNAL_EXPLORATORY",
                      "mx7_extension_manifest_sha256": result["manifest_sha256"],
                      "external_validation": False}, ensure_ascii=False))
    return 0


def cli() -> int:
    try:
        return main()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"all", "preflight"}:
        with console_log(ROOT / OUTPUT / "reports/logs/supervisor.log"):
            raise SystemExit(cli())
    else:
        raise SystemExit(cli())
