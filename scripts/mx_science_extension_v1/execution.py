"""Execution lock and resumable, inner-only arm runner for the MX-05 add-on.

The design lock deliberately does not authorize GPU work.  This second lock is
created only after the parent MX-07 freeze and binds every source that can
affect extension predictions or their interpretation.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.mx_fasttrack_v2.run import read, sha
from scripts.mx_fasttrack_v2_1.supervise_v21 import immutable_json
from scripts.mx_science_extension_v1.audit_and_lock import (
    AUDIT_NAME, CONFIG, LOCK_NAME, OUTPUT, build_audit,
)
from scripts.mx_science_extension_v1.plan_arms import _arg, make_arm_commands
from scripts.mx_science_extension_v1.terminal import require_parent_mx7_terminal
from scripts.mx_science_extension_v1.process_log import run_logged

LOCK = "execution_preregistration.v1.json"
SOURCES = (
    "scripts/mx_science_extension_v1/__init__.py",
    "scripts/mx_science_extension_v1/analysis.py",
    "scripts/mx_science_extension_v1/audit_and_lock.py",
    "scripts/mx_science_extension_v1/baselines.py",
    "scripts/mx_science_extension_v1/execution.py",
    "scripts/mx_science_extension_v1/outer.py",
    "scripts/mx_science_extension_v1/plan_arms.py",
    "scripts/mx_science_extension_v1/process_log.py",
    "scripts/mx_science_extension_v1/random_label_train.py",
    "scripts/mx_science_extension_v1/report.py",
    "scripts/mx_science_extension_v1/selection.py",
    "scripts/mx_science_extension_v1/supervise_extension.py",
    "scripts/mx_science_extension_v1/terminal.py",
    "scripts/run_method_x_hybrid_orchestrator.py",
    "scripts/train_3d_mri.py",
    "src/evaluation/method_x_xai_forward.py",
    "src/models/backbones/lightweight_3d.py",
    "docs/method_x_mx05_science_extension_v1_FA.md",
)


def command_digest(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def execution_payload(root: Path) -> dict:
    terminal = require_parent_mx7_terminal(root)
    if terminal["v242_amendment_sha256"] is None:
        raise RuntimeError("EXTENSION_PARENT_V242_AMENDMENT_REQUIRED")
    reports = root / OUTPUT / "reports"
    audit = build_audit(root)
    design = read(reports / LOCK_NAME)
    if (audit.get("status") != "READY_INTERNAL_ONLY"
        or design.get("status") != "DESIGN_LOCKED_IMPLEMENTATION_PENDING"
        or design.get("gpu_authorized") is not False
        or design.get("design_config_sha256") != sha(root / CONFIG)
        or design.get("day1_audit_sha256") != sha(reports / AUDIT_NAME)
        or design.get("audit_source_sha256") != sha(root / "scripts/mx_science_extension_v1/audit_and_lock.py")
        or read(reports / AUDIT_NAME) != audit):
        raise RuntimeError("EXTENSION_DAY1_DESIGN_OR_AUDIT_DRIFT")
    commands = make_arm_commands(root)
    return {
        "schema_version": 1,
        "protocol_id": "METHOD_X_MX05_SCIENCE_EXTENSION_EXECUTION_V1",
        "status": "LOCKED_BEFORE_EXTENSION_GPU_AND_OUTER",
        "design_lock_sha256": sha(reports / LOCK_NAME),
        "day1_audit_sha256": sha(reports / AUDIT_NAME),
        "parent_freeze_sha256": terminal["freeze_sha256"],
        "parent_thesis_manifest_sha256": terminal["thesis_manifest_sha256"],
        "parent_v242_amendment_sha256": terminal["v242_amendment_sha256"],
        "parent_v243_amendment_sha256": terminal.get("v243_amendment_sha256"),
        "parent_v244_amendment_sha256": terminal.get("v244_amendment_sha256"),
        "source_sha256": {name: sha(root / name) for name in SOURCES},
        "arm_command_sha256": {name: command_digest(cmd)
                               for name, cmd in sorted(commands.items())},
        "arm_count": 6,
        "training_scope": "five_inner_folds_selection_only_exact_resume",
        "outer_selection_scope": "within_fold_inner_validation_only",
        "historical_outer_previously_inspected": True,
        "external_validation": False,
        "disease_specific_or_region_confirmation_allowed": False,
    }


def ensure_execution_lock(root: Path, *, create: bool) -> dict:
    path = root / OUTPUT / "reports" / LOCK
    if not path.is_file() and not create:
        raise RuntimeError("EXTENSION_EXECUTION_LOCK_MISSING")
    expected = execution_payload(root)
    if path.is_file():
        if read(path) != expected:
            raise RuntimeError("EXTENSION_EXECUTION_SOURCE_OR_PARENT_DRIFT")
    else:
        inner = root / OUTPUT / "inner_tasks"
        private = root / OUTPUT / "_private_resume"
        if (inner.exists() and any(inner.iterdir())) or private.exists():
            raise RuntimeError("EXTENSION_EXECUTION_LOCK_TOO_LATE")
        immutable_json(path, expected)
    return expected


def arm_evidence(root: Path, name: str, command: list[str]) -> dict:
    directory = Path(_arg(command, "--out_dir"))
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve()
    if not directory.is_relative_to((root / OUTPUT / "inner_tasks").resolve()):
        raise RuntimeError("EXTENSION_ARM_DIRECTORY_INVALID")
    contract_path, metrics_path = directory / "run_contract.json", directory / "three_d_metrics.json"
    contract_file, metrics = read(contract_path), read(metrics_path)
    contract, aggregate = contract_file.get("run_contract") or {}, metrics.get("aggregate") or {}
    parameters = contract.get("parameters") or {}
    if (contract_file.get("run_contract_digest") != contract.get("digest")
        or contract.get("seed") != int(_arg(command, "--seed"))
        or contract.get("folds") != 5 or contract.get("epochs") != 100
        or parameters.get("hybrid_variant") != _arg(command, "--hybrid_variant")
        or parameters.get("input_mode") != _arg(command, "--input_mode")
        or parameters.get("phase") != _arg(command, "--phase")
        or parameters.get("sanity_random_labels") is not ("--sanity_random_labels" in command)
        or parameters.get("hpo_selection_only") is not True
        or aggregate.get("status") != "COMPLETED"
        or aggregate.get("training_status") != "COMPLETED"
        or aggregate.get("hpo_selection_only") is not True
        or aggregate.get("outer_test_access_count") != 0
        or aggregate.get("completed_folds") != 5
        or aggregate.get("epochs_per_fold") != 20):
        raise RuntimeError(f"EXTENSION_ARM_CONTRACT_OR_METRICS_INVALID:{name}")
    rows = metrics.get("folds") or []
    if (len(rows) != 5 or {row.get("fold") for row in rows} != set(range(5))
        or any(row.get("outer_test_evaluation_count") != 0
               or row.get("hpo_selection_only") is not True for row in rows)):
        raise RuntimeError(f"EXTENSION_ARM_FOLD_METRICS_INVALID:{name}")
    checkpoints = []
    states = []
    for fold in range(5):
        cp = directory / f"method_x_3d_fold_{fold}.pt"
        state_path = directory / f"method_x_3d_fold_{fold}_state.json"
        state = read(state_path)
        digest = sha(cp)
        if (state.get("stage") != "FINALIZED"
            or state.get("contract_digest") != contract["digest"]
            or (state.get("details") or {}).get("checkpoint_sha256") != digest
            or (state.get("details") or {}).get("outer_test_access_count") != 0):
            raise RuntimeError(f"EXTENSION_ARM_FOLD_STATE_INVALID:{name}:{fold}")
        checkpoints.append(digest)
        states.append(sha(state_path))
    return {"status": "COMPLETED_INNER_ONLY", "name": name,
            "command_sha256": command_digest(command),
            "contract_sha256": sha(contract_path), "metrics_sha256": sha(metrics_path),
            "checkpoint_sha256": checkpoints, "fold_state_sha256": states}


def run_arms(root: Path) -> dict:
    lock = ensure_execution_lock(root, create=False)
    commands = make_arm_commands(root)
    receipts = root / OUTPUT / "reports" / "inner_arm_receipts"
    for name, command in sorted(commands.items()):
        if command_digest(command) != lock["arm_command_sha256"][name]:
            raise RuntimeError(f"EXTENSION_ARM_COMMAND_DRIFT:{name}")
        receipt_path = receipts / f"{name}.json"
        if receipt_path.is_file():
            if read(receipt_path) != arm_evidence(root, name, command):
                raise RuntimeError(f"EXTENSION_ARM_RECEIPT_DRIFT:{name}")
            continue
        # The trainer owns its per-epoch/per-fold exact-resume state.  A failed
        # subprocess leaves it intact and the same locked command is retried.
        print(json.dumps({"extension_arm": name, "status": "RUNNING_OR_RESUMING"}), flush=True)
        code = run_logged(command, cwd=root,
                          log_path=receipts.parent / "logs" / f"{name}.log").returncode
        if code:
            raise RuntimeError(f"EXTENSION_ARM_NOT_COMPLETE:{name}:EXIT_{code}")
        immutable_json(receipt_path, arm_evidence(root, name, command))
    return {"status": "SIX_ARMS_COMPLETED_INNER_ONLY", "receipt_sha256": {
        name: sha(receipts / f"{name}.json") for name in sorted(commands)}}


def preflight_arms(root: Path, *, require_lock: bool = True) -> None:
    lock = ensure_execution_lock(root, create=False) if require_lock else None
    for name, command in sorted(make_arm_commands(root).items()):
        print(json.dumps({"extension_preflight_arm": name, "status": "CHECKING"}), flush=True)
        if lock is not None and command_digest(command) != lock["arm_command_sha256"][name]:
            raise RuntimeError(f"EXTENSION_PREFLIGHT_COMMAND_DRIFT:{name}")
        completed = subprocess.run([*command, "--contract_preflight_only"], cwd=root,
                                   check=False, capture_output=True, text=True)
        if completed.returncode or '"status": "CONTRACT_PREFLIGHT_PASS"' not in completed.stdout:
            raise RuntimeError(f"EXTENSION_CONTRACT_PREFLIGHT_FAILED:{name}:"
                               f"{completed.stderr[-1000:]}:{completed.stdout[-1000:]}")
