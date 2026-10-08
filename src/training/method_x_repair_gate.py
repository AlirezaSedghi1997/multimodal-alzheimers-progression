"""Fail-closed repair gate for Method X agent closeouts.

This is not a replacement for the ten-layer training autosupervisor. It is the
Codex-style pre/post gate used before an agent tells the operator to re-run
GPU resume: every required layer must evaluate, absence of evidence is BLOCK
not PASS, and a missing layer cannot be skipped.

Phases:

* ``diagnose`` — read-only inventory of why resume is blocked.
* ``authorize-resume`` — every layer PASS before the operator command is valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.training.method_x_autosupervisor import (
    discover_method_x_processes,
    find_paused_progress_path,
    validate_resumable_checkpoint_evidence,
)
from src.training.method_x_resume import classify_source_only_progress_bind
from src.utils.io_helpers import atomic_write_json


REPAIR_GATE_SCHEMA_VERSION = 1
REQUIRED_REPAIR_LAYERS = (
    "process_lock",
    "scientific_invariants",
    "progress_inventory",
    "fold_immutability",
    "task2_evidence",
    "migration_bind_contract",
    "status_lifecycle",
    "blocker_archive",
    "regression_tests",
    "resume_command_contract",
)
DEFAULT_BATTERY = Path(
    "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729"
)
LOCKED_V3_YAML = Path("configs/experiments/method_x_claim_benchmark_v3.yaml")
PRE_REPAIR_FREEZE = Path(
    "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729"
    "/PRE_REPAIR_ARTIFACT_FREEZE.json"
)
FOCUSED_PYTEST = (
    "tests/test_method_x_resumable_evidence_source_only_hop.py",
    "tests/test_method_x_source_contract_migration_receipt.py",
    "tests/test_3d_training_contract.py",
    "tests/test_method_x_repair_gate.py",
    "tests/test_method_x_staging_trainer_wiring.py",
    "tests/test_method_x_checkpoint_staging_backend.py",
)
_NO_RESUME = re.compile(r"(^|[\s=])--no[_-]resume(\s|$)", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _layer(
    name: str, status: str, *, findings: Sequence[str] = (), evidence: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "findings": list(findings),
        "evidence": dict(evidence or {}),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_row(manifest: Mapping[str, Any], backbone: str) -> dict[str, Any] | None:
    rows = manifest.get("models")
    if not isinstance(rows, list):
        return None
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("backbone") == backbone
    ]
    if len(matches) != 1:
        return None
    row = matches[0]
    return dict(row) if isinstance(row, dict) else None


def _command_blob(row: Mapping[str, Any]) -> str:
    training = row.get("training") if isinstance(row.get("training"), Mapping) else {}
    command = training.get("command") if isinstance(training, Mapping) else None
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command or "")


def _process_layer() -> dict[str, Any]:
    rows = discover_method_x_processes()
    active = [row for row in rows if row.get("role") in {"trainer", "battery_runner"}]
    if active:
        return _layer(
            "process_lock",
            "BLOCK",
            findings=["ACTIVE_WRITER_PRESENT"],
            evidence={"active_count": len(active)},
        )
    return _layer("process_lock", "PASS", evidence={"active_count": 0})


def _scientific_layer(*, root: Path, manifest: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    findings: list[str] = []
    if manifest.get("method_x_pet_free") is not True:
        findings.append("METHOD_X_PET_FREE_CONTRACT_FAILED")
    blob = _command_blob(row)
    if _NO_RESUME.search(blob):
        findings.append("STORED_COMMAND_CONTAINS_NO_RESUME")
    yaml_path = root / LOCKED_V3_YAML
    if not yaml_path.is_file():
        findings.append("LOCKED_V3_YAML_MISSING")
    freeze = root / PRE_REPAIR_FREEZE
    if not freeze.is_file():
        findings.append("PRE_REPAIR_ARTIFACT_FREEZE_MISSING")
    locked = manifest.get("locked_inputs")
    if not isinstance(locked, Mapping) or not locked.get("config_sha256"):
        findings.append("LOCKED_INPUT_HASHES_INCOMPLETE")
    return _layer(
        "scientific_invariants",
        "BLOCK" if findings else "PASS",
        findings=findings,
        evidence={"pet_free": manifest.get("method_x_pet_free") is True},
    )


def inventory_model_progress(model_dir: Path) -> dict[str, Any]:
    leftover_finalized: list[int] = []
    current_training: list[int] = []
    missing_final_checkpoint: list[int] = []
    missing_current_progress: list[int] = []
    for state_path in sorted(model_dir.glob("method_x_3d_fold_*_state.json")):
        state = _read_json(state_path)
        if state is None:
            continue
        fold = state.get("fold")
        stage = state.get("stage")
        if type(fold) is not int:
            continue
        progress = model_dir / f"method_x_3d_fold_{fold}_progress.pt"
        final = model_dir / f"method_x_3d_fold_{fold}.pt"
        if stage == "FINALIZED":
            if not final.is_file():
                missing_final_checkpoint.append(fold)
            if progress.is_file():
                leftover_finalized.append(fold)
        elif stage == "TRAINING":
            current_training.append(fold)
            if find_paused_progress_path(model_dir=model_dir, fold=fold) is None:
                missing_current_progress.append(fold)
    return {
        "leftover_finalized_progress_folds": leftover_finalized,
        "current_training_folds": current_training,
        "missing_final_checkpoint_folds": missing_final_checkpoint,
        "missing_current_progress_folds": missing_current_progress,
    }


def _progress_inventory_layer(*, output_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    model_dir = output_root / str(row.get("output_dir") or "")
    inventory = inventory_model_progress(model_dir)
    findings: list[str] = []
    if inventory["missing_final_checkpoint_folds"]:
        findings.append("FINALIZED_CHECKPOINT_MISSING")
    if len(inventory["current_training_folds"]) != 1:
        findings.append("CURRENT_TRAINING_FOLD_NOT_UNIQUE")
    if inventory["missing_current_progress_folds"]:
        findings.append("CURRENT_PROGRESS_MISSING")
    # Leftover finalized progress is expected on this battery. It is inventory,
    # not a defect — the bind contract must ignore it.
    return _layer(
        "progress_inventory",
        "BLOCK" if findings else "PASS",
        findings=findings,
        evidence=inventory,
    )


def _fold_immutability_layer(*, output_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Live finalized .pt bytes must match the fold state's recorded SHA.

    Historical receipts may show pre!=post (the e1da→22bd hop rewrote
    checkpoints). That is not authority to mutate them again. The live binding
    is ``details.final_checkpoint_sha256`` on FINALIZED state.
    """

    model_dir = output_root / str(row.get("output_dir") or "")
    findings: list[str] = []
    checked: list[dict[str, Any]] = []
    for state_path in sorted(model_dir.glob("method_x_3d_fold_*_state.json")):
        state = _read_json(state_path)
        if not isinstance(state, dict) or state.get("stage") != "FINALIZED":
            continue
        fold = state.get("fold")
        details = state.get("details") if isinstance(state.get("details"), Mapping) else {}
        recorded = str(details.get("final_checkpoint_sha256") or "")
        final_path = model_dir / f"method_x_3d_fold_{fold}.pt"
        if type(fold) is not int or not final_path.is_file():
            findings.append(f"FINALIZED_CHECKPOINT_MISSING:{fold}")
            continue
        if len(recorded) != 64:
            findings.append(f"FINALIZED_CHECKPOINT_SHA_MISSING:{fold}")
            continue
        live = _sha256_file(final_path)
        if live != recorded:
            findings.append(f"FINALIZED_CHECKPOINT_LIVE_MISMATCH:{fold}")
        checked.append({"fold": fold, "sha256": live, "matches_state": live == recorded})
    if not checked:
        findings.append("NO_FINALIZED_FOLDS_CHECKED")
    return _layer(
        "fold_immutability",
        "BLOCK" if findings else "PASS",
        findings=findings,
        evidence={"checked": checked},
    )


def _task2_layer(*, output_root: Path, manifest: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        validated = validate_resumable_checkpoint_evidence(
            output_root=output_root, manifest=manifest, row=row
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return _layer(
            "task2_evidence",
            "BLOCK",
            findings=[str(exc).split(":", 1)[0]],
        )
    return _layer("task2_evidence", "PASS", evidence=validated)


def _migration_bind_contract_layer(*, root: Path) -> dict[str, Any]:
    findings: list[str] = []
    try:
        leftover = classify_source_only_progress_bind(
            filename="method_x_3d_fold_0_progress.pt",
            fold=0,
            action="keep_finalized",
            finalized_folds={0, 1},
        )
        current = classify_source_only_progress_bind(
            filename="method_x_3d_fold_2_progress.pt",
            fold=2,
            action="rewrite",
            finalized_folds={0, 1},
        )
    except RuntimeError as exc:
        return _layer("migration_bind_contract", "BLOCK", findings=[str(exc)])
    if leftover != "ignore_leftover_finalized":
        findings.append("LEFTOVER_FINALIZED_PROGRESS_NOT_IGNORED")
    if current != "bind_current":
        findings.append("CURRENT_PROGRESS_NOT_BOUND")
    trainer_path = root / "scripts" / "train_3d_mri.py"
    if not trainer_path.is_file():
        findings.append("TRAINER_SOURCE_MISSING")
    else:
        trainer = trainer_path.read_text(encoding="utf-8")
        if "classify_source_only_progress_bind" not in trainer:
            findings.append("TRAINER_DOES_NOT_CALL_PROGRESS_CLASSIFIER")
        if "seed_offloaded_best_state_sidecar" not in trainer:
            findings.append("TRAINER_DOES_NOT_SEED_OFFLOADED_BEST_STATE")
    try:
        classify_source_only_progress_bind(
            filename="method_x_3d_fold_2_progress.pt",
            fold=2,
            action="keep_finalized",
            finalized_folds={0, 1},
        )
        findings.append("CURRENT_PROGRESS_KEEP_FINALIZED_NOT_REJECTED")
    except RuntimeError as exc:
        if str(exc) != "RUN_CONTRACT_MIGRATION_PROGRESS_MARKED_FINALIZED":
            findings.append(str(exc))
    return _layer(
        "migration_bind_contract",
        "BLOCK" if findings else "PASS",
        findings=findings,
        evidence={"leftover": leftover, "current": current},
    )


def _status_layer(*, manifest: Mapping[str, Any], row: Mapping[str, Any], phase: str) -> dict[str, Any]:
    findings: list[str] = []
    status = str(row.get("status") or "")
    training = row.get("training") if isinstance(row.get("training"), Mapping) else {}
    training_status = str(training.get("status") or "")
    status_match = re.fullmatch(r"PAUSED_RESUMABLE_(\d+)_OF_5", status)
    training_match = re.fullmatch(r"PAUSED_RESUMABLE_(\d+)_OF_5", training_status)
    if (
        status_match is None
        or training_match is None
        or status_match.groups() != training_match.groups()
    ):
        findings.append(f"RESNET18_STATUS_NOT_RESUMABLE:{status}:{training_status}")
    top = str(manifest.get("status") or "")
    if top == "COMPLETED_WITH_BLOCKS":
        findings.append("MANIFEST_TOP_LEVEL_COMPLETED_WITH_BLOCKS")
    if phase == "diagnose":
        status_name = "BLOCK" if findings else "PASS"
        return _layer("status_lifecycle", status_name, findings=findings, evidence={"phase": phase})
    return _layer(
        "status_lifecycle",
        "BLOCK" if findings else "PASS",
        findings=findings,
        evidence={"row_status": status, "training_status": training_status, "manifest_status": top},
    )


def _blocker_archive_layer(*, output_root: Path, phase: str) -> dict[str, Any]:
    live = output_root / "models" / "resnet18_3d" / "BLOCKED_TRAINING_EXECUTION.json"
    if phase == "diagnose":
        if live.is_file():
            return _layer(
                "blocker_archive",
                "BLOCK",
                findings=["LIVE_BLOCKED_TRAINING_EXECUTION_PRESENT"],
                evidence={"path": str(live.relative_to(output_root)).replace("\\", "/")},
            )
        return _layer("blocker_archive", "PASS", evidence={"live_blocker": False})
    if live.is_file():
        return _layer(
            "blocker_archive",
            "BLOCK",
            findings=["LIVE_BLOCKED_TRAINING_EXECUTION_PRESENT"],
        )
    archives = list(
        (output_root / "archive" / "superseded_blockers").glob(
            "**/*BLOCKED_TRAINING_EXECUTION.json"
        )
    )
    if not archives:
        return _layer(
            "blocker_archive",
            "BLOCK",
            findings=["ARCHIVED_BLOCKER_EVIDENCE_MISSING"],
        )
    return _layer(
        "blocker_archive",
        "PASS",
        evidence={"archived_count": len(archives)},
    )


def _pytest_layer(*, root: Path, run_pytest: bool) -> dict[str, Any]:
    if not run_pytest:
        return _layer(
            "regression_tests",
            "BLOCK",
            findings=["REGRESSION_TESTS_NOT_EXECUTED"],
            evidence={"note": "Absence of pytest evidence is BLOCK, never PASS."},
        )
    command = [
        sys.executable,
        "-m",
        "pytest",
        *FOCUSED_PYTEST,
        "-q",
        "--tb=line",
    ]
    result = subprocess.run(
        command,
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        return _layer(
            "regression_tests",
            "BLOCK",
            findings=["FOCUSED_PYTEST_FAILED"],
            evidence={"returncode": result.returncode, "tail": (result.stdout or "")[-1000:]},
        )
    return _layer(
        "regression_tests",
        "PASS",
        evidence={"returncode": 0, "tests": list(FOCUSED_PYTEST)},
    )


def _resume_command_layer(*, row: Mapping[str, Any]) -> dict[str, Any]:
    findings: list[str] = []
    blob = _command_blob(row)
    if _NO_RESUME.search(blob):
        findings.append("STORED_COMMAND_CONTAINS_NO_RESUME")
    if "--no_resume" in blob:
        findings.append("STORED_COMMAND_CONTAINS_NO_RESUME")
    if str(row.get("backbone")) != "resnet18_3d":
        findings.append("TARGET_BACKBONE_NOT_RESNET18")
    return _layer(
        "resume_command_contract",
        "BLOCK" if findings else "PASS",
        findings=findings,
        evidence={"never_no_resume": True, "backbone": row.get("backbone")},
    )


def evaluate_repair_gate(
    *,
    root: Path,
    output_root: Path,
    phase: str,
    run_pytest: bool,
) -> dict[str, Any]:
    if phase not in {"diagnose", "authorize-resume"}:
        raise RuntimeError(f"REPAIR_GATE_PHASE_INVALID:{phase}")
    manifest_path = output_root / "battery_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest is None:
        layers = [_layer("scientific_invariants", "BLOCK", findings=["MANIFEST_MISSING_OR_INVALID"])]
        return _decision(phase=phase, layers=layers)
    row = _model_row(manifest, "resnet18_3d")
    if row is None:
        layers = [_layer("status_lifecycle", "BLOCK", findings=["RESNET18_ROW_MISSING"])]
        return _decision(phase=phase, layers=layers)

    layers = [
        _process_layer(),
        _scientific_layer(root=root, manifest=manifest, row=row),
        _progress_inventory_layer(output_root=output_root, row=row),
        _fold_immutability_layer(output_root=output_root, row=row),
        _task2_layer(output_root=output_root, manifest=manifest, row=row),
        _migration_bind_contract_layer(root=root),
        _status_layer(manifest=manifest, row=row, phase=phase),
        _blocker_archive_layer(output_root=output_root, phase=phase),
        _pytest_layer(root=root, run_pytest=run_pytest),
        _resume_command_layer(row=row),
    ]
    return _decision(phase=phase, layers=layers)


def _decision(*, phase: str, layers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluated = {str(layer.get("name")) for layer in layers}
    missing = [name for name in REQUIRED_REPAIR_LAYERS if name not in evaluated]
    blocking = sorted(
        {
            finding
            for layer in layers
            if layer.get("status") == "BLOCK"
            for finding in layer.get("findings") or []
        }
    )
    if missing and not blocking:
        blocking = [f"REPAIR_LAYER_NOT_EVALUATED:{name}" for name in missing]
    decision = "BLOCK" if blocking or missing else "AUTHORIZE_RESUME"
    if phase == "diagnose" and decision == "BLOCK":
        next_action = "FIX_THEN_RE_RUN_AUTHORIZE_RESUME"
    elif decision == "AUTHORIZE_RESUME":
        next_action = "OPERATOR_RESUME_BACKBONE_RESNET18"
    else:
        next_action = "WRITE_BLOCKER_AND_STOP"
    return {
        "schema_version": REPAIR_GATE_SCHEMA_VERSION,
        "phase": phase,
        "decision": decision,
        "created_at_utc": utc_now(),
        "blocking_reasons": blocking,
        "layers_not_evaluated": missing,
        "next_action": next_action,
        "fail_closed": True,
        "human_approval_required": False,
        "manual_override_allowed": False,
        "layers": list(layers),
    }


def persist_repair_gate_report(*, output_root: Path, report: Mapping[str, Any]) -> Path:
    phase = str(report.get("phase") or "unknown")
    report_path = output_root / "reports" / f"REPAIR_GATE_{phase.replace('-', '_').upper()}.json"
    atomic_write_json(report_path, dict(report))
    blocked_path = output_root / "reports" / "BLOCKED_REPAIR_GATE.json"
    if report.get("decision") == "BLOCK" and phase == "authorize-resume":
        atomic_write_json(blocked_path, dict(report))
    elif report.get("decision") == "AUTHORIZE_RESUME" and blocked_path.is_file():
        archive = output_root / "archive" / "superseded_blockers" / "repair_gate"
        archive.mkdir(parents=True, exist_ok=True)
        os.replace(blocked_path, archive / blocked_path.name)
    return report_path
