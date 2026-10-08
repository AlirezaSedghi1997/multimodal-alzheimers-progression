"""Autonomous, layered supervision for the Method X execution graph.

The supervisor replaces interactive/operator checkpoints with deterministic
machine decisions.  It is deliberately fail-closed: a missing contract,
unexpected writer, stale resumable state, or privacy/science invariant never
gets converted into a success merely because a process is still running.

The module has no dependency on the training runner.  That keeps it usable as
an independent scheduler/watchdog and avoids a circular import when the runner
invokes it before and after each model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from src.training.method_x_resume import FOLD_STAGES
from src.training.method_x_spm_snapshot import verify_spm_gate_snapshot
from src.training.storage_locality import (
    RESOLVED_PATH_STORAGE_REVALIDATION_FAILED,
    StoragePolicy,
    audit_storage,
    load_storage_policy,
    watch_blockers_are_live_inventory_race,
)
from src.utils.io_helpers import atomic_write_json


AUTOSUPERVISOR_SCHEMA_VERSION = 1
DEFAULT_POLICY_REL = Path("configs/policy/method_x_autosupervision.yaml")
DEFAULT_STORAGE_POLICY_REL = Path("configs/policy/method_x_storage_locality_v1.yaml")
DEFAULT_SNAPSHOT_REL = Path("results/method_x_registry/spm_gate_snapshot.json")
MANIFEST_NAME = "battery_manifest.json"
EXPECTED_CACHE = {
    "total_rows": 6213,
    "eligible_real_rows": 1669,
    "materialized_augmented_rows": 4544,
    "folds": 5,
    "seed": 13,
    "resolution": 128,
}
TRAINING_COMMAND_MARKERS = (
    "train_3d_mri.py",
    "run_method_x_claim_benchmark.py",
    "orchestrate_method_x_mri_5fold.py",
)
TERMINAL_MODEL_STATUSES = {"COMPLETED", "PROTOCOL_EXCLUDED_NOT_STARTED"}
# These layers are part of the scientific execution contract.  A local policy
# file may add descriptive metadata, but it can never remove one of these
# checks or turn fail-closed execution into an approval-driven workflow.
REQUIRED_SUPERVISION_LAYERS = (
    "process_lock",
    "storage_locality",
    "scientific_contract",
    "data_lineage",
    "checkpoint_integrity",
    "runtime_resources",
    "recovery_resume",
    "fold_finalization",
    "claim_privacy",
    "terminal_state",
)
_SPM_VERIFY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_reparse_point(path: Path) -> bool:
    """Return True for symlinks and Windows junction/reparse objects."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"PATH_INSPECTION_FAILED:{path}") from exc
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    )


def assert_path_beneath_root_without_reparse(
    root: Path,
    candidate: Path,
    *,
    outside_error: str,
    reparse_error: str,
    require_exists: bool = False,
    require_regular_file: bool = False,
) -> Path:
    """Lexically contain a path and reject every existing reparse component.

    ``Path.resolve`` follows junctions, so containment is established first on
    absolute lexical paths.  Existing components are then inspected with
    ``lstat`` before a final resolved-containment check.  The function performs
    no creation and is safe to call before lock acquisition.
    """

    root_abs = Path(os.path.abspath(os.fspath(root)))
    candidate_abs = Path(os.path.abspath(os.fspath(candidate)))
    try:
        common = Path(os.path.commonpath((os.fspath(root_abs), os.fspath(candidate_abs))))
    except ValueError as exc:
        raise RuntimeError(outside_error) from exc
    if os.path.normcase(os.fspath(common)) != os.path.normcase(os.fspath(root_abs)):
        raise RuntimeError(outside_error)

    relative = candidate_abs.relative_to(root_abs)
    current = root_abs
    paths = [root_abs]
    for part in relative.parts:
        current = current / part
        paths.append(current)
    for path in paths:
        if not os.path.lexists(path):
            break
        if _is_reparse_point(path):
            raise RuntimeError(f"{reparse_error}:{path}")

    try:
        resolved_root = root_abs.resolve(strict=True)
        resolved_candidate = candidate_abs.resolve(strict=False)
        resolved_candidate.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(outside_error) from exc

    if require_exists and not os.path.lexists(candidate_abs):
        raise RuntimeError(f"PATH_REQUIRED_BUT_MISSING:{candidate_abs}")
    if require_regular_file:
        try:
            metadata = os.lstat(candidate_abs)
        except OSError as exc:
            raise RuntimeError(f"PATH_REGULAR_FILE_REQUIRED:{candidate_abs}") from exc
        if not stat.S_ISREG(metadata.st_mode) or _is_reparse_point(candidate_abs):
            raise RuntimeError(f"PATH_REGULAR_FILE_REQUIRED:{candidate_abs}")
    return candidate_abs


_RESUMABLE_STATUS = re.compile(r"^PAUSED_RESUMABLE_(\d+)_OF_(\d+)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PAUSED_PROGRESS_BIND_ARTIFACT = "method_x_paused_progress_source_only_migration_bind"
PAUSED_PROGRESS_BIND_PREPARED_ARTIFACT = (
    "method_x_paused_progress_source_only_migration_bind_prepared"
)
PAUSED_PROGRESS_BIND_PROVENANCE = frozenset(
    {
        "TRAINER_EMITTED",
        "REPAIR_BIND_AFTER_SOURCE_ONLY_MIGRATION",
    }
)


def _payload_integrity_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a receipt-like JSON object the same way trainer receipts do."""

    body = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    encoded = json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_history_path(
    *, state: Mapping[str, Any], evidence_digest: str, error: str
) -> tuple[str, list[Mapping[str, Any]]]:
    """Walk a unique successor chain from Task 2 evidence to the live digest.

    A direct hop is the length-1 case. Later source-only migrations append
    another link (evidence → previous live → new live); requiring a single
    hop from evidence to live would fail closed on the next trainer edit.
    Forks, cycles, and missing successors are BLOCK, never a guessed path.
    """

    actual = str(state.get("contract_digest") or "")
    if _SHA256_RE.fullmatch(actual) is None:
        raise RuntimeError(error)
    if actual == evidence_digest:
        return actual, []
    history = state.get("source_history")
    if not isinstance(history, list):
        raise RuntimeError(error)
    hops: list[Mapping[str, Any]] = []
    current = evidence_digest
    seen = {current}
    while current != actual:
        successors = [
            item
            for item in history
            if isinstance(item, Mapping)
            and item.get("from_run_contract_digest") == current
        ]
        if len(successors) != 1:
            raise RuntimeError(error)
        nxt = str(successors[0].get("to_run_contract_digest") or "")
        if _SHA256_RE.fullmatch(nxt) is None or nxt in seen:
            raise RuntimeError(error)
        hops.append(successors[0])
        seen.add(nxt)
        current = nxt
    return actual, hops


def _source_history_hop_target(
    *, state: Mapping[str, Any], evidence_digest: str, error: str
) -> str:
    """Return the live fold digest if it matches evidence or hopped via source_history."""

    actual, _hops = _source_history_path(
        state=state, evidence_digest=evidence_digest, error=error
    )
    return actual


def _load_paused_progress_bind(
    *,
    root: Path,
    model_dir: Path,
    progress_name: str,
    fold: int,
    old_digest: str,
    new_digest: str,
) -> tuple[dict[str, Any], Path]:
    bind_candidate = paused_progress_bind_path(
        model_dir=model_dir,
        evidence_digest=old_digest,
        live_digest=new_digest,
        fold=fold,
    )
    bind_path = assert_path_beneath_root_without_reparse(
        root,
        bind_candidate,
        outside_error="RESUME_PATH_OUTSIDE_OUTPUT_ROOT",
        reparse_error="RESUME_REPARSE_POINT_REJECTED",
    )
    if not os.path.lexists(bind_path):
        raise RuntimeError("TASK2_PROGRESS_MIGRATION_BIND_MISSING")
    bind_path = assert_path_beneath_root_without_reparse(
        root,
        bind_path,
        outside_error="RESUME_PATH_OUTSIDE_OUTPUT_ROOT",
        reparse_error="RESUME_REPARSE_POINT_REJECTED",
        require_exists=True,
        require_regular_file=True,
    )
    bind = _read_json(bind_path)
    if bind is None:
        raise RuntimeError("TASK2_PROGRESS_MIGRATION_BIND_INVALID")
    required = {
        "schema_version": 1,
        "artifact": PAUSED_PROGRESS_BIND_ARTIFACT,
        "append_only": True,
        "progress_path": progress_name,
        "fold": fold,
        "old_run_contract_digest": old_digest,
        "new_run_contract_digest": new_digest,
    }
    if (
        bind.get("provenance_status") not in PAUSED_PROGRESS_BIND_PROVENANCE
        or bind.get("integrity_sha256") != _payload_integrity_sha256(bind)
    ):
        raise RuntimeError("TASK2_PROGRESS_MIGRATION_BIND_INVALID")
    for key, expected in required.items():
        if bind.get(key) != expected:
            raise RuntimeError("TASK2_PROGRESS_MIGRATION_BIND_MISMATCH")
    pre_sha256 = str(bind.get("pre_sha256") or "")
    post_sha256 = str(bind.get("post_sha256") or "")
    try:
        pre_byte_size = int(bind.get("pre_byte_size"))
        post_byte_size = int(bind.get("post_byte_size"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("TASK2_PROGRESS_MIGRATION_BIND_INVALID") from exc
    if (
        _SHA256_RE.fullmatch(pre_sha256) is None
        or _SHA256_RE.fullmatch(post_sha256) is None
        or pre_byte_size <= 0
        or post_byte_size <= 0
    ):
        raise RuntimeError("TASK2_PROGRESS_MIGRATION_BIND_INVALID")
    return bind, bind_path


def paused_progress_bind_path(
    *, model_dir: Path, evidence_digest: str, live_digest: str, fold: int
) -> Path:
    """Versioned sidecar next to the source-only migration receipts."""

    migration_key = f"{evidence_digest[:16]}_to_{live_digest[:16]}"
    return (
        model_dir
        / "source_only_contract_migrations"
        / f"{migration_key}.fold_{int(fold)}.progress_bind.json"
    )


def paused_progress_bind_prepared_path(
    *, model_dir: Path, evidence_digest: str, live_digest: str, fold: int
) -> Path:
    completed = paused_progress_bind_path(
        model_dir=model_dir,
        evidence_digest=evidence_digest,
        live_digest=live_digest,
        fold=fold,
    )
    return completed.with_name(
        completed.name.replace(".progress_bind.json", ".progress_bind_prepared.json")
    )


def _strict_sha256_file(path: Path) -> str:
    before = os.lstat(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = os.lstat(path)
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or getattr(before, "st_ctime_ns", None) != getattr(after, "st_ctime_ns", None)
    ):
        raise RuntimeError("RESUME_CHECKPOINT_CHANGED_DURING_HASH")
    return digest.hexdigest()


def paused_progress_relative_paths(*, output_dir: str, fold: int) -> tuple[str, str]:
    """H-side resume proofs for one TRAINING fold, preferred order first.

    Overlay staging stopped writing ``method_x_3d_fold_{N}_progress.pt``. The
    durable H copy is then the epoch/fold mirror. Legacy progress files stay
    first so existing Task 2 evidence and tests keep their recorded path.
    """

    rel_dir = str(output_dir or "").replace("\\", "/")
    fold_i = int(fold)
    return (
        f"{rel_dir}/method_x_3d_fold_{fold_i}_progress.pt",
        f"{rel_dir}/checkpoint_mirror/fold_{fold_i}/resume_current.pt",
    )


def find_paused_progress_path(*, model_dir: Path, fold: int) -> Path | None:
    """Return the first non-empty H-side progress proof under ``model_dir``."""

    fold_i = int(fold)
    candidates = (
        Path(model_dir) / f"method_x_3d_fold_{fold_i}_progress.pt",
        Path(model_dir) / "checkpoint_mirror" / f"fold_{fold_i}" / "resume_current.pt",
    )
    for candidate in candidates:
        try:
            metadata = os.lstat(candidate)
        except OSError:
            continue
        if stat.S_ISREG(metadata.st_mode) and int(metadata.st_size) > 0:
            return candidate
    return None


def resolve_paused_progress_checkpoint(
    *,
    output_root: Path,
    output_dir: str,
    fold: int,
) -> dict[str, Any]:
    """Hash the live H-side progress proof for Task 2 / orphan reconciliation."""

    root = Path(output_root)
    rel_dir = str(output_dir or "").replace("\\", "/")
    candidates = paused_progress_relative_paths(output_dir=rel_dir, fold=fold)
    chosen_rel: str | None = None
    chosen_path: Path | None = None
    for relative in candidates:
        candidate = assert_path_beneath_root_without_reparse(
            root,
            root / Path(relative),
            outside_error="ORPHAN_PROGRESS_OUTSIDE_OUTPUT_ROOT",
            reparse_error="ORPHAN_PROGRESS_REPARSE_POINT_REJECTED",
        )
        try:
            metadata = os.lstat(candidate)
        except OSError:
            continue
        if stat.S_ISREG(metadata.st_mode) and int(metadata.st_size) > 0:
            chosen_rel = relative
            chosen_path = candidate
            break
    if chosen_rel is None or chosen_path is None:
        raise RuntimeError(f"PATH_REQUIRED_BUT_MISSING:{root / Path(candidates[0])}")
    chosen_path = assert_path_beneath_root_without_reparse(
        root,
        chosen_path,
        outside_error="ORPHAN_PROGRESS_OUTSIDE_OUTPUT_ROOT",
        reparse_error="ORPHAN_PROGRESS_REPARSE_POINT_REJECTED",
        require_exists=True,
        require_regular_file=True,
    )
    progress_stat = os.lstat(chosen_path)
    if int(progress_stat.st_size) <= 0:
        raise RuntimeError("ORPHAN_PROGRESS_EMPTY")
    return {
        "path": chosen_rel,
        "sha256": _strict_sha256_file(chosen_path),
        "byte_size": int(progress_stat.st_size),
    }


def validate_resumable_checkpoint_evidence(
    *, output_root: Path, manifest: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one paused row against Task 2's exact reconciliation proof."""

    backbone = str(row.get("backbone") or "")
    status = str(row.get("status") or "")
    training = row.get("training") if isinstance(row.get("training"), Mapping) else {}
    training_status = str(training.get("status") or "")
    status_match = _RESUMABLE_STATUS.fullmatch(status)
    training_match = _RESUMABLE_STATUS.fullmatch(training_status)
    if status_match is None or training_match is None or status_match.groups() != training_match.groups():
        raise RuntimeError("STATUS_SUFFIX_INVALID")
    current_fold, total_folds = (int(value) for value in status_match.groups())
    if total_folds != 5 or current_fold < 0 or current_fold >= total_folds:
        raise RuntimeError("STATUS_FOLD_RANGE_INVALID")

    reconciliation = manifest.get("state_reconciliation")
    evidence_rows = (
        reconciliation.get("resumable_evidence")
        if isinstance(reconciliation, Mapping)
        else None
    )
    if not isinstance(evidence_rows, list):
        raise RuntimeError("TASK2_RESUMABLE_EVIDENCE_MISSING")
    matches = [
        evidence
        for evidence in evidence_rows
        if isinstance(evidence, Mapping) and evidence.get("backbone") == backbone
    ]
    if len(matches) != 1:
        raise RuntimeError("TASK2_RESUMABLE_EVIDENCE_NOT_UNIQUE")
    evidence = matches[0]
    expected_finalized = list(range(current_fold))
    contract_digest = str(evidence.get("state_contract_digest") or "")
    if (
        evidence.get("current_fold") != current_fold
        or evidence.get("finalized_folds") != expected_finalized
        or re.fullmatch(r"[0-9a-f]{64}", contract_digest) is None
    ):
        raise RuntimeError("TASK2_FOLD_OR_CONTRACT_BINDING_INVALID")

    output_dir = str(row.get("output_dir") or "")
    expected_output_dir = f"models/{backbone}"
    if output_dir.replace("\\", "/") != expected_output_dir:
        raise RuntimeError("MODEL_OUTPUT_DIR_INVALID")
    checkpoint = evidence.get("progress_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("TASK2_PROGRESS_EVIDENCE_INVALID")
    allowed_rel = set(
        paused_progress_relative_paths(
            output_dir=expected_output_dir, fold=current_fold
        )
    )
    expected_rel = str(checkpoint.get("path") or "").replace("\\", "/")
    if expected_rel not in allowed_rel:
        raise RuntimeError("TASK2_PROGRESS_PATH_INVALID")
    try:
        expected_size = int(checkpoint.get("byte_size"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("TASK2_PROGRESS_SIZE_INVALID") from exc
    expected_sha256 = str(checkpoint.get("sha256") or "")
    if expected_size <= 0 or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise RuntimeError("TASK2_PROGRESS_DIGEST_INVALID")

    root = Path(output_root)
    progress_path = assert_path_beneath_root_without_reparse(
        root,
        root / Path(expected_rel),
        outside_error="RESUME_PATH_OUTSIDE_OUTPUT_ROOT",
        reparse_error="RESUME_REPARSE_POINT_REJECTED",
        require_exists=True,
        require_regular_file=True,
    )
    progress_stat = os.lstat(progress_path)
    live_sha256 = _strict_sha256_file(progress_path)
    live_size = int(progress_stat.st_size)

    model_dir = root / Path(expected_output_dir)
    live_digests: list[str] = []
    for fold in expected_finalized:
        state_path = assert_path_beneath_root_without_reparse(
            root,
            model_dir / f"method_x_3d_fold_{fold}_state.json",
            outside_error="RESUME_PATH_OUTSIDE_OUTPUT_ROOT",
            reparse_error="RESUME_REPARSE_POINT_REJECTED",
            require_exists=True,
            require_regular_file=True,
        )
        state = _read_json(state_path)
        if (
            not isinstance(state, dict)
            or state.get("fold") != fold
            or state.get("stage") != "FINALIZED"
        ):
            raise RuntimeError(f"FINALIZED_FOLD_STATE_INVALID:{fold}")
        live_digests.append(
            _source_history_hop_target(
                state=state,
                evidence_digest=contract_digest,
                error=f"FINALIZED_FOLD_STATE_INVALID:{fold}",
            )
        )
    current_state_path = assert_path_beneath_root_without_reparse(
        root,
        model_dir / f"method_x_3d_fold_{current_fold}_state.json",
        outside_error="RESUME_PATH_OUTSIDE_OUTPUT_ROOT",
        reparse_error="RESUME_REPARSE_POINT_REJECTED",
        require_exists=True,
        require_regular_file=True,
    )
    current_state = _read_json(current_state_path)
    if (
        not isinstance(current_state, dict)
        or current_state.get("fold") != current_fold
        or current_state.get("stage") != "TRAINING"
    ):
        raise RuntimeError("CURRENT_FOLD_STATE_INVALID")
    live_digests.append(
        _source_history_hop_target(
            state=current_state,
            evidence_digest=contract_digest,
            error="CURRENT_FOLD_STATE_INVALID",
        )
    )
    if len(set(live_digests)) != 1:
        raise RuntimeError("TASK2_CONTRACT_DIGEST_INCONSISTENT")
    live_digest = live_digests[0]

    hop_info: dict[str, Any] | None = None
    if live_digest == contract_digest:
        if live_size != expected_size or live_sha256 != expected_sha256:
            raise RuntimeError("TASK2_PROGRESS_BYTES_MISMATCH")
    else:
        _actual, hops = _source_history_path(
            state=current_state,
            evidence_digest=contract_digest,
            error="CURRENT_FOLD_STATE_INVALID",
        )
        if not hops:
            raise RuntimeError("TASK2_PROGRESS_MIGRATION_BIND_MISSING")
        pre_sha256 = expected_sha256
        pre_byte_size = expected_size
        bind_paths: list[Path] = []
        for hop in hops:
            bind, bind_path = _load_paused_progress_bind(
                root=root,
                model_dir=model_dir,
                progress_name=progress_path.name,
                fold=current_fold,
                old_digest=str(hop.get("from_run_contract_digest") or ""),
                new_digest=str(hop.get("to_run_contract_digest") or ""),
            )
            if (
                bind.get("pre_sha256") != pre_sha256
                or int(bind.get("pre_byte_size")) != pre_byte_size
            ):
                raise RuntimeError("TASK2_PROGRESS_MIGRATION_BIND_MISMATCH")
            pre_sha256 = str(bind.get("post_sha256") or "")
            pre_byte_size = int(bind.get("post_byte_size"))
            bind_paths.append(bind_path)
        if pre_sha256 != live_sha256 or pre_byte_size != live_size:
            raise RuntimeError("TASK2_PROGRESS_MIGRATION_BIND_MISMATCH")
        hop_info = {
            "from_run_contract_digest": contract_digest,
            "to_run_contract_digest": live_digest,
            "bind_path": bind_paths[-1].relative_to(root).as_posix(),
            "bind_paths": [path.relative_to(root).as_posix() for path in bind_paths],
            "hop_count": len(hops),
        }

    return {
        "backbone": backbone,
        "current_fold": current_fold,
        "finalized_folds": expected_finalized,
        "state_contract_digest": live_digest,
        "progress_checkpoint": {
            "path": expected_rel,
            "sha256": live_sha256,
            "byte_size": live_size,
        },
        "source_only_evidence_hop": hop_info,
    }


def load_supervision_policy(root: Path) -> dict[str, Any]:
    """Load the immutable policy, falling back only to conservative defaults."""

    path = root / DEFAULT_POLICY_REL
    defaults: dict[str, Any] = {
        "schema_version": 1,
        "mode": "AUTONOMOUS_MULTI_LAYER_FAIL_CLOSED",
        "human_approval_required": False,
        "manual_override_allowed": False,
        "fail_closed": True,
        "poll_seconds": 15,
        "max_idle_seconds": 3600,
        "layers": list(REQUIRED_SUPERVISION_LAYERS),
    }
    if not path.is_file():
        return defaults
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {**defaults, "policy_load_error": True}
    if not isinstance(payload, dict):
        return {**defaults, "policy_load_error": True}
    merged = {**defaults, **payload}
    # The execution contract cannot be weakened by a local YAML edit.
    merged["human_approval_required"] = False
    merged["manual_override_allowed"] = False
    merged["fail_closed"] = True
    merged["mode"] = "AUTONOMOUS_MULTI_LAYER_FAIL_CLOSED"
    configured_layers = merged.get("layers")
    if not isinstance(configured_layers, list):
        configured_layers = []
    # Preserve any optional diagnostic layers while restoring every mandatory
    # layer if a stale or hand-edited policy omitted one.
    merged["layers"] = list(dict.fromkeys([*REQUIRED_SUPERVISION_LAYERS, *map(str, configured_layers)]))
    return merged


def discover_method_x_processes(*, exclude_pids: Sequence[int] = ()) -> list[dict[str, Any]]:
    """Return sanitized metadata for active Method X writers.

    Only a role and command digest are persisted; full command lines are not
    written to reports.  ``psutil`` is optional because the supervisor must
    still be able to run a contract/data audit in a minimal environment.
    """

    excluded = {int(pid) for pid in exclude_pids}
    excluded.add(os.getpid())
    rows: list[dict[str, Any]] = []
    try:
        import psutil
    except ImportError:
        return rows
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        try:
            pid = int(process.info.get("pid") or -1)
            if pid <= 0 or pid in excluded:
                continue
            command = " ".join(str(part) for part in (process.info.get("cmdline") or []))
            folded = command.casefold()
            marker = next((item for item in TRAINING_COMMAND_MARKERS if item.casefold() in folded), None)
            if marker is None:
                continue
            process_name = str(process.info.get("name") or "unknown")
            # A PowerShell command line can contain the text of a Python
            # command while merely invoking it.  Do not mistake that shell
            # wrapper for a second writer; the actual Python child is the
            # authoritative process.
            if process_name.casefold() in {"pwsh.exe", "powershell.exe", "cmd.exe"} and "python" not in folded:
                continue
            role = "trainer" if marker == "train_3d_mri.py" else "battery_runner"
            rows.append(
                {
                    "pid": pid,
                    "parent_pid": int(process.info.get("ppid") or -1),
                    "name": process_name,
                    "role": role,
                    "marker": marker,
                    "command_sha256": hashlib.sha256(command.encode("utf-8", "replace")).hexdigest(),
                    "active": True,
                }
            )
        except (psutil.Error, OSError, TypeError, ValueError):
            continue
    return rows


def _layer(name: str, status: str, *, findings: Sequence[str] = (), evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "findings": list(findings),
        "evidence": dict(evidence or {}),
    }


def _process_layer(
    *, process_rows: Sequence[Mapping[str, Any]], expected_pid: int | None,
) -> dict[str, Any]:
    active = [row for row in process_rows if bool(row.get("active", True))]
    allowed = {os.getpid()}
    if expected_pid is not None:
        allowed.add(int(expected_pid))
    # On Windows a virtual-environment ``python.exe`` can remain as a launcher
    # while the real base interpreter is its child.  Treat that verified
    # descendant chain as one logical writer, while still rejecting siblings
    # and unrelated trainers fail-closed.
    if expected_pid is not None:
        changed = True
        while changed:
            changed = False
            for row in active:
                try:
                    row_pid = int(row.get("pid", -1))
                    parent_pid = int(row.get("parent_pid", -1))
                except (TypeError, ValueError):
                    continue
                if row_pid > 0 and parent_pid in allowed and row_pid not in allowed:
                    allowed.add(row_pid)
                    changed = True

    unexpected: list[Mapping[str, Any]] = []
    for row in active:
        try:
            row_pid = int(row.get("pid", -1))
        except (TypeError, ValueError):
            row_pid = -1
        if row_pid not in allowed:
            unexpected.append(row)
    if unexpected:
        return _layer(
            "process_lock",
            "BLOCK",
            findings=["ACTIVE_WRITER_PRESENT"],
            evidence={"active_count": len(active), "unexpected": list(unexpected)},
        )
    if expected_pid is None and len(active) > 1:
        return _layer(
            "process_lock",
            "BLOCK",
            findings=["MULTIPLE_METHOD_X_WRITERS"],
            evidence={"active_count": len(active)},
        )
    return _layer("process_lock", "PASS", evidence={"active_count": len(active)})


def _load_lexical_storage_policy(root: Path) -> tuple[StoragePolicy | None, dict[str, Any] | None]:
    """Load storage policy once from an absolute H-side lexical project root."""

    if not root.is_absolute() or root.drive.casefold() != "h:":
        return None, _layer("storage_locality", "BLOCK", findings=["STORAGE_POLICY_ROOT_NOT_H"])
    try:
        return load_storage_policy(root / DEFAULT_STORAGE_POLICY_REL), None
    except ValueError as exc:
        return None, _layer("storage_locality", "BLOCK", findings=[str(exc)])


def _storage_layer(*, root: Path, output_root: Path, storage_policy: StoragePolicy) -> dict[str, Any]:
    """Read storage evidence only; this layer never materializes run state."""

    try:
        report = audit_storage(root=root, output_root=output_root, policy=storage_policy)
    except (OSError, ValueError) as exc:
        return _layer("storage_locality", "BLOCK", findings=[f"STORAGE_AUDIT_UNREADABLE:{type(exc).__name__}"])
    return _layer(
        "storage_locality",
        "PASS" if report["decision"] == "PASS" else "BLOCK",
        findings=list(report["blockers"]),
        evidence=report,
    )


def _contract_layer(
    *, root: Path, output_root: Path, config: Mapping[str, Any] | None, phase: str,
) -> dict[str, Any]:
    manifest_path = output_root / MANIFEST_NAME
    manifest = _read_json(manifest_path)
    if manifest is None:
        return _layer("scientific_contract", "BLOCK", findings=["MANIFEST_MISSING_OR_INVALID"])
    findings: list[str] = []
    protocol = manifest.get("locked_protocol") or {}
    for key, expected in (("resolution", 128), ("folds", 5), ("seed", 13)):
        try:
            observed = int(protocol.get(key, -1))
        except (TypeError, ValueError):
            observed = -1
        if observed != expected:
            findings.append(f"LOCKED_{key.upper()}_MISMATCH")
    if manifest.get("method_x_pet_free") is not True:
        findings.append("METHOD_X_PET_FREE_CONTRACT_FAILED")
    if manifest.get("final_claim_allowed") is True:
        findings.append("FINAL_CLAIM_MUST_REMAIN_BLOCKED")
    locked = manifest.get("locked_inputs")
    if not isinstance(locked, dict) or any(not locked.get(key) for key in ("cache_sha256", "split_sha256", "config_sha256", "device_policy_sha256")):
        findings.append("LOCKED_INPUT_HASHES_INCOMPLETE")
    snapshot_path = root / DEFAULT_SNAPSHOT_REL
    cache_key = str(snapshot_path.resolve())
    now = time.monotonic()
    cached = _SPM_VERIFY_CACHE.get(cache_key)
    # A full QC/NIfTI inventory is intentionally periodic during a long
    # training child; the cheap snapshot digest is still checked every poll.
    if phase == "watch" and cached is not None and now - cached[0] < 60.0:
        snapshot = cached[1]
    else:
        snapshot = verify_spm_gate_snapshot(snapshot_path, root=root, verify_sources=True)
        _SPM_VERIFY_CACHE[cache_key] = (now, snapshot)
    if not snapshot.get("verified") or not snapshot.get("ready"):
        findings.append("SPM_SNAPSHOT_NOT_VERIFIED")
    if config is not None:
        for key, expected in (("resolution", 128), ("folds", 5), ("seed", 13)):
            try:
                if int(config.get(key, -1)) != expected:
                    findings.append(f"CONFIG_{key.upper()}_MISMATCH")
            except (TypeError, ValueError):
                findings.append(f"CONFIG_{key.upper()}_INVALID")
    status = "BLOCK" if findings else "PASS"
    return _layer(
        "scientific_contract",
        status,
        findings=findings,
        evidence={
            "manifest": str(manifest_path.relative_to(root)) if manifest_path.is_relative_to(root) else str(manifest_path),
            "manifest_status": manifest.get("status"),
            "snapshot": {
                "status": snapshot.get("status"),
                "verified": bool(snapshot.get("verified")),
                "ready": bool(snapshot.get("ready")),
                "sha256": snapshot.get("snapshot_sha256"),
            },
        },
    )


def _data_layer(*, output_root: Path) -> dict[str, Any]:
    manifest = _read_json(output_root / MANIFEST_NAME) or {}
    observed = manifest.get("cache_characterization")
    if not isinstance(observed, dict):
        return _layer("data_lineage", "BLOCK", findings=["CACHE_CHARACTERIZATION_MISSING"])
    mismatches = [key for key, expected in EXPECTED_CACHE.items() if observed.get(key) != expected]
    findings = [f"CACHE_{key.upper()}_MISMATCH" for key in mismatches]
    return _layer(
        "data_lineage",
        "BLOCK" if findings else "PASS",
        findings=findings,
        evidence={"expected": EXPECTED_CACHE, "observed": {key: observed.get(key) for key in EXPECTED_CACHE}},
    )


def _runtime_layer(*, phase: str, config: Mapping[str, Any] | None) -> dict[str, Any]:
    hard_stop = float((config or {}).get("hardware", {}).get("hard_stop_vram_gb", 7.5))
    evidence: dict[str, Any] = {"hard_stop_vram_gb": hard_stop, "cuda_available": None}
    findings: list[str] = []
    try:
        import torch

        evidence["cuda_available"] = bool(torch.cuda.is_available())
        if phase in {"before_model", "watch", "training"} and not torch.cuda.is_available():
            findings.append("CUDA_REQUIRED_FOR_TRAINING")
    except (ImportError, RuntimeError) as exc:
        evidence["torch_error"] = type(exc).__name__
        if phase in {"before_model", "watch", "training"}:
            findings.append("RUNTIME_PROBE_UNAVAILABLE")
    return _layer("runtime_resources", "BLOCK" if findings else ("WARN" if evidence["cuda_available"] is False else "PASS"), findings=findings, evidence=evidence)


def _recovery_layer(*, output_root: Path, process_rows: Sequence[Mapping[str, Any]], expected_pid: int | None) -> dict[str, Any]:
    manifest = _read_json(output_root / MANIFEST_NAME) or {}
    model_rows = [row for row in manifest.get("models", []) if isinstance(row, Mapping)]
    active_pids: set[int] = set()
    for row in process_rows:
        if not bool(row.get("active", True)):
            continue
        try:
            active_pids.add(int(row.get("pid", -1)))
        except (TypeError, ValueError):
            continue
    if expected_pid is not None:
        active_pids.add(int(expected_pid))
    pause_reasons: list[str] = []
    hard_findings: list[str] = []
    resumable: list[dict[str, Any]] = []
    controlled_retry: list[dict[str, Any]] = []
    retry_rows = [
        row for row in model_rows if row.get("status") == "TRANSIENT_IO_RETRY_PENDING"
    ]
    if len(retry_rows) > 1:
        hard_findings.append("MULTIPLE_TRANSIENT_IO_RETRIES_PENDING")
    elif retry_rows:
        retry_row = retry_rows[0]
        training = retry_row.get("training") if isinstance(retry_row.get("training"), Mapping) else {}
        authorization = training.get("retry_authorization")
        history = training.get("attempts")
        command = training.get("command")
        last_failure = (
            history[-1].get("failure")
            if isinstance(history, list) and history and isinstance(history[-1], Mapping)
            else None
        )
        command_digest = (
            hashlib.sha256(
                json.dumps(
                    [str(value) for value in command],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            if isinstance(command, list) and command
            else None
        )
        if (
            training.get("status") != "TRANSIENT_IO_RETRY_PENDING"
            or not isinstance(authorization, Mapping)
            or authorization.get("status") != "PENDING"
            or authorization.get("category") != "TRANSIENT_IO"
            or authorization.get("next_attempt") not in {2, 3}
            or authorization.get("command_sha256") != command_digest
            or not isinstance(last_failure, Mapping)
            or last_failure.get("category") != "TRANSIENT_IO"
            or last_failure.get("retryable") is not True
            or authorization.get("failure_fingerprint_sha256")
            != last_failure.get("fingerprint_sha256")
        ):
            hard_findings.append("TRANSIENT_IO_RETRY_STATE_INVALID")
        else:
            controlled_retry.append(
                {
                    "backbone": retry_row.get("backbone"),
                    "next_attempt": authorization.get("next_attempt"),
                    "command_sha256": command_digest,
                    "failure_fingerprint_sha256": last_failure.get(
                        "fingerprint_sha256"
                    ),
                }
            )

    for row in model_rows:
        status = str(row.get("status") or "")
        paused_resumable = status in {
            "TRAINING_PAUSED_RESUMABLE",
            "PAUSED_RESUMABLE",
        } or status.startswith("PAUSED_RESUMABLE_")
        if status != "TRAINING_RUNNING" and not paused_resumable:
            continue
        training = row.get("training") if isinstance(row.get("training"), dict) else {}
        pid = training.get("pid")
        # ``expected_pid`` is supplied by the supervised child launcher.  It
        # is authoritative even when the child command line is not visible to
        # psutil yet (a normal Windows process-start race).
        try:
            writer_pid = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            writer_pid = None
        writer_active = (
            expected_pid is not None
            or (writer_pid is not None and writer_pid in active_pids)
            or bool(active_pids)
        )
        if status == "TRAINING_RUNNING" and not writer_active:
            # A dead owner is not sufficient evidence for resume. Task 2 (or
            # the later exact-checkpoint journal) must first bind the current
            # fold, state contract and checkpoint bytes.
            hard_findings.append("ORPHANED_TRAINING_REQUIRES_RECONCILIATION")
        elif paused_resumable:
            # During a supervised child run, other pre-existing paused rows are
            # expected serial-schedule state and must not be re-hashed on every
            # watch tick. The active writer is still checked by process_lock.
            if writer_active or controlled_retry:
                continue
            try:
                validated = validate_resumable_checkpoint_evidence(
                    output_root=output_root, manifest=manifest, row=row
                )
            except (OSError, RuntimeError, ValueError) as exc:
                code = str(exc).split(":", 1)[0]
                hard_findings.append(
                    f"RESUMABLE_EVIDENCE_INVALID:{row.get('backbone')}:{code}"
                )
            else:
                pause_reasons.append("RESUMABLE_PAUSED_ROW")
                resumable.append(validated)
    if hard_findings:
        return _layer(
            "recovery_resume",
            "BLOCK",
            findings=hard_findings,
            evidence={"resumable": resumable, "controlled_retry": controlled_retry},
        )
    if pause_reasons:
        return _layer(
            "recovery_resume",
            "PAUSE",
            findings=sorted(set(pause_reasons)),
            evidence={"resumable": resumable, "controlled_retry": controlled_retry},
        )
    return _layer(
        "recovery_resume",
        "PASS",
        evidence={"resumable": [], "controlled_retry": controlled_retry},
    )


def _checkpoint_integrity_layer(*, output_root: Path) -> dict[str, Any]:
    """Layer 4 — no truncated or unverified payload may become a resume source.

    Reads only the journal/mirror bookkeeping written alongside each fold. A
    zero-byte or hash-mismatched payload is a BLOCK, because silently resuming
    from it would fabricate training history.
    """

    findings: list[str] = []
    checked: list[dict[str, Any]] = []
    for state_path in sorted(output_root.glob("**/checkpoint_mirror_state.json")):
        state = _read_json(state_path)
        if state is None:
            findings.append("MIRROR_STATE_UNREADABLE")
            continue
        mirror = state.get("mirror") if isinstance(state.get("mirror"), dict) else {}
        recorded_digest = str(mirror.get("sha256") or "")
        relative = str(mirror.get("relative_path") or "")
        if not relative:
            continue
        payload = output_root / relative
        entry: dict[str, Any] = {"relative_path": relative}
        if not payload.exists():
            findings.append("MIRRORED_CHECKPOINT_MISSING")
            entry["present"] = False
            checked.append(entry)
            continue
        size = payload.stat().st_size
        entry["size_bytes"] = size
        if size == 0:
            findings.append("MIRRORED_CHECKPOINT_ZERO_BYTE")
        elif len(recorded_digest) == 64:
            actual = _sha256_file(payload)
            entry["digest_matches"] = actual == recorded_digest
            if actual != recorded_digest:
                findings.append("MIRRORED_CHECKPOINT_DIGEST_MISMATCH")
        else:
            findings.append("MIRRORED_CHECKPOINT_DIGEST_MISSING")
        checked.append(entry)
    stray_temps: list[str] = []
    for temp_path in sorted(output_root.glob("**/*.tmp")):
        if not temp_path.is_file():
            continue
        # `archive/` is the designated quarantine for stale runtime evidence.
        # Step A of the corrective plan requires incomplete temps to be moved
        # there with hash and timestamp rather than deleted, so their presence
        # is the intended end state, not a violation.
        if "archive" in temp_path.relative_to(output_root).parts:
            continue
        stray_temps.append(temp_path.name)
    # A stray temp in a live directory means an atomic replace did not finish.
    # It is surfaced, not fail-closed: the guarantee that matters -- that a
    # partial temp is never chosen as a resume source -- is enforced in
    # `plan_recovery` and covered by its own tests. Blocking the whole battery
    # over a leftover from an already-completed model would be a false stop.
    status = "BLOCK" if findings else ("WARN" if stray_temps else "PASS")
    if stray_temps:
        findings.append("STRAY_TEMP_PAYLOAD_IN_LIVE_DIRECTORY")
    return _layer(
        "checkpoint_integrity",
        status,
        findings=sorted(set(findings)),
        evidence={
            "checked_count": len(checked),
            "checked": checked[:20],
            "stray_temp_count": len(stray_temps),
            "stray_temps": stray_temps[:10],
        },
    )


def _fold_finalization_layer(*, output_root: Path) -> dict[str, Any]:
    """Layer 8 — a fold may not be counted complete before its outer test is locked.

    Guards the two ways history gets falsified: a fold marked FINALIZED without
    a committed outer result, and an outer evaluation recorded more than once.
    """

    findings: list[str] = []
    inspected = 0
    for state_path in sorted(output_root.glob("**/fold_state.json")):
        state = _read_json(state_path)
        if state is None:
            findings.append("FOLD_STATE_UNREADABLE")
            continue
        inspected += 1
        stage = state.get("stage")
        if stage not in FOLD_STAGES:
            findings.append("FOLD_STAGE_UNKNOWN")
            continue
        locked = state_path.parent / "locked_outer_test_result.json"
        if stage in {"OUTER_EVAL_DONE", "FINALIZED"} and not locked.exists():
            findings.append("FINALIZED_FOLD_MISSING_COMMITTED_OUTER_RESULT")
        if stage == "TRAINING" and locked.exists():
            findings.append("OUTER_RESULT_PRESENT_BEFORE_CHECKPOINT_LOCK")
        if locked.exists():
            result = _read_json(locked) or {}
            if int(result.get("evaluation_count", 0)) != 1:
                findings.append("OUTER_EVALUATION_COUNT_NOT_ONE")
    return _layer(
        "fold_finalization",
        "BLOCK" if findings else "PASS",
        findings=sorted(set(findings)),
        evidence={"folds_inspected": inspected},
    )


def _claim_privacy_layer(*, output_root: Path) -> dict[str, Any]:
    manifest = _read_json(output_root / MANIFEST_NAME) or {}
    findings: list[str] = []
    if manifest.get("final_claim_allowed") is True:
        findings.append("FINAL_CLAIM_MUST_REMAIN_BLOCKED")
    science = manifest.get("science_audit") if isinstance(manifest.get("science_audit"), dict) else {}
    if science.get("diagnostic_claim_allowed") is not False:
        findings.append("DIAGNOSTIC_CLAIM_BOUNDARY_NOT_EXPLICIT")
    for row in manifest.get("models", []):
        if row.get("uses_pet_as_input") is True or row.get("uses_raw_identifiers") is True:
            findings.append("FORBIDDEN_MODEL_INPUT_DECLARED")
    return _layer("claim_privacy", "BLOCK" if findings else "PASS", findings=sorted(set(findings)))


def _terminal_state_layer(*, output_root: Path, phase: str) -> dict[str, Any]:
    """Prevent a post-flight success label when any model is unfinished."""

    if phase != "postflight":
        return _layer("terminal_state", "PASS", evidence={"checked": False})
    manifest = _read_json(output_root / MANIFEST_NAME)
    if manifest is None:
        return _layer("terminal_state", "BLOCK", findings=["MANIFEST_MISSING_OR_INVALID"])
    rows = manifest.get("models")
    if not isinstance(rows, list) or not rows:
        return _layer("terminal_state", "BLOCK", findings=["MODEL_ROWS_MISSING"])
    nonterminal = [
        {"backbone": row.get("backbone"), "status": row.get("status")}
        for row in rows
        if row.get("status") not in TERMINAL_MODEL_STATUSES
    ]
    if nonterminal:
        return _layer(
            "terminal_state",
            "BLOCK",
            findings=["BATTERY_NOT_TERMINAL"],
            evidence={"nonterminal": nonterminal},
        )
    return _layer("terminal_state", "PASS", evidence={"checked": True, "model_count": len(rows)})


def _supervision_report(
    *,
    phase: str,
    layers: Sequence[Mapping[str, Any]],
    policy_payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    """Materialize a deterministic fail-closed decision from evaluated layers."""

    blocking_reasons = sorted({finding for layer in layers if layer["status"] == "BLOCK" for finding in layer["findings"]})
    pause_reasons = sorted({finding for layer in layers if layer["status"] == "PAUSE" for finding in layer["findings"]})
    # A layer that never ran cannot be evidence of safety. Early-return paths
    # (storage BLOCK, unresolvable path) legitimately evaluate a subset, so an
    # incomplete set is only an additional blocker when nothing else blocked.
    evaluated = {layer["name"] for layer in layers}
    missing_layers = [name for name in REQUIRED_SUPERVISION_LAYERS if name not in evaluated]
    if missing_layers and not blocking_reasons:
        blocking_reasons = ["REQUIRED_SUPERVISION_LAYER_NOT_EVALUATED"]
    if blocking_reasons:
        decision = "BLOCK"
        next_action = "WRITE_BLOCKER_AND_STOP"
    elif pause_reasons:
        decision = "PAUSE_RESUMABLE"
        next_action = "RESUME_EXACT_CONTRACT"
    elif phase in {"postflight", "after_model"}:
        decision = "COMPLETE"
        next_action = "RECORD_AND_ADVANCE"
    else:
        decision = "CONTINUE"
        next_action = "RUN_NEXT_LOCKED_STAGE"
    return {
        "schema_version": AUTOSUPERVISOR_SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
        "created_at_utc": utc_now(),
        "phase": phase,
        "decision": decision,
        "blocking_reasons": blocking_reasons + pause_reasons,
        "warnings": sorted({finding for layer in layers if layer["status"] == "WARN" for finding in layer["findings"]}),
        "layers": list(layers),
        "layers_not_evaluated": missing_layers,
        "automation": {
            "mode": "AUTONOMOUS_MULTI_LAYER_FAIL_CLOSED",
            "human_approval_required": False,
            "manual_override_allowed": False,
            "fail_closed": True,
            "next_action": next_action,
            "policy_sha256": hashlib.sha256(json.dumps(dict(policy_payload), sort_keys=True).encode("utf-8")).hexdigest(),
            "layers_evaluated": [layer["name"] for layer in layers],
            "required_layers": list(REQUIRED_SUPERVISION_LAYERS),
        },
        "processes": [{key: row.get(key) for key in ("pid", "role", "marker", "active")} for row in rows],
        "output_root": str(output_root),
    }


def _resolve_supervision_path(path: Path) -> Path:
    """Resolve a path only after lexical storage validation has passed."""

    return path.resolve()


def watch_resolved_storage_is_inventory_race(
    *, phase: str, resolved_storage_layer: Mapping[str, Any]
) -> bool:
    """True when watch revalidation hit only a transient C inventory lock.

    Preflight still fail-closes on the same code. Watch must not kill a live
    trainer because ``os.replace`` on cadence-1 staging collided with the
    second storage walk.
    """

    if str(phase) != "watch" or not isinstance(resolved_storage_layer, Mapping):
        return False
    synthetic = {
        "blocking_reasons": [RESOLVED_PATH_STORAGE_REVALIDATION_FAILED],
        "layers": [
            {
                "name": "path_resolution",
                "status": "BLOCK",
                "findings": [RESOLVED_PATH_STORAGE_REVALIDATION_FAILED],
                "evidence": {"resolved_storage": dict(resolved_storage_layer)},
            }
        ],
    }
    return watch_blockers_are_live_inventory_race(synthetic)


def evaluate_autonomous_supervision(
    *,
    root: Path,
    output_root: Path,
    phase: str,
    config: Mapping[str, Any] | None = None,
    expected_pid: int | None = None,
    process_rows: Sequence[Mapping[str, Any]] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate every supervision layer and return a machine decision."""

    lexical_root = Path(root)
    lexical_output_root = Path(output_root)
    storage_policy, storage_policy_failure = _load_lexical_storage_policy(lexical_root)
    storage_layer = storage_policy_failure or _storage_layer(
        root=lexical_root,
        output_root=lexical_output_root,
        storage_policy=storage_policy,
    )
    rows = list(process_rows) if process_rows is not None else discover_method_x_processes(exclude_pids=(expected_pid or -1,))
    process_layer = _process_layer(process_rows=rows, expected_pid=expected_pid)
    if storage_layer["status"] == "BLOCK":
        return _supervision_report(
            phase=phase,
            layers=[process_layer, storage_layer],
            policy_payload=dict(policy or {}),
            rows=rows,
            output_root=lexical_output_root,
        )
    try:
        root = _resolve_supervision_path(lexical_root)
        output_root = _resolve_supervision_path(lexical_output_root)
    except (OSError, RuntimeError):
        return _supervision_report(
            phase=phase,
            layers=[process_layer, storage_layer, _layer("path_resolution", "BLOCK", findings=["PATH_RESOLUTION_UNREADABLE"])],
            policy_payload=dict(policy or {}),
            rows=rows,
            output_root=lexical_output_root,
        )
    resolved_storage_layer = _storage_layer(root=root, output_root=output_root, storage_policy=storage_policy)
    if resolved_storage_layer["status"] == "BLOCK":
        if watch_resolved_storage_is_inventory_race(
            phase=phase, resolved_storage_layer=resolved_storage_layer
        ):
            # Lexical storage already passed. A watch-only inventory lock is
            # not scientific drift; keep the child alive and continue layers.
            pass
        else:
            return _supervision_report(
                phase=phase,
                layers=[
                    process_layer,
                    storage_layer,
                    _layer(
                        "path_resolution",
                        "BLOCK",
                        findings=["RESOLVED_PATH_STORAGE_REVALIDATION_FAILED"],
                        evidence={"resolved_storage": resolved_storage_layer},
                    ),
                ],
                policy_payload=dict(policy or {}),
                rows=rows,
                output_root=output_root,
            )
    policy_payload = dict(policy or load_supervision_policy(root))
    layers = [
        process_layer,
        storage_layer,
        _contract_layer(root=root, output_root=output_root, config=config, phase=phase),
        _data_layer(output_root=output_root),
        _checkpoint_integrity_layer(output_root=output_root),
        _runtime_layer(phase=phase, config=config),
        _recovery_layer(output_root=output_root, process_rows=rows, expected_pid=expected_pid),
        _fold_finalization_layer(output_root=output_root),
        _claim_privacy_layer(output_root=output_root),
        _terminal_state_layer(output_root=output_root, phase=phase),
    ]
    return _supervision_report(
        phase=phase,
        layers=layers,
        policy_payload=policy_payload,
        rows=rows,
        output_root=output_root,
    )


def write_supervision_report(*, root: Path, output_root: Path, report: Mapping[str, Any]) -> Path:
    """Persist the latest report and append an immutable registry event."""

    phase = str(report.get("phase") or "unknown")
    report_path = output_root / "reports" / f"autosupervisor_{phase}.json"
    atomic_write_json(report_path, dict(report))
    registry_path = root / "results/method_x_registry/autosupervision_events.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(report), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return report_path


def enforce_autonomous_supervision(*, root: Path, output_root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Write a blocker/pause marker when the automated decision is not continue."""

    materialized = dict(report)
    write_supervision_report(root=root, output_root=output_root, report=materialized)
    decision = str(materialized.get("decision"))
    if decision == "BLOCK":
        atomic_write_json(output_root / "reports" / "BLOCKED_AUTOSUPERVISOR.json", materialized)
    elif decision == "PAUSE_RESUMABLE":
        atomic_write_json(output_root / "reports" / "PAUSED_AUTOSUPERVISOR.json", materialized)
    return materialized


def _tree_signature(root: Path) -> tuple[int, int]:
    """Cheap progress signal: file count and newest mtime in an output root."""

    count = 0
    newest = 0
    if root.is_dir():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            # Supervisor telemetry must not count as training progress; an
            # otherwise idle child must still hit the idle timeout.
            if path.parent.name == "reports" and path.name.startswith("autosupervisor"):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            count += 1
            newest = max(newest, int(stat.st_mtime_ns))
    return count, newest


def _terminate_process_tree(process: subprocess.Popen[Any], *, timeout: float = 15.0) -> None:
    """Terminate and reap the exact supervised child and all descendants."""

    if process.poll() is not None:
        try:
            process.wait(timeout=max(0.1, timeout))
        except subprocess.TimeoutExpired:
            pass
        return
    pid = int(process.pid)
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=max(1.0, timeout),
                creationflags=creationflags,
            )
        except (OSError, subprocess.SubprocessError):
            # Fall through to the exact parent handle. The taskkill path is the
            # normal Windows descendant-aware mechanism; parent-only kill is a
            # final reaping fallback, never an authorization to continue.
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                process.terminate()
            except OSError:
                pass
    try:
        process.wait(timeout=max(1.0, timeout))
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        try:
            process.kill()
        except OSError:
            pass
        process.wait(timeout=max(1.0, timeout))


def _prepare_child_log_path(*, output_root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else output_root / path
    candidate = assert_path_beneath_root_without_reparse(
        output_root,
        candidate,
        outside_error="CHILD_LOG_PATH_OUTSIDE_OUTPUT_ROOT",
        reparse_error="CHILD_LOG_REPARSE_POINT_REJECTED",
    )
    candidate.parent.mkdir(parents=True, exist_ok=True)
    # A second check closes a parent-creation/swap window before open.
    candidate = assert_path_beneath_root_without_reparse(
        output_root,
        candidate,
        outside_error="CHILD_LOG_PATH_OUTSIDE_OUTPUT_ROOT",
        reparse_error="CHILD_LOG_REPARSE_POINT_REJECTED",
    )
    if os.path.lexists(candidate):
        raise RuntimeError("CHILD_LOG_PATH_ALREADY_EXISTS")
    return candidate


def _open_exclusive_child_log(*, output_root: Path, path: Path) -> Any:
    candidate = _prepare_child_log_path(output_root=output_root, path=path)
    # Revalidate immediately before and after exclusive creation. A reparse
    # introduced by an in-process test hook or concurrent local actor is
    # rejected before a trainer can be launched.
    assert_path_beneath_root_without_reparse(
        output_root,
        candidate.parent,
        outside_error="CHILD_LOG_PATH_OUTSIDE_OUTPUT_ROOT",
        reparse_error="CHILD_LOG_REPARSE_POINT_REJECTED",
        require_exists=True,
    )
    handle = candidate.open("x", encoding="utf-8", errors="replace", newline="\n")
    try:
        assert_path_beneath_root_without_reparse(
            output_root,
            candidate,
            outside_error="CHILD_LOG_PATH_OUTSIDE_OUTPUT_ROOT",
            reparse_error="CHILD_LOG_REPARSE_POINT_REJECTED",
            require_exists=True,
            require_regular_file=True,
        )
        opened = os.fstat(handle.fileno())
        named = os.lstat(candidate)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError("CHILD_LOG_PATH_SWAP_DETECTED")
        return handle
    except BaseException:
        handle.close()
        try:
            if os.path.lexists(candidate) and not _is_reparse_point(candidate):
                candidate.unlink()
        except OSError:
            pass
        raise


def run_supervised_process(
    command: Sequence[str],
    *,
    cwd: Path,
    output_root: Path,
    interval_seconds: float = 15.0,
    max_idle_seconds: float = 3600.0,
    check: Callable[[int], Mapping[str, Any] | None] | None = None,
    on_start: Callable[[int], None] | None = None,
    env: Mapping[str, str] | None = None,
    stdout_log_path: Path | None = None,
    stderr_log_path: Path | None = None,
) -> dict[str, Any]:
    """Run a child with heartbeat, progress and fail-closed supervision.

    A callback may return a supervisor report.  A ``BLOCK`` report terminates
    the child and maps to return code 78 (the project's contract-mismatch
    blocker code); return code 75 remains the resumable-pause convention.
    """

    output_root = assert_path_beneath_root_without_reparse(
        cwd,
        output_root,
        outside_error="SUPERVISED_OUTPUT_ROOT_OUTSIDE_CWD",
        reparse_error="SUPERVISED_OUTPUT_ROOT_REPARSE_POINT_REJECTED",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = assert_path_beneath_root_without_reparse(
        cwd,
        output_root,
        outside_error="SUPERVISED_OUTPUT_ROOT_OUTSIDE_CWD",
        reparse_error="SUPERVISED_OUTPUT_ROOT_REPARSE_POINT_REJECTED",
        require_exists=True,
    )
    if (stdout_log_path is None) != (stderr_log_path is None):
        raise RuntimeError("CHILD_LOG_PATH_PAIR_REQUIRED")
    log_handles: list[Any] = []
    tee_threads: list[threading.Thread] = []
    process: subprocess.Popen[str] | None = None
    started = time.monotonic()

    def _pump(source: Any, terminal: Any, log_handle: Any) -> None:
        for line in iter(source.readline, ""):
            terminal.write(line)
            terminal.flush()
            log_handle.write(line)
            log_handle.flush()

    try:
        popen_kwargs: dict[str, Any] = {}
        if stdout_log_path is not None and stderr_log_path is not None:
            stdout_log_path = _prepare_child_log_path(
                output_root=output_root, path=stdout_log_path
            )
            stderr_log_path = _prepare_child_log_path(
                output_root=output_root, path=stderr_log_path
            )
            stdout_handle = _open_exclusive_child_log(
                output_root=output_root, path=stdout_log_path
            )
            try:
                stderr_handle = _open_exclusive_child_log(
                    output_root=output_root, path=stderr_log_path
                )
            except BaseException:
                stdout_handle.close()
                raise
            log_handles.extend((stdout_handle, stderr_handle))
            popen_kwargs.update(
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        launch_command = list(command)
        if launch_command:
            exe_path = Path(launch_command[0])
            if not exe_path.is_absolute():
                launch_command[0] = str((Path(cwd) / exe_path).resolve())
        process = subprocess.Popen(
            launch_command,
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            **popen_kwargs,
        )
        if log_handles:
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("CHILD_TEE_PIPE_MISSING")
            tee_threads = [
                threading.Thread(
                    target=_pump,
                    args=(process.stdout, sys.stdout, log_handles[0]),
                    name=f"method-x-stdout-{process.pid}",
                    daemon=True,
                ),
                threading.Thread(
                    target=_pump,
                    args=(process.stderr, sys.stderr, log_handles[1]),
                    name=f"method-x-stderr-{process.pid}",
                    daemon=True,
                ),
            ]
            for thread in tee_threads:
                thread.start()
        if on_start is not None:
            try:
                on_start(int(process.pid))
            except Exception as exc:  # the child must not run without a recorded owner
                _terminate_process_tree(process)
                return {
                    "returncode": 78,
                    "decision": "BLOCK",
                    "poll_count": 0,
                    "terminated_by_supervisor": True,
                    "reason": "SUPERVISOR_START_HOOK_ERROR",
                    "probe_error": f"{type(exc).__name__}:{exc}",
                    "duration_seconds": time.monotonic() - started,
                    "last_report": None,
                }
        poll_count = 0
        previous_signature = _tree_signature(output_root)
        last_progress = time.monotonic()
        last_report: Mapping[str, Any] | None = None
        terminated_by_supervisor = False
        while True:
            poll_count += 1
            try:
                report = check(process.pid) if check is not None else None
            except Exception as exc:  # fail closed if a probe itself is broken
                _terminate_process_tree(process)
                return {
                    "returncode": 78,
                    "decision": "BLOCK",
                    "poll_count": poll_count,
                    "terminated_by_supervisor": True,
                    "reason": "SUPERVISOR_PROBE_ERROR",
                    "probe_error": f"{type(exc).__name__}:{exc}",
                    "duration_seconds": time.monotonic() - started,
                    "last_report": dict(last_report) if last_report is not None else None,
                }
            if report is not None:
                last_report = report
                if str(report.get("decision")) == "BLOCK":
                    terminated_by_supervisor = True
                    _terminate_process_tree(process)
                    return {
                        "returncode": 78,
                        "decision": "BLOCK",
                        "poll_count": poll_count,
                        "terminated_by_supervisor": True,
                        "duration_seconds": time.monotonic() - started,
                        "last_report": dict(last_report),
                    }
            current_signature = _tree_signature(output_root)
            if current_signature != previous_signature:
                previous_signature = current_signature
                last_progress = time.monotonic()
            returncode = process.poll()
            if returncode is not None:
                if returncode == 0:
                    decision = "COMPLETE"
                elif returncode == 75:
                    decision = "PAUSE_RESUMABLE"
                elif returncode == 78:
                    decision = "BLOCK"
                else:
                    # Scientific/readiness children use their own non-zero
                    # blocker codes (for example 3).  They are not resumable
                    # checkpoints.  Treat every other non-zero exit as a hard
                    # machine block so an autonomous caller cannot retry or
                    # promote an unclassified failure as if it were a pause.
                    decision = "BLOCK"
                return {
                    "returncode": int(returncode),
                    "decision": decision,
                    "poll_count": poll_count,
                    "terminated_by_supervisor": terminated_by_supervisor,
                    "reason": None if decision != "BLOCK" else "CHILD_NONZERO_EXIT",
                    "duration_seconds": time.monotonic() - started,
                    "last_report": dict(last_report) if last_report is not None else None,
                }
            if max_idle_seconds > 0 and time.monotonic() - last_progress > max_idle_seconds:
                terminated_by_supervisor = True
                _terminate_process_tree(process)
                return {
                    "returncode": 75,
                    "decision": "PAUSE_RESUMABLE",
                    "poll_count": poll_count,
                    "terminated_by_supervisor": True,
                    "reason": "NO_OUTPUT_PROGRESS_WITHIN_POLICY",
                    "duration_seconds": time.monotonic() - started,
                    "last_report": dict(last_report) if last_report is not None else None,
                }
            heartbeat = {
                "status": "RUNNING",
                "pid": process.pid,
                "updated_at_utc": utc_now(),
                "poll_count": poll_count,
                "output_signature": {"file_count": current_signature[0], "newest_mtime_ns": current_signature[1]},
            }
            atomic_write_json(output_root / "reports" / "autosupervisor_heartbeat.json", heartbeat)
            time.sleep(max(0.01, float(interval_seconds)))
    except BaseException:
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        raise
    finally:
        for thread in tee_threads:
            thread.join(timeout=30)
        if process is not None and process.poll() is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        for handle in log_handles:
            try:
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
