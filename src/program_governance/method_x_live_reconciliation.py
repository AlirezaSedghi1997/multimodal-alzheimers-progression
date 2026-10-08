"""Transactional reconciliation for the live five-model Method X continuation.

Only scheduler/lifecycle metadata may change. Frozen checkpoints, predictions,
metrics, fold states, and outer evaluations are verified and never rewritten.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.program_governance.method_x_completion_scope import (
    DEFAULT_SCOPE_PATH,
    OFFICIAL_CNN_ORDER,
    PROJECT_ROOT,
    load_completion_scope,
)


COMPLETED_BACKBONES = OFFICIAL_CNN_ORDER[:4]
RESUMABLE_BACKBONE = OFFICIAL_CNN_ORDER[4]
EXCLUDED_BACKBONES = ("resnet50_3d", "efficientnet3d_b0")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FREEZE_HASH_METHOD = "sha256_utf8_json_sort_keys_compact_ensure_ascii_false"
TRANSACTION_DIR = Path("journals") / "continuation_reconciliation"
PREFLIGHT_MANIFEST_RECOVERY_DIR = (
    Path("journals") / "preflight_manifest_recovery"
)


def _canonical_compact(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_file(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"REQUIRED_JSON_INVALID:{path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"REQUIRED_JSON_INVALID:{path.name}")
    return value


def _is_reparse(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError:
        return False
    return bool(getattr(metadata, "st_file_attributes", 0) & 0x0400) or stat.S_ISLNK(
        metadata.st_mode
    )


def _beneath(root: Path, candidate: Path, *, must_exist: bool = False) -> Path:
    lexical_root = Path(os.path.abspath(root))
    lexical = Path(os.path.abspath(candidate))
    try:
        lexical.relative_to(lexical_root)
    except ValueError as exc:
        raise RuntimeError("RECONCILIATION_PATH_OUTSIDE_ROOT") from exc
    cursor = lexical_root
    relative = lexical.relative_to(lexical_root)
    for component in relative.parts:
        cursor = cursor / component
        if os.path.lexists(cursor) and _is_reparse(cursor):
            raise RuntimeError("RECONCILIATION_REPARSE_POINT_REJECTED")
    if must_exist and not lexical.exists():
        raise RuntimeError("RECONCILIATION_REQUIRED_PATH_MISSING")
    return lexical


def _regular_file(root: Path, path: Path) -> Path:
    candidate = _beneath(root, path, must_exist=True)
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise RuntimeError("RECONCILIATION_REQUIRED_FILE_INVALID") from exc
    if not stat.S_ISREG(metadata.st_mode) or _is_reparse(candidate):
        raise RuntimeError("RECONCILIATION_REQUIRED_FILE_INVALID")
    return candidate


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        if _is_reparse(path) or not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError("TRANSACTION_IMMUTABLE_PAYLOAD_CONFLICT")
        return
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if os.path.lexists(temporary):
        raise RuntimeError("TRANSACTION_TEMP_ALREADY_EXISTS")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(temporary, flags)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _wrapped_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "payload": dict(payload),
        "payload_sha256_method": FREEZE_HASH_METHOD,
        "payload_sha256": _digest_bytes(_canonical_compact(payload)),
    }


def _validate_wrapper(wrapper: Mapping[str, Any], *, artifact: str) -> dict[str, Any]:
    payload = wrapper.get("payload")
    if (
        not isinstance(payload, dict)
        or wrapper.get("payload_sha256_method") != FREEZE_HASH_METHOD
        or wrapper.get("payload_sha256") != _digest_bytes(_canonical_compact(payload))
        or payload.get("artifact") != artifact
    ):
        raise RuntimeError("TRANSACTION_INTEGRITY_INVALID")
    return payload


def _resolve_frozen_artifact(path_value: str) -> Path:
    if path_value.startswith("C_STAGING/"):
        staging_root = Path(
            r"C:\Users\USER\AppData\Local\AMUSE-X\checkpoint_staging"
        )
        relative = Path(path_value.removeprefix("C_STAGING/"))
        candidate = _beneath(staging_root, staging_root / relative, must_exist=True)
        if candidate.name not in {"resume_current.pt", "resume_previous.pt"} and not (
            candidate.name.endswith(".tmp")
        ):
            raise RuntimeError("FREEZE_C_STAGING_PATH_INVALID")
        return candidate
    return _regular_file(PROJECT_ROOT, PROJECT_ROOT / Path(path_value))


def _load_and_verify_freeze(
    freeze_path: Path, *, output_root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    freeze_file = _regular_file(PROJECT_ROOT, freeze_path)
    wrapper = _read_json(freeze_file)
    payload = _validate_wrapper(
        wrapper, artifact="method_x_pre_continuation_artifact_freeze"
    )
    if payload.get("status") != "FROZEN_BEFORE_CONTINUATION_RECONCILIATION":
        raise RuntimeError("FREEZE_STATUS_INVALID")
    try:
        recorded_root = (PROJECT_ROOT / Path(str(payload["battery_root"]))).resolve()
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("FREEZE_BATTERY_ROOT_INVALID") from exc
    if recorded_root != output_root.resolve():
        raise RuntimeError("FREEZE_BATTERY_ROOT_INVALID")
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or len(rows) != payload.get("artifact_count"):
        raise RuntimeError("FREEZE_ARTIFACT_INVENTORY_INVALID")
    observed_total = 0
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError("FREEZE_ARTIFACT_INVENTORY_INVALID")
        logical = str(row["path"])
        if logical in by_path:
            raise RuntimeError("FREEZE_ARTIFACT_INVENTORY_INVALID")
        try:
            expected_size = int(row["size_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("FREEZE_ARTIFACT_INVENTORY_INVALID") from exc
        expected_sha = str(row.get("sha256") or "")
        if expected_size < 0 or SHA256_RE.fullmatch(expected_sha) is None:
            raise RuntimeError("FREEZE_ARTIFACT_INVENTORY_INVALID")
        try:
            artifact = _resolve_frozen_artifact(logical)
            metadata = os.lstat(artifact)
            actual_sha = _sha256_file(artifact)
        except (OSError, RuntimeError):
            raise RuntimeError("FREEZE_ARTIFACT_MISMATCH") from None
        if int(metadata.st_size) != expected_size or actual_sha != expected_sha:
            raise RuntimeError("FREEZE_ARTIFACT_MISMATCH")
        observed_total += expected_size
        by_path[logical] = row
    if observed_total != int(payload.get("artifact_total_bytes", -1)):
        raise RuntimeError("FREEZE_ARTIFACT_INVENTORY_INVALID")
    manifest_logical = (output_root / "battery_manifest.json").relative_to(
        PROJECT_ROOT
    ).as_posix()
    lock_logical = (output_root / ".battery.lock").relative_to(PROJECT_ROOT).as_posix()
    if manifest_logical not in by_path or lock_logical not in by_path:
        raise RuntimeError("FREEZE_MUTATION_TARGETS_NOT_PINNED")
    if payload.get("battery_manifest_sha256") != by_path[manifest_logical]["sha256"]:
        raise RuntimeError("FREEZE_MANIFEST_BINDING_INVALID")
    return payload, by_path


def _rows_by_name(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("models")
    if not isinstance(rows, list):
        raise RuntimeError("BATTERY_MODEL_ROWS_INVALID")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("backbone"), str):
            raise RuntimeError("BATTERY_MODEL_ROWS_INVALID")
        name = str(row["backbone"])
        if name in result:
            raise RuntimeError("BATTERY_MODEL_ROWS_INVALID")
        result[name] = row
    expected = set(OFFICIAL_CNN_ORDER) | set(EXCLUDED_BACKBONES)
    if set(result) != expected:
        raise RuntimeError("BATTERY_MODEL_ROWS_INVALID")
    for name in EXCLUDED_BACKBONES:
        row = result[name]
        training = row.get("training")
        if (
            row.get("status") != "PROTOCOL_EXCLUDED_NOT_STARTED"
            or not isinstance(training, dict)
            or training.get("status") != "PROTOCOL_EXCLUDED_NOT_STARTED"
        ):
            raise RuntimeError("BATTERY_PROTOCOL_EXCLUSION_INVALID")
    return result


def _validate_recovery(evidence: Mapping[str, Any]) -> dict[str, Any]:
    required_digest_fields = ("payload_sha256", "journal_record_sha256")
    try:
        valid = (
            evidence.get("backbone") == RESUMABLE_BACKBONE
            and int(evidence.get("current_fold")) == 1
            and evidence.get("finalized_folds") == [0]
            and evidence.get("source") == "C_CURRENT"
            and int(evidence.get("resume_from_step")) > 0
            and int(evidence.get("target_step"))
            == int(evidence.get("resume_from_step"))
            and int(evidence.get("replay_steps")) == 0
            and int(evidence.get("generation")) > 0
            and int(evidence.get("payload_size_bytes")) > 0
            and int(evidence.get("sampler_cursor")) >= 0
            and re.fullmatch(r"[0-9a-f]{16}", str(evidence.get("run_id") or ""))
            is not None
            and all(
                SHA256_RE.fullmatch(str(evidence.get(field) or "")) is not None
                for field in required_digest_fields
            )
        )
    except (TypeError, ValueError):
        valid = False
    path = str(evidence.get("payload_path") or "").replace("\\", "/")
    expected_fragment = (
        f"{evidence.get('run_id')}/{RESUMABLE_BACKBONE}/1/resume_current.pt"
    )
    if not valid or path != f"C_STAGING/{expected_fragment}":
        raise RuntimeError("EXACT_RECOVERY_EVIDENCE_INVALID")
    return dict(evidence)


def _default_completed_validation(
    *, backbone: str, row: dict[str, Any], output_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    import yaml

    from scripts import run_method_x_claim_benchmark as battery

    config_path = PROJECT_ROOT / "configs" / "experiments" / "method_x_claim_benchmark_v3.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("BATTERY_CONFIG_INVALID")
    context = battery._validation_context(manifest, config)
    return battery._validate_completed_row(
        row=row, output_root=output_root, config=config, context=context
    )


def _default_recovery_provider(
    *, output_root: Path, row: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    import torch

    from src.training.checkpoint_staging import (
        StagingLayout,
        fold_journal_path,
        plan_recovery,
        resolve_staging_run_id,
    )
    from src.training.storage_locality import load_storage_policy

    del manifest
    model_dir = _beneath(
        output_root,
        output_root / str(row.get("output_dir") or ""),
        must_exist=True,
    )
    finalized: list[int] = []
    current_fold: int | None = None
    contract_digest: str | None = None
    for fold in range(5):
        state_path = model_dir / f"method_x_3d_fold_{fold}_state.json"
        if not state_path.is_file():
            break
        state = _read_json(_regular_file(output_root, state_path))
        if state.get("fold") != fold:
            raise RuntimeError("MEDNEXT_FOLD_STATE_INVALID")
        if state.get("stage") == "FINALIZED" and current_fold is None:
            finalized.append(fold)
        elif state.get("stage") == "TRAINING" and current_fold is None:
            current_fold = fold
            contract_digest = str(state.get("contract_digest") or "")
        else:
            raise RuntimeError("MEDNEXT_FOLD_STATE_INVALID")
    if current_fold != 1 or finalized != [0] or SHA256_RE.fullmatch(
        str(contract_digest or "")
    ) is None:
        raise RuntimeError("MEDNEXT_FOLD_STATE_INVALID")
    policy = load_storage_policy(
        PROJECT_ROOT / "configs" / "policy" / "method_x_storage_locality_v1.yaml"
    )
    journal = fold_journal_path(model_dir, current_fold)
    run_id = resolve_staging_run_id(
        live_run_id=str(contract_digest)[:16], journal_path=journal
    )
    layout = StagingLayout.build(
        policy=policy,
        run_id=run_id,
        backbone=RESUMABLE_BACKBONE,
        fold=current_fold,
        h_output_dir=model_dir,
    )

    def load(path: Path) -> Any:
        return torch.load(path, map_location="cpu", weights_only=False)

    plan = plan_recovery(layout, load_fn=load, require_exact=True, max_replay_steps=0)
    if plan.payload_path is None:
        raise RuntimeError("EXACT_RECOVERY_EVIDENCE_INVALID")
    checkpoint = load(plan.payload_path)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("EXACT_RECOVERY_EVIDENCE_INVALID")
    try:
        relative_c = plan.payload_path.relative_to(policy.checkpoint_staging_root)
    except ValueError as exc:
        raise RuntimeError("EXACT_RECOVERY_EVIDENCE_INVALID") from exc
    return {
        "backbone": RESUMABLE_BACKBONE,
        "current_fold": current_fold,
        "finalized_folds": finalized,
        "run_id": run_id,
        "source": plan.source,
        "payload_path": f"C_STAGING/{relative_c.as_posix()}",
        "resume_from_step": plan.resume_from_step,
        "target_step": plan.target_step,
        "replay_steps": plan.replay_steps,
        "generation": plan.generation,
        "payload_sha256": plan.payload_sha256,
        "payload_size_bytes": plan.payload_size_bytes,
        "journal_record_sha256": plan.journal_record_sha256,
        "sampler_cursor": checkpoint.get("sampler_cursor"),
        "epoch": checkpoint.get("epoch"),
        "epoch_batch_cursor": checkpoint.get("epoch_batch_cursor"),
        "loss_sum": plan.expected_loss_sum,
        "loss_count": plan.expected_loss_count,
    }


def _default_process_provider() -> Sequence[Mapping[str, Any]]:
    from scripts.orchestrate_method_x_mri_5fold import active_method_x_processes

    return active_method_x_processes()


def _default_pid_liveness(pid: int) -> bool | None:
    from scripts.run_method_x_claim_benchmark import _pid_is_live

    return _pid_is_live(int(pid))


def _acquire_guard(output_root: Path) -> Path:
    from scripts.run_method_x_claim_benchmark import acquire_reconciliation_guard

    return acquire_reconciliation_guard(output_root)


def _release_guard(path: Path | None) -> None:
    from scripts.run_method_x_claim_benchmark import release_reconciliation_guard

    release_reconciliation_guard(path)


def _validated_plan(
    *, output_root: Path, plan_path: Path, wrapper: Mapping[str, Any]
) -> dict[str, Any]:
    plan = _validate_wrapper(
        wrapper, artifact="method_x_live_continuation_reconciliation_plan"
    )
    transaction_id = str(plan.get("transaction_id") or "")
    expected_path = output_root / TRANSACTION_DIR / transaction_id / "plan.json"
    if plan_path != _beneath(output_root, expected_path, must_exist=True):
        raise RuntimeError("TRANSACTION_PLAN_BINDING_INVALID")
    intent = {
        "freeze_sha256": plan.get("freeze_sha256"),
        "pre_manifest_sha256": plan.get("pre_manifest_sha256"),
        "pre_lock_sha256": plan.get("pre_lock_sha256"),
        "completed_evidence": plan.get("completed_model_evidence"),
        "exact_recovery": plan.get("exact_recovery"),
    }
    expected_transaction_id = _digest_bytes(_canonical_compact(intent))
    transaction_relative = TRANSACTION_DIR / transaction_id
    expected_paths = (
        (transaction_relative / "battery_manifest.pre.json").as_posix(),
        (transaction_relative / "stale_battery_lock.json").as_posix(),
        (transaction_relative / "receipt.json").as_posix(),
    )
    bound_paths = (
        plan.get("archive_manifest_path"),
        plan.get("archive_lock_path"),
        plan.get("receipt_path"),
    )
    if (
        plan.get("status") != "PREPARED"
        or transaction_id != expected_transaction_id
        or bound_paths != expected_paths
        or plan.get("allowed_live_mutations")
        != ["battery_manifest.json", ".battery.lock"]
        or plan.get("outer_loader_invocation_count") != 0
        or plan.get("finalized_artifact_rewrite_count") != 0
    ):
        raise RuntimeError("TRANSACTION_PLAN_BINDING_INVALID")
    return plan


def _verify_freeze_for_plan(
    *,
    output_root: Path,
    plan: Mapping[str, Any],
    additional_allowed_manifest: bytes | None = None,
) -> None:
    """Revalidate frozen science while allowing only the two planned mutations."""

    freeze_path = _regular_file(
        PROJECT_ROOT, PROJECT_ROOT / str(plan.get("freeze_path") or "")
    )
    if _sha256_file(freeze_path) != plan.get("freeze_sha256"):
        raise RuntimeError("FREEZE_ARTIFACT_MISMATCH")
    wrapper = _read_json(freeze_path)
    payload = _validate_wrapper(
        wrapper, artifact="method_x_pre_continuation_artifact_freeze"
    )
    try:
        recorded_root = PROJECT_ROOT / Path(str(payload["battery_root"]))
        recorded_root = _beneath(PROJECT_ROOT, recorded_root, must_exist=True)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("FREEZE_BATTERY_ROOT_INVALID") from exc
    if recorded_root != output_root:
        raise RuntimeError("FREEZE_BATTERY_ROOT_INVALID")
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or len(rows) != payload.get("artifact_count"):
        raise RuntimeError("FREEZE_ARTIFACT_INVENTORY_INVALID")
    pre_manifest = base64.b64decode(
        str(plan.get("pre_manifest_base64") or ""), validate=True
    )
    post_manifest = base64.b64decode(
        str(plan.get("post_manifest_base64") or ""), validate=True
    )
    pre_lock = base64.b64decode(str(plan.get("pre_lock_base64") or ""), validate=True)
    manifest_logical = (output_root / "battery_manifest.json").relative_to(
        PROJECT_ROOT
    ).as_posix()
    lock_logical = (output_root / ".battery.lock").relative_to(PROJECT_ROOT).as_posix()
    by_path: dict[str, Mapping[str, Any]] = {}
    observed_total = 0
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError("FREEZE_ARTIFACT_INVENTORY_INVALID")
        logical = str(row["path"])
        if logical in by_path:
            raise RuntimeError("FREEZE_ARTIFACT_INVENTORY_INVALID")
        by_path[logical] = row
        try:
            expected_size = int(row["size_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("FREEZE_ARTIFACT_INVENTORY_INVALID") from exc
        expected_sha = str(row.get("sha256") or "")
        if expected_size < 0 or SHA256_RE.fullmatch(expected_sha) is None:
            raise RuntimeError("FREEZE_ARTIFACT_INVENTORY_INVALID")
        observed_total += expected_size
        if logical == manifest_logical:
            artifact = _regular_file(output_root, output_root / "battery_manifest.json")
            actual = (artifact.stat().st_size, _sha256_file(artifact))
            allowed = {
                (len(pre_manifest), _digest_bytes(pre_manifest)),
                (len(post_manifest), _digest_bytes(post_manifest)),
            }
            if additional_allowed_manifest is not None:
                allowed.add(
                    (
                        len(additional_allowed_manifest),
                        _digest_bytes(additional_allowed_manifest),
                    )
                )
            if expected_sha != plan.get("pre_manifest_sha256") or actual not in allowed:
                raise RuntimeError("FREEZE_ARTIFACT_MISMATCH")
            continue
        if logical == lock_logical:
            live_lock = output_root / ".battery.lock"
            archived_lock = output_root / str(plan.get("archive_lock_path") or "")
            if os.path.lexists(live_lock):
                artifact = _regular_file(output_root, live_lock)
            else:
                artifact = _regular_file(output_root, archived_lock)
            if (
                expected_sha != plan.get("pre_lock_sha256")
                or artifact.stat().st_size != len(pre_lock)
                or _sha256_file(artifact) != _digest_bytes(pre_lock)
            ):
                raise RuntimeError("FREEZE_ARTIFACT_MISMATCH")
            continue
        try:
            artifact = _resolve_frozen_artifact(logical)
            metadata = os.lstat(artifact)
            actual_sha = _sha256_file(artifact)
        except (OSError, RuntimeError):
            raise RuntimeError("FREEZE_ARTIFACT_MISMATCH") from None
        if int(metadata.st_size) != expected_size or actual_sha != expected_sha:
            raise RuntimeError("FREEZE_ARTIFACT_MISMATCH")
    if (
        observed_total != int(payload.get("artifact_total_bytes", -1))
        or manifest_logical not in by_path
        or lock_logical not in by_path
        or payload.get("battery_manifest_sha256")
        != by_path[manifest_logical].get("sha256")
    ):
        raise RuntimeError("FREEZE_ARTIFACT_INVENTORY_INVALID")


def _validated_completed_receipt_evidence(
    output_root: Path,
    manifest: Mapping[str, Any],
    *,
    require_current_manifest: bool,
) -> dict[str, Any] | None:
    block = manifest.get("continuation_reconciliation")
    if not isinstance(block, dict) or block.get("status") != "RECONCILED_PAUSED_RESUMABLE":
        return None
    relative = block.get("receipt_path")
    if not isinstance(relative, str):
        raise RuntimeError("CONTINUATION_RECEIPT_INVALID")
    receipt_candidate = _beneath(output_root, output_root / relative)
    # A crash after the manifest commit and before receipt commit is recoverable
    # from the immutable prepared plan. No other missing-receipt state is accepted.
    if not os.path.lexists(receipt_candidate):
        return None
    try:
        receipt_path = _regular_file(output_root, receipt_candidate)
        wrapper = _read_json(receipt_path)
        receipt = _validate_wrapper(
            wrapper, artifact="method_x_live_continuation_reconciliation_receipt"
        )
        transaction_id = str(block.get("transaction_id") or "")
        plan_relative = str(receipt.get("plan_path") or "")
        expected_plan_relative = (
            TRANSACTION_DIR / transaction_id / "plan.json"
        ).as_posix()
        if plan_relative != expected_plan_relative:
            raise RuntimeError("CONTINUATION_RECEIPT_INVALID")
        plan_path = _regular_file(output_root, output_root / plan_relative)
        plan_wrapper = _read_json(plan_path)
        plan = _validated_plan(
            output_root=output_root, plan_path=plan_path, wrapper=plan_wrapper
        )
        archive_manifest = _regular_file(
            output_root, output_root / str(plan["archive_manifest_path"])
        )
        archive_lock = _regular_file(
            output_root, output_root / str(plan["archive_lock_path"])
        )
        freeze_file = _regular_file(
            PROJECT_ROOT, PROJECT_ROOT / str(receipt.get("freeze_path") or "")
        )
        manifest_path = _regular_file(output_root, output_root / "battery_manifest.json")
        current_hash = _sha256_file(manifest_path)
        valid = (
            receipt.get("status") == "COMPLETED"
            and receipt.get("transaction_id") == transaction_id
            and plan.get("transaction_id") == transaction_id
            and plan.get("receipt_path") == relative
            and receipt.get("plan_sha256") == _sha256_file(plan_path)
            and receipt.get("freeze_sha256") == _sha256_file(freeze_file)
            and receipt.get("freeze_sha256") == plan.get("freeze_sha256")
            and receipt.get("pre_manifest_sha256")
            == plan.get("pre_manifest_sha256")
            and receipt.get("post_manifest_sha256")
            == plan.get("post_manifest_sha256")
            and receipt.get("archived_manifest_sha256")
            == _sha256_file(archive_manifest)
            and receipt.get("archived_manifest_sha256")
            == plan.get("pre_manifest_sha256")
            and receipt.get("archived_lock_sha256") == _sha256_file(archive_lock)
            and receipt.get("archived_lock_sha256") == plan.get("pre_lock_sha256")
            and receipt.get("completed_model_evidence")
            == plan.get("completed_model_evidence")
            and receipt.get("exact_recovery") == plan.get("exact_recovery")
            and receipt.get("outer_loader_invocation_count") == 0
            and receipt.get("finalized_artifact_rewrite_count") == 0
        )
        if require_current_manifest:
            valid = valid and receipt.get("post_manifest_sha256") == current_hash
        if not valid:
            raise RuntimeError("CONTINUATION_RECEIPT_INVALID")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("CONTINUATION_RECEIPT_INVALID") from exc
    return {
        "status": "ALREADY_RECONCILED",
        "transaction_id": str(receipt["transaction_id"]),
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "plan": plan,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256_file(plan_path),
        "canonical_manifest_sha256": str(receipt["post_manifest_sha256"]),
        "manifest_sha256": current_hash,
    }


def _completed_receipt(
    output_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any] | None:
    evidence = _validated_completed_receipt_evidence(
        output_root,
        manifest,
        require_current_manifest=True,
    )
    if evidence is None:
        return None
    return {
        "status": evidence["status"],
        "transaction_id": evidence["transaction_id"],
        "receipt_path": evidence["receipt_path"],
        "manifest_sha256": evidence["manifest_sha256"],
    }


def _build_plan(
    *,
    output_root: Path,
    freeze_path: Path,
    freeze_payload: Mapping[str, Any],
    manifest: dict[str, Any],
    manifest_bytes: bytes,
    lock_bytes: bytes,
    completed_evidence: Mapping[str, Any],
    recovery: Mapping[str, Any],
) -> dict[str, Any]:
    freeze_hash = _sha256_file(freeze_path)
    intent = {
        "freeze_sha256": freeze_hash,
        "pre_manifest_sha256": _digest_bytes(manifest_bytes),
        "pre_lock_sha256": _digest_bytes(lock_bytes),
        "completed_evidence": completed_evidence,
        "exact_recovery": recovery,
    }
    transaction_id = _digest_bytes(_canonical_compact(intent))
    relative_dir = TRANSACTION_DIR / transaction_id
    receipt_rel = (relative_dir / "receipt.json").as_posix()
    plan_rel = (relative_dir / "plan.json").as_posix()
    archive_manifest_rel = (relative_dir / "battery_manifest.pre.json").as_posix()
    archive_lock_rel = (relative_dir / "stale_battery_lock.json").as_posix()
    updated = copy.deepcopy(manifest)
    rows = _rows_by_name(updated)
    stamp = str(freeze_payload.get("generated_at_utc") or "")

    resnet_training = rows["resnet34_3d"].get("training")
    if (
        rows["resnet34_3d"].get("status") != "COMPLETED"
        or not isinstance(resnet_training, dict)
        or resnet_training.get("status") != "BLOCKED_POST_TRAINING_VALIDATION"
    ):
        raise RuntimeError("RESNET34_STALE_STATUS_NOT_PROVEN")
    resnet_training.pop("blocker", None)
    resnet_training.update(
        status="COMPLETED_VALIDATED_RECONCILED",
        phase="complete",
        pid=None,
        reason="STALE_POST_TRAINING_STATUS_RECONCILED_FROM_VALIDATED_METRICS",
        reconciled_at_utc=stamp,
        updated_at_utc=stamp,
    )

    mednext = rows[RESUMABLE_BACKBONE]
    mednext_training = mednext.get("training")
    if (
        mednext.get("status") != "TRAINING_RUNNING"
        or not isinstance(mednext_training, dict)
        or mednext_training.get("status") != "RUNNING"
    ):
        raise RuntimeError("MEDNEXT_STALE_RUNNING_STATUS_NOT_PROVEN")
    mednext["status"] = "PAUSED_RESUMABLE_1_OF_5"
    mednext_training.update(
        status="PAUSED_RESUMABLE_1_OF_5",
        phase="training_paused_exact_resume",
        pid=None,
        reason="STALE_RUNNING_RECONCILED_EXACT_STAGING_RECOVERY",
        reconciled_at_utc=stamp,
        updated_at_utc=stamp,
        exact_recovery={
            "run_id": recovery["run_id"],
            "fold": 1,
            "source": recovery["source"],
            "payload_path": recovery["payload_path"],
            "payload_sha256": recovery["payload_sha256"],
            "payload_size_bytes": recovery["payload_size_bytes"],
            "optimizer_step": recovery["resume_from_step"],
            "target_step": recovery["target_step"],
            "replay_steps": recovery["replay_steps"],
            "generation": recovery["generation"],
            "journal_record_sha256": recovery["journal_record_sha256"],
            "sampler_cursor": recovery["sampler_cursor"],
        },
    )
    updated["status"] = "PAUSED_RESUMABLE"
    updated["updated_at_utc"] = stamp
    updated["continuation_reconciliation"] = {
        "schema_version": 1,
        "status": "RECONCILED_PAUSED_RESUMABLE",
        "transaction_id": transaction_id,
        "freeze_path": freeze_path.relative_to(output_root).as_posix(),
        "freeze_sha256": freeze_hash,
        "transaction_plan_path": plan_rel,
        "receipt_path": receipt_rel,
        "completed_backbones_validated": list(COMPLETED_BACKBONES),
        "resumable_backbone": RESUMABLE_BACKBONE,
        "outer_loader_invocation_count": 0,
        "finalized_artifact_rewrite_count": 0,
    }
    post_bytes = _canonical_file(updated)
    body = {
        "schema_version": 1,
        "artifact": "method_x_live_continuation_reconciliation_plan",
        "status": "PREPARED",
        "transaction_id": transaction_id,
        "created_at_utc": stamp,
        "freeze_path": freeze_path.relative_to(PROJECT_ROOT).as_posix(),
        "freeze_sha256": freeze_hash,
        "pre_manifest_sha256": _digest_bytes(manifest_bytes),
        "post_manifest_sha256": _digest_bytes(post_bytes),
        "pre_lock_sha256": _digest_bytes(lock_bytes),
        "pre_manifest_base64": base64.b64encode(manifest_bytes).decode("ascii"),
        "post_manifest_base64": base64.b64encode(post_bytes).decode("ascii"),
        "pre_lock_base64": base64.b64encode(lock_bytes).decode("ascii"),
        "archive_manifest_path": archive_manifest_rel,
        "archive_lock_path": archive_lock_rel,
        "receipt_path": receipt_rel,
        "completed_model_evidence": completed_evidence,
        "exact_recovery": dict(recovery),
        "allowed_live_mutations": ["battery_manifest.json", ".battery.lock"],
        "outer_loader_invocation_count": 0,
        "finalized_artifact_rewrite_count": 0,
    }
    return _wrapped_payload(body)


def _execute_plan(
    *,
    output_root: Path,
    plan_path: Path,
    plan_wrapper: Mapping[str, Any],
    process_provider: Callable[[], Sequence[Mapping[str, Any]]],
    failpoint: Callable[[str], None] | None,
) -> dict[str, Any]:
    expected_plan_sha256 = _digest_bytes(_canonical_file(plan_wrapper))
    if _sha256_file(plan_path) != expected_plan_sha256:
        raise RuntimeError("TRANSACTION_PLAN_CHANGED")
    plan = _validated_plan(
        output_root=output_root, plan_path=plan_path, wrapper=plan_wrapper
    )
    _verify_freeze_for_plan(output_root=output_root, plan=plan)
    if process_provider():
        raise RuntimeError("ACTIVE_METHOD_X_WRITER")
    if _sha256_file(plan_path) != expected_plan_sha256:
        raise RuntimeError("TRANSACTION_PLAN_CHANGED")
    manifest_path = output_root / "battery_manifest.json"
    lock_path = output_root / ".battery.lock"
    pre_manifest = base64.b64decode(str(plan["pre_manifest_base64"]), validate=True)
    post_manifest = base64.b64decode(str(plan["post_manifest_base64"]), validate=True)
    pre_lock = base64.b64decode(str(plan["pre_lock_base64"]), validate=True)
    if (
        _digest_bytes(pre_manifest) != plan.get("pre_manifest_sha256")
        or _digest_bytes(post_manifest) != plan.get("post_manifest_sha256")
        or _digest_bytes(pre_lock) != plan.get("pre_lock_sha256")
    ):
        raise RuntimeError("TRANSACTION_PLAN_PAYLOAD_INVALID")
    archive_manifest = _beneath(
        output_root, output_root / str(plan["archive_manifest_path"])
    )
    archive_lock = _beneath(output_root, output_root / str(plan["archive_lock_path"]))
    receipt_path = _beneath(output_root, output_root / str(plan["receipt_path"]))
    _write_once(archive_manifest, pre_manifest)

    if os.path.lexists(lock_path):
        if lock_path.read_bytes() != pre_lock or os.path.lexists(archive_lock):
            raise RuntimeError("TRANSACTION_LOCK_EVIDENCE_CONFLICT")
        archive_lock.parent.mkdir(parents=True, exist_ok=True)
        os.replace(lock_path, archive_lock)
    elif not os.path.lexists(archive_lock) or archive_lock.read_bytes() != pre_lock:
        raise RuntimeError("TRANSACTION_LOCK_EVIDENCE_MISSING")
    if failpoint is not None:
        failpoint("after_lock_archived")

    current_manifest = manifest_path.read_bytes()
    if current_manifest == pre_manifest:
        if process_provider():
            raise RuntimeError("ACTIVE_METHOD_X_WRITER")
        _atomic_replace(manifest_path, post_manifest)
    elif current_manifest != post_manifest:
        raise RuntimeError("TRANSACTION_MANIFEST_CONFLICT")
    if failpoint is not None:
        failpoint("after_manifest_written")

    receipt_body = {
        "schema_version": 1,
        "artifact": "method_x_live_continuation_reconciliation_receipt",
        "status": "COMPLETED",
        "transaction_id": plan["transaction_id"],
        "completed_at_utc": plan["created_at_utc"],
        "freeze_path": plan["freeze_path"],
        "freeze_sha256": plan["freeze_sha256"],
        "plan_path": plan_path.relative_to(output_root).as_posix(),
        "plan_sha256": _sha256_file(plan_path),
        "pre_manifest_sha256": plan["pre_manifest_sha256"],
        "post_manifest_sha256": plan["post_manifest_sha256"],
        "archived_manifest_sha256": _sha256_file(archive_manifest),
        "archived_lock_sha256": _sha256_file(archive_lock),
        "completed_model_evidence": plan["completed_model_evidence"],
        "exact_recovery": plan["exact_recovery"],
        "outer_loader_invocation_count": 0,
        "finalized_artifact_rewrite_count": 0,
    }
    if _sha256_file(plan_path) != expected_plan_sha256:
        raise RuntimeError("TRANSACTION_PLAN_CHANGED")
    _write_once(receipt_path, _canonical_file(_wrapped_payload(receipt_body)))
    if failpoint is not None:
        failpoint("after_receipt_written")
    return {
        "status": "RECONCILED_PAUSED_RESUMABLE",
        "transaction_id": str(plan["transaction_id"]),
        "receipt_path": str(receipt_path),
        "manifest_sha256": str(plan["post_manifest_sha256"]),
        "exact_recovery": dict(plan["exact_recovery"]),
    }


def _manifest_top_level_differences(
    canonical: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    return sorted(
        str(key)
        for key in set(canonical) | set(current)
        if canonical.get(key) != current.get(key)
    )


def _build_preflight_manifest_recovery_plan(
    *,
    output_root: Path,
    original: Mapping[str, Any],
    drifted_manifest: bytes,
    canonical_manifest: bytes,
) -> tuple[Path, dict[str, Any]]:
    original_plan_path = Path(str(original["plan_path"]))
    original_receipt_path = Path(str(original["receipt_path"]))
    intent = {
        "original_transaction_id": original["transaction_id"],
        "original_plan_sha256": original["plan_sha256"],
        "original_receipt_sha256": original["receipt_sha256"],
        "drifted_manifest_sha256": _digest_bytes(drifted_manifest),
        "canonical_manifest_sha256": _digest_bytes(canonical_manifest),
        "allowed_differences": ["updated_at_utc"],
    }
    transaction_id = _digest_bytes(_canonical_compact(intent))
    transaction_rel = PREFLIGHT_MANIFEST_RECOVERY_DIR / transaction_id
    plan_rel = transaction_rel / "plan.json"
    archive_rel = transaction_rel / "battery_manifest.preflight_drift.json"
    receipt_rel = transaction_rel / "receipt.json"
    body = {
        "schema_version": 1,
        "artifact": "method_x_preflight_manifest_recovery_plan",
        "status": "PREPARED",
        "transaction_id": transaction_id,
        "original_transaction_id": original["transaction_id"],
        "original_plan_path": original_plan_path.relative_to(output_root).as_posix(),
        "original_plan_sha256": original["plan_sha256"],
        "original_receipt_path": original_receipt_path.relative_to(
            output_root
        ).as_posix(),
        "original_receipt_sha256": original["receipt_sha256"],
        "drifted_manifest_sha256": _digest_bytes(drifted_manifest),
        "drifted_manifest_size_bytes": len(drifted_manifest),
        "drifted_manifest_base64": base64.b64encode(drifted_manifest).decode("ascii"),
        "canonical_manifest_sha256": _digest_bytes(canonical_manifest),
        "canonical_manifest_size_bytes": len(canonical_manifest),
        "canonical_manifest_base64": base64.b64encode(canonical_manifest).decode(
            "ascii"
        ),
        "allowed_differences": ["updated_at_utc"],
        "archive_path": archive_rel.as_posix(),
        "receipt_path": receipt_rel.as_posix(),
        "outer_loader_invocation_count": 0,
        "finalized_artifact_rewrite_count": 0,
    }
    return output_root / plan_rel, _wrapped_payload(body)


def _validated_preflight_manifest_recovery_plan(
    *,
    output_root: Path,
    plan_path: Path,
    wrapper: Mapping[str, Any],
    original: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _validate_wrapper(
        wrapper, artifact="method_x_preflight_manifest_recovery_plan"
    )
    transaction_id = str(plan.get("transaction_id") or "")
    expected_plan_path = (
        output_root
        / PREFLIGHT_MANIFEST_RECOVERY_DIR
        / transaction_id
        / "plan.json"
    )
    if plan_path != _regular_file(output_root, expected_plan_path):
        raise RuntimeError("PREFLIGHT_MANIFEST_RECOVERY_PLAN_INVALID")
    intent = {
        "original_transaction_id": plan.get("original_transaction_id"),
        "original_plan_sha256": plan.get("original_plan_sha256"),
        "original_receipt_sha256": plan.get("original_receipt_sha256"),
        "drifted_manifest_sha256": plan.get("drifted_manifest_sha256"),
        "canonical_manifest_sha256": plan.get("canonical_manifest_sha256"),
        "allowed_differences": plan.get("allowed_differences"),
    }
    expected_transaction_id = _digest_bytes(_canonical_compact(intent))
    transaction_rel = PREFLIGHT_MANIFEST_RECOVERY_DIR / transaction_id
    drifted = base64.b64decode(
        str(plan.get("drifted_manifest_base64") or ""), validate=True
    )
    canonical = base64.b64decode(
        str(plan.get("canonical_manifest_base64") or ""), validate=True
    )
    expected_original_plan = Path(str(original["plan_path"])).relative_to(
        output_root
    ).as_posix()
    expected_original_receipt = Path(str(original["receipt_path"])).relative_to(
        output_root
    ).as_posix()
    if (
        plan.get("status") != "PREPARED"
        or transaction_id != expected_transaction_id
        or plan.get("original_transaction_id") != original.get("transaction_id")
        or plan.get("original_plan_path") != expected_original_plan
        or plan.get("original_plan_sha256") != original.get("plan_sha256")
        or plan.get("original_receipt_path") != expected_original_receipt
        or plan.get("original_receipt_sha256") != original.get("receipt_sha256")
        or plan.get("allowed_differences") != ["updated_at_utc"]
        or plan.get("archive_path")
        != (transaction_rel / "battery_manifest.preflight_drift.json").as_posix()
        or plan.get("receipt_path") != (transaction_rel / "receipt.json").as_posix()
        or plan.get("drifted_manifest_sha256") != _digest_bytes(drifted)
        or plan.get("drifted_manifest_size_bytes") != len(drifted)
        or plan.get("canonical_manifest_sha256") != _digest_bytes(canonical)
        or plan.get("canonical_manifest_size_bytes") != len(canonical)
        or plan.get("canonical_manifest_sha256")
        != original.get("canonical_manifest_sha256")
        or plan.get("outer_loader_invocation_count") != 0
        or plan.get("finalized_artifact_rewrite_count") != 0
    ):
        raise RuntimeError("PREFLIGHT_MANIFEST_RECOVERY_PLAN_INVALID")
    return plan


def _execute_preflight_manifest_recovery_plan(
    *,
    output_root: Path,
    plan_path: Path,
    plan_wrapper: Mapping[str, Any],
    original: Mapping[str, Any],
    process_provider: Callable[[], Sequence[Mapping[str, Any]]],
    failpoint: Callable[[str], None] | None,
) -> dict[str, Any]:
    plan_sha256 = _sha256_file(plan_path)
    plan = _validated_preflight_manifest_recovery_plan(
        output_root=output_root,
        plan_path=plan_path,
        wrapper=plan_wrapper,
        original=original,
    )
    drifted = base64.b64decode(str(plan["drifted_manifest_base64"]), validate=True)
    canonical = base64.b64decode(
        str(plan["canonical_manifest_base64"]), validate=True
    )
    original_plan = original["plan"]
    if not isinstance(original_plan, Mapping):
        raise RuntimeError("PREFLIGHT_MANIFEST_RECOVERY_PLAN_INVALID")
    _verify_freeze_for_plan(
        output_root=output_root,
        plan=original_plan,
        additional_allowed_manifest=drifted,
    )
    if process_provider():
        raise RuntimeError("ACTIVE_METHOD_X_WRITER")
    if _sha256_file(plan_path) != plan_sha256:
        raise RuntimeError("PREFLIGHT_MANIFEST_RECOVERY_PLAN_CHANGED")
    archive_path = _beneath(output_root, output_root / str(plan["archive_path"]))
    receipt_path = _beneath(output_root, output_root / str(plan["receipt_path"]))
    receipt_preexisting = os.path.lexists(receipt_path)
    _write_once(archive_path, drifted)
    manifest_path = _regular_file(output_root, output_root / "battery_manifest.json")
    current = manifest_path.read_bytes()
    if current == drifted:
        if process_provider():
            raise RuntimeError("ACTIVE_METHOD_X_WRITER")
        _atomic_replace(manifest_path, canonical)
    elif current != canonical:
        raise RuntimeError("PREFLIGHT_MANIFEST_RECOVERY_LIVE_CONFLICT")
    if failpoint is not None:
        failpoint("after_manifest_written")
    receipt_body = {
        "schema_version": 1,
        "artifact": "method_x_preflight_manifest_recovery_receipt",
        "status": "COMPLETED",
        "transaction_id": plan["transaction_id"],
        "original_transaction_id": plan["original_transaction_id"],
        "plan_path": plan_path.relative_to(output_root).as_posix(),
        "plan_sha256": plan_sha256,
        "original_plan_sha256": plan["original_plan_sha256"],
        "original_receipt_sha256": plan["original_receipt_sha256"],
        "archived_manifest_sha256": _sha256_file(archive_path),
        "canonical_manifest_sha256": _sha256_file(manifest_path),
        "allowed_differences": ["updated_at_utc"],
        "outer_loader_invocation_count": 0,
        "finalized_artifact_rewrite_count": 0,
    }
    _write_once(receipt_path, _canonical_file(_wrapped_payload(receipt_body)))
    if failpoint is not None:
        failpoint("after_receipt_written")
    return {
        "status": (
            "ALREADY_REPAIRED"
            if receipt_preexisting
            else "RESTORED_RECEIPT_BOUND_MANIFEST"
        ),
        "transaction_id": str(plan["transaction_id"]),
        "original_transaction_id": str(plan["original_transaction_id"]),
        "archive_path": str(plan["archive_path"]),
        "receipt_path": str(plan["receipt_path"]),
        "manifest_sha256": str(plan["canonical_manifest_sha256"]),
        "allowed_differences": ["updated_at_utc"],
        "outer_loader_invocation_count": 0,
        "finalized_artifact_rewrite_count": 0,
    }


def _existing_preflight_manifest_recovery_plan(
    *, output_root: Path, original: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    root = _beneath(output_root, output_root / PREFLIGHT_MANIFEST_RECOVERY_DIR)
    if not os.path.lexists(root):
        return None
    _beneath(output_root, root, must_exist=True)
    matches: list[tuple[Path, dict[str, Any]]] = []
    with os.scandir(root) as entries:
        for entry in entries:
            candidate_root = _beneath(output_root, Path(entry.path), must_exist=True)
            if not entry.is_dir(follow_symlinks=False) or _is_reparse(candidate_root):
                raise RuntimeError("RECONCILIATION_REPARSE_POINT_REJECTED")
            plan_path = _regular_file(output_root, candidate_root / "plan.json")
            wrapper = _read_json(plan_path)
            plan = _validated_preflight_manifest_recovery_plan(
                output_root=output_root,
                plan_path=plan_path,
                wrapper=wrapper,
                original=original,
            )
            if (
                plan.get("original_transaction_id") == original.get("transaction_id")
                and plan.get("canonical_manifest_sha256")
                == original.get("canonical_manifest_sha256")
            ):
                matches.append((plan_path, wrapper))
    if len(matches) > 1:
        raise RuntimeError("PREFLIGHT_MANIFEST_RECOVERY_NOT_UNIQUE")
    return matches[0] if matches else None


def repair_preflight_manifest_drift(
    *,
    output_root: Path,
    process_provider: Callable[
        [], Sequence[Mapping[str, Any]]
    ] = _default_process_provider,
    failpoint: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Restore only the known preflight timestamp mutation, transactionally.

    The original continuation plan and receipt remain the authority. Any
    scientific, scheduler, command, model, or recovery change is rejected.
    """

    load_completion_scope(DEFAULT_SCOPE_PATH)
    root = _beneath(PROJECT_ROOT, Path(output_root), must_exist=True)
    if process_provider():
        raise RuntimeError("ACTIVE_METHOD_X_WRITER")
    guard: Path | None = None
    try:
        guard = _acquire_guard(root)
        if process_provider():
            raise RuntimeError("ACTIVE_METHOD_X_WRITER")
        manifest_path = _regular_file(root, root / "battery_manifest.json")
        current = manifest_path.read_bytes()
        manifest = _read_json(manifest_path)
        original = _validated_completed_receipt_evidence(
            root, manifest, require_current_manifest=False
        )
        if original is None:
            raise RuntimeError("PREFLIGHT_MANIFEST_RECOVERY_RECEIPT_MISSING")
        original_plan = original.get("plan")
        if not isinstance(original_plan, Mapping):
            raise RuntimeError("PREFLIGHT_MANIFEST_RECOVERY_RECEIPT_INVALID")
        canonical = base64.b64decode(
            str(original_plan.get("post_manifest_base64") or ""), validate=True
        )
        if _digest_bytes(canonical) != original.get("canonical_manifest_sha256"):
            raise RuntimeError("PREFLIGHT_MANIFEST_RECOVERY_CANONICAL_INVALID")
        try:
            canonical_value = json.loads(canonical.decode("utf-8"))
            current_value = json.loads(current.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("PREFLIGHT_MANIFEST_RECOVERY_JSON_INVALID") from exc
        if not isinstance(canonical_value, dict) or not isinstance(current_value, dict):
            raise RuntimeError("PREFLIGHT_MANIFEST_RECOVERY_JSON_INVALID")

        existing = _existing_preflight_manifest_recovery_plan(
            output_root=root, original=original
        )
        if current == canonical:
            if existing is None:
                return {
                    "status": "ALREADY_CANONICAL",
                    "original_transaction_id": original["transaction_id"],
                    "manifest_sha256": original["canonical_manifest_sha256"],
                    "allowed_differences": [],
                    "outer_loader_invocation_count": 0,
                    "finalized_artifact_rewrite_count": 0,
                }
            plan_path, plan_wrapper = existing
            return _execute_preflight_manifest_recovery_plan(
                output_root=root,
                plan_path=plan_path,
                plan_wrapper=plan_wrapper,
                original=original,
                process_provider=process_provider,
                failpoint=failpoint,
            )

        differences = _manifest_top_level_differences(
            canonical_value, current_value
        )
        if differences != ["updated_at_utc"]:
            label = (differences[0] if differences else "UNKNOWN").upper()
            raise RuntimeError(
                f"PREFLIGHT_MANIFEST_DRIFT_NOT_TELEMETRY_ONLY:{label}"
            )
        if not isinstance(current_value.get("updated_at_utc"), str):
            raise RuntimeError(
                "PREFLIGHT_MANIFEST_DRIFT_NOT_TELEMETRY_ONLY:UPDATED_AT_UTC"
            )
        _verify_freeze_for_plan(
            output_root=root,
            plan=original_plan,
            additional_allowed_manifest=current,
        )
        if existing is None:
            plan_path, plan_wrapper = _build_preflight_manifest_recovery_plan(
                output_root=root,
                original=original,
                drifted_manifest=current,
                canonical_manifest=canonical,
            )
            plan_path = _beneath(root, plan_path)
            _write_once(plan_path, _canonical_file(plan_wrapper))
        else:
            plan_path, plan_wrapper = existing
        return _execute_preflight_manifest_recovery_plan(
            output_root=root,
            plan_path=plan_path,
            plan_wrapper=plan_wrapper,
            original=original,
            process_provider=process_provider,
            failpoint=failpoint,
        )
    except (KeyError, TypeError, ValueError, base64.binascii.Error) as exc:
        raise RuntimeError("PREFLIGHT_MANIFEST_RECOVERY_EVIDENCE_INVALID") from exc
    finally:
        _release_guard(guard)


def reconcile_live_continuation(
    *,
    output_root: Path,
    freeze_path: Path,
    process_provider: Callable[[], Sequence[Mapping[str, Any]]] = _default_process_provider,
    pid_liveness: Callable[[int], bool | None] = _default_pid_liveness,
    completed_validator: Callable[..., dict[str, Any]] = _default_completed_validation,
    recovery_provider: Callable[..., dict[str, Any]] = _default_recovery_provider,
    failpoint: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Reconcile the exact current battery state and return its immutable receipt."""

    load_completion_scope(DEFAULT_SCOPE_PATH)
    root = _beneath(PROJECT_ROOT, Path(output_root), must_exist=True)
    freeze = _regular_file(PROJECT_ROOT, Path(freeze_path))
    if process_provider():
        raise RuntimeError("ACTIVE_METHOD_X_WRITER")
    guard: Path | None = None
    try:
        guard = _acquire_guard(root)
        if process_provider():
            raise RuntimeError("ACTIVE_METHOD_X_WRITER")
        manifest_path = _regular_file(root, root / "battery_manifest.json")
        manifest = _read_json(manifest_path)
        completed = _completed_receipt(root, manifest)
        if completed is not None:
            return completed

        transaction_root = _beneath(root, root / TRANSACTION_DIR)
        existing_plans = (
            sorted(transaction_root.glob("*/plan.json"))
            if transaction_root.exists()
            else []
        )
        if existing_plans:
            if len(existing_plans) != 1:
                raise RuntimeError("CONTINUATION_TRANSACTION_NOT_UNIQUE")
            plan_path = _regular_file(root, existing_plans[0])
            return _execute_plan(
                output_root=root,
                plan_path=plan_path,
                plan_wrapper=_read_json(plan_path),
                process_provider=process_provider,
                failpoint=failpoint,
            )

        freeze_payload, freeze_rows = _load_and_verify_freeze(
            freeze, output_root=root
        )
        manifest_bytes = manifest_path.read_bytes()
        if _digest_bytes(manifest_bytes) != freeze_payload["battery_manifest_sha256"]:
            raise RuntimeError("FREEZE_MANIFEST_MISMATCH")
        lock_path = _regular_file(root, root / ".battery.lock")
        lock_bytes = lock_path.read_bytes()
        lock_payload = _read_json(lock_path)
        try:
            recorded_pid = int(lock_payload.get("pid"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("BATTERY_LOCK_OWNER_INVALID") from exc
        if recorded_pid <= 0:
            raise RuntimeError("BATTERY_LOCK_OWNER_INVALID")
        liveness = pid_liveness(recorded_pid)
        if liveness is None:
            raise RuntimeError("BATTERY_LOCK_OWNER_LIVENESS_UNKNOWN")
        if liveness:
            raise RuntimeError("BATTERY_LOCK_OWNER_ACTIVE")

        rows = _rows_by_name(manifest)
        completed_evidence: dict[str, Any] = {}
        for backbone in COMPLETED_BACKBONES:
            row = rows[backbone]
            if row.get("status") != "COMPLETED":
                raise RuntimeError("COMPLETED_MODEL_STATUS_INVALID")
            validated = completed_validator(
                backbone=backbone,
                row=row,
                output_root=root,
                manifest=manifest,
            )
            folds = validated.get("folds") if isinstance(validated, dict) else None
            if not isinstance(folds, list) or len(folds) != 5:
                raise RuntimeError("COMPLETED_MODEL_VALIDATION_INVALID")
            metrics_path = _regular_file(
                root,
                root / str(row["output_dir"]) / "three_d_metrics.json",
            )
            completed_evidence[backbone] = {
                "metrics_path": metrics_path.relative_to(root).as_posix(),
                "metrics_sha256": _sha256_file(metrics_path),
                "validated_fold_count": 5,
            }

        recovery = _validate_recovery(
            recovery_provider(
                output_root=root, row=rows[RESUMABLE_BACKBONE], manifest=manifest
            )
        )
        # A fast second process check closes the evidence-to-plan interval. The
        # shared guard prevents a new battery runner from completing its lock.
        if process_provider():
            raise RuntimeError("ACTIVE_METHOD_X_WRITER")
        # Recheck the two live mutation targets immediately before plan creation.
        manifest_logical = manifest_path.relative_to(PROJECT_ROOT).as_posix()
        lock_logical = lock_path.relative_to(PROJECT_ROOT).as_posix()
        if (
            _sha256_file(manifest_path) != freeze_rows[manifest_logical]["sha256"]
            or _sha256_file(lock_path) != freeze_rows[lock_logical]["sha256"]
        ):
            raise RuntimeError("FREEZE_MUTATION_TARGET_CHANGED")

        plan_wrapper = _build_plan(
            output_root=root,
            freeze_path=freeze,
            freeze_payload=freeze_payload,
            manifest=manifest,
            manifest_bytes=manifest_bytes,
            lock_bytes=lock_bytes,
            completed_evidence=completed_evidence,
            recovery=recovery,
        )
        plan = plan_wrapper["payload"]
        plan_path = _beneath(
            root,
            root / TRANSACTION_DIR / str(plan["transaction_id"]) / "plan.json",
        )
        _write_once(plan_path, _canonical_file(plan_wrapper))
        if failpoint is not None:
            failpoint("after_plan_written")
        return _execute_plan(
            output_root=root,
            plan_path=plan_path,
            plan_wrapper=plan_wrapper,
            process_provider=process_provider,
            failpoint=failpoint,
        )
    finally:
        _release_guard(guard)
