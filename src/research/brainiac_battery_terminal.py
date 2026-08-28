"""Evidence verifier for the exact five-model Method X battery terminal state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.research.brainiac_readiness import (
    EFFECTIVE_FIVE,
    EXCLUDED_BACKBONES,
    PROJECT_ROOT,
    _beneath,
    _is_reparse,
    bind_file,
)
from src.training.method_x_resume import reuse_committed_outer_result
from src.training.method_x_source_compatibility import (
    resolve_finalized_source_compatibility,
)
from src.utils.reproducibility import canonical_digest

FIVE_MODEL_AMENDMENT = (
    Path("amendments") / "method_x_battery_v3_five_model_amendment.v1.json"
)


def _canonical_payload_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"BATTERY_TERMINAL_JSON_INVALID:{path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"BATTERY_TERMINAL_JSON_ROOT_INVALID:{path.name}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_model_dir(
    row: Mapping[str, Any], *, backbone: str, battery_root: Path, project_root: Path
) -> Path:
    declared = Path(str(row.get("output_dir") or ""))
    if declared.is_absolute() or ".." in declared.parts:
        raise RuntimeError("BATTERY_MODEL_OUTPUT_PATH_INVALID")
    if declared.as_posix() != (Path("models") / backbone).as_posix():
        raise RuntimeError("BATTERY_MODEL_OUTPUT_PATH_INVALID")
    candidate = battery_root / declared
    resolved = _beneath(candidate, project_root, must_exist=True)
    if not resolved.is_dir() or _is_reparse(resolved):
        raise RuntimeError("BATTERY_MODEL_OUTPUT_DIRECTORY_INVALID")
    return resolved


def _validate_exact_five_amendment(
    *,
    manifest: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    battery_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    amendment_path = battery_root / FIVE_MODEL_AMENDMENT
    binding = bind_file(amendment_path, project_root=project_root)
    amendment = _json_object(Path(binding["path"]))
    unsigned = dict(amendment)
    recorded = unsigned.pop("scientific_payload_sha256", None)
    locked_inputs = manifest.get("locked_inputs")
    expected_proof = {
        name: "SMOKE_PASS_ONLY_NO_TRAINING_CHECKPOINT_OR_METRICS_ARTIFACTS"
        for name in EXCLUDED_BACKBONES
    }
    if (
        amendment.get("schema_version") != 1
        or amendment.get("amendment_id")
        != "method_x_battery_v3_five_model_amendment.v1"
        or amendment.get("append_only") is not True
        or recorded != _canonical_payload_sha256(unsigned)
        or amendment.get("effective_order") != list(EFFECTIVE_FIVE)
        or amendment.get("excluded_backbones") != list(EXCLUDED_BACKBONES)
        or amendment.get("not_started_proof") != expected_proof
        or not isinstance(locked_inputs, Mapping)
        or amendment.get("locked_config_sha256") != locked_inputs.get("config_sha256")
    ):
        raise RuntimeError("FIVE_MODEL_AMENDMENT_INVALID")
    for name in EXCLUDED_BACKBONES:
        row = rows[name]
        training = row.get("training")
        if (
            row.get("status") != "PROTOCOL_EXCLUDED_NOT_STARTED"
            or not isinstance(training, Mapping)
            or training.get("status") != "PROTOCOL_EXCLUDED_NOT_STARTED"
        ):
            raise RuntimeError("FIVE_MODEL_EXCLUSION_STATUS_INVALID")
        model_dir = _exact_model_dir(
            row,
            backbone=name,
            battery_root=battery_root,
            project_root=project_root,
        )
        for candidate in model_dir.rglob("*"):
            guarded = _beneath(candidate, project_root, must_exist=True)
            if guarded.is_dir():
                continue
            if not guarded.is_file() or _is_reparse(guarded):
                raise RuntimeError("FIVE_MODEL_EXCLUSION_ARTIFACT_UNSAFE")
            if guarded.relative_to(model_dir).as_posix() != "smoke.json":
                raise RuntimeError("FIVE_MODEL_EXCLUSION_NOT_STARTED_PROOF_FAILED")
    return binding


def _validate_contract(model_dir: Path, *, backbone: str) -> tuple[dict[str, Any], str]:
    contract = _json_object(model_dir / "run_contract.json")
    body = contract.get("run_contract")
    recorded = str(contract.get("run_contract_digest") or "")
    if not isinstance(body, Mapping):
        raise RuntimeError("BATTERY_RUN_CONTRACT_INVALID")
    embedded = str(body.get("digest") or "")
    projected = {key: value for key, value in body.items() if key != "digest"}
    parameters = body.get("parameters")
    if (
        not recorded
        or recorded != embedded
        or canonical_digest(projected) != recorded
        or not isinstance(parameters, Mapping)
        or parameters.get("backbone") != backbone
        or int(body.get("folds", -1)) != 5
        or int(body.get("seed", -1)) != 13
        or int(parameters.get("folds", -1)) != 5
        or int(parameters.get("seed", -1)) != 13
        or int(parameters.get("resolution", -1)) != 128
        or parameters.get("phase") != "final"
        or parameters.get("require_gpu") is not True
    ):
        raise RuntimeError("BATTERY_RUN_CONTRACT_DIGEST_INVALID")
    return dict(body), recorded


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load trusted, hash-verified internal checkpoint metadata on CPU."""

    try:
        import torch

        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("BATTERY_CHECKPOINT_PAYLOAD_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("BATTERY_CHECKPOINT_PAYLOAD_INVALID")
    return payload


def _validate_completed_model(
    *,
    backbone: str,
    row: Mapping[str, Any],
    battery_root: Path,
    project_root: Path,
    hash_checkpoints: bool,
) -> dict[str, Any]:
    model_dir = _exact_model_dir(
        row, backbone=backbone, battery_root=battery_root, project_root=project_root
    )
    required = [model_dir / "run_contract.json", model_dir / "three_d_metrics.json"]
    for path in required:
        bind_file(path, project_root=project_root)
    current_contract, contract_digest = _validate_contract(
        model_dir, backbone=backbone
    )
    metrics = _json_object(model_dir / "three_d_metrics.json")
    aggregate = metrics.get("aggregate")
    folds = metrics.get("folds")
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("status") != "COMPLETED"
        or aggregate.get("backbone") != backbone
        or int(aggregate.get("folds", 5)) != 5
        or not isinstance(folds, list)
        or len(folds) != 5
    ):
        raise RuntimeError("BATTERY_METRICS_NOT_FINALIZED_5_FOLDS")
    fold_rows: dict[int, Mapping[str, Any]] = {}
    for item in folds:
        if not isinstance(item, Mapping):
            raise RuntimeError("BATTERY_METRICS_FOLD_INVALID")
        fold = int(item.get("fold", -1))
        if fold in fold_rows or fold not in range(5):
            raise RuntimeError("BATTERY_METRICS_FOLD_INVALID")
        if int(item.get("outer_test_evaluation_count", 0)) != 1:
            raise RuntimeError("BATTERY_OUTER_EVALUATION_COUNT_INVALID")
        fold_rows[fold] = item
    if set(fold_rows) != set(range(5)):
        raise RuntimeError("BATTERY_METRICS_FOLD_INVALID")

    fold_evidence: list[dict[str, Any]] = []
    for fold in range(5):
        state_path = model_dir / f"method_x_3d_fold_{fold}_state.json"
        outer_path = model_dir / f"method_x_3d_fold_{fold}_outer_eval.json"
        checkpoint_path = model_dir / f"method_x_3d_fold_{fold}.pt"
        for path in (state_path, outer_path, checkpoint_path):
            resolved = _beneath(path, project_root, must_exist=True)
            if not resolved.is_file() or _is_reparse(resolved) or resolved.stat().st_size <= 0:
                raise RuntimeError("FINALIZED_FOLD_EVIDENCE_MISSING")
        state = _json_object(state_path)
        details = state.get("details")
        if (
            state.get("stage") != "FINALIZED"
            or int(state.get("fold", -1)) != fold
            or state.get("contract_digest") != contract_digest
            or not isinstance(details, Mapping)
        ):
            raise RuntimeError("BATTERY_FINALIZED_STATE_INVALID")
        expected_checkpoint = checkpoint_path.resolve()
        expected_outer = outer_path.resolve()
        declared_checkpoint = _beneath(
            project_root / Path(str(details.get("final_checkpoint") or "")),
            project_root,
            must_exist=True,
        )
        declared_outer = _beneath(
            project_root / Path(str(details.get("locked_result") or "")),
            project_root,
            must_exist=True,
        )
        if declared_checkpoint != expected_checkpoint or declared_outer != expected_outer:
            raise RuntimeError("BATTERY_FINALIZED_LINKAGE_INVALID")
        recorded_checkpoint_sha = str(details.get("final_checkpoint_sha256") or "")
        if len(recorded_checkpoint_sha) != 64:
            raise RuntimeError("BATTERY_CHECKPOINT_SHA256_INVALID")
        checkpoint_sha = _sha256_file(checkpoint_path)
        if checkpoint_sha != recorded_checkpoint_sha:
            raise RuntimeError("BATTERY_CHECKPOINT_SHA256_MISMATCH")
        outer = _json_object(outer_path)
        outer_file_sha = _sha256_file(outer_path)
        checkpoint = _load_checkpoint(checkpoint_path)
        compatibility = resolve_finalized_source_compatibility(
            model_dir=model_dir,
            state=state,
            checkpoint=checkpoint,
            locked_outer=outer,
            expected_contract=current_contract,
            expected_fold=fold,
            checkpoint_sha256=checkpoint_sha,
            outer_file_sha256=outer_file_sha,
        )
        validated = reuse_committed_outer_result(
            checkpoint,
            outer,
            expected_fold=fold,
            expected_contract_digest=compatibility.artifact_contract_digest,
            checkpoint_sha256=checkpoint_sha,
            outer_file_sha256=outer_file_sha,
            compatibility_evidence=compatibility.historical_authority,
        )
        if details.get("outer_result_digest") != validated["result_digest"]:
            raise RuntimeError("BATTERY_OUTER_STATE_DIGEST_MISMATCH")
        fold_evidence.append(
            {
                "fold": fold,
                "state_sha256": _sha256_file(state_path),
                "outer_file_sha256": outer_file_sha,
                "outer_result_digest": validated["result_digest"],
                "checkpoint_sha256": recorded_checkpoint_sha,
                "checkpoint_hash_recomputed": True,
                "artifact_contract_digest": compatibility.artifact_contract_digest,
                "target_contract_digest": compatibility.target_contract_digest,
                "source_transition_count": compatibility.transition_count,
                "outer_compatibility_status": validated["compatibility_status"],
            }
        )
    return {
        "backbone": backbone,
        "run_contract_digest": contract_digest,
        "folds": fold_evidence,
    }


def verify_battery_terminal(
    *,
    manifest_path: str | Path,
    battery_root: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    hash_checkpoints: bool = True,
) -> dict[str, Any]:
    """Return terminal=true only after five complete, linked, exactly-once folds."""

    root = Path(project_root)
    battery = _beneath(Path(battery_root), root, must_exist=True)
    manifest_binding = bind_file(manifest_path, project_root=root)
    manifest = _json_object(Path(manifest_binding["path"]))
    rows = manifest.get("models")
    blockers: list[str] = []
    by_name: dict[str, Mapping[str, Any]] = {}
    if not isinstance(rows, list) or len(rows) != 7:
        blockers.append("FIVE_CNN_BATTERY_IDENTITY_INVALID")
    else:
        for row in rows:
            if not isinstance(row, Mapping):
                blockers.append("FIVE_CNN_BATTERY_IDENTITY_INVALID")
                break
            name = str(row.get("backbone") or "")
            if not name or name in by_name:
                blockers.append("FIVE_CNN_BATTERY_IDENTITY_INVALID")
                break
            by_name[name] = row
    expected = set(EFFECTIVE_FIVE) | set(EXCLUDED_BACKBONES)
    if set(by_name) != expected:
        blockers.append("FIVE_CNN_BATTERY_IDENTITY_INVALID")
    if not blockers and any(
        str(by_name[name].get("status")) != "PROTOCOL_EXCLUDED_NOT_STARTED"
        for name in EXCLUDED_BACKBONES
    ):
        blockers.append("FIVE_CNN_BATTERY_EXCLUSION_INVALID")
    amendment_binding: dict[str, Any] | None = None
    if not blockers:
        try:
            amendment_binding = _validate_exact_five_amendment(
                manifest=manifest,
                rows=by_name,
                battery_root=battery,
                project_root=root,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            blockers.append(f"FIVE_MODEL_AMENDMENT_INVALID:{exc}")
    identity_valid = not blockers
    if not identity_valid:
        return {
            "status": "BLOCKED",
            "identity_valid": False,
            "terminal": False,
            "blockers": list(dict.fromkeys(blockers)),
            "manifest_binding": manifest_binding,
            "models": [],
            "amendment_binding": amendment_binding,
        }
    if any(str(by_name[name].get("status")) != "COMPLETED" for name in EFFECTIVE_FIVE):
        return {
            "status": "NONTERMINAL",
            "identity_valid": True,
            "terminal": False,
            "blockers": ["FIVE_CNN_BATTERY_NOT_TERMINAL"],
            "manifest_binding": manifest_binding,
            "models": [],
            "amendment_binding": amendment_binding,
        }
    if not hash_checkpoints:
        return {
            "status": "BLOCKED",
            "identity_valid": True,
            "terminal": False,
            "blockers": ["CHECKPOINT_HASH_RECOMPUTATION_REQUIRED"],
            "manifest_binding": manifest_binding,
            "models": [],
            "amendment_binding": amendment_binding,
            "checkpoint_hashes_recomputed": False,
        }
    evidence: list[dict[str, Any]] = []
    try:
        for backbone in EFFECTIVE_FIVE:
            evidence.append(
                _validate_completed_model(
                    backbone=backbone,
                    row=by_name[backbone],
                    battery_root=battery,
                    project_root=root,
                    hash_checkpoints=hash_checkpoints,
                )
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        code = str(exc)
        if isinstance(exc, FileNotFoundError) or "FINALIZED_FOLD_EVIDENCE_MISSING" in code:
            blocker = "FINALIZED_FOLD_EVIDENCE_MISSING"
        else:
            blocker = f"BATTERY_TERMINAL_EVIDENCE_INVALID:{code}"
        return {
            "status": "BLOCKED",
            "identity_valid": True,
            "terminal": False,
            "blockers": [blocker],
            "manifest_binding": manifest_binding,
            "models": evidence,
            "amendment_binding": amendment_binding,
        }
    return {
        "status": "VERIFIED_TERMINAL",
        "identity_valid": True,
        "terminal": True,
        "blockers": [],
        "manifest_binding": manifest_binding,
        "models": evidence,
        "amendment_binding": amendment_binding,
        "outer_evaluation_count": 25,
        "checkpoint_hashes_recomputed": bool(hash_checkpoints),
    }
