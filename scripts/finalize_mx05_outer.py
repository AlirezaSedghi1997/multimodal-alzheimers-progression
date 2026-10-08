#!/usr/bin/env python3
"""Finalize the locked MX-05 choice with journaled outer reporting.

The input is the immutable inner-only ``mx05_finalization_lock.json`` written
after Hybrid-vs-MX04 arbitration.  This script never chooses a model.  It
reuses the canonical seed-13 locked fold states, evaluates each outer fold
once as one logical post-lock operation, and commits one batch at a time.  A
rerun continues at the last committed cursor and records any process-level
replay risk without exposing identifiers.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_method_x_hybrid_orchestrator import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    _canonical_sha,
    _read,
    _sha,
)
from src.method_y_revival.strict_yaml import load_unique_yaml  # noqa: E402
from src.training.method_x_resume import (  # noqa: E402
    build_locked_outer_result,
    build_run_contract,
    validate_locked_outer_result,
)
from src.utils.io_helpers import atomic_write_json  # noqa: E402
from src.utils.reproducibility import split_membership_digest  # noqa: E402


LOCK_NAME = "reports/mx05_finalization_lock.json"
FINALIZATION_REPORT = "reports/mx05_outer_finalization.json"
FINAL_MANIFEST = "reports/final_model_selection_manifest.json"
ALLOWED_DECISIONS = {
    "PROMOTE_MX05_HYBRID": ("MX05_HYBRID", "amuse_x_hybrid3d"),
    "RETAIN_MX04_NO_HYBRID_PROMOTION": ("MX04_FINETUNE", None),
}
MX04_BACKBONES = {
    "resnet10_3d",
    "resnet18_3d",
    "resnet34_3d",
    "densenet121_3d",
    "mednext_3d_s",
}
REQUIRED_EVIDENCE_ROLES = (
    "replication_report",
    "paired_bootstrap_report",
    "cohort_probe_report",
    "promotion_report",
)
_STOP_REQUESTED = threading.Event()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _request_stop(_signum: int, _frame: Any) -> None:
    _STOP_REQUESTED.set()


def _install_signal_handlers() -> None:
    _STOP_REQUESTED.clear()
    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, _request_stop)


def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _output_relative(path: Path, output_root: Path) -> str:
    return path.resolve().relative_to(output_root.resolve()).as_posix()


def finalization_lock_integrity(payload: Mapping[str, Any]) -> str:
    return _canonical_sha(
        {key: value for key, value in payload.items() if key != "integrity_sha256"}
    )


def _resolve_under_output(value: str, output_root: Path, error: str) -> Path:
    path = Path(value)
    candidate = path if path.is_absolute() else output_root / path
    try:
        candidate = candidate.resolve()
        candidate.relative_to(output_root.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(error) from exc
    return candidate


def validate_finalization_lock(
    payload: Mapping[str, Any], *, output_root: Path
) -> dict[str, Any]:
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "LOCKED_INNER_ONLY"
        or payload.get("integrity_sha256") != finalization_lock_integrity(payload)
        or not str(payload.get("run_id") or "").strip()
        or payload.get("canonical_checkpoint_seed") != 13
        or payload.get("replication_seeds") != [13, 29, 47]
        or payload.get("outer_test_used_for_selection") is not False
    ):
        raise RuntimeError("MX05_FINALIZATION_LOCK_INVALID")
    decision = str(payload.get("selection_decision") or "")
    if decision not in ALLOWED_DECISIONS:
        raise RuntimeError("MX05_FINALIZATION_DECISION_INVALID")
    expected_kind, fixed_backbone = ALLOWED_DECISIONS[decision]
    backbone = str(payload.get("selected_backbone") or "")
    if payload.get("model_kind") != expected_kind:
        raise RuntimeError("MX05_FINALIZATION_DECISION_MODEL_MISMATCH")
    if fixed_backbone is not None and backbone != fixed_backbone:
        raise RuntimeError("MX05_FINALIZATION_DECISION_MODEL_MISMATCH")
    if expected_kind == "MX04_FINETUNE" and backbone not in MX04_BACKBONES:
        raise RuntimeError("MX05_FINALIZATION_DECISION_MODEL_MISMATCH")
    promotion = payload.get("promotion_gate")
    if not isinstance(promotion, Mapping):
        raise RuntimeError("MX05_FINALIZATION_PROMOTION_GATE_INVALID")
    if decision == "PROMOTE_MX05_HYBRID":
        valid_promotion = (
            promotion.get("status") == "PASS"
            and promotion.get("would_promote") is True
            and promotion.get("blockers") == []
        )
    else:
        valid_promotion = (
            promotion.get("status") == "NO_PROMOTION"
            and promotion.get("would_promote") is False
            and isinstance(promotion.get("blockers"), list)
            and bool(promotion.get("blockers"))
        )
    if not valid_promotion:
        raise RuntimeError("MX05_FINALIZATION_PROMOTION_GATE_INVALID")
    source_root = _resolve_under_output(
        str(payload.get("source_model_root") or ""),
        output_root,
        "MX05_FINALIZATION_SOURCE_ROOT_INVALID",
    )
    if not source_root.is_dir():
        raise RuntimeError("MX05_FINALIZATION_SOURCE_ROOT_MISSING")
    source_artifacts = payload.get("source_artifacts")
    if not isinstance(source_artifacts, list) or len(source_artifacts) != 7:
        raise RuntimeError("MX05_FINALIZATION_SOURCE_ARTIFACTS_INVALID")
    observed_roles: set[str] = set()
    for row in source_artifacts:
        if not isinstance(row, Mapping):
            raise RuntimeError("MX05_FINALIZATION_SOURCE_ARTIFACTS_INVALID")
        role = str(row.get("role") or "")
        expected = str(row.get("sha256") or "")
        path = _resolve_under_output(
            str(row.get("path") or ""),
            output_root,
            "MX05_FINALIZATION_SOURCE_ARTIFACT_PATH_INVALID",
        )
        if (
            not role
            or role in observed_roles
            or not path.is_file()
            or len(expected) != 64
            or _sha(path) != expected
        ):
            raise RuntimeError("MX05_FINALIZATION_SOURCE_ARTIFACTS_INVALID")
        observed_roles.add(role)
    expected_roles = {"metrics", "run_contract", *(f"fold_{fold}_checkpoint" for fold in range(5))}
    if observed_roles != expected_roles:
        raise RuntimeError("MX05_FINALIZATION_SOURCE_ARTIFACTS_INVALID")
    evidence_paths = payload.get("evidence_paths")
    if not isinstance(evidence_paths, Mapping) or set(evidence_paths) != set(
        REQUIRED_EVIDENCE_ROLES
    ):
        raise RuntimeError("MX05_FINALIZATION_EVIDENCE_PATHS_INVALID")
    for role, value in evidence_paths.items():
        path = _resolve_under_output(
            str(value), output_root, "MX05_FINALIZATION_EVIDENCE_PATH_INVALID"
        )
        if not path.is_file():
            raise RuntimeError(f"MX05_FINALIZATION_EVIDENCE_MISSING:{role}")
    return {
        **dict(payload),
        "source_model_root": str(source_root),
        "evidence_paths": {
            role: str(
                _resolve_under_output(
                    str(value), output_root, "MX05_FINALIZATION_EVIDENCE_PATH_INVALID"
                )
            )
            for role, value in evidence_paths.items()
        },
    }


def create_finalization_lock(
    *,
    output_root: Path,
    run_id: str,
    selection_decision: str,
    model_kind: str,
    selected_backbone: str,
    source_model_root: Path,
    selected_variant: str | None,
    promotion_gate: Mapping[str, Any],
    evidence_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Create or verify the immutable inner-only boundary before outer access."""

    source_model_root = source_model_root.resolve()
    source_model_root.relative_to(output_root.resolve())
    artifacts = [
        ("metrics", source_model_root / "three_d_metrics.json"),
        ("run_contract", source_model_root / "run_contract.json"),
        *[
            (f"fold_{fold}_checkpoint", source_model_root / f"method_x_3d_fold_{fold}.pt")
            for fold in range(5)
        ],
    ]
    if any(not path.is_file() for _, path in artifacts):
        raise RuntimeError("MX05_FINALIZATION_SOURCE_ARTIFACT_MISSING")
    if set(evidence_paths) != set(REQUIRED_EVIDENCE_ROLES):
        raise RuntimeError("MX05_FINALIZATION_EVIDENCE_PATHS_INVALID")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_INNER_ONLY",
        "run_id": str(run_id),
        "selection_decision": str(selection_decision),
        "model_kind": str(model_kind),
        "selected_backbone": str(selected_backbone),
        "selected_variant": selected_variant,
        "source_model_root": _output_relative(source_model_root, output_root),
        "canonical_checkpoint_seed": 13,
        "replication_seeds": [13, 29, 47],
        "outer_test_used_for_selection": False,
        "promotion_gate": dict(promotion_gate),
        "source_artifacts": [
            {
                "role": role,
                "path": _output_relative(path, output_root),
                "sha256": _sha(path),
            }
            for role, path in artifacts
        ],
        "evidence_paths": {
            role: _output_relative(Path(path), output_root)
            for role, path in evidence_paths.items()
        },
    }
    payload["integrity_sha256"] = finalization_lock_integrity(payload)
    lock_path = output_root / LOCK_NAME
    if lock_path.is_file():
        existing = _read(lock_path)
        if existing != payload:
            raise RuntimeError("MX05_FINALIZATION_LOCK_IMMUTABLE_MISMATCH")
    else:
        atomic_write_json(lock_path, payload)
    return validate_finalization_lock(payload, output_root=output_root)


def build_final_run_contract(
    *,
    source_contract: Mapping[str, Any],
    full_split_digest: str,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = source_contract.get("parameters")
    loader = source_contract.get("loader")
    if not isinstance(parameters, Mapping) or not isinstance(loader, Mapping):
        raise RuntimeError("MX05_FINALIZATION_SOURCE_CONTRACT_INVALID")
    final_parameters = dict(parameters)
    final_parameters.update(
        {
            "validity_inner_only": False,
            "hpo_selection_only": False,
            "throughput_profile_only": False,
            "outer_test_used_for_selection": False,
            "mx05_finalization_policy": "LOCKED_SEED13_SELECTION_STATE_OUTER_REPORTING_ONLY",
            "mx05_source_selection_contract_digest": source_contract.get("digest"),
            "mx05_finalization_lock_integrity": lock.get("integrity_sha256"),
        }
    )
    return build_run_contract(
        cache_sha256=str(source_contract.get("cache_sha256") or ""),
        split_digest=str(full_split_digest),
        spm_snapshot_sha256=str(source_contract.get("spm_snapshot_sha256") or ""),
        folds=int(source_contract.get("folds", -1)),
        epochs=int(source_contract.get("epochs", -1)),
        seed=13,
        parameters=final_parameters,
        loader=dict(loader),
    )


def _journal_integrity(payload: Mapping[str, Any]) -> str:
    return _canonical_sha(
        {key: value for key, value in payload.items() if key != "integrity_sha256"}
    )


def validate_outer_journal(
    payload: Mapping[str, Any],
    *,
    fold: int,
    logical_evaluation_id: str,
    expected_count: int,
    run_contract_digest: str,
) -> dict[str, Any]:
    labels = payload.get("y_true")
    probabilities = payload.get("y_prob")
    logits = payload.get("test_logits")
    cursor = payload.get("cursor")
    if (
        payload.get("schema_version") != 1
        or payload.get("integrity_sha256") != _journal_integrity(payload)
        or payload.get("fold") != int(fold)
        or payload.get("logical_evaluation_id") != logical_evaluation_id
        or payload.get("run_contract_digest") != run_contract_digest
        or payload.get("expected_count") != int(expected_count)
        or isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or cursor < 0
        or cursor > int(expected_count)
        or not isinstance(labels, list)
        or not isinstance(probabilities, list)
        or not isinstance(logits, list)
        or len(labels) != cursor
        or len(probabilities) != cursor
        or len(logits) != cursor
        or any(int(value) not in {0, 1} for value in labels)
        or any(
            not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            for value in probabilities
        )
        or any(not math.isfinite(float(value)) for value in logits)
        or payload.get("raw_identifiers_stored") is not False
        or payload.get("outer_test_used_for_selection") is not False
    ):
        raise RuntimeError("MX05_OUTER_JOURNAL_INVALID")
    status = str(payload.get("status") or "")
    if status not in {"IN_PROGRESS", "COMMITTED"}:
        raise RuntimeError("MX05_OUTER_JOURNAL_INVALID")
    if status == "COMMITTED" and cursor != int(expected_count):
        raise RuntimeError("MX05_OUTER_JOURNAL_INVALID")
    return dict(payload)


def _write_journal(path: Path, payload: dict[str, Any]) -> None:
    payload.pop("integrity_sha256", None)
    payload["updated_at_utc"] = _now()
    payload["integrity_sha256"] = _journal_integrity(payload)
    atomic_write_json(path, payload)


def _fold_membership(
    *,
    cache_path: Path,
    split_dir: Path,
    fold: int,
    seed: int,
    val_fraction: float,
) -> dict[str, Any]:
    from scripts.train_3d_mri import (
        cache_index,
        ok_indices,
        split_fit_val_indices,
        split_indices,
    )

    sample_to_idx = cache_index(cache_path)
    valid = ok_indices(cache_path)
    train_idx, train_y = split_indices(
        split_dir / f"fold_{fold}_train.csv", sample_to_idx, valid
    )
    test_idx, test_y = split_indices(
        split_dir / f"fold_{fold}_test.csv", sample_to_idx, valid
    )
    fit_idx, fit_y, val_idx, val_y = split_fit_val_indices(
        train_idx,
        train_y,
        cache_path=cache_path,
        ok=valid,
        val_fraction=float(val_fraction),
        seed=int(seed),
        fold=int(fold),
    )
    return {
        "fit_idx": fit_idx,
        "fit_y": fit_y,
        "val_idx": val_idx,
        "val_y": val_y,
        "test_idx": tuple(test_idx),
        "test_y": tuple(test_y),
        "selection_membership_digest": split_membership_digest(
            fit=fit_idx, validation=val_idx, outer_test=()
        ),
        "final_membership_digest": split_membership_digest(
            fit=fit_idx, validation=val_idx, outer_test=test_idx
        ),
    }


def _predict_outer_suffix(
    *,
    checkpoint: Mapping[str, Any],
    cache_path: Path,
    membership: Mapping[str, Any],
    cursor: int,
    journal: dict[str, Any],
    journal_path: Path,
    device: Any,
) -> dict[str, Any]:
    import torch

    from scripts.train_3d_mri import H5MethodXDataset, normalize_tabular
    from src.evaluation.method_x_xai_forward import build_model_from_checkpoint

    model, state_source = build_model_from_checkpoint(checkpoint, device=device)
    journal["model_state_source"] = state_source
    indices = list(membership["test_idx"])[cursor:]
    labels = list(membership["test_y"])[cursor:]
    dataset = H5MethodXDataset(
        cache_path,
        indices,
        labels,
        input_mode=str(checkpoint.get("input_mode", "mri_plus_tabular")),
        drop_tabular_feature=str(checkpoint.get("drop_tabular_feature", "none")),
        seed=13 + int(journal["fold"]) + 2000,
    )
    tab_mean = torch.as_tensor(checkpoint.get("tab_mean"), device=device)
    tab_std = torch.as_tensor(checkpoint.get("tab_std"), device=device)
    model.eval()
    with torch.no_grad():
        for item in range(len(dataset)):
            image, tabular, label, presence = dataset[item]
            image = image.unsqueeze(0).to(device)
            tabular = normalize_tabular(
                tabular.unsqueeze(0).to(device), tab_mean, tab_std
            )
            presence = presence.reshape(1).to(device)
            logit = model(image, tabular, tab_presence=presence).reshape(-1)
            probability = torch.sigmoid(logit)
            journal["y_true"].append(int(label.item()))
            journal["y_prob"].append(float(probability[0].detach().cpu().item()))
            journal["test_logits"].append(float(logit[0].detach().cpu().item()))
            journal["cursor"] = len(journal["y_true"])
            journal["committed_batches"] = int(journal.get("committed_batches", 0)) + 1
            _write_journal(journal_path, journal)
            if _STOP_REQUESTED.is_set():
                raise RuntimeError("MX05_OUTER_PAUSED_RESUMABLE:OPERATOR_STOP")
    return journal


def _fold_metrics(
    *,
    fold: int,
    journal: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from scripts.train_3d_mri import (
        average_precision,
        brier,
        evaluate_with_train_selected_threshold,
        roc_auc,
        temperature_scaling_report,
    )

    selection = source_checkpoint.get("selection_only_evaluation")
    if not isinstance(selection, Mapping):
        raise RuntimeError(f"MX05_SOURCE_SELECTION_EVALUATION_MISSING:FOLD_{fold}")
    val_y = selection.get("y_true")
    val_logits = selection.get("logits")
    val_prob = selection.get("y_prob")
    if not all(isinstance(value, list) and value for value in (val_y, val_logits, val_prob)):
        raise RuntimeError(f"MX05_SOURCE_SELECTION_EVALUATION_INVALID:FOLD_{fold}")
    outer_y = [int(value) for value in journal["y_true"]]
    outer_prob = [float(value) for value in journal["y_prob"]]
    outer_logits = [float(value) for value in journal["test_logits"]]
    temperature, calibrated = temperature_scaling_report(
        validation_y=[int(value) for value in val_y],
        validation_logits=[float(value) for value in val_logits],
        test_y=outer_y,
        test_logits=outer_logits,
    )
    threshold = evaluate_with_train_selected_threshold(
        validation_y=[int(value) for value in val_y],
        validation_prob=[float(value) for value in val_prob],
        test_y=outer_y,
        test_prob=outer_prob,
    )
    metrics = {
        "fold": int(fold),
        "n": len(outer_y),
        "test_n": len(outer_y),
        "roc_auc": float(roc_auc(outer_y, outer_prob)),
        "pr_auc": float(average_precision(outer_y, outer_prob)),
        "brier": float(brier(outer_y, outer_prob)),
        "evaluation_scope": "locked_outer_test_reporting_only",
        "outer_test_access_count": 1,
        "outer_test_evaluation_count": 1,
        "outer_test_used_for_selection": False,
        "temperature_scaling": temperature,
        "threshold_metrics": threshold,
        "physical_process_starts": int(journal.get("physical_process_starts", 1)),
        "uncommitted_batch_replay_possible": bool(
            journal.get("uncommitted_batch_replay_possible")
        ),
    }
    return metrics, {"temperature_scaling": temperature, "calibrated_prob": calibrated}


def _final_checkpoint_reusable(
    *,
    checkpoint_path: Path,
    state_path: Path,
    outer_path: Path,
    journal_path: Path,
    fold: int,
    final_contract: Mapping[str, Any],
    membership_digest: str,
) -> bool:
    if not all(
        path.is_file()
        for path in (checkpoint_path, state_path, outer_path, journal_path)
    ):
        return False
    try:
        import torch

        from src.evaluation.xai_validation import validate_xai_checkpoint_binding

        state = _read(state_path)
        journal = _read(journal_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        outer = _read(outer_path)
        validate_locked_outer_result(
            outer,
            expected_fold=fold,
            expected_contract_digest=str(final_contract["digest"]),
            expected_split_membership_digest=membership_digest,
        )
        validate_xai_checkpoint_binding(
            checkpoint=checkpoint,
            expected_fold=fold,
            expected_cache_sha256=str(final_contract["cache_sha256"]),
            expected_run_contract_digest=str(final_contract["digest"]),
            expected_split_membership_digest=membership_digest,
            expected_backbone=str(final_contract["parameters"]["backbone"]),
        )
        return bool(
            state.get("stage") == "FINALIZED"
            and state.get("contract_digest") == final_contract["digest"]
            and (state.get("details") or {}).get("checkpoint_sha256")
            == _sha(checkpoint_path)
            and journal.get("status") == "COMMITTED"
            and journal.get("cursor") == journal.get("expected_count")
            and journal.get("locked_result_sha256") == _sha(outer_path)
            and journal.get("final_checkpoint_sha256") == _sha(checkpoint_path)
            and journal.get("integrity_sha256") == _journal_integrity(journal)
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _artifact(role: str, path: Path, output_root: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"MX05_FINAL_ARTIFACT_MISSING:{role}")
    return {
        "role": role,
        "path": _output_relative(path, output_root),
        "sha256": _sha(path),
    }


def build_final_selection_manifest(
    *,
    output_root: Path,
    lock: Mapping[str, Any],
    selected_model_root: Path,
) -> dict[str, Any]:
    evidence_paths = {
        role: Path(str(path)) for role, path in lock["evidence_paths"].items()
    }
    evidence_paths.update(
        {
            "outer_evaluation_report": output_root / "reports/mx05_outer_evaluation_report.json",
            "calibration_report": output_root / "reports/mx05_calibration_report.json",
        }
    )
    artifacts = [
        _artifact("metrics", selected_model_root / "three_d_metrics.json", output_root),
        _artifact("run_contract", selected_model_root / "run_contract.json", output_root),
        *[
            _artifact(
                f"fold_{fold}_checkpoint",
                selected_model_root / f"method_x_3d_fold_{fold}.pt",
                output_root,
            )
            for fold in range(5)
        ],
        *[
            _artifact(role, path, output_root)
            for role, path in evidence_paths.items()
        ],
    ]
    manifest = {
        "schema_version": 1,
        "status": "COMPLETED",
        "run_id": lock["run_id"],
        "model_kind": lock["model_kind"],
        "selected_backbone": lock["selected_backbone"],
        "selected_variant": lock.get("selected_variant"),
        "selection_decision": lock["selection_decision"],
        "canonical_checkpoint_seed": 13,
        "selected_model_root": _output_relative(selected_model_root, output_root),
        "replication_seeds": [13, 29, 47],
        "replication_status": "COMPLETED",
        "outer_test_used_for_selection": False,
        "outer_evaluation": {"status": "COMPLETED", "count": 1},
        "promotion_gate": dict(lock["promotion_gate"]),
        "artifacts": artifacts,
    }
    return manifest


def _write_final_manifest(
    *, output_root: Path, lock: Mapping[str, Any], selected_model_root: Path
) -> dict[str, Any]:
    manifest = build_final_selection_manifest(
        output_root=output_root,
        lock=lock,
        selected_model_root=selected_model_root,
    )
    path = output_root / FINAL_MANIFEST
    if path.is_file():
        existing = _read(path)
        if existing != manifest:
            raise RuntimeError("MX05_FINAL_SELECTION_MANIFEST_IMMUTABLE_MISMATCH")
    else:
        atomic_write_json(path, manifest)
    from scripts.run_method_x_anatomical_xai import resolve_xai_selection

    resolved = resolve_xai_selection(
        corrected_root=output_root,
        require_completed_manifest=True,
    )
    if (
        resolved.get("selection_scope") != "FINAL_SELECTION_LOCK"
        or resolved.get("backbone") != lock["selected_backbone"]
    ):
        raise RuntimeError("MX05_FINAL_SELECTION_XAI_HANDOFF_INVALID")
    return manifest


def finalize_outer(
    *, output_root: Path, config_path: Path
) -> dict[str, Any]:
    config = load_unique_yaml(config_path)
    if not isinstance(config, Mapping):
        raise RuntimeError("MX05_OUTER_CONFIG_INVALID")
    outer_config = config.get("outer_finalization")
    if not isinstance(outer_config, Mapping):
        raise RuntimeError("MX05_OUTER_CONFIG_MISSING")
    lock = validate_finalization_lock(
        _read(output_root / LOCK_NAME), output_root=output_root
    )
    source_root = Path(str(lock["source_model_root"]))
    from src.evaluation.method_x_finetune_selection import validate_selection_run

    validate_selection_run(
        source_root,
        expected_backbone=str(lock["selected_backbone"]),
        expected_seed=13,
        folds=5,
    )
    source_contract_doc = _read(source_root / "run_contract.json")
    source_contract = source_contract_doc.get("run_contract")
    if not isinstance(source_contract, Mapping):
        raise RuntimeError("MX05_OUTER_SOURCE_CONTRACT_INVALID")
    split_dir = ROOT / "splits"
    cache_path = ROOT / "data/cached/t1_volume_cache_128.h5"
    from scripts.train_3d_mri import method_x_split_contract_digest

    full_split_digest = method_x_split_contract_digest(
        split_dir, folds=5, selection_only=False
    )
    final_contract = build_final_run_contract(
        source_contract=source_contract,
        full_split_digest=full_split_digest,
        lock=lock,
    )
    destination = (
        output_root
        / str(outer_config["destination_root"])
        / str(lock["run_id"])
    ).resolve()
    destination.relative_to(output_root.resolve())
    destination.mkdir(parents=True, exist_ok=True)
    contract_path = destination / "run_contract.json"
    contract_doc = {
        "run_contract": final_contract,
        "run_contract_digest": final_contract["digest"],
        "source_selection_run_contract_digest": source_contract.get("digest"),
        "finalization_lock_integrity": lock["integrity_sha256"],
    }
    if contract_path.is_file() and _read(contract_path) != contract_doc:
        raise RuntimeError("MX05_OUTER_FINAL_CONTRACT_IMMUTABLE_MISMATCH")
    if not contract_path.is_file():
        atomic_write_json(contract_path, contract_doc)

    import torch

    from scripts.train_3d_mri import atomic_torch_save

    if not torch.cuda.is_available():
        raise RuntimeError("MX05_OUTER_CUDA_UNAVAILABLE")
    device = torch.device("cuda")
    folds: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    all_y: list[int] = []
    all_prob: list[float] = []
    parameters = source_contract.get("parameters") or {}
    val_fraction = float(parameters.get("val_fraction", 0.15))
    seed = int(source_contract.get("seed", 13))
    for fold in range(5):
        membership = _fold_membership(
            cache_path=cache_path,
            split_dir=split_dir,
            fold=fold,
            seed=seed,
            val_fraction=val_fraction,
        )
        source_checkpoint_path = source_root / f"method_x_3d_fold_{fold}.pt"
        source_checkpoint = torch.load(
            source_checkpoint_path, map_location="cpu", weights_only=True
        )
        source_selection = source_checkpoint.get("selection_only_evaluation")
        if (
            not isinstance(source_selection, Mapping)
            or source_selection.get("split_membership_digest")
            != membership["selection_membership_digest"]
            or source_checkpoint.get("hpo_selection_only") is not True
        ):
            raise RuntimeError(f"MX05_OUTER_SOURCE_SELECTION_BINDING_INVALID:FOLD_{fold}")
        checkpoint_path = destination / f"method_x_3d_fold_{fold}.pt"
        state_path = destination / f"method_x_3d_fold_{fold}_state.json"
        outer_path = destination / f"method_x_3d_fold_{fold}_outer_eval.json"
        journal_path = destination / f"method_x_3d_fold_{fold}_outer_journal.json"
        if _final_checkpoint_reusable(
            checkpoint_path=checkpoint_path,
            state_path=state_path,
            outer_path=outer_path,
            journal_path=journal_path,
            fold=fold,
            final_contract=final_contract,
            membership_digest=membership["final_membership_digest"],
        ):
            final_checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            metrics = dict(final_checkpoint["locked_outer_test_evaluation"]["metrics"])
            outer = _read(outer_path)
            folds.append(metrics)
            all_y.extend(int(value) for value in outer["y_true"])
            all_prob.extend(float(value) for value in outer["y_prob"])
            calibration_rows.append(
                {"fold": fold, **metrics.get("temperature_scaling", {})}
            )
            continue

        logical_id = _canonical_sha(
            {
                "run_contract_digest": final_contract["digest"],
                "fold": fold,
                "finalization_lock_integrity": lock["integrity_sha256"],
            }
        )
        if journal_path.is_file():
            journal = validate_outer_journal(
                _read(journal_path),
                fold=fold,
                logical_evaluation_id=logical_id,
                expected_count=len(membership["test_idx"]),
                run_contract_digest=str(final_contract["digest"]),
            )
            if journal["status"] == "IN_PROGRESS" and journal["cursor"] < journal["expected_count"]:
                journal["physical_process_starts"] = int(
                    journal.get("physical_process_starts", 1)
                ) + 1
                journal["uncommitted_batch_replay_possible"] = True
                _write_journal(journal_path, journal)
        else:
            journal = {
                "schema_version": 1,
                "status": "IN_PROGRESS",
                "fold": fold,
                "logical_evaluation_id": logical_id,
                "run_contract_digest": final_contract["digest"],
                "split_membership_digest": membership["final_membership_digest"],
                "expected_count": len(membership["test_idx"]),
                "cursor": 0,
                "committed_batches": 0,
                "physical_process_starts": 1,
                "uncommitted_batch_replay_possible": False,
                "outer_test_used_for_selection": False,
                "raw_identifiers_stored": False,
                "y_true": [],
                "y_prob": [],
                "test_logits": [],
                "created_at_utc": _now(),
            }
            _write_journal(journal_path, journal)
        if journal["cursor"] < journal["expected_count"]:
            journal = _predict_outer_suffix(
                checkpoint=source_checkpoint,
                cache_path=cache_path,
                membership=membership,
                cursor=int(journal["cursor"]),
                journal=journal,
                journal_path=journal_path,
                device=device,
            )
        locked_outer = build_locked_outer_result(
            fold=fold,
            run_contract_digest=str(final_contract["digest"]),
            split_membership_digest=str(membership["final_membership_digest"]),
            y_true=[int(value) for value in journal["y_true"]],
            y_prob=[float(value) for value in journal["y_prob"]],
            test_logits=[float(value) for value in journal["test_logits"]],
        )
        if outer_path.is_file() and _read(outer_path) != locked_outer:
            raise RuntimeError(f"MX05_OUTER_RESULT_IMMUTABLE_MISMATCH:FOLD_{fold}")
        if not outer_path.is_file():
            atomic_write_json(outer_path, locked_outer)
        metrics, calibration = _fold_metrics(
            fold=fold,
            journal=journal,
            source_checkpoint=source_checkpoint,
        )
        embedded_outer = {
            **locked_outer,
            "metrics": metrics,
            "calibrated_prob": list(calibration["calibrated_prob"]),
        }
        final_checkpoint = dict(source_checkpoint)
        final_checkpoint.update(
            {
                "validity_inner_only": False,
                "hpo_selection_only": False,
                "throughput_profile_only": False,
                "evaluation_scope": "locked_outer_test_reporting_only",
                "outer_test_access_count": 1,
                "outer_test_evaluation_count": 1,
                "outer_test_used_for_selection": False,
                "selection_only_evaluation_source": source_selection,
                "selection_only_evaluation": None,
                "locked_outer_test_evaluation": embedded_outer,
                "run_contract": dict(final_contract),
                "run_contract_digest": final_contract["digest"],
                "split_membership_digest": membership["final_membership_digest"],
                "checkpoint_kind": "FINALIZED_FROM_LOCKED_SELECTION_CHECKPOINT",
                "source_selection_checkpoint": _output_relative(
                    source_checkpoint_path, output_root
                ),
                "source_selection_checkpoint_sha256": _sha(source_checkpoint_path),
                "finalization_lock_integrity": lock["integrity_sha256"],
                "fold_terminal": True,
            }
        )
        atomic_torch_save(final_checkpoint, checkpoint_path)
        state = {
            "schema_version": 1,
            "fold": fold,
            "stage": "FINALIZED",
            "previous_stage": "OUTER_EVAL_DONE",
            "contract_digest": final_contract["digest"],
            "updated_at_utc": _now(),
            "details": {
                "checkpoint": _project_relative(checkpoint_path),
                "checkpoint_sha256": _sha(checkpoint_path),
                "source_selection_checkpoint_sha256": _sha(source_checkpoint_path),
                "split_membership_digest": membership["final_membership_digest"],
                "outer_test_access_count": 1,
                "logical_evaluation_id": logical_id,
                "physical_process_starts": journal["physical_process_starts"],
                "uncommitted_batch_replay_possible": journal[
                    "uncommitted_batch_replay_possible"
                ],
                "selection_only": False,
                "hpo_selection_only": False,
            },
        }
        atomic_write_json(state_path, state)
        journal["status"] = "COMMITTED"
        journal["locked_result_sha256"] = _sha(outer_path)
        journal["final_checkpoint_sha256"] = _sha(checkpoint_path)
        _write_journal(journal_path, journal)
        if not _final_checkpoint_reusable(
            checkpoint_path=checkpoint_path,
            state_path=state_path,
            outer_path=outer_path,
            journal_path=journal_path,
            fold=fold,
            final_contract=final_contract,
            membership_digest=membership["final_membership_digest"],
        ):
            raise RuntimeError(f"MX05_OUTER_FINAL_CHECKPOINT_INVALID:FOLD_{fold}")
        folds.append(metrics)
        all_y.extend(int(value) for value in journal["y_true"])
        all_prob.extend(float(value) for value in journal["y_prob"])
        calibration_rows.append({"fold": fold, **calibration["temperature_scaling"]})
        del final_checkpoint, source_checkpoint
        torch.cuda.empty_cache()
        if _STOP_REQUESTED.is_set():
            raise RuntimeError("MX05_OUTER_PAUSED_RESUMABLE:OPERATOR_STOP")

    from scripts.train_3d_mri import average_precision, brier, roc_auc

    aggregate = {
        "status": "COMPLETED",
        "model": "MethodX3DClassifier",
        "mode": "FULL_3D_TRAINING",
        "phase": "final",
        "backbone": lock["selected_backbone"],
        "folds": 5,
        "completed_folds": 5,
        "seed": 13,
        "evaluation_scope": "locked_outer_test_reporting_only",
        "validity_inner_only": False,
        "hpo_selection_only": False,
        "outer_test_access_count": 5,
        "outer_test_logical_program_count": 1,
        "outer_test_used_for_selection": False,
        "roc_auc": float(roc_auc(all_y, all_prob)),
        "pr_auc": float(average_precision(all_y, all_prob)),
        "brier": float(brier(all_y, all_prob)),
        "checkpoints": [
            _project_relative(destination / f"method_x_3d_fold_{fold}.pt")
            for fold in range(5)
        ],
        "finalization_policy": "LOCKED_SEED13_SELECTION_STATE_OUTER_REPORTING_ONLY",
        "final_claim_allowed": False,
        "claim_boundary": (
            "Internal PET-free technical evaluation; no diagnosis, causal-region, "
            "or external-generalization claim."
        ),
    }
    atomic_write_json(
        destination / "three_d_metrics.json", {"aggregate": aggregate, "folds": folds}
    )
    journal_rows = [_read(destination / f"method_x_3d_fold_{fold}_outer_journal.json") for fold in range(5)]
    outer_report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "run_id": lock["run_id"],
        "selected_backbone": lock["selected_backbone"],
        "count": 1,
        "logical_scope": "ONE_POST_LOCK_PROGRAM_EVALUATION_ACROSS_FIVE_OUTER_FOLDS",
        "fold_evaluation_count": 5,
        "per_fold_evaluation_count": 1,
        "outer_test_used_for_selection": False,
        "raw_identifiers_stored": False,
        "physical_process_starts": sum(int(row["physical_process_starts"]) for row in journal_rows),
        "uncommitted_batch_replay_possible": any(
            bool(row["uncommitted_batch_replay_possible"]) for row in journal_rows
        ),
        "folds": [
            {
                "fold": int(row["fold"]),
                "logical_evaluation_id": row["logical_evaluation_id"],
                "n": int(row["expected_count"]),
                "physical_process_starts": int(row["physical_process_starts"]),
                "uncommitted_batch_replay_possible": bool(
                    row["uncommitted_batch_replay_possible"]
                ),
                "locked_result_sha256": row["locked_result_sha256"],
            }
            for row in journal_rows
        ],
    }
    calibration_report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "run_id": lock["run_id"],
        "selected_backbone": lock["selected_backbone"],
        "fit_scope": "INNER_VALIDATION_ONLY",
        "outer_labels_used_for_calibration_fit": False,
        "folds": calibration_rows,
    }
    atomic_write_json(
        output_root / "reports/mx05_outer_evaluation_report.json", outer_report
    )
    atomic_write_json(
        output_root / "reports/mx05_calibration_report.json", calibration_report
    )
    manifest = _write_final_manifest(
        output_root=output_root,
        lock=lock,
        selected_model_root=destination,
    )
    report = {
        "schema_version": 1,
        "status": "COMPLETED",
        "run_id": lock["run_id"],
        "selected_backbone": lock["selected_backbone"],
        "selected_model_root": _output_relative(destination, output_root),
        "selection_decision": lock["selection_decision"],
        "outer_evaluation": {"status": "COMPLETED", "count": 1},
        "outer_test_used_for_selection": False,
        "final_selection_manifest_sha256": _sha(output_root / FINAL_MANIFEST),
        "final_selection_manifest": manifest,
    }
    atomic_write_json(output_root / FINALIZATION_REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    _install_signal_handlers()
    try:
        report = finalize_outer(
            output_root=output_root.resolve(), config_path=config_path.resolve()
        )
    except (KeyboardInterrupt, OSError, RuntimeError, ValueError) as exc:
        paused = isinstance(exc, KeyboardInterrupt) or "PAUSED_RESUMABLE" in str(exc)
        print(
            json.dumps(
                {"status": "PAUSED_RESUMABLE" if paused else "BLOCKED", "blocker": str(exc)},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 75 if paused else 3
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
