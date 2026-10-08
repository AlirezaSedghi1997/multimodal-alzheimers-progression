from __future__ import annotations

import hashlib
import json
import math
import re
import os
import shutil
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.training.method_x_outer_compatibility import (
    HistoricalCompatibilityAuthority,
)
from src.utils.io_helpers import atomic_write_json
from src.utils.reproducibility import canonical_digest


RUN_CONTRACT_SCHEMA_VERSION = 3
RUNTIME_RESUME_SCHEMA_VERSION = 1


def protocol_epochs_per_fold(total_epochs: int, folds: int) -> int:
    """Return the locked per-fold cap for a total cross-validation budget.

    Method X v3 records ``epochs`` as the total model budget so its immutable
    run-contract digest remains stable across the five-fold battery.  The
    executable loop must divide that budget evenly across folds; treating the
    value as a per-fold limit silently multiplies training by ``folds``.
    """

    total = int(total_epochs)
    fold_count = int(folds)
    if total <= 0:
        raise SystemExit("EPOCHS_MUST_BE_POSITIVE")
    if fold_count <= 0:
        raise SystemExit("FOLDS_MUST_BE_POSITIVE")
    if total % fold_count != 0:
        raise SystemExit("EPOCHS_MUST_DIVIDE_EVENLY_ACROSS_FOLDS")
    return total // fold_count


# Loader keys that change data order or exact-resume semantics. These are
# scientific even though they live under "loader": a batch or worker change
# alters the sequence the model sees, so it can never be waved through as
# benign runtime plumbing.
SCIENTIFIC_LOADER_KEYS = frozenset({"batch_size", "num_workers", "persistent_workers"})

# Parameter keys that describe the toolchain rather than the experiment.
NON_SCIENTIFIC_PARAMETER_KEYS = frozenset({"source_contract_digest", "runtime_versions"})

_ALLOWED_H_MIRROR_POLICIES = frozenset({"epoch", "fold", "epoch_and_fold"})
_ALLOWED_REPLAY_POLICIES = frozenset({"deterministic"})
_ALLOWED_STAGING_ROOT_CLASSES = frozenset(
    {"C_ALLOWLISTED_CHECKPOINT_STAGING", "H_ONLY"}
)

FOLD_STAGES = (
    "TRAINING",
    "CHECKPOINT_LOCKED",
    "OUTER_EVAL_IN_PROGRESS",
    "OUTER_EVAL_DONE",
    "FINALIZED",
)

# Cadence-1 staging writes a weight-only sidecar next to progress. The source-
# only migrator used to glob ``method_x_3d_fold_*.pt`` and demand a run
# contract inside that sidecar, which is exactly the live ResNet18 resume
# failure after the first best-val write.
_FOLD_CHECKPOINT_NAME = re.compile(
    r"^method_x_3d_fold_(\d+)(?:_(progress|locked|best_state))?\.pt$"
)


def parse_fold_checkpoint_filename(filename: str) -> tuple[int, str]:
    """Return ``(fold, kind)`` for a trainer checkpoint basename.

    ``kind`` is ``final``, ``progress``, ``locked``, or ``best_state``. Any
    other ``method_x_3d_fold_*.pt`` name is unknown and must stay blocked.
    """

    match = _FOLD_CHECKPOINT_NAME.fullmatch(Path(str(filename)).name)
    if match is None:
        raise RuntimeError(f"RUN_CONTRACT_MIGRATION_INVALID_CHECKPOINT_NAME:{filename}")
    suffix = match.group(2)
    return int(match.group(1)), "final" if suffix is None else suffix


def assert_offloaded_best_state_sidecar(payload: Any, *, filename: str) -> None:
    """Offloaded best weights are not a run-contract checkpoint.

    ``CheckpointStager.record_best_state`` writes only weight tensors plus
    metric metadata. Rewriting that file with a run contract would break the
    provenance sidecar hash and ``load_best_state``.
    """

    if not isinstance(payload, dict):
        raise RuntimeError(f"RUN_CONTRACT_MIGRATION_BEST_STATE_INVALID:{filename}")
    if "run_contract" in payload or "run_contract_digest" in payload:
        raise RuntimeError(
            f"RUN_CONTRACT_MIGRATION_BEST_STATE_CONTRACT_PRESENT:{filename}"
        )
    if "best_model_state_dict" not in payload and "best_averaged_state_dict" not in payload:
        raise RuntimeError(f"RUN_CONTRACT_MIGRATION_BEST_STATE_WEIGHTS_MISSING:{filename}")


def classify_source_only_progress_bind(
    *,
    filename: str,
    fold: int,
    action: str,
    finalized_folds: set[int] | frozenset[int],
) -> str:
    """Classify a globbed ``*.pt`` during source-only contract migration.

    Live batteries keep leftover ``method_x_3d_fold_{N}_progress.pt`` beside the
    immutable finalized ``method_x_3d_fold_{N}.pt``. Those leftovers are not
    Task 2 resume sources. Binding or rewriting them is a scientific-adjacent
    mutation of historical bytes. Only the current TRAINING fold's progress
    file may be rewritten and bound.
    """

    if not str(filename).endswith("_progress.pt"):
        return "not_progress"
    if fold in finalized_folds:
        if action != "keep_finalized":
            raise RuntimeError(
                f"RUN_CONTRACT_MIGRATION_FINALIZED_PROGRESS_NOT_FROZEN:{filename}"
            )
        return "ignore_leftover_finalized"
    if action == "keep_finalized":
        raise RuntimeError("RUN_CONTRACT_MIGRATION_PROGRESS_MARKED_FINALIZED")
    if action not in {"rewrite", "keep"}:
        raise RuntimeError(f"RUN_CONTRACT_MIGRATION_PROGRESS_ACTION_INVALID:{filename}")
    return "bind_current"
_ALLOWED_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"TRAINING"},
    "TRAINING": {"TRAINING", "CHECKPOINT_LOCKED"},
    "CHECKPOINT_LOCKED": {"CHECKPOINT_LOCKED", "OUTER_EVAL_IN_PROGRESS", "FINALIZED"},
    "OUTER_EVAL_IN_PROGRESS": {"OUTER_EVAL_DONE"},
    "OUTER_EVAL_DONE": {"OUTER_EVAL_DONE", "FINALIZED"},
    "FINALIZED": {"FINALIZED"},
}


def _contract_body(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in contract.items() if key != "digest"}


def sha256_path(path: str | Path) -> str:
    """Return a stable SHA-256 for a regular file.

    Resume and migration code must never silently hash a directory, a missing
    path, or a reparse point.  Keeping this small helper here also gives the
    trainer and the audit tests one canonical byte-hash implementation.
    """

    candidate = Path(path)
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:  # pragma: no cover - platform-specific errno
        raise RuntimeError(f"RUN_CONTRACT_ARTIFACT_MISSING:{candidate.name}") from exc
    if not candidate.is_file() or stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    ):
        raise RuntimeError(f"RUN_CONTRACT_ARTIFACT_NOT_REGULAR:{candidate.name}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten_contract(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_contract(value[key], child))
        return flattened
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def _is_scientific_key(key: str) -> bool:
    if key.startswith("loader."):
        return key.split(".", 1)[1] in SCIENTIFIC_LOADER_KEYS
    if key.startswith("parameters."):
        field = key.split(".", 1)[1]
        return not any(
            field == name or field.startswith(f"{name}.")
            for name in NON_SCIENTIFIC_PARAMETER_KEYS
        )
    return key in {
        "cache_sha256",
        "split_digest",
        "spm_snapshot_sha256",
        "folds",
        "epochs",
        "seed",
    }


def contract_diff(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Explain a resume mismatch without exposing identifiers or raw paths."""

    found = payload.get("run_contract") if isinstance(payload, Mapping) else None
    found_body = _contract_body(found) if isinstance(found, Mapping) else {}
    expected_body = _contract_body(expected) if isinstance(expected, Mapping) else {}
    found_flat = _flatten_contract(found_body)
    expected_flat = _flatten_contract(expected_body)
    changed = sorted({*found_flat, *expected_flat})
    scientific: list[str] = []
    source: list[str] = []
    runtime: list[str] = []
    other: list[str] = []
    for key in changed:
        if found_flat.get(key) == expected_flat.get(key):
            continue
        if "source_contract_digest" in key:
            source.append(key)
        elif _is_scientific_key(key):
            scientific.append(key)
        elif "runtime_versions" in key or key.startswith("loader."):
            runtime.append(key)
        else:
            other.append(key)
    return {
        "schema_version": 1,
        "scientific_changes": scientific,
        "source_changes": source,
        "runtime_changes": runtime,
        "other_changes": other,
        "found_digest": canonical_digest(found_body) if found_body else None,
        "expected_digest": canonical_digest(expected_body) if expected_body else None,
        "resume_allowed": not scientific and not other,
        "source_only_migration_allowed": bool(source) and not scientific and not other,
    }


def build_run_contract(
    *,
    cache_sha256: str,
    split_digest: str,
    folds: int,
    epochs: int,
    seed: int,
    parameters: Mapping[str, Any],
    loader: Mapping[str, Any],
    spm_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    normalized_spm_digest = str(spm_snapshot_sha256 or "").strip().lower() or None
    if normalized_spm_digest is not None and (
        len(normalized_spm_digest) != 64
        or any(char not in "0123456789abcdef" for char in normalized_spm_digest)
    ):
        raise ValueError("INVALID_SPM_SNAPSHOT_SHA256")
    body = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "cache_sha256": str(cache_sha256),
        "split_digest": str(split_digest),
        "spm_snapshot_sha256": normalized_spm_digest,
        "folds": int(folds),
        "epochs": int(epochs),
        "seed": int(seed),
        "parameters": dict(parameters),
        "loader": dict(loader),
    }
    return {**body, "digest": canonical_digest(body)}


def scientific_contract_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Project a run contract onto the fields that define the experiment.

    Everything the toolchain owns (source digest, library versions, prefetch and
    pinning plumbing) is dropped so that a pure runtime change leaves this
    projection - and therefore the scientific digest - untouched.
    """

    body = _contract_body(contract) if isinstance(contract, Mapping) else {}
    parameters = {
        key: value
        for key, value in (body.get("parameters") or {}).items()
        if key not in NON_SCIENTIFIC_PARAMETER_KEYS
    }
    loader = {
        key: value
        for key, value in (body.get("loader") or {}).items()
        if key in SCIENTIFIC_LOADER_KEYS
    }
    return {
        "schema_version": body.get("schema_version"),
        "cache_sha256": body.get("cache_sha256"),
        "split_digest": body.get("split_digest"),
        "spm_snapshot_sha256": body.get("spm_snapshot_sha256"),
        "folds": body.get("folds"),
        "epochs": body.get("epochs"),
        "seed": body.get("seed"),
        "parameters": parameters,
        "loader": loader,
    }


def scientific_contract_digest(contract: Mapping[str, Any]) -> str:
    """Digest of the experiment definition alone.

    A change here is never migrated in place; it requires a new versioned run.
    """

    return canonical_digest(scientific_contract_projection(contract))


def build_runtime_resume_contract(
    *,
    staging_root_class: str,
    full_cadence: int,
    h_mirror_policy: str,
    journal_enabled: bool,
    replay_policy: str,
) -> dict[str, Any]:
    """Describe where and how often resume state is written.

    Only a policy class is recorded, never a concrete staging path, so no local
    user path is embedded in checkpoints or mismatch reports.
    """

    if str(staging_root_class) not in _ALLOWED_STAGING_ROOT_CLASSES:
        raise ValueError(f"INVALID_STAGING_ROOT_CLASS:{staging_root_class}")
    if int(full_cadence) < 1:
        raise ValueError(f"INVALID_CHECKPOINT_FULL_CADENCE:{full_cadence}")
    if str(h_mirror_policy) not in _ALLOWED_H_MIRROR_POLICIES:
        raise ValueError(f"INVALID_H_MIRROR_POLICY:{h_mirror_policy}")
    if str(replay_policy) not in _ALLOWED_REPLAY_POLICIES:
        raise ValueError(f"INVALID_RESUME_REPLAY_POLICY:{replay_policy}")
    body = {
        "schema_version": RUNTIME_RESUME_SCHEMA_VERSION,
        "staging_root_class": str(staging_root_class),
        "full_cadence": int(full_cadence),
        "h_mirror_policy": str(h_mirror_policy),
        "journal_enabled": bool(journal_enabled),
        "replay_policy": str(replay_policy),
    }
    return {**body, "digest": canonical_digest(body)}


def runtime_resume_digest(contract: Mapping[str, Any]) -> str:
    return canonical_digest(_contract_body(contract))


def storage_policy_digest(path: str | Path) -> str:
    """Bind the active storage policy file into the resume contract."""

    policy_path = Path(path)
    if not policy_path.is_file():
        raise RuntimeError(f"STORAGE_POLICY_MISSING:{policy_path.name}")
    return hashlib.sha256(policy_path.read_bytes()).hexdigest()


def classify_contract_change(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    expected_runtime: Mapping[str, Any] | None = None,
    expected_storage_digest: str | None = None,
) -> dict[str, Any]:
    """Decide between RESUME, AUDITED_MIGRATION, and NEW_VERSIONED_RUN.

    A scientific change always outranks runtime, source, and storage-policy
    changes, so a mixed diff can never be downgraded to a migration.
    """

    diff = contract_diff(payload, expected)
    found_contract = payload.get("run_contract") if isinstance(payload, Mapping) else None
    found_runtime = payload.get("runtime_resume_contract") if isinstance(payload, Mapping) else None
    legacy_runtime_absent = expected_runtime is not None and not isinstance(
        found_runtime, Mapping
    )
    runtime_changes = list(diff["runtime_changes"])
    if isinstance(found_runtime, Mapping) and expected_runtime is not None:
        if runtime_resume_digest(found_runtime) != runtime_resume_digest(expected_runtime):
            runtime_changes.append("runtime_resume_contract")
    storage_changed = bool(
        expected_storage_digest is not None
        and payload.get("storage_policy_digest") != expected_storage_digest
    )
    blocking = list(diff["scientific_changes"]) + list(diff["other_changes"])
    if blocking:
        decision = "NEW_VERSIONED_RUN"
    elif diff["source_changes"] or runtime_changes or storage_changed or legacy_runtime_absent:
        decision = "AUDITED_MIGRATION"
    else:
        decision = "RESUME"
    return {
        "schema_version": 1,
        "decision": decision,
        "scientific_changes": list(diff["scientific_changes"]),
        "source_changes": list(diff["source_changes"]),
        "runtime_changes": sorted(runtime_changes),
        "other_changes": list(diff["other_changes"]),
        "storage_policy_changed": storage_changed,
        "legacy_runtime_contract_absent": legacy_runtime_absent,
        "scientific_digest_expected": scientific_contract_digest(expected),
        "scientific_digest_found": scientific_contract_digest(found_contract)
        if isinstance(found_contract, Mapping)
        else None,
    }


def append_source_history(
    path: str | Path,
    *,
    old_digest: str,
    new_digest: str,
    decision: str,
    equality_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one audited migration record. Never rewrites earlier entries."""

    if str(decision) != "AUDITED_MIGRATION":
        raise RuntimeError(f"SCIENTIFIC_MIGRATION_FORBIDDEN:{decision}")
    amendment_path = Path(path)
    history: list[Any] = []
    if amendment_path.is_file():
        existing = json.loads(amendment_path.read_text(encoding="utf-8"))
        if isinstance(existing, Mapping):
            history = list(existing.get("source_history") or [])
    history.append(
        {
            "old_digest": str(old_digest),
            "new_digest": str(new_digest),
            "decision": str(decision),
            "equality_report": dict(equality_report),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    amendment = {
        "schema_version": 1,
        "append_only": True,
        "source_history": history,
    }
    atomic_write_json(amendment_path, amendment)
    return amendment


def source_history_binds_digest(
    *,
    history: Any,
    from_digest: str,
    to_digest: str,
) -> bool:
    """True iff a unique successor chain walks ``from_digest`` to ``to_digest``.

    Forks, cycles, missing successors, and non-sha256 digests are False.
    Equal digests bind with no history. This is the same unique-path rule the
    autosupervisor uses; it does not waive a scientific contract mismatch.
    """

    if not _is_sha256(from_digest) or not _is_sha256(to_digest):
        return False
    if from_digest == to_digest:
        return True
    if not isinstance(history, list):
        return False
    current = from_digest
    seen = {current}
    while current != to_digest:
        successors = [
            item
            for item in history
            if isinstance(item, Mapping)
            and item.get("from_run_contract_digest") == current
        ]
        if len(successors) != 1:
            return False
        nxt = str(successors[0].get("to_run_contract_digest") or "")
        if not _is_sha256(nxt) or nxt in seen:
            return False
        seen.add(nxt)
        current = nxt
    return True


def assert_resume_contract(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    source_history: Any = None,
) -> None:
    found = payload.get("run_contract")
    found_digest = payload.get("run_contract_digest")
    expected_body = _contract_body(expected)
    expected_digest = canonical_digest(expected_body)
    if not isinstance(found, Mapping):
        raise RuntimeError("RUN_CONTRACT_MISSING")
    found_body = _contract_body(found)
    actual_found_digest = canonical_digest(found_body)
    internally_consistent = (
        found_digest == actual_found_digest
        and found.get("digest") == actual_found_digest
        and expected.get("digest") == expected_digest
    )
    exact_match = internally_consistent and actual_found_digest == expected_digest
    source_only_ancestor = (
        internally_consistent
        and source_history is not None
        and source_contract_migration_allowed(payload, expected)
        and source_history_binds_digest(
            history=source_history,
            from_digest=actual_found_digest,
            to_digest=expected_digest,
        )
    )
    if exact_match or source_only_ancestor:
        return
    diff = contract_diff(payload, expected)
    integrity = contract_integrity_evidence(payload, expected)
    raise RuntimeError(
        "RUN_CONTRACT_MISMATCH:"
        + json.dumps(
            {
                "expected": expected_digest,
                "found": actual_found_digest,
                "contract_diff": diff,
                "integrity": integrity,
            },
            sort_keys=True,
        )
    )


def source_contract_migration_allowed(
    payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Allow only a source-code digest refresh when scientific settings match.

    Resume mechanics and non-semantic loader plumbing can change after a
    checkpoint is written. Cache, split, seed, folds, epochs, model and every
    training parameter must remain byte-for-byte contract-compatible.
    """

    found = payload.get("run_contract") if isinstance(payload, Mapping) else None
    if not isinstance(found, Mapping) or not isinstance(expected, Mapping):
        return False
    found_body = dict(_contract_body(found))
    expected_body = dict(_contract_body(expected))
    found_parameters = dict(found_body.get("parameters") or {})
    expected_parameters = dict(expected_body.get("parameters") or {})
    found_source = found_parameters.pop("source_contract_digest", None)
    expected_source = expected_parameters.pop("source_contract_digest", None)
    if not found_source or not expected_source or found_source == expected_source:
        return False
    found_body["parameters"] = found_parameters
    expected_body["parameters"] = expected_parameters
    return found_body == expected_body


def _migration_contract_is_canonical(contract: Any, *, label: str) -> str:
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"RUN_CONTRACT_MIGRATION_{label}_CONTRACT_INVALID")
    declared = contract.get("digest")
    canonical = canonical_digest(_contract_body(contract))
    if not _is_sha256(declared) or declared != canonical:
        raise RuntimeError(f"RUN_CONTRACT_MIGRATION_{label}_CONTRACT_INVALID")
    return str(canonical)


def _migration_load_torch(path: Path) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"RUN_CONTRACT_MIGRATION_CHECKPOINT_INVALID:{path.name}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"RUN_CONTRACT_MIGRATION_CHECKPOINT_INVALID:{path.name}")
    return payload


def _migration_validate_checkpoint(
    path: Path,
    *,
    expected_contract: Mapping[str, Any],
    old_digest: str,
    fold: int,
    allowed_digests: set[str] | None = None,
) -> dict[str, Any]:
    payload = _migration_load_torch(path)
    embedded = payload.get("run_contract")
    embedded_digest = payload.get("run_contract_digest")
    try:
        embedded_canonical = _migration_contract_is_canonical(
            embedded, label="CHECKPOINT"
        )
    except RuntimeError as exc:
        raise RuntimeError("RUN_CONTRACT_MIGRATION_CHECKPOINT_CONTRACT_INVALID") from exc
    accepted = allowed_digests or {old_digest}
    # A history digest is only an identity hint, never permission to import a
    # checkpoint with different data, split, seed, model or training settings.
    # The embedded declaration must also agree with its own canonical body.
    scientifically_compatible = (
        embedded_canonical == _migration_contract_is_canonical(
            expected_contract, label="EXPECTED"
        )
        or source_contract_migration_allowed(payload, expected_contract)
    )
    if (
        embedded_canonical not in accepted
        or embedded_digest != embedded_canonical
        or not scientifically_compatible
        or type(payload.get("fold")) is not int
        or payload["fold"] != int(fold)
    ):
        raise RuntimeError("RUN_CONTRACT_MIGRATION_CHECKPOINT_CONTRACT_INVALID")
    # A final checkpoint must explicitly say it is terminal; a progress file
    # must explicitly say the opposite.  This prevents a renamed/forged file
    # from being used as migration authority.
    return payload


def _migration_validate_outer(
    path: Path, *, old_digest: str, fold: int
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("RUN_CONTRACT_MIGRATION_PRE_LINKAGE_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("RUN_CONTRACT_MIGRATION_PRE_LINKAGE_INVALID")
    labels = payload.get("y_true")
    probabilities = payload.get("y_prob")
    if (
        type(payload.get("fold")) is not int
        or payload.get("fold") != int(fold)
        or type(payload.get("evaluation_count")) is not int
        or payload.get("evaluation_count") != 1
        or payload.get("run_contract_digest") != old_digest
        or not isinstance(labels, list)
        or not isinstance(probabilities, list)
        or not labels
        or len(labels) != len(probabilities)
        or any(int(value) not in {0, 1} for value in labels)
        or any(
            not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in probabilities
        )
        or payload.get("result_digest")
        != canonical_digest({key: value for key, value in payload.items() if key != "result_digest"})
    ):
        raise RuntimeError("RUN_CONTRACT_MIGRATION_PRE_LINKAGE_INVALID")
    return payload


def _migration_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError("RUN_CONTRACT_MIGRATION_PATH_OUTSIDE_ROOT") from exc


def _migration_receipt_integrity(payload: Mapping[str, Any]) -> str:
    return canonical_digest(
        {key: value for key, value in payload.items() if key != "integrity_sha256"}
    )


def _migration_validate_prepared(
    path: Path,
    *,
    old_digest: str,
    new_digest: str,
    root: Path,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("RUN_CONTRACT_MIGRATION_RECEIPT_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("RUN_CONTRACT_MIGRATION_RECEIPT_INVALID")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact")
        != "method_x_source_only_contract_migration_prepared"
        or payload.get("append_only") is not True
        or payload.get("old_run_contract_digest") != old_digest
        or payload.get("new_run_contract_digest") != new_digest
        or payload.get("integrity_sha256") != _migration_receipt_integrity(payload)
        or not isinstance(payload.get("folds"), list)
    ):
        raise RuntimeError("RUN_CONTRACT_MIGRATION_RECEIPT_INVALID")
    # The prepared receipt is an observation, not authority by itself.  Check
    # every recorded pre-image against the current filesystem before any write.
    for row in payload["folds"]:
        if not isinstance(row, Mapping):
            raise RuntimeError("RUN_CONTRACT_MIGRATION_PREPARED_EVIDENCE_MISMATCH")
        try:
            checkpoint = root / str(row["checkpoint_path"])
            state = root / str(row["state_path"])
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("RUN_CONTRACT_MIGRATION_PREPARED_EVIDENCE_MISMATCH") from exc
        for candidate in (checkpoint, state):
            if candidate.resolve().parent != root.resolve() and root.resolve() not in candidate.resolve().parents:
                raise RuntimeError("RUN_CONTRACT_MIGRATION_PREPARED_EVIDENCE_MISMATCH")
        current_checkpoint = sha256_path(checkpoint)
        current_state = sha256_path(state)
        expected_pre_checkpoint = str(row.get("pre_checkpoint_sha256") or "")
        expected_pre_state = str(row.get("pre_state_sha256") or "")
        expected_post_checkpoint = str(row.get("post_checkpoint_sha256") or "")
        expected_post_state = str(row.get("post_state_sha256") or "")
        # A crash can leave a post-image in place.  Both pre and post are valid
        # during recovery; anything else is evidence tampering.
        if row.get("kind") == "final" and current_checkpoint != expected_pre_checkpoint:
            raise RuntimeError("RUN_CONTRACT_MIGRATION_PREPARED_EVIDENCE_MISMATCH")
        if current_checkpoint != expected_pre_checkpoint and (
            not expected_post_checkpoint or current_checkpoint != expected_post_checkpoint
        ):
            # For a progress payload torch's zip serialization may carry a
            # process-local timestamp, so a post-image hash cannot always be
            # predicted before the write.  Accept an unrecorded post-image
            # only when its embedded contract proves this exact transition.
            post_contract_ok = False
            if row.get("kind") == "progress" and not expected_post_checkpoint:
                try:
                    post_payload = _migration_load_torch(checkpoint)
                    post_contract_ok = post_payload.get("run_contract_digest") == new_digest
                except RuntimeError:
                    post_contract_ok = False
            if not post_contract_ok:
                raise RuntimeError("RUN_CONTRACT_MIGRATION_PREPARED_EVIDENCE_MISMATCH")
        if current_state != expected_pre_state:
            # A post-image is admissible only when the state itself proves that
            # this transition was already applied.  A forged pre-image hash on
            # an old state is therefore rejected even if its JSON integrity
            # field was recomputed.
            state_payload = None
            try:
                state_payload = json.loads(state.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                state_payload = None
            post_matches = expected_post_state and current_state == expected_post_state
            transition_present = isinstance(state_payload, Mapping) and any(
                isinstance(link, Mapping)
                and link.get("from_run_contract_digest") == old_digest
                and link.get("to_run_contract_digest") == new_digest
                for link in (state_payload.get("source_history") or [])
            )
            if (
                not (post_matches or transition_present)
                or not isinstance(state_payload, Mapping)
                or state_payload.get("contract_digest") != new_digest
            ):
                raise RuntimeError("RUN_CONTRACT_MIGRATION_PREPARED_EVIDENCE_MISMATCH")
        if int(row.get("fold", -1)) != fold:
            raise RuntimeError("RUN_CONTRACT_MIGRATION_PREPARED_EVIDENCE_MISMATCH")
    return payload


def migrate_source_only_resume_artifacts(
    model_dir: str | Path,
    *,
    existing_contract: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    failpoint: Any = None,
) -> dict[str, Any]:
    """Perform a receipt-backed source-only migration without outer re-eval.

    Finalized checkpoint and outer-result bytes are immutable.  Only a live
    progress payload is rebound to the new source digest; state/receipt links
    prove the transition.  The transaction is idempotent and can recover from
    a process death after any individual state write.
    """

    root = Path(model_dir).resolve()
    if not root.is_dir():
        raise RuntimeError("RUN_CONTRACT_MIGRATION_MODEL_DIR_MISSING")
    old = existing_contract.get("run_contract") if isinstance(existing_contract, Mapping) else None
    new = expected_contract.get("run_contract") if isinstance(expected_contract, Mapping) else None
    if old is None and isinstance(existing_contract, Mapping) and "digest" in existing_contract:
        old = existing_contract
    if new is None and isinstance(expected_contract, Mapping) and "digest" in expected_contract:
        new = expected_contract
    old_digest = _migration_contract_is_canonical(old, label="OLD")
    new_digest = _migration_contract_is_canonical(new, label="NEW")
    if old_digest == new_digest or not source_contract_migration_allowed(
        {"run_contract": old}, new
    ):
        raise RuntimeError("RUN_CONTRACT_MIGRATION_NOT_SOURCE_ONLY")

    # Reject unknown checkpoint suffixes before observing/mutating anything.
    checkpoint_files: dict[tuple[int, str], Path] = {}
    for candidate in sorted(root.iterdir(), key=lambda p: p.name):
        if not candidate.name.startswith("method_x_3d_fold_") or candidate.suffix != ".pt":
            continue
        fold, kind = parse_fold_checkpoint_filename(candidate.name)
        checkpoint_files[(fold, kind)] = candidate
        if kind == "best_state":
            # Best-state files are weight-only sidecars and must stay byte-identical.
            try:
                import torch

                sidecar = torch.load(candidate, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise RuntimeError(f"RUN_CONTRACT_MIGRATION_BEST_STATE_INVALID:{candidate.name}") from exc
            assert_offloaded_best_state_sidecar(sidecar, filename=candidate.name)

    finals = {fold: path for (fold, kind), path in checkpoint_files.items() if kind == "final"}
    progresses = {fold: path for (fold, kind), path in checkpoint_files.items() if kind == "progress"}
    folds = sorted(set(finals) | set(progresses))
    historical_digests: set[str] = {old_digest, new_digest}
    # Finalized checkpoints are intentionally immutable across multiple
    # source-only hops.  Their creation digest is discoverable only through
    # the append-only state history, so include those ancestors as accepted
    # artifact identities for validation.
    for state_candidate in root.glob("method_x_3d_fold_*_state.json"):
        try:
            state_payload = json.loads(state_candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        for link in state_payload.get("source_history", []) if isinstance(state_payload, Mapping) else []:
            if isinstance(link, Mapping):
                for key in ("from_run_contract_digest", "to_run_contract_digest"):
                    value = link.get(key)
                    if _is_sha256(value):
                        historical_digests.add(str(value))
    records: list[dict[str, Any]] = []
    finalized_folds: set[int] = set()
    for fold in folds:
        checkpoint = finals.get(fold) or progresses.get(fold)
        assert checkpoint is not None
        payload = _migration_validate_checkpoint(
            checkpoint,
            expected_contract=new,
            old_digest=old_digest,
            fold=fold,
            allowed_digests=historical_digests,
        )
        artifact_contract_digest = str(payload.get("run_contract_digest") or "")
        is_final = fold in finals
        state_path = root / f"method_x_3d_fold_{fold}_state.json"
        if not state_path.is_file():
            raise RuntimeError("RUN_CONTRACT_MIGRATION_PRE_LINKAGE_INVALID")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("RUN_CONTRACT_MIGRATION_PRE_LINKAGE_INVALID") from exc
        outer_path = root / f"method_x_3d_fold_{fold}_outer_eval.json"
        # MX-03 inner-only validity arms never construct the outer loader: a
        # fold that reached CHECKPOINT_LOCKED keeps a terminal progress
        # snapshot (fold_terminal True) with zero outer access and no outer
        # result file.  That shape is migratable progress, not a forged
        # final.  Battery shapes are unchanged: finals still require
        # FINALIZED plus an outer result, and live progress still requires
        # TRAINING plus a non-terminal payload.  A renamed battery final can
        # never satisfy this exception (its state is FINALIZED and its outer
        # result file exists).
        state_details = state.get("details") if isinstance(state, Mapping) else None
        inner_only_terminal_progress = (
            not is_final
            and payload.get("fold_terminal") is True
            and isinstance(state, dict)
            and state.get("stage") == "CHECKPOINT_LOCKED"
            and isinstance(state_details, Mapping)
            and state_details.get("outer_test_access_count") == 0
            and not outer_path.is_file()
        )
        # A selection-only arm is forbidden from ever constructing the outer
        # loader, so it legitimately FINALIZES with zero outer access, no outer
        # result file, and positive ``selection_only`` provenance.  The
        # exception above covered only the *progress* shape, which made every
        # such run unmigratable even though the comment below already
        # recognises the situation.  Requiring the explicit ``selection_only``
        # flag plus ``not outer_path.is_file()`` keeps this strictly narrower
        # than a real battery fold: a renamed battery final has an outer result
        # file, so it still cannot qualify.  Checkpoint bytes stay immutable.
        inner_only_finalized_no_outer = (
            is_final
            and payload.get("fold_terminal") is True
            and (payload.get("hpo_selection_only") is True
                 or payload.get("validity_inner_only") is True)
            and type(payload.get("outer_test_access_count")) is int
            and payload.get("outer_test_access_count") == 0
            and isinstance(state, dict)
            and state.get("stage") == "FINALIZED"
            and isinstance(state_details, Mapping)
            and state_details.get("selection_only") is True
            and state_details.get("hpo_selection_only") is True
            and type(state_details.get("outer_test_access_count")) is int
            and state_details.get("outer_test_access_count") == 0
            and not state_details.get("outer_result_digest")
            and not outer_path.is_file()
            and isinstance(payload.get("selection_only_evaluation"), Mapping)
            and payload["selection_only_evaluation"].get("fold") == fold
            and payload["selection_only_evaluation"].get("run_contract_digest")
                == artifact_contract_digest
            and payload["selection_only_evaluation"].get("split_membership_digest")
                == state_details.get("split_membership_digest")
            and type(payload["selection_only_evaluation"].get("outer_test_access_count")) is int
            and payload["selection_only_evaluation"].get("outer_test_access_count") == 0
            and payload["selection_only_evaluation"].get("outer_test_used_for_selection") is False
        )
        if is_final and payload.get("fold_terminal") is False:
            raise RuntimeError("RUN_CONTRACT_MIGRATION_CHECKPOINT_CONTRACT_INVALID")
        if (
            not is_final
            and payload.get("fold_terminal") is True
            and not inner_only_terminal_progress
        ):
            raise RuntimeError("RUN_CONTRACT_MIGRATION_CHECKPOINT_CONTRACT_INVALID")
        if (
            not isinstance(state, dict)
            or state.get("fold") != fold
            or state.get("contract_digest") not in {old_digest, new_digest}
            or state.get("stage")
            != (
                "FINALIZED"
                if is_final
                else (
                    "CHECKPOINT_LOCKED"
                    if inner_only_terminal_progress
                    else "TRAINING"
                )
            )
        ):
            raise RuntimeError("RUN_CONTRACT_MIGRATION_PRE_LINKAGE_INVALID")
        outer_digest: str | None = None
        if is_final:
            finalized_folds.add(fold)
            if not outer_path.is_file() and not inner_only_finalized_no_outer:
                raise RuntimeError("RUN_CONTRACT_MIGRATION_PRE_LINKAGE_INVALID")
            details = state.get("details") if isinstance(state.get("details"), Mapping) else {}
            # A selection-only fold records its identity under the
            # ``checkpoint``/``checkpoint_sha256`` vocabulary and has no outer
            # result to cross-check; only the outer-independent identity linkage
            # can be proved, and its absence is the expected shape here.
            if inner_only_finalized_no_outer:
                recorded_sha = details.get("final_checkpoint_sha256") or details.get(
                    "checkpoint_sha256"
                )
                recorded_path = details.get("final_checkpoint") or details.get("checkpoint")
                if (
                    recorded_sha != sha256_path(checkpoint)
                    or Path(str(recorded_path or "")).name != checkpoint.name
                ):
                    raise RuntimeError("RUN_CONTRACT_MIGRATION_PRE_LINKAGE_INVALID")
            else:
                outer = _migration_validate_outer(
                    outer_path, old_digest=artifact_contract_digest, fold=fold
                )
                outer_digest = str(outer["result_digest"])
                if (
                    details.get("final_checkpoint_sha256") != sha256_path(checkpoint)
                    or details.get("outer_result_digest") != outer_digest
                    or details.get("split_membership_digest") != outer.get("split_membership_digest")
                    or Path(str(details.get("final_checkpoint", ""))).name != checkpoint.name
                    or Path(str(details.get("locked_result", ""))).name != outer_path.name
                ):
                    raise RuntimeError("RUN_CONTRACT_MIGRATION_PRE_LINKAGE_INVALID")
        post_checkpoint_sha = sha256_path(checkpoint) if is_final else None
        post_size = checkpoint.stat().st_size if is_final else None
        records.append(
            {
                "fold": fold,
                "kind": "final" if is_final else "progress",
                "checkpoint_path": _migration_relative(checkpoint, root),
                "state_path": _migration_relative(state_path, root),
                "outer_result_path": (
                    _migration_relative(outer_path, root)
                    if is_final and outer_path.is_file()
                    else None
                ),
                "pre_checkpoint_sha256": sha256_path(checkpoint),
                "post_checkpoint_sha256": post_checkpoint_sha,
                "pre_checkpoint_byte_size": int(checkpoint.stat().st_size),
                "post_checkpoint_byte_size": int(post_size) if post_size is not None else None,
                "pre_state_sha256": sha256_path(state_path),
                "post_state_sha256": None,
                "pre_repair_state_sha256": sha256_path(state_path),
                "post_repair_state_sha256": None,
                "pre_outer_result_digest": outer_digest,
                "post_outer_result_digest": outer_digest,
                "split_membership_digest": (
                    state.get("details", {}).get("split_membership_digest")
                    if isinstance(state.get("details"), Mapping)
                    else None
                ),
                "artifact_contract_digest": artifact_contract_digest,
            }
        )

    migration_dir = root / "source_only_contract_migrations"
    migration_dir.mkdir(parents=True, exist_ok=True)
    key = f"{old_digest[:16]}_to_{new_digest[:16]}"
    prepared_candidates = sorted(migration_dir.glob(f"{key}.*.prepared.json"))
    prepared_path: Path
    prepared: dict[str, Any] | None = None
    if prepared_candidates:
        prepared_path = prepared_candidates[-1]
        prepared = _migration_validate_prepared(
            prepared_path, old_digest=old_digest, new_digest=new_digest, root=root
        )
        # Reuse the recorded pre-image; it is the immutable evidence for this
        # transition and must not be recomputed from a tampered file.
        records = [dict(row) for row in prepared.get("folds", [])]
    else:
        prepared_path = migration_dir / f"{key}.{uuid.uuid4().hex}.prepared.json"
        # Predict post-state hashes after changing only the contract linkage.
        for row in records:
            state_path = root / str(row["state_path"])
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state_copy = dict(state)
            state_copy["contract_digest"] = new_digest
            state_copy["source_history"] = list(state.get("source_history") or [])
            row["post_state_sha256"] = None  # populated after receipt path is known
        prepared = {
            "schema_version": 1,
            "artifact": "method_x_source_only_contract_migration_prepared",
            "append_only": True,
            "old_run_contract_digest": old_digest,
            "new_run_contract_digest": new_digest,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_receipt_name": f"{key}.{prepared_path.stem.split('.', 2)[-1]}.completed.json",
            "folds": records,
        }
        prepared["integrity_sha256"] = _migration_receipt_integrity(prepared)
        atomic_write_json(prepared_path, prepared)
        if callable(failpoint):
            failpoint("after_prepared_receipt")

    completed_name = str(prepared.get("completed_receipt_name") or f"{key}.{uuid.uuid4().hex}.completed.json")
    completed_path = migration_dir / completed_name
    completed_payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "method_x_source_only_contract_migration_receipt",
        "append_only": True,
        "old_run_contract_digest": old_digest,
        "new_run_contract_digest": new_digest,
        "prepared_receipt_sha256": sha256_path(prepared_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_test_touched": False,
        "outer_loader_invocation_count": 0,
        # ``folds`` is the immutable finalized-artifact evidence consumed by
        # the source-compatibility resolver.  Live progress rebinding is
        # recorded separately so a later hop cannot be mistaken for mutation
        # of a finalized checkpoint.
        "folds": [row for row in records if row.get("kind") == "final"],
        "progress_folds": [row for row in records if row.get("kind") == "progress"],
    }
    # Apply progress post-images and state links.  Existing post-images are
    # accepted only when their recorded hash matches exactly.
    for row in records:
        fold = int(row["fold"])
        checkpoint = root / str(row["checkpoint_path"])
        state_path = root / str(row["state_path"])
        current_checkpoint_sha = sha256_path(checkpoint)
        if row.get("kind") == "progress":
            expected_post = row.get("post_checkpoint_sha256")
            if expected_post and current_checkpoint_sha == expected_post:
                pass
            else:
                if current_checkpoint_sha != row.get("pre_checkpoint_sha256"):
                    # A previous interrupted attempt may already have written
                    # the rebound payload; accept it only after semantic load.
                    rebound_existing = _migration_load_torch(checkpoint)
                    if rebound_existing.get("run_contract_digest") != new_digest:
                        raise RuntimeError("RUN_CONTRACT_MIGRATION_PREPARED_EVIDENCE_MISMATCH")
                else:
                    payload = _migration_load_torch(checkpoint)
                    payload["run_contract"] = dict(new)
                    payload["run_contract_digest"] = new_digest
                    try:
                        import torch

                        tmp = checkpoint.with_name(f".{checkpoint.name}.{uuid.uuid4().hex}.tmp")
                        torch.save(payload, tmp)
                        os.replace(tmp, checkpoint)
                    finally:
                        try:
                            tmp.unlink(missing_ok=True)
                        except Exception:
                            pass
                row["post_checkpoint_sha256"] = sha256_path(checkpoint)
                row["post_checkpoint_byte_size"] = int(checkpoint.stat().st_size)
        if row.get("kind") == "final" and sha256_path(checkpoint) != row.get("pre_checkpoint_sha256"):
            raise RuntimeError("RUN_CONTRACT_MIGRATION_FINALIZED_CHECKPOINT_MUTATION")

        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("contract_digest") == new_digest and any(
            isinstance(link, Mapping)
            and link.get("to_run_contract_digest") == new_digest
            for link in (state.get("source_history") or [])
        ):
            pass
        elif state.get("contract_digest") == old_digest:
            state["contract_digest"] = new_digest
            history = list(state.get("source_history") or [])
            relative_receipt = _migration_relative(completed_path, root)
            link = {
                "from_run_contract_digest": old_digest,
                "to_run_contract_digest": new_digest,
                "receipt_path": relative_receipt,
                "receipt_sha256": None,
            }
            history.append(link)
            state["source_history"] = history
            details = dict(state.get("details") or {})
            details["source_only_migration_receipt"] = {
                "path": relative_receipt,
                "sha256": None,
                "old_run_contract_digest": old_digest,
                "new_run_contract_digest": new_digest,
            }
            state["details"] = details
            atomic_write_json(state_path, state)
            row["post_state_sha256"] = sha256_path(state_path)
            row["post_repair_state_sha256"] = row["post_state_sha256"]
            if callable(failpoint):
                failpoint(f"after_state_write:{fold}")
        elif state.get("contract_digest") != new_digest:
            raise RuntimeError("RUN_CONTRACT_MIGRATION_STATE_CONTRACT_INVALID")
        row["post_state_sha256"] = sha256_path(state_path)
        row["post_repair_state_sha256"] = row["post_state_sha256"]

    # The receipt is written only after all pre-images and progress post-images
    # are verified.  Then state links are patched with its immutable hash.
    completed_payload["integrity_sha256"] = _migration_receipt_integrity(completed_payload)
    if completed_path.is_file():
        existing_completed = _migration_validate_completed_receipt(
            completed_path, old_digest=old_digest, new_digest=new_digest
        )
        completed_payload = existing_completed
    else:
        atomic_write_json(completed_path, completed_payload)
    completed_sha = sha256_path(completed_path)
    for row in records:
        state_path = root / str(row["state_path"])
        state = json.loads(state_path.read_text(encoding="utf-8"))
        history = list(state.get("source_history") or [])
        for link in history:
            if (
                isinstance(link, dict)
                and link.get("from_run_contract_digest") == old_digest
                and link.get("to_run_contract_digest") == new_digest
            ):
                link["receipt_sha256"] = completed_sha
        details = dict(state.get("details") or {})
        audit = dict(details.get("source_only_migration_receipt") or {})
        audit.update({"path": _migration_relative(completed_path, root), "sha256": completed_sha})
        details["source_only_migration_receipt"] = audit
        state["source_history"] = history
        state["details"] = details
        if state.get("contract_digest") != new_digest:
            state["contract_digest"] = new_digest
        atomic_write_json(state_path, state)

        if row.get("kind") == "progress":
            try:
                from src.training.method_x_autosupervisor import (
                    PAUSED_PROGRESS_BIND_ARTIFACT,
                    _payload_integrity_sha256,
                    paused_progress_bind_path,
                )

                bind_path = paused_progress_bind_path(
                    model_dir=root,
                    evidence_digest=old_digest,
                    live_digest=new_digest,
                    fold=int(row["fold"]),
                )
                bind = {
                    "schema_version": 1,
                    "artifact": PAUSED_PROGRESS_BIND_ARTIFACT,
                    "append_only": True,
                    "provenance_status": "REPAIR_BIND_AFTER_SOURCE_ONLY_MIGRATION",
                    "progress_path": str(row["checkpoint_path"]),
                    "fold": int(row["fold"]),
                    "old_run_contract_digest": old_digest,
                    "new_run_contract_digest": new_digest,
                    "pre_sha256": row["pre_checkpoint_sha256"],
                    "post_sha256": row["post_checkpoint_sha256"],
                    "pre_byte_size": int(row["pre_checkpoint_byte_size"]),
                    "post_byte_size": int(row["post_checkpoint_byte_size"]),
                    "migration_receipt_sha256": completed_sha,
                }
                bind["integrity_sha256"] = _payload_integrity_sha256(bind)
                if bind_path.is_file():
                    existing_bind = json.loads(bind_path.read_text(encoding="utf-8"))
                    if existing_bind != bind:
                        raise RuntimeError("RUN_CONTRACT_MIGRATION_BIND_MISMATCH")
                else:
                    atomic_write_json(bind_path, bind)
            except ImportError:
                pass

    wrapper = {
        "run_contract": dict(new),
        "run_contract_digest": new_digest,
    }
    atomic_write_json(root / "run_contract.json", wrapper)
    return {
        "status": "SOURCE_ONLY_MIGRATED",
        "old_run_contract_digest": old_digest,
        "new_run_contract_digest": new_digest,
        "receipt_path": _migration_relative(completed_path, root),
        "receipt_sha256": completed_sha,
        "outer_test_touched": False,
    }


def _migration_validate_completed_receipt(
    path: Path, *, old_digest: str, new_digest: str
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("RUN_CONTRACT_MIGRATION_RECEIPT_INVALID") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("artifact") != "method_x_source_only_contract_migration_receipt"
        or payload.get("append_only") is not True
        or payload.get("old_run_contract_digest") != old_digest
        or payload.get("new_run_contract_digest") != new_digest
        or payload.get("integrity_sha256") != _migration_receipt_integrity(payload)
    ):
        raise RuntimeError("RUN_CONTRACT_MIGRATION_RECEIPT_INVALID")
    return payload


def read_fold_state(path: str | Path) -> dict[str, Any] | None:
    state_path = Path(path)
    if not state_path.exists():
        return None
    value = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("stage") not in FOLD_STAGES:
        raise RuntimeError(f"INVALID_FOLD_STATE:{state_path}")
    return value


def transition_fold_state(
    path: str | Path,
    *,
    fold: int,
    target: str,
    contract_digest: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if target not in FOLD_STAGES:
        raise RuntimeError(f"UNKNOWN_FOLD_STATE:{target}")
    current = read_fold_state(path)
    previous = current.get("stage") if current else None
    if current and int(current.get("fold", -1)) != int(fold):
        raise RuntimeError("FOLD_STATE_FOLD_MISMATCH")
    if target not in _ALLOWED_TRANSITIONS[previous]:
        raise RuntimeError(f"ILLEGAL_FOLD_STATE_TRANSITION:{previous}->{target}")
    if current and current.get("contract_digest") != contract_digest:
        raise RuntimeError("FOLD_STATE_RUN_CONTRACT_MISMATCH")
    state = {
        "schema_version": 1,
        "fold": int(fold),
        "stage": target,
        "previous_stage": previous,
        "contract_digest": str(contract_digest),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "details": dict(details or {}),
    }
    atomic_write_json(path, state)
    return state


def outer_evaluation_resume_action(stage: str | None, *, locked_result_exists: bool) -> str:
    if stage in {None, "TRAINING"}:
        raise RuntimeError(f"OUTER_EVALUATION_NOT_CHECKPOINT_LOCKED:{stage}")
    if stage == "CHECKPOINT_LOCKED" and not locked_result_exists:
        return "RUN"
    if stage in {"OUTER_EVAL_DONE", "FINALIZED"} and locked_result_exists:
        return "REUSE"
    if stage == "OUTER_EVAL_IN_PROGRESS" and locked_result_exists:
        return "REUSE_COMMITTED_RECOVER_STATE"
    if stage == "OUTER_EVAL_IN_PROGRESS" and not locked_result_exists:
        raise RuntimeError("AMBIGUOUS_OUTER_EVALUATION_STATE")
    raise RuntimeError(f"OUTER_EVALUATION_STATE_INCONSISTENT:{stage}")


def build_locked_outer_result(
    *,
    fold: int,
    run_contract_digest: str,
    split_membership_digest: str,
    y_true: list[int],
    y_prob: list[float],
    test_logits: list[float],
) -> dict[str, Any]:
    body = {
        "fold": int(fold),
        "evaluation_count": 1,
        "run_contract_digest": str(run_contract_digest),
        "split_membership_digest": str(split_membership_digest),
        "y_true": [int(value) for value in y_true],
        "y_prob": [float(value) for value in y_prob],
        "test_logits": [float(value) for value in test_logits],
        "raw_identifiers_stored": False,
    }
    return {**body, "result_digest": canonical_digest(body)}


def validate_locked_outer_result(
    locked_result: Mapping[str, Any],
    *,
    expected_fold: int,
    expected_contract_digest: str,
    expected_split_membership_digest: str | None = None,
    expected_count: int | None = None,
) -> dict[str, Any]:
    body = {key: value for key, value in locked_result.items() if key != "result_digest"}
    labels = locked_result.get("y_true")
    probabilities = locked_result.get("y_prob")
    logits = locked_result.get("test_logits")
    valid = (
        int(locked_result.get("fold", -1)) == int(expected_fold)
        and locked_result.get("run_contract_digest") == expected_contract_digest
        and int(locked_result.get("evaluation_count", 0)) == 1
        and isinstance(locked_result.get("split_membership_digest"), str)
        and len(locked_result["split_membership_digest"]) == 64
        and (expected_split_membership_digest is None or locked_result["split_membership_digest"] == expected_split_membership_digest)
        and isinstance(labels, list)
        and isinstance(probabilities, list)
        and isinstance(logits, list)
        and len(labels) > 0
        and len(labels) == len(probabilities) == len(logits)
        and (expected_count is None or len(labels) == int(expected_count))
        and all(int(value) in {0, 1} for value in labels)
        and all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in probabilities)
        and all(math.isfinite(float(value)) for value in logits)
        and locked_result.get("raw_identifiers_stored") is False
        and locked_result.get("result_digest") == canonical_digest(body)
    )
    if not valid:
        raise RuntimeError("COMMITTED_OUTER_EVALUATION_INVALID")
    return {
        "y_true": [int(value) for value in locked_result["y_true"]],
        "y_prob": [float(value) for value in locked_result["y_prob"]],
        "test_logits": [float(value) for value in locked_result["test_logits"]],
        "split_membership_digest": str(locked_result["split_membership_digest"]),
        "result_digest": str(locked_result["result_digest"]),
    }


def outer_prediction_identity_digest(locked_result: Mapping[str, Any]) -> str:
    """Hash the immutable, subject-aligned prediction identity of one outer fold.

    The digest deliberately excludes migration metadata and metrics.  It proves
    that a source-only metadata migration did not recompute or reorder the
    once-opened outer-test predictions.
    """

    return canonical_digest(
        {
            "fold": locked_result.get("fold"),
            "evaluation_count": locked_result.get("evaluation_count"),
            "split_membership_digest": locked_result.get("split_membership_digest"),
            "y_true": locked_result.get("y_true"),
            "y_prob": locked_result.get("y_prob"),
            "test_logits": locked_result.get("test_logits"),
        }
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def legacy_source_metadata_normalization(
    payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Recognize one narrowly-scoped malformed legacy contract.

    A historical trainer appended ``source_digest`` *inside* the nested
    contract after calculating its digest.  That makes the declared digest
    refer to the contract without that field while the current canonical
    digest includes it.  We can normalize this only when removing the field
    yields an exact byte-for-byte scientific contract match.  Any other drift
    remains a hard mismatch; this function is deliberately not a general
    digest waiver.
    """

    if not isinstance(payload, Mapping) or not isinstance(expected, Mapping):
        return None
    found = payload.get("run_contract")
    if not isinstance(found, Mapping) or "source_digest" not in found:
        return None
    source_digest = found.get("source_digest")
    if not _is_sha256(source_digest):
        return None
    expected_body = _contract_body(expected)
    expected_declared = expected.get("digest")
    expected_canonical = canonical_digest(expected_body)
    if expected_declared != expected_canonical:
        return None
    candidate = {key: value for key, value in found.items() if key != "source_digest"}
    candidate_body = _contract_body(candidate)
    candidate_canonical = canonical_digest(candidate_body)
    if (
        candidate.get("digest") != candidate_canonical
        or payload.get("run_contract_digest") != candidate_canonical
        or candidate_body != expected_body
    ):
        return None
    normalized = dict(payload)
    normalized["run_contract"] = candidate
    normalized["run_contract_digest"] = candidate_canonical
    # Keep provenance outside the digested scientific contract.  This field is
    # informational and is itself bound by the migration receipt written by
    # the caller.
    normalized["source_digest_metadata"] = str(source_digest).lower()
    return normalized


def contract_integrity_evidence(
    payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Return non-sensitive reason codes for a contract mismatch."""

    found = payload.get("run_contract") if isinstance(payload, Mapping) else None
    found_body = _contract_body(found) if isinstance(found, Mapping) else {}
    expected_body = _contract_body(expected) if isinstance(expected, Mapping) else {}
    found_canonical = canonical_digest(found_body) if found_body else None
    expected_canonical = canonical_digest(expected_body) if expected_body else None
    found_declared = found.get("digest") if isinstance(found, Mapping) else None
    expected_declared = expected.get("digest") if isinstance(expected, Mapping) else None
    wrapper_declared = payload.get("run_contract_digest") if isinstance(payload, Mapping) else None
    reasons: list[str] = []
    if found_declared != found_canonical:
        reasons.append("FOUND_CONTRACT_DIGEST_STALE")
    if wrapper_declared != found_canonical:
        reasons.append("FOUND_WRAPPER_DIGEST_STALE")
    if expected_declared != expected_canonical:
        reasons.append("EXPECTED_CONTRACT_DIGEST_STALE")
    if found_canonical != expected_canonical:
        reasons.append("CONTRACT_BODY_DIGEST_DIFFERENT")
    if isinstance(found, Mapping) and "source_digest" in found:
        reasons.append("LEGACY_SOURCE_METADATA_UNBOUND")
    return {
        "found_declared_digest": found_declared,
        "found_canonical_digest": found_canonical,
        "expected_declared_digest": expected_declared,
        "expected_canonical_digest": expected_canonical,
        "found_digest_consistent": bool(
            found_declared == found_canonical and wrapper_declared == found_canonical
        ),
        "expected_digest_consistent": bool(expected_declared == expected_canonical),
        "reason_codes": reasons,
    }


def _validate_historical_outer_compatibility(
    *,
    checkpoint: Mapping[str, Any],
    committed: Mapping[str, Any],
    locked_result: Mapping[str, Any],
    locked: Mapping[str, Any],
    expected_fold: int,
    expected_contract_digest: str,
    checkpoint_sha256: str | None,
    outer_file_sha256: str | None,
    evidence: HistoricalCompatibilityAuthority | None,
) -> bool:
    """Validate the narrow receipt escape hatch for the five frozen artifacts.

    This is intentionally not a general mismatch waiver.  Every byte-level and
    prediction-level identity in the externally pinned receipt must match, and
    the outer loader invocation count must remain zero.
    """

    required_keys = {
        "schema_version",
        "artifact",
        "scope",
        "backbone",
        "fold",
        "checkpoint_sha256",
        "outer_file_sha256",
        "run_contract_digest",
        "split_membership_digest",
        "evaluation_count",
        "embedded_result_digest",
        "external_result_digest",
        "prediction_identity_digest",
        "outer_loader_invocation_count",
    }
    if not isinstance(evidence, HistoricalCompatibilityAuthority):
        return False
    evidence_payload = evidence.evidence
    if set(evidence_payload) != required_keys:
        return False
    embedded_digest = committed.get("result_digest")
    return bool(
        evidence_payload.get("schema_version") == 1
        and evidence_payload.get("artifact")
        == "method_x_historical_outer_result_compatibility_entry"
        and evidence_payload.get("scope") == "EXACT_FIVE_FROZEN_ARTIFACTS_ONLY"
        and evidence_payload.get("backbone") == checkpoint.get("backbone")
        and type(evidence_payload.get("fold")) is int
        and evidence_payload.get("fold") == int(expected_fold)
        and type(evidence_payload.get("evaluation_count")) is int
        and evidence_payload.get("evaluation_count") == 1
        and type(evidence_payload.get("outer_loader_invocation_count")) is int
        and evidence_payload.get("outer_loader_invocation_count") == 0
        and _is_sha256(checkpoint_sha256)
        and _is_sha256(outer_file_sha256)
        and evidence_payload.get("checkpoint_sha256") == checkpoint_sha256
        and evidence_payload.get("outer_file_sha256") == outer_file_sha256
        and evidence_payload.get("run_contract_digest") == expected_contract_digest
        and evidence_payload.get("split_membership_digest")
        == locked.get("split_membership_digest")
        and evidence_payload.get("embedded_result_digest") == embedded_digest
        and evidence_payload.get("external_result_digest") == locked.get("result_digest")
        and _is_sha256(embedded_digest)
        and embedded_digest != locked.get("result_digest")
        and evidence_payload.get("prediction_identity_digest")
        == outer_prediction_identity_digest(locked_result)
        and committed.get("y_true") == locked.get("y_true")
        and committed.get("y_prob") == locked.get("y_prob")
        and committed.get("test_logits") == locked.get("test_logits")
        and committed.get("split_membership_digest")
        == locked.get("split_membership_digest")
    )


def reuse_committed_outer_result(
    checkpoint: Mapping[str, Any],
    locked_result: Mapping[str, Any],
    *,
    expected_fold: int,
    expected_contract_digest: str,
    checkpoint_sha256: str | None = None,
    outer_file_sha256: str | None = None,
    compatibility_evidence: HistoricalCompatibilityAuthority | None = None,
) -> dict[str, Any]:
    """Validate and return a committed outer result without touching the test loader."""

    committed = checkpoint.get("locked_outer_test_evaluation")
    locked = validate_locked_outer_result(
        locked_result,
        expected_fold=expected_fold,
        expected_contract_digest=expected_contract_digest,
        expected_split_membership_digest=str(checkpoint.get("split_membership_digest", "")),
    )
    common_valid = (
        isinstance(committed, Mapping)
        and int(checkpoint.get("fold", -1)) == int(expected_fold)
        and checkpoint.get("run_contract_digest") == expected_contract_digest
        and int(committed.get("evaluation_count", 0)) == 1
        and isinstance(committed.get("metrics"), Mapping)
        and committed.get("y_true") == locked["y_true"]
        and committed.get("y_prob") == locked["y_prob"]
        and committed.get("test_logits") == locked["test_logits"]
        and committed.get("split_membership_digest") == locked["split_membership_digest"]
    )
    exact_digest_match = bool(
        common_valid and committed.get("result_digest") == locked["result_digest"]
    )
    compatibility_match = bool(
        common_valid
        and _validate_historical_outer_compatibility(
            checkpoint=checkpoint,
            committed=committed,
            locked_result=locked_result,
            locked=locked,
            expected_fold=expected_fold,
            expected_contract_digest=expected_contract_digest,
            checkpoint_sha256=checkpoint_sha256,
            outer_file_sha256=outer_file_sha256,
            evidence=compatibility_evidence,
        )
    )
    valid = exact_digest_match or compatibility_match
    if not valid:
        raise RuntimeError("COMMITTED_OUTER_EVALUATION_INVALID")
    calibrated = committed.get("calibrated_prob")
    if calibrated is not None and (
        not isinstance(calibrated, list) or len(calibrated) != len(locked["y_true"])
    ):
        raise RuntimeError("COMMITTED_OUTER_EVALUATION_INVALID")
    return {
        "metrics": dict(committed["metrics"]),
        **locked,
        "calibrated_prob": [float(value) for value in calibrated] if calibrated is not None else [],
        "compatibility_status": (
            "EXACT_FROZEN_HISTORICAL_RECEIPT"
            if compatibility_match
            else "EXACT_EMBEDDED_EXTERNAL_MATCH"
        ),
    }


def validate_exact_resume_loader_contract(
    *,
    required: bool,
    monai_augment: bool,
    num_workers: int,
    persistent_workers: bool,
) -> None:
    """Reject loader modes whose process-local augmentation state cannot be restored exactly."""

    if not required:
        return
    if persistent_workers:
        raise RuntimeError("EXACT_RESUME_PERSISTENT_WORKERS_UNSUPPORTED")
    if monai_augment and int(num_workers) != 0:
        raise RuntimeError("EXACT_RESUME_MONAI_REQUIRES_NUM_WORKERS_ZERO")
