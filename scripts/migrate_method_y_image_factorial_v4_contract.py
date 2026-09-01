"""Migrate resumable Method Y factorial-v4 artifacts after a code-contract fix.

The v4 trainer binds the source/runtime contract into every inner checkpoint.
Changing the trainer (for example, fixing a CPU/CUDA state comparison) therefore
requires an explicit, auditable migration before an old output root can resume.
This tool never changes model values, histories, memberships, caches, splits, or
outer-test artifacts.  It only changes the contract digest after reconstructing
and validating the old cell contract, and it does so with per-file atomic writes.

Usage (from PowerShell):
    python scripts/migrate_method_y_image_factorial_v4_contract.py --dry-run
    python scripts/migrate_method_y_image_factorial_v4_contract.py --apply

The migration is idempotent.  A partially applied migration can be resumed with
the same ``--apply`` command; an unexpected digest or any outer-test exposure
causes a fail-closed error instead of guessing.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival.atomic_io import atomic_write_json, read_json  # noqa: E402
from src.method_y_revival.contracts import sha256_mapping  # noqa: E402
from src.method_y_revival.image_factorial_orchestrator_v4 import (  # noqa: E402
    build_factorial_cell_contract,
)
from src.method_y_revival.image_factorial_v4 import FACTORIAL_ARMS  # noqa: E402
from src.method_y_revival.image_training_runtime_v3 import (  # noqa: E402
    FoldTabularTransformer,
    _atomic_torch_save,
)
from src.method_y_revival.process_guard import (  # noqa: E402
    WriterLock,
    pid_is_active,
)
from scripts.run_method_y_image_factorial_v4 import (  # noqa: E402
    _preflight,
    _restrict_inner_splits,
)


class MigrationError(RuntimeError):
    """Raised when migration cannot be proven safe."""


_CHECKPOINT_RE = re.compile(
    r"^development[\\/]"
    r"(?P<branch>amyloid|tau)[\\/]"
    r"(?P<arm>A[0-4])[\\/]"
    r"outer_(?P<outer>[0-9]+)[\\/]inner[\\/]"
    r"(?P<candidate>[^\\/]+)[\\/]fold_(?P<inner>[0-9]+)"
    r"[\\/]epoch_checkpoint\.pt$",
    re.IGNORECASE,
)

MIGRATION_FILE = "contract_migration_v4_20260901.json"
MIGRATION_SCHEMA = "1.0"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_digest_without(payload: dict[str, Any], field: str) -> str:
    return sha256_mapping({key: value for key, value in payload.items() if key != field})


def _normalise_for_hash(value: Any, hasher: hashlib._Hash) -> None:
    """Hash nested torch/numpy/python state without serialisation addresses."""

    import numpy as np
    import torch

    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"TORCH\0")
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(repr(tuple(tensor.shape)).encode("ascii"))
        hasher.update(tensor.numpy().tobytes())
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"NUMPY\0")
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(repr(tuple(array.shape)).encode("ascii"))
        hasher.update(array.tobytes())
        return
    if isinstance(value, dict):
        hasher.update(b"DICT\0")
        for key in sorted(value, key=lambda item: str(item)):
            _normalise_for_hash(str(key), hasher)
            _normalise_for_hash(value[key], hasher)
        return
    if isinstance(value, (list, tuple)):
        hasher.update(("LIST\0" if isinstance(value, list) else "TUPLE\0").encode())
        for item in value:
            _normalise_for_hash(item, hasher)
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        hasher.update(b"BYTES\0")
        hasher.update(bytes(value))
        return
    if value is None:
        hasher.update(b"NONE\0")
        return
    hasher.update(type(value).__name__.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(repr(value).encode("utf-8"))


def _semantic_fingerprint(payload: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    _normalise_for_hash(
        {key: value for key, value in payload.items() if key != "run_contract_digest"},
        hasher,
    )
    return hasher.hexdigest()


def _state_fingerprint(state: Any) -> str:
    hasher = hashlib.sha256()
    _normalise_for_hash(state, hasher)
    return hasher.hexdigest()


def _path(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _read_raw_config(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MigrationError("CONFIG_MAPPING_REQUIRED")
    return payload


def _targeting_processes(output_root: Path) -> list[dict[str, Any]]:
    """Return live processes whose command line targets this output root."""

    try:
        import psutil
    except ImportError:
        return []
    needle = str(output_root.resolve()).lower().replace("/", "\\")
    rows: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            pid = int(process.info["pid"])
            if pid == os.getpid():
                continue
            command = " ".join(process.info.get("cmdline") or [])
            if needle not in command.lower().replace("/", "\\"):
                continue
            if not process.is_running():
                continue
            rows.append(
                {
                    "pid": pid,
                    "name": process.info.get("name"),
                    "command_line": command,
                    "create_time": process.info.get("create_time"),
                }
            )
        except (psutil.Error, OSError, ValueError):
            continue
    return rows


def _assert_no_active_writer(output_root: Path) -> dict[str, Any] | None:
    """Check both the versioned lock and the OS process list."""

    lock_path = output_root / ".factorial_v4.writer.lock"
    stale_lock: dict[str, Any] | None = None
    if lock_path.exists():
        try:
            lock = read_json(lock_path)
        except Exception as exc:  # noqa: BLE001
            raise MigrationError(f"UNREADABLE_TRAINING_LOCK:{lock_path}") from exc
        if pid_is_active(lock.get("pid", -1), lock.get("process_start_time")):
            raise MigrationError(
                f"ACTIVE_TRAINING_WRITER:pid={lock.get('pid')}:output_root={output_root}"
            )
        stale_lock = copy.deepcopy(lock)
    processes = _targeting_processes(output_root)
    if processes:
        raise MigrationError(f"ACTIVE_TARGETING_PROCESS:{json.dumps(processes, sort_keys=True)}")
    return stale_lock


def _assert_no_outer_exposure(output_root: Path) -> None:
    outer = output_root / "private" / "outer_evaluation"
    if outer.exists():
        files = [path for path in outer.rglob("*") if path.is_file()]
        if files:
            raise MigrationError(
                "OUTER_EXPOSURE_PRESENT:" + ",".join(str(path) for path in files[:10])
            )
    for pattern in ("winner_*.json", "inner_only_arm_selection.json"):
        if list((output_root / "orchestration").rglob(pattern)):
            raise MigrationError(f"SELECTION_ARTIFACTS_PRESENT:{pattern}")


def _fingerprint_inputs(
    *, old_contract: dict[str, Any], new_preflight: dict[str, Any], root: Path
) -> None:
    old_inputs = old_contract.get("input_fingerprints")
    new_inputs = new_preflight.get("input_fingerprints")
    if not isinstance(old_inputs, dict) or not isinstance(new_inputs, dict):
        raise MigrationError("INPUT_FINGERPRINTS_MISSING")
    if set(old_inputs) != set(new_inputs):
        raise MigrationError("INPUT_FINGERPRINT_KEYS_CHANGED")
    for key in sorted(old_inputs):
        old = old_inputs[key]
        new = new_inputs[key]
        if old.get("sha256") != new.get("sha256") or int(old.get("size", -1)) != int(
            new.get("size", -2)
        ):
            raise MigrationError(f"INPUT_FINGERPRINT_CHANGED:{key}")
        path = _path(root, str(new.get("path", "")))
        if not path.is_file() or _sha256_file(path) != new.get("sha256"):
            raise MigrationError(f"INPUT_FILE_CHANGED_OR_MISSING:{key}:{path}")


def _make_bound_old_config(raw: dict[str, Any], old_contract: dict[str, Any]) -> dict[str, Any]:
    bound = copy.deepcopy(raw)
    software = old_contract.get("software_runtime_contract")
    if not isinstance(software, dict):
        raise MigrationError("OLD_SOFTWARE_CONTRACT_MISSING")
    bound["software_runtime_contract"] = software
    training_source = software.get("training_source", {})
    bound["training_source_sha256"] = training_source.get("bundle_sha256")
    return bound


def _parse_checkpoint_path(path: Path, root: Path) -> dict[str, Any]:
    relative = str(path.relative_to(root)).replace("/", "\\")
    match = _CHECKPOINT_RE.match(relative)
    if not match:
        raise MigrationError(f"UNEXPECTED_CHECKPOINT_PATH:{path}")
    row = match.groupdict()
    row["outer"] = int(row["outer"])
    row["inner"] = int(row["inner"])
    row["branch"] = row["branch"].lower()
    row["arm"] = row["arm"].upper()
    return row


def _load_partition_contexts(
    *, raw_config: dict[str, Any], root: Path, branches: Iterable[str]
) -> dict[tuple[str, int, int], dict[str, Any]]:
    import pandas as pd

    contexts: dict[tuple[str, int, int], dict[str, Any]] = {}
    for branch in branches:
        branch_cfg = raw_config["branches"][branch]
        membership = pd.read_parquet(_path(root, branch_cfg["membership"]))
        development = membership[membership["role"] == "development"].copy()
        splits = read_json(_path(root, branch_cfg["splits"]))
        for outer in splits.get("outer_folds", []):
            outer_fold = int(outer["fold"])
            outer_keys = set(
                outer.get("train_subjects") or outer.get("train_subject_keys") or []
            )
            train_frame = development[
                development["patient_hash"].astype(str).isin(outer_keys)
            ].copy()
            restricted = _restrict_inner_splits(
                outer.get("inner_folds", []), set(train_frame["patient_hash"].astype(str))
            )
            for inner in restricted:
                inner_fold = int(inner["fold"])
                train_keys = set(inner["train_subjects"])
                validation_keys = set(inner["validation_subjects"])
                train = development[
                    development["patient_hash"].astype(str).isin(train_keys)
                ].copy()
                transformer = FoldTabularTransformer.fit(train, branch_cfg["model_features"])
                contexts[(branch, outer_fold, inner_fold)] = {
                    "train_subjects": sorted(train_keys),
                    "validation_subjects": sorted(validation_keys),
                    "partition_contract": {
                        "train_subjects_sha256": sha256_mapping(
                            {"subjects": sorted(train_keys)}
                        ),
                        "validation_subjects_sha256": sha256_mapping(
                            {"subjects": sorted(validation_keys)}
                        ),
                        "train_count": len(train_keys),
                        "validation_count": len(validation_keys),
                        "tabular_transformer": transformer.to_dict(),
                    },
                }
    return contexts


def _cell_digest(
    *,
    config: dict[str, Any],
    row: dict[str, Any],
    contract_inputs: dict[str, str],
    partition_contract: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    return build_factorial_cell_contract(
        config=config,
        branch=row["branch"],
        arm_id=row["arm"],
        outer_fold=int(row["outer"]),
        candidate_id=row["candidate"],
        membership_sha256=contract_inputs["membership"],
        cache_sha256=contract_inputs["cache"],
        mask_sha256=contract_inputs["mask"],
        split_sha256=contract_inputs["split"],
        augmentation_qc_sha256=contract_inputs["augmentation_qc"],
        stage="INNER_HPO",
        inner_fold=int(row["inner"]),
        partition_contract=partition_contract,
        training_parameters={
            "learning_rate": float(candidate["learning_rate"]),
            "weight_decay": float(candidate["weight_decay"]),
            "max_epochs": int(config["model"]["max_epochs"]),
            "patience": int(config["model"]["early_stopping_patience"]),
        },
    )


def _contract_inputs(
    *, old_contract: dict[str, Any], preflight: dict[str, Any], branch: str, old: bool
) -> dict[str, str]:
    source = old_contract["input_fingerprints"] if old else preflight["input_fingerprints"]
    return {
        "membership": source[f"{branch}_membership"]["sha256"],
        "cache": source[f"{branch}_pattern_cache"]["sha256"],
        "mask": source["foreground_mask"]["sha256"],
        "split": source[f"{branch}_splits"]["sha256"],
        "augmentation_qc": (
            old_contract["augmentation_qc_contract_sha256"]
            if old
            else preflight["config"]["shared"]["augmentation_qc_expected_contract_sha256"]
        ),
    }


def _master_digest(
    *, config: dict[str, Any], branch: str, inputs: dict[str, str]
) -> str:
    return sha256_mapping(
        {
            "schema_version": "4.0",
            "branch": branch,
            "arms": list(config["execution"]["arms"]),
            "seed": int(config["seed"]),
            "hpo_candidates": config["model"]["hpo_candidates"],
            "membership_sha256": inputs["membership"],
            "cache_sha256": inputs["cache"],
            "mask_sha256": inputs["mask"],
            "split_sha256": inputs["split"],
            "augmentation_qc_sha256": inputs["augmentation_qc"],
            "model": config.get("model", {}),
            "factorial_arms": config.get("factorial_arms", {}),
            "runtime": config.get("runtime", {}),
            "software_runtime_contract": config.get("software_runtime_contract", {}),
        }
    )


def _checkpoint_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in (root / "development").rglob("epoch_checkpoint.pt")
        if path.is_file()
    )


def _candidate_by_id(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["candidate_id"]): row for row in config["model"]["hpo_candidates"]
    }


def _validate_and_plan(
    *, config_path: Path, output_root: Path
) -> dict[str, Any]:
    if not output_root.is_dir():
        raise MigrationError(f"OUTPUT_ROOT_MISSING:{output_root}")
    stale_lock = _assert_no_active_writer(output_root)
    _assert_no_outer_exposure(output_root)
    contract_path = output_root / "run_contract.json"
    if not contract_path.is_file():
        raise MigrationError("RUN_CONTRACT_MISSING")
    old_contract = read_json(contract_path)
    if old_contract.get("schema_version") != "4.0":
        raise MigrationError("UNSUPPORTED_OLD_CONTRACT_SCHEMA")
    old_digest = str(old_contract.get("digest", ""))
    if len(old_digest) != 64 or _json_digest_without(old_contract, "digest") != old_digest:
        raise MigrationError("OLD_RUN_CONTRACT_SELF_HASH_INVALID")

    raw = _read_raw_config(config_path)
    preflight = _preflight(config_path)
    new_digest = str(preflight["run_contract_sha256"])
    if old_digest == new_digest:
        raise MigrationError("CONTRACT_ALREADY_CURRENT")
    _fingerprint_inputs(old_contract=old_contract, new_preflight=preflight, root=ROOT)
    if old_contract.get("config_sha256") != _sha256_file(config_path):
        raise MigrationError("CONFIG_CHANGED_SINCE_OLD_RUN")

    state_path = output_root / "orchestration_state.json"
    old_state = read_json(state_path) if state_path.is_file() else None
    if old_state is not None and old_state.get("run_contract_sha256") != old_digest:
        raise MigrationError("ORCHESTRATION_STATE_NOT_ON_OLD_CONTRACT")

    old_config = _make_bound_old_config(raw, old_contract)
    candidates = _candidate_by_id(raw)
    checkpoints = _checkpoint_files(output_root)
    contexts = _load_partition_contexts(
        raw_config=raw,
        root=ROOT,
        branches={_parse_checkpoint_path(path, output_root)["branch"] for path in checkpoints},
    )
    mappings: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        row = _parse_checkpoint_path(checkpoint_path, output_root)
        if row["candidate"] not in candidates:
            raise MigrationError(f"UNKNOWN_HPO_CANDIDATE:{row['candidate']}")
        context = contexts.get((row["branch"], row["outer"], row["inner"]))
        if context is None:
            raise MigrationError(f"PARTITION_CONTEXT_MISSING:{row}")
        old_inputs = _contract_inputs(
            old_contract=old_contract, preflight=preflight, branch=row["branch"], old=True
        )
        new_inputs = _contract_inputs(
            old_contract=old_contract, preflight=preflight, branch=row["branch"], old=False
        )
        old_cell = _cell_digest(
            config=old_config,
            row=row,
            contract_inputs=old_inputs,
            partition_contract=context["partition_contract"],
            candidate=candidates[row["candidate"]],
        )
        new_cell = _cell_digest(
            config=preflight["config"],
            row=row,
            contract_inputs=new_inputs,
            partition_contract=context["partition_contract"],
            candidate=candidates[row["candidate"]],
        )
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run_contract_digest") != old_cell:
            raise MigrationError(
                f"CHECKPOINT_CELL_CONTRACT_MISMATCH:{checkpoint_path}:"
                f"stored={checkpoint.get('run_contract_digest')}:expected={old_cell}"
            )
        if checkpoint.get("fold_terminal") is True:
            if checkpoint.get("terminal_reason") not in {"EARLY_STOPPING", "MAX_EPOCHS_COMPLETED"}:
                raise MigrationError(f"TERMINAL_REASON_INVALID:{checkpoint_path}")
        files = [checkpoint_path]
        locked_path = checkpoint_path.parent / "locked_checkpoint.pt"
        summary_path = checkpoint_path.parent / "training_summary.json"
        if checkpoint.get("fold_terminal") is True:
            if not locked_path.is_file() or not summary_path.is_file():
                raise MigrationError(f"TERMINAL_ARTIFACT_MISSING:{checkpoint_path}")
        elif locked_path.is_file():
            raise MigrationError(f"NONTERMINAL_HAS_LOCKED_ARTIFACT:{checkpoint_path}")
        if locked_path.is_file():
            locked = torch.load(locked_path, map_location="cpu", weights_only=False)
            if locked.get("run_contract_digest") != old_cell:
                raise MigrationError(f"LOCKED_CELL_CONTRACT_MISMATCH:{locked_path}")
            if _state_fingerprint(locked.get("model_state")) != _state_fingerprint(
                checkpoint.get("best_model_state")
            ):
                raise MigrationError(f"LOCKED_STATE_DIFFERS_FROM_BEST:{locked_path}")
            files.append(locked_path)
        if summary_path.is_file():
            summary = read_json(summary_path)
            if summary.get("run_contract_digest") != old_cell:
                raise MigrationError(f"SUMMARY_CELL_CONTRACT_MISMATCH:{summary_path}")
            files.append(summary_path)
        file_rows = []
        for path in files:
            if path.suffix.lower() == ".pt":
                import torch

                payload = torch.load(path, map_location="cpu", weights_only=False)
                semantic = _semantic_fingerprint(payload)
            else:
                payload = read_json(path)
                semantic = _json_digest_without(payload, "run_contract_digest")
            file_rows.append(
                {
                    "path": str(path),
                    "kind": "torch" if path.suffix.lower() == ".pt" else "json",
                    "old_sha256": _sha256_file(path),
                    "new_sha256": None,
                    "old_contract_digest": old_cell,
                    "new_contract_digest": new_cell,
                    "semantic_fingerprint": semantic,
                }
            )
        mappings.append(
            {
                "branch": row["branch"],
                "arm": row["arm"],
                "outer_fold": row["outer"],
                "inner_fold": row["inner"],
                "candidate_id": row["candidate"],
                "checkpoint": str(checkpoint_path),
                "old_cell_digest": old_cell,
                "new_cell_digest": new_cell,
                "files": file_rows,
            }
        )

    master_rows = []
    for branch in ("amyloid", "tau"):
        master_path = output_root / "orchestration" / branch / "master_contract.json"
        if not master_path.is_file():
            continue
        old_inputs = _contract_inputs(
            old_contract=old_contract, preflight=preflight, branch=branch, old=True
        )
        new_inputs = _contract_inputs(
            old_contract=old_contract, preflight=preflight, branch=branch, old=False
        )
        master = read_json(master_path)
        old_master = _master_digest(config=old_config, branch=branch, inputs=old_inputs)
        new_master = _master_digest(config=preflight["config"], branch=branch, inputs=new_inputs)
        if master.get("master_contract_sha256") != old_master:
            raise MigrationError(f"MASTER_CONTRACT_MISMATCH:{branch}")
        master_rows.append(
            {
                "branch": branch,
                "path": str(master_path),
                "old_master_digest": old_master,
                "new_master_digest": new_master,
                "old_sha256": _sha256_file(master_path),
                "new_sha256": None,
            }
        )

    migration_id = (
        f"METHOD_Y_V4_CONTRACT_MIGRATION_{old_digest[:12]}_{new_digest[:12]}"
    )
    return {
        "schema_version": MIGRATION_SCHEMA,
        "migration_id": migration_id,
        "status": "PREPARED",
        "created_utc": _utc_now(),
        "host": socket.gethostname(),
        "pid_preflight": os.getpid(),
        "output_root": str(output_root),
        "config_path": str(config_path),
        "old_run_contract_sha256": old_digest,
        "new_run_contract_sha256": new_digest,
        "old_run_contract": old_contract,
        "new_run_contract": {
            **preflight["run_contract_body"],
            "digest": new_digest,
        },
        "stale_training_lock_snapshot": stale_lock,
        "outer_test_exposure": {
            "access_journal_count": 0,
            "result_count": 0,
            "outer_accessed": False,
        },
        "checkpoint_mappings": mappings,
        "master_mappings": master_rows,
        "state_path": str(state_path),
        "state_old_sha256": _sha256_file(state_path) if state_path.is_file() else None,
        "state_new_sha256": None,
        "preservation": {
            "model_values_changed": False,
            "optimizer_values_changed": False,
            "history_changed": False,
            "membership_cache_split_changed": False,
            "outer_results_changed": False,
        },
    }


def _load_or_prepare_plan(
    *, config_path: Path, output_root: Path, apply: bool
) -> tuple[dict[str, Any], Path]:
    provenance_dir = output_root / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = provenance_dir / MIGRATION_FILE
    if provenance_path.is_file():
        existing = read_json(provenance_path)
        if existing.get("status") == "COMMITTED":
            # A committed migration is verified against the current contract below.
            return existing, provenance_path
        if existing.get("status") == "PREPARED":
            if Path(existing.get("output_root", "")).resolve() != output_root.resolve():
                raise MigrationError("MIGRATION_ROOT_MISMATCH")
            return existing, provenance_path
        raise MigrationError("MIGRATION_PROVENANCE_STATUS_INVALID")
    plan = _validate_and_plan(config_path=config_path, output_root=output_root)
    if apply:
        atomic_write_json(provenance_path, plan)
    return plan, provenance_path


def _update_torch_file(row: dict[str, Any], old_digest: str, new_digest: str) -> None:
    import torch

    path = Path(row["path"])
    if not path.is_file():
        raise MigrationError(f"MIGRATION_FILE_MISSING:{path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    current = payload.get("run_contract_digest")
    if current == new_digest:
        if _semantic_fingerprint(payload) != row["semantic_fingerprint"]:
            raise MigrationError(f"MIGRATION_SEMANTIC_DRIFT_AFTER_REWRITE:{path}")
        row["new_sha256"] = _sha256_file(path)
        return
    if current != old_digest:
        raise MigrationError(f"MIGRATION_UNEXPECTED_DIGEST:{path}:{current}")
    if _semantic_fingerprint(payload) != row["semantic_fingerprint"]:
        raise MigrationError(f"MIGRATION_SEMANTIC_DRIFT_BEFORE_REWRITE:{path}")
    payload["run_contract_digest"] = new_digest
    _atomic_torch_save(path, payload)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    if reloaded.get("run_contract_digest") != new_digest or _semantic_fingerprint(
        reloaded
    ) != row["semantic_fingerprint"]:
        raise MigrationError(f"MIGRATION_TORCH_VERIFY_FAILED:{path}")
    row["new_sha256"] = _sha256_file(path)


def _update_json_digest_file(row: dict[str, Any], old_digest: str, new_digest: str) -> None:
    path = Path(row["path"])
    if not path.is_file():
        raise MigrationError(f"MIGRATION_FILE_MISSING:{path}")
    payload = read_json(path)
    current = payload.get("run_contract_digest")
    if current == new_digest:
        if _json_digest_without(payload, "run_contract_digest") != row["semantic_fingerprint"]:
            raise MigrationError(f"MIGRATION_JSON_SEMANTIC_DRIFT_AFTER_REWRITE:{path}")
        row["new_sha256"] = _sha256_file(path)
        return
    if current != old_digest:
        raise MigrationError(f"MIGRATION_UNEXPECTED_DIGEST:{path}:{current}")
    if _json_digest_without(payload, "run_contract_digest") != row["semantic_fingerprint"]:
        raise MigrationError(f"MIGRATION_JSON_SEMANTIC_DRIFT_BEFORE_REWRITE:{path}")
    payload["run_contract_digest"] = new_digest
    atomic_write_json(path, payload)
    verified = read_json(path)
    if verified.get("run_contract_digest") != new_digest or _json_digest_without(
        verified, "run_contract_digest"
    ) != row["semantic_fingerprint"]:
        raise MigrationError(f"MIGRATION_JSON_VERIFY_FAILED:{path}")
    row["new_sha256"] = _sha256_file(path)


def _update_master(row: dict[str, Any]) -> None:
    path = Path(row["path"])
    payload = read_json(path)
    current = payload.get("master_contract_sha256")
    if current == row["new_master_digest"]:
        row["new_sha256"] = _sha256_file(path)
        return
    if current != row["old_master_digest"]:
        raise MigrationError(f"MASTER_UNEXPECTED_DIGEST:{path}:{current}")
    payload["master_contract_sha256"] = row["new_master_digest"]
    payload["contract_migration_id"] = row.get(
        "migration_id", "METHOD_Y_V4_CONTRACT_MIGRATION"
    )
    payload["previous_master_contract_sha256"] = row["old_master_digest"]
    atomic_write_json(path, payload)
    verified = read_json(path)
    if verified.get("master_contract_sha256") != row["new_master_digest"]:
        raise MigrationError(f"MASTER_VERIFY_FAILED:{path}")
    row["new_sha256"] = _sha256_file(path)


def _update_global_contract(plan: dict[str, Any]) -> None:
    path = Path(plan["output_root"]) / "run_contract.json"
    expected = plan["new_run_contract"]
    if path.is_file():
        current = read_json(path)
        if current == expected:
            return
        if current.get("digest") != plan["old_run_contract_sha256"]:
            raise MigrationError("GLOBAL_CONTRACT_UNEXPECTED_DIGEST")
    atomic_write_json(path, expected)
    if read_json(path) != expected:
        raise MigrationError("GLOBAL_CONTRACT_VERIFY_FAILED")


def _update_state(plan: dict[str, Any]) -> None:
    path = Path(plan["state_path"])
    if not path.is_file():
        return
    payload = read_json(path)
    current = payload.get("run_contract_sha256")
    if current == plan["new_run_contract_sha256"]:
        if payload.get("contract_migration_id") != plan["migration_id"]:
            raise MigrationError("STATE_MIGRATION_ID_MISSING")
        plan["state_new_sha256"] = _sha256_file(path)
        return
    if current != plan["old_run_contract_sha256"]:
        raise MigrationError("STATE_UNEXPECTED_CONTRACT")
    payload["run_contract_sha256"] = plan["new_run_contract_sha256"]
    payload["previous_run_contract_sha256"] = plan["old_run_contract_sha256"]
    payload["contract_migration_id"] = plan["migration_id"]
    atomic_write_json(path, payload)
    verified = read_json(path)
    if verified.get("run_contract_sha256") != plan["new_run_contract_sha256"]:
        raise MigrationError("STATE_VERIFY_FAILED")
    plan["state_new_sha256"] = _sha256_file(path)


def _verify_committed(plan: dict[str, Any]) -> None:
    root = Path(plan["output_root"])
    _assert_no_outer_exposure(root)
    contract = read_json(root / "run_contract.json")
    if contract != plan["new_run_contract"]:
        raise MigrationError("COMMITTED_GLOBAL_CONTRACT_MISMATCH")
    for mapping in plan["checkpoint_mappings"]:
        for row in mapping["files"]:
            path = Path(row["path"])
            if not path.is_file() or _sha256_file(path) != row.get("new_sha256"):
                raise MigrationError(f"COMMITTED_FILE_HASH_MISMATCH:{path}")
            if row["kind"] == "torch":
                import torch

                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("run_contract_digest") != row["new_contract_digest"]:
                    raise MigrationError(f"COMMITTED_CELL_DIGEST_MISMATCH:{path}")
                if _semantic_fingerprint(payload) != row["semantic_fingerprint"]:
                    raise MigrationError(f"COMMITTED_CELL_SEMANTIC_MISMATCH:{path}")
            else:
                payload = read_json(path)
                if payload.get("run_contract_digest") != row["new_contract_digest"]:
                    raise MigrationError(f"COMMITTED_JSON_DIGEST_MISMATCH:{path}")
    for row in plan["master_mappings"]:
        payload = read_json(row["path"])
        if payload.get("master_contract_sha256") != row["new_master_digest"]:
            raise MigrationError(f"COMMITTED_MASTER_MISMATCH:{row['path']}")
    if Path(plan["state_path"]).is_file():
        state = read_json(plan["state_path"])
        if state.get("run_contract_sha256") != plan["new_run_contract_sha256"]:
            raise MigrationError("COMMITTED_STATE_MISMATCH")


def apply_plan(plan: dict[str, Any], provenance_path: Path) -> dict[str, Any]:
    output_root = Path(plan["output_root"])
    _assert_no_active_writer(output_root)
    lock_path = output_root / ".factorial_v4.migration.lock"
    with WriterLock(
        lock_path,
        output_root=output_root,
        run_contract_digest=plan["new_run_contract_sha256"],
    ):
        # A stale training lock is removed only after its snapshot is in the plan.
        training_lock = output_root / ".factorial_v4.writer.lock"
        if training_lock.exists():
            try:
                lock = read_json(training_lock)
            except Exception as exc:  # noqa: BLE001
                raise MigrationError("TRAINING_LOCK_BECAME_UNREADABLE") from exc
            if pid_is_active(lock.get("pid", -1), lock.get("process_start_time")):
                raise MigrationError("TRAINING_WRITER_STARTED_DURING_MIGRATION")
            training_lock.unlink()

        # Keep the journal current after every atomically rewritten artifact.
        for mapping in plan["checkpoint_mappings"]:
            for row in mapping["files"]:
                if row["kind"] == "torch":
                    _update_torch_file(
                        row,
                        mapping["old_cell_digest"],
                        mapping["new_cell_digest"],
                    )
                else:
                    _update_json_digest_file(
                        row,
                        mapping["old_cell_digest"],
                        mapping["new_cell_digest"],
                    )
            atomic_write_json(provenance_path, plan)

        for row in plan["master_mappings"]:
            row["migration_id"] = plan["migration_id"]
            _update_master(row)
            atomic_write_json(provenance_path, plan)

        # Commit the global contract and state only after all per-cell files pass.
        _update_global_contract(plan)
        _update_state(plan)
        plan["status"] = "COMMITTED"
        plan["committed_utc"] = _utc_now()
        atomic_write_json(provenance_path, plan)
        _verify_committed(plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "method_y_image_factorial_v4.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "method_y_revival_v4_image_factorial_v1",
    )
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    try:
        existing_provenance = output_root / "provenance" / MIGRATION_FILE
        plan, provenance_path = _load_or_prepare_plan(
            config_path=config_path.resolve(), output_root=output_root.resolve(), apply=args.apply
        )
        if args.apply:
            if plan.get("status") == "COMMITTED":
                _verify_committed(plan)
            else:
                plan = apply_plan(plan, provenance_path)
        result = {
            "status": "APPLIED" if args.apply else "DRY_RUN_READY",
            "migration_id": plan.get("migration_id"),
            "old_run_contract_sha256": plan.get("old_run_contract_sha256"),
            "new_run_contract_sha256": plan.get("new_run_contract_sha256"),
            "checkpoint_count": len(plan.get("checkpoint_mappings", [])),
            "file_count": sum(
                len(row.get("files", [])) for row in plan.get("checkpoint_mappings", [])
            ),
            "master_count": len(plan.get("master_mappings", [])),
            "outer_test_exposure": plan.get("outer_test_exposure"),
            "provenance_path": str(provenance_path),
            "output_root": str(output_root.resolve()),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"status": "BLOCKED", "reason": f"{type(exc).__name__}:{exc}"},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
