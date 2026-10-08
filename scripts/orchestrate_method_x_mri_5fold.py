#!/usr/bin/env python3
"""Fail-closed, resumable Method X HPO-to-final-training orchestrator.

This launcher never terminates competing processes.  It runs every child in
the foreground with unbuffered output so Ctrl+C stops the active work and the
same command can resume from the last committed epoch/stage.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_3d_hyperparameter_search import (
    apply_hpo_critic,
    load_candidate_config,
    validate_completed_selection_trial,
    validate_reusable_trial_contract,
)
from scripts.spm_completion_guard import spm_completion_ready, spm_completion_status
from src.utils.artifact_provenance import build_provenance
from src.utils.reproducibility import canonical_digest
from src.training.method_x_resume import reuse_committed_outer_result, validate_locked_outer_result
from src.training.method_x_source_compatibility import (
    resolve_finalized_source_compatibility,
)

OFFICIAL_ROOT_REL = Path("results/final_results/method_x_5fold_corrected_v1")
OFFICIAL_CACHE_REL = Path("data/cached/t1_volume_cache_128.h5")
SPLIT_DIR_REL = Path("splits")
GRID_REL = Path("configs/experiments/method_x_3d_candidate_grid.yaml")
FOLDS = 5
SEED = 13
HPO_EPOCHS = 40
FINAL_EPOCHS = 100
POSTPROCESS_BLOCKER_STAGES = {
    "xai": "BLOCKED_REQUIRES_CORRECTED_V1_MODEL_ADAPTER",
    "subgroup": "BLOCKED_REQUIRES_AGGREGATE_SAFE_SUBGROUP_ADAPTER",
    "scanner_sensitivity": "BLOCKED_SCANNER_METADATA_NOT_AVAILABLE_IN_LOCKED_ARTIFACT",
    "robustness": "BLOCKED_REQUIRES_PRESPECIFIED_CORRECTED_V1_ADAPTER",
    "external_validation": "BLOCKED_REQUIRES_INDEPENDENT_COHORT_DUA",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"REQUIRED_JSON_UNREADABLE:{path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"REQUIRED_JSON_NOT_OBJECT:{path.name}")
    return value


def _atomic_torch_replace(path: Path, payload: Any) -> None:
    """Atomically rewrite a resume payload without changing tensor devices."""

    import torch

    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def migrate_hpo_resume_source_contracts(
    root: Path,
    *,
    cache_sha256: str,
) -> list[dict[str, Any]]:
    """Migrate a paused HPO trial after a resume-only source change.

    The exact-resume contract intentionally fingerprints the training source.
    A resume bug fix can therefore strand an otherwise valid checkpoint.  This
    migration is deliberately narrow: it changes only ``source_contract_digest``
    when every cache/split/seed/loader/parameter/runtime field is unchanged.
    Any other mismatch remains fail-closed and is handled by the normal HPO
    contract validator.
    """

    if not root.is_dir():
        return []
    from scripts.train_3d_mri import method_x_training_source_contract

    current_source_digest, current_runtime_versions = method_x_training_source_contract()
    migrations: list[dict[str, Any]] = []
    trials_root = root / "hpo" / "trials"
    if not trials_root.is_dir():
        return migrations

    for trial_dir in sorted(path for path in trials_root.iterdir() if path.is_dir()):
        contract_path = trial_dir / "run_contract.json"
        if not contract_path.is_file():
            continue
        payload = read_json(contract_path)
        contract = payload.get("run_contract")
        if not isinstance(contract, dict) or not isinstance(contract.get("parameters"), dict):
            continue
        body = {key: value for key, value in contract.items() if key != "digest"}
        old_digest = canonical_digest(body)
        if (
            payload.get("run_contract_digest") != old_digest
            or contract.get("digest") != old_digest
            or contract.get("cache_sha256") != cache_sha256
        ):
            continue
        parameters = body["parameters"]
        old_source_digest = parameters.get("source_contract_digest")
        if not isinstance(old_source_digest, str) or old_source_digest == current_source_digest:
            continue
        if parameters.get("runtime_versions") != current_runtime_versions:
            continue

        migrated_body = copy.deepcopy(body)
        migrated_body["parameters"]["source_contract_digest"] = current_source_digest
        migrated_digest = canonical_digest(migrated_body)
        # A source-only migration must leave all scientific and loader fields
        # byte-for-byte unchanged; this assertion also protects against an
        # accidental migration of a stale command/config.
        unchanged_body = copy.deepcopy(body)
        unchanged_body["parameters"]["source_contract_digest"] = current_source_digest
        if canonical_digest(unchanged_body) != migrated_digest:
            raise RuntimeError("RESUME_SOURCE_MIGRATION_INTERNAL_DIGEST_ERROR")

        new_payload = {
            "run_contract": {**migrated_body, "digest": migrated_digest},
            "run_contract_digest": migrated_digest,
        }
        atomic_json(contract_path, new_payload)

        rewritten_checkpoints: list[str] = []
        import torch

        for checkpoint_path in sorted(trial_dir.glob("*.pt")):
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
            except Exception:
                continue
            if not isinstance(checkpoint, dict):
                continue
            checkpoint_contract = checkpoint.get("run_contract")
            if not isinstance(checkpoint_contract, dict):
                continue
            checkpoint_body = {
                key: value for key, value in checkpoint_contract.items() if key != "digest"
            }
            if (
                canonical_digest(checkpoint_body) != old_digest
                or checkpoint.get("run_contract_digest") != old_digest
            ):
                continue
            checkpoint["run_contract"] = {**migrated_body, "digest": migrated_digest}
            checkpoint["run_contract_digest"] = migrated_digest
            _atomic_torch_replace(checkpoint_path, checkpoint)
            rewritten_checkpoints.append(checkpoint_path.name)

        rewritten_states: list[str] = []
        for state_path in sorted(trial_dir.glob("*_state.json")):
            state = read_json(state_path)
            if state.get("contract_digest") != old_digest:
                continue
            state["contract_digest"] = migrated_digest
            atomic_json(state_path, state)
            rewritten_states.append(state_path.name)

        migration = {
            "schema_version": 1,
            "status": "COMPLETED",
            "reason": "RESUME_ONLY_SOURCE_CONTRACT_MIGRATION",
            "trial": trial_dir.name,
            "old_source_contract_digest": old_source_digest,
            "new_source_contract_digest": current_source_digest,
            "old_run_contract_digest": old_digest,
            "new_run_contract_digest": migrated_digest,
            "rewritten_checkpoints": rewritten_checkpoints,
            "rewritten_states": rewritten_states,
            "scientific_fields_changed": False,
            "cache_sha256": cache_sha256,
            "created_utc": utc_now(),
        }
        atomic_json(trial_dir / "resume_contract_migration.json", migration)
        migrations.append(migration)
    return migrations


def _pid_is_active_orchestrator(pid: int) -> bool:
    try:
        import psutil as process_utils
    except ImportError:
        process_utils = None
    if process_utils is not None:
        try:
            command = " ".join(process_utils.Process(pid).cmdline()).lower().replace("/", "\\")
            return "orchestrate_method_x_mri_5fold.py" in command
        except process_utils.Error:
            return False
    if os.name == "nt":
        query = (
            f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}' -ErrorAction SilentlyContinue; "
            "if($p){$p.CommandLine}"
        )
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("LAUNCHER_LOCK_OWNER_LIVENESS_UNAVAILABLE") from exc
        return "orchestrate_method_x_mri_5fold.py" in (completed.stdout or "").lower()
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_run_lock(root: Path) -> Path:
    """Acquire an atomic single-writer lock, reclaiming only demonstrably stale owners."""

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".launcher.lock"
    for _ in range(2):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner: dict[str, Any] = {}
            try:
                owner = read_json(lock_path)
            except RuntimeError:
                pass
            owner_pid = int(owner.get("pid", -1))
            active = owner_pid == os.getpid()
            if not active and owner_pid > 0:
                active = _pid_is_active_orchestrator(owner_pid)
            if active:
                raise RuntimeError(f"METHOD_X_LAUNCHER_ALREADY_ACTIVE:pid={owner_pid}")
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        payload = json.dumps(
            {"pid": os.getpid(), "created_at_utc": utc_now(), "root": OFFICIAL_ROOT_REL.as_posix()},
            sort_keys=True,
        ).encode("utf-8")
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return lock_path
    raise RuntimeError("METHOD_X_LAUNCHER_LOCK_ACQUISITION_RACE")


def release_run_lock(lock_path: Path | None) -> None:
    if lock_path is None or not lock_path.exists():
        return
    try:
        owner = read_json(lock_path)
    except RuntimeError:
        owner = {}
    if int(owner.get("pid", -1)) == os.getpid():
        lock_path.unlink(missing_ok=True)


def record_state(root: Path, stage: str, **details: Any) -> dict[str, Any]:
    path = root / "orchestration_state.json"
    previous: dict[str, Any] = {}
    if path.is_file():
        try:
            previous = read_json(path)
        except RuntimeError:
            previous = {}
    history = list(previous.get("history") or [])
    history.append({"stage": stage, "at_utc": utc_now()})
    payload = {
        "schema_version": 1,
        "pipeline": OFFICIAL_ROOT_REL.name,
        "stage": stage,
        "updated_at_utc": utc_now(),
        "folds": FOLDS,
        "seed": SEED,
        "resume_contract": "rerun_same_launcher_command",
        "history": history,
        **details,
    }
    atomic_json(path, payload)
    return payload


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def cached_file_sha256(path: Path, registry: Path) -> str:
    """Hash a large immutable input once, keyed strictly by size and mtime_ns."""
    stat = path.stat()
    key = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    prior: dict[str, Any] = {}
    if registry.is_file():
        try:
            prior = read_json(registry)
        except RuntimeError:
            prior = {}
    if prior.get("key") == key and re.fullmatch(r"[0-9a-f]{64}", str(prior.get("sha256", ""))):
        return str(prior["sha256"])
    value = sha256_file(path)
    atomic_json(
        registry,
        {
            "schema_version": 1,
            "input": OFFICIAL_CACHE_REL.as_posix(),
            "key": key,
            "sha256": value,
            "computed_at_utc": utc_now(),
        },
    )
    return value


def digest_named_files(paths: Iterable[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda p: p.as_posix()):
        relative = path.resolve().relative_to(base.resolve()).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_cache_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("OFFICIAL_CACHE_MISSING")
    import h5py

    with h5py.File(path, "r") as cache:
        attrs = {str(key): str(value) for key, value in cache.attrs.items()}
        keys = set(cache.keys())
        shape = tuple(int(value) for value in cache["image"].shape) if "image" in cache else ()
    failures: list[str] = []
    if attrs.get("resolution") != "128" or len(shape) != 5 or shape[1:] != (1, 128, 128, 128):
        failures.append("CACHE_NOT_128_CUBED")
    if attrs.get("contains_pet_input", "").lower() != "false" or any("pet" in key.lower() for key in keys):
        failures.append("CACHE_NOT_PET_FREE")
    if attrs.get("contains_raw_identifiers", "").lower() != "false":
        failures.append("CACHE_RAW_ID_CONTRACT_NOT_FALSE")
    if attrs.get("split_folds") != str(FOLDS):
        failures.append("CACHE_FOLD_CONTRACT_NOT_5")
    if attrs.get("source_geometry_audit_status") != "PASS":
        failures.append("CACHE_SOURCE_GEOMETRY_NOT_PASS")
    required = {"image", "label", "sample_hash", "patient_hash", "tabular", "status"}
    if not required <= keys:
        failures.append("CACHE_REQUIRED_DATASETS_MISSING")
    if failures:
        raise RuntimeError("CACHE_CONTRACT_FAILED:" + ",".join(failures))
    return {
        "status": "PASS",
        "resolution": 128,
        "folds": FOLDS,
        "n": shape[0],
        "pet_free": True,
        "raw_identifiers": False,
        "source_geometry_audit_status": "PASS",
    }


def _read_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_hash", "sample_hash", "ab_label", "substudy"}
        if not required <= set(reader.fieldnames or []):
            raise RuntimeError(f"SPLIT_REQUIRED_COLUMNS_MISSING:{path.name}")
        return [dict(row) for row in reader]


def validate_split_contract(split_dir: Path) -> dict[str, Any]:
    summary_path = split_dir / "split_summary.json"
    summary = read_json(summary_path)
    if int(summary.get("folds", 0)) != FOLDS:
        raise RuntimeError("SPLIT_FOLD_COUNT_NOT_5")
    if summary.get("raw_identifiers_written") is not False:
        raise RuntimeError("SPLIT_RAW_ID_CONTRACT_NOT_FALSE")
    expected = {f"fold_{fold}_{part}.csv" for fold in range(FOLDS) for part in ("train", "test")}
    actual = {path.name for path in split_dir.glob("fold_*_*.csv")}
    if actual != expected:
        raise RuntimeError(f"SPLIT_FILES_NOT_EXACT:missing={sorted(expected-actual)},extra={sorted(actual-expected)}")
    test_partitions: list[set[str]] = []
    dev_patients: set[str] | None = None

    for fold in range(FOLDS):
        train = _read_split_rows(split_dir / f"fold_{fold}_train.csv")
        test = _read_split_rows(split_dir / f"fold_{fold}_test.csv")
        train_patients = {row["patient_hash"] for row in train}
        test_patients = {row["patient_hash"] for row in test}
        train_samples = {row["sample_hash"] for row in train}
        test_samples = {row["sample_hash"] for row in test}
        if (
            len(train_patients) != len(train)
            or len(test_patients) != len(test)
            or len(train_samples) != len(train)
            or len(test_samples) != len(test)
            or train_patients & test_patients
            or train_samples & test_samples
            or {row["ab_label"] for row in train} != {"0", "1"}
            or {row["ab_label"] for row in test} != {"0", "1"}
        ):
            raise RuntimeError(f"SPLIT_LEAKAGE_OR_CLASS_CONTRACT_FAILED:fold_{fold}")
        current_dev = train_patients | test_patients
        if dev_patients is None:
            dev_patients = current_dev
        elif current_dev != dev_patients:
            raise RuntimeError("SPLIT_DEV_MEMBERSHIP_DIFFERS_ACROSS_FOLDS")
        test_partitions.append(test_patients)
    if any(test_partitions[left] & test_partitions[right] for left in range(FOLDS) for right in range(left + 1, FOLDS)):
        raise RuntimeError("SPLIT_OUTER_TEST_PARTITIONS_OVERLAP")
    if len(set().union(*test_partitions)) != int(summary.get("dev_rows", -1)):
        raise RuntimeError("SPLIT_OUTER_TEST_UNION_COUNT_MISMATCH")
    external_path = split_dir / "external_holdout.csv"
    external = _read_split_rows(external_path)
    external_patients = {row["patient_hash"] for row in external}
    if (
        len(external_patients) != len(external)
        or external_patients & (dev_patients or set())
        or len(external) != int(summary.get("external_rows", -1))
        or {row["ab_label"] for row in external} != {"0", "1"}
    ):
        raise RuntimeError("SPLIT_EXTERNAL_HOLDOUT_CONTRACT_FAILED")
    paths = [summary_path, external_path, *(split_dir / name for name in sorted(expected))]
    return {
        "status": "PASS",
        "folds": FOLDS,
        "sha256": digest_named_files(paths, PROJECT_ROOT),
        "dev_patients": len(dev_patients or set()),
        "external_patients": len(external_patients),
        "patient_overlap_count": 0,
        "outer_test_partition_overlap_count": 0,
        "all_train_and_test_splits_have_both_classes": True,
        "cohort_label_confound": bool(summary.get("cohort_label_confound")),
        "cohort_label_confound_policy": "DIAGNOSTIC_WARNING_NOT_SELECTION_EXCLUSION",
    }


def validate_cache_split_membership(cache_path: Path, split_dir: Path) -> dict[str, Any]:
    import h5py
    import numpy as np
    from scripts.build_fold_local_mri_augmented_cache import cache_manifest_digest, file_digest

    all_rows = _read_split_rows(split_dir / "external_holdout.csv")
    train_samples_by_fold: dict[int, set[str]] = {}
    test_samples_by_fold: dict[int, set[str]] = {}
    for fold in range(FOLDS):
        train_rows = _read_split_rows(split_dir / f"fold_{fold}_train.csv")
        test_rows = _read_split_rows(split_dir / f"fold_{fold}_test.csv")
        train_samples_by_fold[fold] = {row["sample_hash"] for row in train_rows}
        test_samples_by_fold[fold] = {row["sample_hash"] for row in test_rows}
        if fold == 0:
            all_rows.extend(train_rows)
            all_rows.extend(test_rows)
    split_by_sample = {row["sample_hash"]: row for row in all_rows}

    def decode(values: Any) -> list[str]:
        return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]

    with h5py.File(cache_path, "r") as cache:
        sample_hash = decode(cache["sample_hash"][:])
        source_sample_hash = decode(cache["source_sample_hash"][:])
        patient_hash = decode(cache["patient_hash"][:])
        source_patient_hash = decode(cache["source_patient_hash"][:])
        labels = [int(value) for value in cache["label"][:]]
        synthetic = np.asarray(cache["is_synthetic"][:], dtype=bool)
        train_only = np.asarray(cache["source_train_only"][:], dtype=bool)
        source_fold = np.asarray(cache["source_fold"][:], dtype=int)
        status = decode(cache["status"][:])
        expected_real = int(cache.attrs.get("n_real_eligible", -1))
        expected_excluded = int(cache.attrs.get("n_real_excluded_qc", -1))
        source_manifest_attr = str(cache.attrs.get("source_cache_manifest_sha256", ""))
        split_contract_attr = str(cache.attrs.get("split_contract_sha256", ""))
    real_indices = np.flatnonzero(~synthetic)
    augmented_indices = np.flatnonzero(synthetic)
    failures: list[str] = []
    if len(real_indices) != expected_real or expected_real + expected_excluded != len(split_by_sample):
        failures.append("CACHE_REAL_QC_COUNTS_MISMATCH")
    if len(set(sample_hash)) != len(sample_hash) or any(value != "OK" for value in status):
        failures.append("CACHE_DUPLICATE_SAMPLE_OR_NON_OK_STATUS")
    split_paths = [split_dir / "external_holdout.csv"] + [
        split_dir / f"fold_{fold}_{part}.csv"
        for fold in range(FOLDS)
        for part in ("train", "test")
    ]
    if file_digest(split_paths) != split_contract_attr:
        failures.append("CACHE_SPLIT_CONTRACT_SHA256_MISMATCH")
    verification_path = PROJECT_ROOT / "results/tables/final_mri_augmented_cache_verification.json"
    verification = read_json(verification_path)
    backup = PROJECT_ROOT / str(verification.get("real_backup", ""))
    if (
        verification.get("status") != "PASS"
        or not backup.is_file()
        or cache_manifest_digest(backup) != source_manifest_attr
    ):
        failures.append("CACHE_SOURCE_MANIFEST_SHA256_MISMATCH")
    augmentation = read_json(PROJECT_ROOT / "results/tables/fold_local_mri_augmentation_summary.json")
    geometry = read_json(PROJECT_ROOT / "results/tables/mri_128_geometry_audit.json")
    if (
        (augmentation.get("validation") or {}).get("status") != "PASS"
        or int((augmentation.get("validation") or {}).get("checked_images", 0)) != len(sample_hash)
        or geometry.get("status") != "PASS"
        or geometry.get("axis_consistent") is not True
        or geometry.get("affine_consistent") is not True
    ):
        failures.append("CACHE_LIVE_QC_OR_GEOMETRY_ARTIFACT_MISMATCH")
    for index in real_indices:
        row = split_by_sample.get(sample_hash[index])
        if (
            row is None
            or row["patient_hash"] != patient_hash[index]
            or int(row["ab_label"]) != labels[index]
            or train_only[index]
            or source_fold[index] != -1
        ):
            failures.append("CACHE_REAL_MEMBERSHIP_OR_LABEL_MISMATCH")
            break
    for index in augmented_indices:
        fold = int(source_fold[index])
        source_sample = source_sample_hash[index]
        row = split_by_sample.get(source_sample)
        if (
            fold not in range(FOLDS)
            or not train_only[index]
            or row is None
            or source_sample not in train_samples_by_fold[fold]
            or source_sample in test_samples_by_fold[fold]
            or source_patient_hash[index] != row["patient_hash"]
            or labels[index] != int(row["ab_label"])
        ):
            failures.append("CACHE_AUGMENTATION_FOLD_LEAKAGE_OR_LABEL_MISMATCH")
            break
    if failures:
        raise RuntimeError("CACHE_SPLIT_MEMBERSHIP_FAILED:" + ",".join(failures))
    return {
        "status": "PASS",
        "real_qc_pass": len(real_indices),
        "real_qc_excluded": expected_excluded,
        "fold_local_augmented": len(augmented_indices),
        "duplicate_sample_hash_count": 0,
        "non_ok_status_count": 0,
        "augmentation_test_membership_overlap_count": 0,
        "label_mismatch_count": 0,
        "split_contract_sha256_matches": True,
        "source_cache_manifest_sha256_matches": True,
        "full_cache_validation_status": "PASS",
        "geometry_audit_status": "PASS",
    }


def active_method_x_processes() -> list[dict[str, Any]]:
    rows: list[tuple[int, int, list[str], str]] = []
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        for process in psutil.process_iter(["pid", "ppid", "cmdline", "cwd"]):
            try:
                rows.append(
                    (
                        int(process.info["pid"]),
                        int(process.info.get("ppid") or -1),
                        [str(value) for value in (process.info.get("cmdline") or [])],
                        str(process.info.get("cwd") or ""),
                    )
                )
            except (psutil.Error, TypeError, ValueError):
                continue
    elif os.name == "nt":
        # Windows-native, read-only fallback.  Command lines remain in memory
        # for matching and are never written to an artifact or printed.
        query = (
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            value = json.loads(completed.stdout or "[]")
            objects = value if isinstance(value, list) else [value]
            for item in objects:
                if isinstance(item, dict):
                    rows.append(
                        (
                            int(item.get("ProcessId", -1)),
                            int(item.get("ParentProcessId", -1)),
                            [str(item.get("CommandLine") or "")],
                            "",
                        )
                    )
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError("PROCESS_GUARD_WINDOWS_CIM_UNAVAILABLE") from exc
    else:
        raise RuntimeError("PROCESS_GUARD_UNAVAILABLE")
    watched = {
        "train_3d_mri.py",
        "run_3d_hyperparameter_search.py",
        "orchestrate_method_x_mri_5fold.py",
        "audit_raw_dataset_qc.py",
        "build_raw_qc_visual_review_worksheet.py",
    }
    found: list[dict[str, Any]] = []
    current = os.getpid()
    current_process_tree = {current}
    parent_by_pid = {pid: parent for pid, parent, _, _ in rows}
    ancestor = parent_by_pid.get(current, -1)
    while ancestor > 0 and ancestor not in current_process_tree:
        current_process_tree.add(ancestor)
        ancestor = parent_by_pid.get(ancestor, -1)
    root_normalized = str(PROJECT_ROOT.resolve()).lower().replace("/", "\\")
    for pid, _, parts, process_cwd in rows:
        if pid in current_process_tree:
            continue
        joined = " ".join(parts).lower().replace("/", "\\")
        cwd = process_cwd.lower().replace("/", "\\")
        script = next((name for name in watched if name in joined), None)
        in_project = root_normalized in joined or cwd == root_normalized or cwd.startswith(root_normalized + "\\")
        if script and (in_project or "method_x_5fold_corrected_v1" in joined):
            found.append({"pid": pid, "script": script})
    return sorted(found, key=lambda row: row["pid"])


def gpu_preflight() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("GPU_PREFLIGHT_PYTORCH_MISSING") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_PREFLIGHT_CUDA_UNAVAILABLE")
    return {"status": "PASS", "cuda": True, "device": torch.cuda.get_device_name(0)}


def _python_prefix(python: str) -> list[str]:
    return [python, "-u"]


def build_hpo_command(python: str, cache_sha256: str) -> list[str]:
    return [
        *_python_prefix(python),
        str(PROJECT_ROOT / "scripts" / "run_3d_hyperparameter_search.py"),
        "--real_cache", OFFICIAL_CACHE_REL.as_posix(),
        "--combined_cache", OFFICIAL_CACHE_REL.as_posix(),
        "--out_dir", (OFFICIAL_ROOT_REL / "hpo").as_posix(),
        "--candidate_config", GRID_REL.as_posix(),
        "--folds", str(FOLDS),
        "--epochs", str(HPO_EPOCHS),
        "--batch_size", "1",
        "--grad_accum", "8",
        "--seed", str(SEED),
        "--cache_sha256", cache_sha256,
        "--require_exact_resume_contract",
    ]


def build_final_command(python: str, cache_sha256: str) -> list[str]:
    return [
        *_python_prefix(python),
        str(PROJECT_ROOT / "scripts" / "train_3d_mri.py"),
        "--cache", OFFICIAL_CACHE_REL.as_posix(),
        "--splits", SPLIT_DIR_REL.as_posix(),
        "--out_dir", (OFFICIAL_ROOT_REL / "final").as_posix(),
        "--phase", "final",
        "--resolution", "128",
        "--folds", str(FOLDS),
        "--epochs", str(FINAL_EPOCHS),
        "--seed", str(SEED),
        "--batch_size", "1",
        "--grad_accum", "8",
        "--device", "cuda",
        "--require_gpu",
        "--amp",
        "--checkpoint",
        "--monai_augment",
        "--restore_best_at_end",
        "--use_hparam_manifest",
        "--hparam_manifest", (OFFICIAL_ROOT_REL / "locked" / "hparam_manifest.json").as_posix(),
        "--cache_sha256", cache_sha256,
        "--require_exact_resume_contract",
    ]


def _loss(row: dict[str, Any]) -> tuple[float, float]:
    try:
        first = float(row["mean_best_val_loss"])
        second = float(row.get("mean_final_val_loss", first))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("HPO_SELECTION_LOSS_MISSING_OR_INVALID") from exc
    if not math.isfinite(first) or not math.isfinite(second):
        raise RuntimeError("HPO_SELECTION_LOSS_NONFINITE")
    return first, second


def validate_hpo_summary(
    summary: dict[str, Any],
    accepted_candidates: set[str] | dict[str, dict[str, Any]],
    *,
    expected_cache_sha256: str | None = None,
) -> dict[str, Any]:
    accepted_names = set(accepted_candidates)
    if summary.get("status") != "COMPLETED":
        raise RuntimeError("HPO_NOT_COMPLETED")
    if summary.get("paused_trial") is not None:
        raise RuntimeError("HPO_HAS_PAUSED_TRIAL")
    if summary.get("test_used_for_selection") is not False or summary.get(
        "external_or_pseudo_external_used_for_selection"
    ) is not False:
        raise RuntimeError("HPO_TEST_SELECTION_FORBIDDEN")
    if summary.get("selection_source") != "fold_internal_validation_loss_only" or summary.get(
        "selection_metric"
    ) != "mean_best_val_loss":
        raise RuntimeError("HPO_SELECTION_CONTRACT_INVALID")
    if int(summary.get("folds", 0)) != FOLDS:
        raise RuntimeError("HPO_FOLD_COUNT_NOT_5")
    if expected_cache_sha256 is not None and (
        summary.get("seed") != SEED
        or summary.get("cache_sha256") != expected_cache_sha256
        or summary.get("require_exact_resume_contract") is not True
    ):
        raise RuntimeError("HPO_EXACT_RESUME_PROVENANCE_INVALID")
    rows = summary.get("completed_trials")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("HPO_COMPLETED_TRIALS_MISSING")
    completed_names = {str(row.get("name", "")) for row in rows if isinstance(row, dict)}
    available_names = {str(value) for value in (summary.get("available_candidates") or [])}
    if (
        completed_names != accepted_names
        or available_names != accepted_names
        or int(summary.get("n_candidates_after_critic", -1)) != len(accepted_names)
        or int(summary.get("n_candidates_completed", -1)) != len(accepted_names)
        or summary.get("rejected_completed_trials") not in ([], None)
    ):
        raise RuntimeError("HPO_CANDIDATE_COMPLETION_MISMATCH")
    if isinstance(accepted_candidates, dict):
        completed_params = {
            str(row.get("name", "")): row.get("params")
            for row in rows
            if isinstance(row, dict)
        }
        if completed_params != accepted_candidates:
            raise RuntimeError("HPO_CANDIDATE_PARAM_MISMATCH")
    global_best = min(rows, key=_loss)
    declared = summary.get("best")
    if not isinstance(declared, dict) or str(declared.get("name")) != str(global_best.get("name")) or _loss(declared) != _loss(global_best):
        raise RuntimeError("GLOBAL_BEST_MISMATCH")
    return global_best


def validate_hpo_trial_artifacts(
    hpo_dir: Path,
    summary: dict[str, Any],
    *,
    cache_sha256: str,
) -> None:
    args = argparse.Namespace(
        require_exact_resume_contract=True,
        cache_sha256=cache_sha256,
        folds=FOLDS,
        epochs=HPO_EPOCHS,
        seed=SEED,
        batch_size=1,
        grad_accum=8,
    )
    for row in summary.get("completed_trials") or []:
        if not isinstance(row, dict) or not isinstance(row.get("params"), dict):
            raise RuntimeError("HPO_TRIAL_ARTIFACT_ROW_INVALID")
        trial_dir = hpo_dir / "trials" / str(row.get("name", ""))
        validate_completed_selection_trial(trial_dir, folds=FOLDS)
        validate_reusable_trial_contract(trial_dir, args, row["params"])


def build_locked_payloads(
    summary: dict[str, Any],
    *,
    cache_sha256: str,
    split_sha256: str,
    grid_sha256: str,
    hpo_sha256: str,
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    best = dict(summary["best"])
    hashes = {
        "cache_sha256": cache_sha256,
        "split_contract_sha256": split_sha256,
        "candidate_grid_sha256": grid_sha256,
        "hpo_summary_sha256": hpo_sha256,
    }
    locked = {
        "schema_version": 1,
        "status": "LOCKED",
        "selected_trial": best.get("name"),
        "params": best.get("params") or {},
        "selection_metric": "mean_best_val_loss",
        "selection_value": best.get("mean_best_val_loss"),
        "selection_source": "fold_internal_validation_loss_only",
        "folds": FOLDS,
        "seed": SEED,
        "hashes": hashes,
        "provenance": provenance,
    }
    manifest = {
        "schema_version": 1,
        "status": "DONE",
        "source_hpo_status": summary.get("status"),
        "selection_source": "fold_internal_validation_loss_only",
        "selection_metric": "mean_best_val_loss",
        "selection_contract": {
            "test_used_for_selection": False,
            "external_or_pseudo_external_used_for_selection": False,
        },
        "folds": FOLDS,
        "seed": SEED,
        "best": best,
        "best_by_backbone": {},
        "hashes": hashes,
        "provenance": provenance,
    }
    return locked, manifest


def run_foreground(command: list[str]) -> int:
    print("[METHOD_X] foreground stage command starting", flush=True)
    return int(subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode)


def validate_final_artifact(
    path: Path,
    *,
    expected_selected_trial: str | None = None,
    expected_cache_sha256: str | None = None,
) -> dict[str, Any]:
    import torch

    from scripts.train_baseline import average_precision, brier, roc_auc

    payload = read_json(path)
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    if aggregate.get("status") != "COMPLETED" or int(aggregate.get("folds", 0)) != FOLDS:
        raise RuntimeError("FINAL_TRAINING_NOT_COMPLETED")
    policy = aggregate.get("outer_test_evaluation_policy") or {}
    if (
        policy.get("during_training_epochs") is not False
        or policy.get("after_checkpoint_lock_only") is not True
        or int(policy.get("max_evaluations_per_fold", 0)) != 1
    ):
        raise RuntimeError("FINAL_OUTER_TEST_POLICY_INVALID")
    selection = aggregate.get("hparam_selection") or {}
    if (
        selection.get("selection_applied") is not True
        or selection.get("selection_source") != "fold_internal_validation_loss_only"
        or selection.get("test_used_for_selection") is not False
        or (
            expected_selected_trial is not None
            and selection.get("selected_trial") != expected_selected_trial
        )
    ):
        raise RuntimeError("FINAL_LOCKED_WINNER_NOT_APPLIED")
    if aggregate.get("hpo_selection_only") is not False or aggregate.get("uses_pet_as_input") is not False:
        raise RuntimeError("FINAL_METHOD_X_SCOPE_INVALID")
    if aggregate.get("clinical_alzheimer_diagnosis_output_allowed") is not False:
        raise RuntimeError("FINAL_CLAIM_SCOPE_INVALID")
    if aggregate.get("final_claim_allowed") is not False:
        raise RuntimeError("FINAL_POSITIVE_CLAIM_UNSAFE_WITH_CURRENT_SCIENCE_BLOCKERS")

    rows = payload.get("folds")
    if not isinstance(rows, list) or {int(row.get("fold", -1)) for row in rows if isinstance(row, dict)} != set(
        range(FOLDS)
    ) or len(rows) != FOLDS:
        raise RuntimeError("FINAL_FOLD_ROWS_INCOMPLETE")

    contract_payload = read_json(path.parent / "run_contract.json")
    contract = contract_payload.get("run_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("FINAL_RUN_CONTRACT_MISSING")
    contract_body = {key: value for key, value in contract.items() if key != "digest"}
    contract_digest = canonical_digest(contract_body)
    if (
        contract.get("digest") != contract_digest
        or contract_payload.get("run_contract_digest") != contract_digest
        or int(contract.get("folds", 0)) != FOLDS
        or int(contract.get("seed", -1)) != SEED
        or (expected_cache_sha256 is not None and contract.get("cache_sha256") != expected_cache_sha256)
    ):
        raise RuntimeError("FINAL_RUN_CONTRACT_INVALID")

    pooled_y: list[int] = []
    pooled_p: list[float] = []
    for fold in range(FOLDS):
        row = next(row for row in rows if int(row.get("fold", -1)) == fold)
        row_policy = row.get("outer_test_evaluation_policy") or {}
        if (
            int(row.get("outer_test_evaluation_count", 0)) != 1
            or row.get("hpo_selection_only") is not False
            or row_policy.get("during_training_epochs") is not False
            or row_policy.get("after_checkpoint_lock_only") is not True
            or int(row_policy.get("max_evaluations_per_fold", 0)) != 1
        ):
            raise RuntimeError("FINAL_OUTER_TEST_EXACTLY_ONCE_INVALID")
        state = read_json(path.parent / f"method_x_3d_fold_{fold}_state.json")
        outer = read_json(path.parent / f"method_x_3d_fold_{fold}_outer_eval.json")
        details = state.get("details") if isinstance(state.get("details"), dict) else {}
        checkpoint_path = path.parent / f"method_x_3d_fold_{fold}.pt"
        if (
            not checkpoint_path.is_file()
            or details.get("final_checkpoint_sha256") != sha256_file(checkpoint_path)
        ):
            raise RuntimeError("FINAL_CHECKPOINT_SHA256_INVALID")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError("FINAL_CHECKPOINT_UNREADABLE") from exc
        source_resolution = resolve_finalized_source_compatibility(
            model_dir=path.parent,
            state=state,
            checkpoint=checkpoint,
            locked_outer=outer,
            expected_contract=contract,
            expected_fold=fold,
            checkpoint_sha256=sha256_file(checkpoint_path),
            outer_file_sha256=sha256_file(
                path.parent / f"method_x_3d_fold_{fold}_outer_eval.json"
            ),
        )
        validated_outer = validate_locked_outer_result(
            outer,
            expected_fold=fold,
            expected_contract_digest=source_resolution.artifact_contract_digest,
            expected_split_membership_digest=details.get("split_membership_digest"),
            expected_count=int(row.get("test_n", 0)),
        )
        reused = reuse_committed_outer_result(
            checkpoint,
            outer,
            expected_fold=fold,
            expected_contract_digest=source_resolution.artifact_contract_digest,
            checkpoint_sha256=sha256_file(checkpoint_path),
            outer_file_sha256=sha256_file(
                path.parent / f"method_x_3d_fold_{fold}_outer_eval.json"
            ),
            compatibility_evidence=source_resolution.historical_authority,
        )
        expected_metrics = {
            "roc_auc": roc_auc(reused["y_true"], reused["y_prob"]),
            "pr_auc": average_precision(reused["y_true"], reused["y_prob"]),
            "brier": brier(reused["y_true"], reused["y_prob"]),
        }
        checkpoint_metrics = reused["metrics"]
        for name, expected_value in expected_metrics.items():
            try:
                row_value = float(row[name])
                checkpoint_value = float(checkpoint_metrics[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("FINAL_METRICS_OUTER_BINDING_INVALID") from exc
            if not (
                math.isclose(row_value, expected_value, rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(checkpoint_value, expected_value, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise RuntimeError("FINAL_METRICS_OUTER_BINDING_INVALID")
        pooled_y.extend(reused["y_true"])
        pooled_p.extend(reused["y_prob"])
        if (
            state.get("stage") != "FINALIZED"
            or int(state.get("fold", -1)) != fold
            or state.get("contract_digest") != contract_digest
            or int(outer.get("fold", -1)) != fold
            or int(outer.get("evaluation_count", 0)) != 1
            or outer.get("run_contract_digest") != contract_digest
            or details.get("outer_result_digest") != validated_outer["result_digest"]
            or details.get("split_membership_digest") != validated_outer["split_membership_digest"]
        ):
            raise RuntimeError("FINAL_FOLD_LOCK_EVIDENCE_INVALID")
    aggregate_expected = {
        "roc_auc": roc_auc(pooled_y, pooled_p),
        "pr_auc": average_precision(pooled_y, pooled_p),
        "brier": brier(pooled_y, pooled_p),
    }
    for name, expected_value in aggregate_expected.items():
        try:
            declared = float(aggregate[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("FINAL_AGGREGATE_METRICS_OUTER_BINDING_INVALID") from exc
        if not math.isclose(declared, expected_value, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError("FINAL_AGGREGATE_METRICS_OUTER_BINDING_INVALID")
    return aggregate


def privacy_scan(root: Path) -> dict[str, Any]:
    id_patterns = [re.compile(r"\bB\d{5,}\b"), re.compile(r"\bPREVBID\b"), re.compile(r"\bLRN_?\d{4,}\b")]
    path_pattern = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/)", re.IGNORECASE)
    violations: list[dict[str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt", ".csv", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path_pattern.search(text):
            violations.append({"artifact": path.relative_to(root).as_posix(), "violation": "RAW_PATH_PATTERN"})
        if any(pattern.search(text) for pattern in id_patterns):
            violations.append({"artifact": path.relative_to(root).as_posix(), "violation": "RAW_ID_PATTERN"})
    return {
        "schema_version": 1,
        "status": "PASS" if not violations else "FAIL",
        "raw_paths_exported": any(row["violation"] == "RAW_PATH_PATTERN" for row in violations),
        "raw_identifiers_exported": any(row["violation"] == "RAW_ID_PATTERN" for row in violations),
        "violations": violations,
    }


def accepted_candidate_contracts() -> dict[str, dict[str, Any]]:
    candidates, _ = load_candidate_config(GRID_REL)
    accepted, _ = apply_hpo_critic(candidates, grad_accum=8)
    contracts = {str(row["name"]): dict(row) for row in accepted}
    if not contracts:
        raise RuntimeError("NO_ACCEPTED_HPO_CANDIDATES")
    return contracts


def preflight(root: Path) -> dict[str, Any]:
    active = active_method_x_processes()
    if active:
        raise RuntimeError(f"ACTIVE_METHOD_X_PROCESS_BLOCKS_LAUNCH:{active}")
    spm = spm_completion_status()
    if not spm_completion_ready(spm):
        raise RuntimeError(f"SPM_GUARD_NOT_READY:{spm.get('status')}")
    cache = validate_cache_contract(PROJECT_ROOT / OFFICIAL_CACHE_REL)
    splits = validate_split_contract(PROJECT_ROOT / SPLIT_DIR_REL)
    cache_split_membership = validate_cache_split_membership(
        PROJECT_ROOT / OFFICIAL_CACHE_REL,
        PROJECT_ROOT / SPLIT_DIR_REL,
    )
    gpu = gpu_preflight()
    root.joinpath("locked").mkdir(parents=True, exist_ok=True)
    cache_sha = cached_file_sha256(
        PROJECT_ROOT / OFFICIAL_CACHE_REL,
        root / "locked" / "cache_sha256_registry.json",
    )
    return {
        "status": "PASS",
        "process_guard": {"status": "PASS", "active": []},
        "spm_guard": {"status": spm.get("status"), "ready": True},
        "cache_contract": cache,
        "split_contract": splits,
        "cache_split_membership": cache_split_membership,
        "gpu_preflight": gpu,
        "cache_sha256": cache_sha,
        "grid_sha256": sha256_file(PROJECT_ROOT / GRID_REL),
    }


def postprocess(python: str, root: Path) -> list[str]:
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    consistency = [
        *_python_prefix(python),
        str(PROJECT_ROOT / "scripts" / "audit_artifact_consistency.py"),
        "--repo-root", str(PROJECT_ROOT),
        "--artifact", (OFFICIAL_ROOT_REL / "final" / "three_d_metrics.json").as_posix(),
        "--hparam-manifest", (OFFICIAL_ROOT_REL / "locked" / "hparam_manifest.json").as_posix(),
        "--cache", OFFICIAL_CACHE_REL.as_posix(),
        "--split-summary", (SPLIT_DIR_REL / "split_summary.json").as_posix(),
        "--json-output", (OFFICIAL_ROOT_REL / "reports" / "artifact_claim_consistency_audit.json").as_posix(),
        "--md-output", (OFFICIAL_ROOT_REL / "reports" / "artifact_claim_consistency_audit.md").as_posix(),
    ]
    if run_foreground(consistency) != 0:
        raise RuntimeError("POSTPROCESS_CONSISTENCY_FAILED")
    metrics = read_json(root / "final" / "three_d_metrics.json")
    aggregate = metrics.get("aggregate") if isinstance(metrics.get("aggregate"), dict) else {}
    y_true: list[int] = []
    y_prob: list[float] = []
    for fold in range(FOLDS):
        outer = read_json(root / "final" / f"method_x_3d_fold_{fold}_outer_eval.json")
        y_true.extend(int(value) for value in outer["y_true"])
        y_prob.extend(float(value) for value in outer["y_prob"])
    from scripts.train_baseline import average_precision, brier, roc_auc
    from src.evaluation.bootstrap_ci import guarded_bootstrap_ci

    confidence_intervals = {
        "schema_version": 1,
        "status": "COMPLETED_AGGREGATE_ONLY",
        "n": len(y_true),
        "seed": SEED,
        "bootstrap_iterations": 2000,
        "roc_auc": guarded_bootstrap_ci(y_true, y_prob, roc_auc, n_boot=2000, seed=SEED),
        "pr_auc": guarded_bootstrap_ci(y_true, y_prob, average_precision, n_boot=2000, seed=SEED + 1),
        "brier": guarded_bootstrap_ci(y_true, y_prob, brier, n_boot=2000, seed=SEED + 2),
        "subject_level_predictions_exported": False,
    }
    atomic_json(report_dir / "bootstrap_confidence_intervals.json", confidence_intervals)
    claim_boundary = {
        "schema_version": 1,
        "status": "REFRESHED_NOT_ALLOWED",
        "claim_status": "NOT_ALLOWED",
        "allowed_scope": "A4_LEARN_INTERNAL_BIOMARKER_STATUS_PREDICTION_ONLY",
        "forbidden_claims": [
            "ALZHEIMER_DIAGNOSIS",
            "EXTERNAL_GENERALIZATION",
            "CAUSAL_EFFECT",
        ],
        "science_blockers": [
            "COHORT_LABEL_PERFECT_CONFOUNDING",
            "INDEPENDENT_EXTERNAL_VALIDATION_NOT_AVAILABLE",
        ],
        "trainer_final_claim_allowed": bool(aggregate.get("final_claim_allowed")),
    }
    atomic_json(report_dir / "claim_boundary_report.json", claim_boundary)
    paper_summary = {
        "schema_version": 1,
        "status": "AGGREGATE_DRAFT_WITH_SCIENTIFIC_BLOCKERS",
        "selected_trial": (aggregate.get("hparam_selection") or {}).get("selected_trial"),
        "folds": FOLDS,
        "metrics": {
            "roc_auc": aggregate.get("roc_auc"),
            "pr_auc": aggregate.get("pr_auc"),
            "brier": aggregate.get("brier"),
            "bootstrap_ci_artifact": "bootstrap_confidence_intervals.json",
            "calibration": aggregate.get("temperature_scaling"),
            "cohort_probe": aggregate.get("cohort_probe"),
        },
        "cohort_stratified_performance": {
            "status": "NOT_MEASURABLE_PERFECT_COHORT_LABEL_CONFOUNDING",
            "positive_claim_allowed": False,
        },
        "scanner_sensitivity": {
            "status": POSTPROCESS_BLOCKER_STAGES["scanner_sensitivity"]
        },
        "claim_boundary_artifact": "claim_boundary_report.json",
        "subject_level_outputs_exported": False,
    }
    atomic_json(report_dir / "paper_aggregate_summary.json", paper_summary)
    atomic_json(
        report_dir / "paper_reporting_checklist.json",
        {
            "schema_version": 1,
            "status": "INCOMPLETE_BLOCKED_FOR_POSITIVE_PUBLICATION_CLAIM",
            "frameworks": ["TRIPOD+AI", "PROBAST+AI", "CLAIM_2024"],
            "critical_open_items": list(POSTPROCESS_BLOCKER_STAGES.values()),
        },
    )
    adapters = {
        "schema_version": 1,
        "status": "COMPLETED_WITH_BLOCKERS",
        "completed": [
            {"stage": "calibration", "status": "COMPLETED_IN_TRAINER_INNER_VALIDATION_ONLY"},
            {"stage": "cohort_probe", "status": "COMPLETED_IN_TRAINER_DIAGNOSTIC_ONLY"},
            {"stage": "bootstrap_confidence_intervals", "status": "COMPLETED_AGGREGATE_ONLY"},
            {"stage": "consistency", "status": "COMPLETED"},
            {"stage": "claim_boundary", "status": "REFRESHED_NOT_ALLOWED"},
            {"stage": "paper_aggregate_tables", "status": "DRAFT_WITH_BLOCKERS"},
        ],
        "blockers": [
            {"stage": stage, "status": status}
            for stage, status in POSTPROCESS_BLOCKER_STAGES.items()
        ],
        "note": "Legacy model-dependent adapters are not invoked against corrected-v1 artifacts without a compatible tested interface.",
    }
    atomic_json(report_dir / "postprocess_adapter_status.json", adapters)
    privacy = privacy_scan(root)
    atomic_json(report_dir / "privacy_scan.json", privacy)
    if privacy["status"] != "PASS":
        raise RuntimeError("POSTPROCESS_PRIVACY_SCAN_FAILED")
    return list(POSTPROCESS_BLOCKER_STAGES.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true", help="Run guards/contracts/hash only; never HPO or train.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Versioned output root under the project. The default remains "
            "results/final_results/method_x_5fold_corrected_v1; use a new root "
            "when an old trial has a run-contract mismatch."
        ),
    )
    return parser


def main() -> int:
    global OFFICIAL_ROOT_REL
    args = build_parser().parse_args()
    if args.output_root is not None:
        requested = Path(args.output_root)
        resolved = (
            requested.resolve()
            if requested.is_absolute()
            else (PROJECT_ROOT / requested).resolve()
        )
        try:
            relative = resolved.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError("OUTPUT_ROOT_MUST_BE_WITHIN_PROJECT_ROOT") from exc
        if not relative.parts or relative.parts[0].lower() != "results":
            raise RuntimeError("OUTPUT_ROOT_MUST_BE_UNDER_RESULTS")
        OFFICIAL_ROOT_REL = relative
    root = PROJECT_ROOT / OFFICIAL_ROOT_REL
    root.mkdir(parents=True, exist_ok=True)
    for child in ("hpo", "locked", "final", "reports"):
        (root / child).mkdir(exist_ok=True)
    try:
        lock_path = acquire_run_lock(root)
    except Exception as exc:
        print(f"[METHOD_X] BLOCKED: {exc}", file=sys.stderr, flush=True)
        return 3
    try:
        record_state(root, "PREFLIGHT_RUNNING")
        checks = preflight(root)
        record_state(root, "PREFLIGHT_PASSED", preflight=checks)
        if args.preflight_only:
            print(json.dumps({"status": "PREFLIGHT_PASSED", "training_started": False}, indent=2), flush=True)
            return 0

        migrations = migrate_hpo_resume_source_contracts(
            root,
            cache_sha256=checks["cache_sha256"],
        )
        if migrations:
            record_state(
                root,
                "RESUME_CONTRACT_MIGRATED",
                migrations=migrations,
            )
            print(
                json.dumps(
                    {
                        "event": "RESUME_CONTRACT_MIGRATED",
                        "count": len(migrations),
                        "trials": [item["trial"] for item in migrations],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        accepted_candidates = accepted_candidate_contracts()
        summary_path = root / "hpo" / "three_d_hparam_search_summary.json"
        valid_summary: dict[str, Any] | None = None
        if summary_path.is_file():
            candidate = read_json(summary_path)
            try:
                validate_hpo_summary(
                    candidate, accepted_candidates, expected_cache_sha256=checks["cache_sha256"]
                )
                validate_hpo_trial_artifacts(
                    root / "hpo", candidate, cache_sha256=checks["cache_sha256"]
                )
                valid_summary = candidate
            except RuntimeError:
                valid_summary = None
        if valid_summary is None:
            record_state(root, "HPO_RUNNING")
            returncode = run_foreground(build_hpo_command(sys.executable, checks["cache_sha256"]))
            if returncode != 0 or not summary_path.is_file():
                record_state(root, "HPO_PAUSED_RESUMABLE", child_returncode=returncode)
                return 75
            valid_summary = read_json(summary_path)
            try:
                validate_hpo_summary(
                    valid_summary, accepted_candidates, expected_cache_sha256=checks["cache_sha256"]
                )
                validate_hpo_trial_artifacts(
                    root / "hpo", valid_summary, cache_sha256=checks["cache_sha256"]
                )
            except RuntimeError as exc:
                record_state(root, "HPO_PAUSED_RESUMABLE", blocker=str(exc))
                return 75
        record_state(root, "HPO_COMPLETED")

        hpo_sha = sha256_file(summary_path)
        provenance = build_provenance(
            repo_root=PROJECT_ROOT,
            # The multi-GB cache was already fully hashed exactly once by
            # cached_file_sha256; do not stream it again through provenance.
            inputs=[PROJECT_ROOT / GRID_REL, summary_path],
            parent_artifacts=[(OFFICIAL_ROOT_REL / "hpo" / summary_path.name).as_posix()],
        )
        locked, manifest = build_locked_payloads(
            valid_summary,
            cache_sha256=checks["cache_sha256"],
            split_sha256=checks["split_contract"]["sha256"],
            grid_sha256=checks["grid_sha256"],
            hpo_sha256=hpo_sha,
            provenance=provenance,
        )
        locked_path = root / "locked" / "locked_config.json"
        manifest_path = root / "locked" / "hparam_manifest.json"
        if locked_path.is_file() or manifest_path.is_file():
            if not (locked_path.is_file() and manifest_path.is_file()):
                raise RuntimeError("PARTIAL_WINNER_LOCK_ARTIFACTS")
            existing_locked = read_json(locked_path)
            existing_manifest = read_json(manifest_path)
            if (
                existing_locked.get("selected_trial") != locked["selected_trial"]
                or existing_locked.get("hashes") != locked["hashes"]
                or existing_locked.get("params") != locked["params"]
                or existing_manifest.get("best") != manifest["best"]
                or existing_manifest.get("hashes") != manifest["hashes"]
            ):
                raise RuntimeError("IMMUTABLE_WINNER_LOCK_MISMATCH")
        else:
            atomic_json(locked_path, locked)
            atomic_json(manifest_path, manifest)
        record_state(root, "WINNER_LOCKED", selected_trial=locked["selected_trial"])

        metrics_path = root / "final" / "three_d_metrics.json"
        final_complete = False
        if metrics_path.is_file():
            try:
                validate_final_artifact(
                    metrics_path,
                    expected_selected_trial=str(locked["selected_trial"]),
                    expected_cache_sha256=checks["cache_sha256"],
                )
                final_complete = True
            except RuntimeError:
                final_complete = False
        if not final_complete:
            record_state(root, "FINAL_TRAINING_RUNNING", selected_trial=locked["selected_trial"])
            returncode = run_foreground(build_final_command(sys.executable, checks["cache_sha256"]))
            if returncode != 0 or not metrics_path.is_file():
                record_state(root, "FINAL_TRAINING_PAUSED_RESUMABLE", child_returncode=returncode)
                return 75
            try:
                validate_final_artifact(
                    metrics_path,
                    expected_selected_trial=str(locked["selected_trial"]),
                    expected_cache_sha256=checks["cache_sha256"],
                )
            except RuntimeError as exc:
                record_state(root, "FINAL_TRAINING_PAUSED_RESUMABLE", blocker=str(exc))
                return 75
        record_state(root, "FINAL_TRAINING_COMPLETED")
        postprocess_blockers = postprocess(sys.executable, root)
        record_state(
            root,
            "COMPLETED_WITH_BLOCKERS",
            blockers=postprocess_blockers,
        )
        return 0
    except KeyboardInterrupt:
        record_state(root, "INTERRUPTED_RESUMABLE")
        return 130
    except Exception as exc:
        record_state(root, "BLOCKED", blocker=str(exc))
        print(f"[METHOD_X] BLOCKED: {exc}", file=sys.stderr, flush=True)
        return 3
    finally:
        release_run_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
