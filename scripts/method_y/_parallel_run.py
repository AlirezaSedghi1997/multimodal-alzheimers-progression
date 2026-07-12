from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import re
import shutil
import string
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_DIR = PROJECT_ROOT / "results" / "method_y_execution"
DELIVERY_MANIFEST = PROJECT_ROOT / "final_delivery" / "results" / "tables" / "delivery_manifest.json"
RUNS_DIRNAME = "_runs"
WRITER_LOCK_NAME = ".writer.lock"
PROCESS_GUARD_PATTERN = re.compile(
    r"method_y|pet_preprocess|train_pet_only|train_joint|train_distill|train_3d_mri|"
    r"run_3d|preprocess_pet|build_pet_volume_cache|spm|antsRegistration|matlab|python",
    re.IGNORECASE,
)
DEFAULT_METHOD_Y_PET_CACHE = "caches/method_y/pet_mni_128_broad_s1_fbp_only_hq_v1.h5"
DEFAULT_METHOD_Y_PET_INPUT_LIST = "data/method_y/pet_input_list_broad_s1_fbp_only.csv"
GPU_TRAINING_PATTERN = re.compile(
    r"train_pet_only|train_joint|train_distill|train_3d_mri|run_3d|run_method_y|"
    r"run_pet_guided_kd_train|run_3d_hyperparameter_search",
    re.IGNORECASE,
)
MAX_HASH_BYTES = 128 * 1024 * 1024
REQUIRED_STAGE_REPORT_KEYS = {
    "stage",
    "mode",
    "status",
    "acceptance_gate_passed",
    "started_at_utc",
    "ended_at_utc",
    "next_allowed_stage",
    "metrics",
}


class ParallelRunError(RuntimeError):
    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        super().__init__(code if detail is None else f"{code}:{detail}")


@dataclass(frozen=True)
class IsolatedRun:
    stage: str
    mode: str
    run_id: str
    run_dir: Path
    config_path: Path
    frozen_input_manifest: Path

    def public_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "run_id": self.run_id,
            "run_dir": _project_relative(self.run_dir),
            "frozen_input_manifest": _project_relative(self.frozen_input_manifest),
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _project_relative(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_write_text(path: str | Path, text: str) -> None:
    target = _project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(_json_safe(payload), indent=2, sort_keys=False, allow_nan=False))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_file(path: str | Path, *, max_hash_bytes: int = MAX_HASH_BYTES) -> dict[str, Any]:
    target = _project_path(path)
    entry: dict[str, Any] = {
        "path": _project_relative(target),
        "exists": target.exists(),
        "size_bytes": None,
        "mtime_ns": None,
        "sha256": None,
        "sha256_status": "MISSING",
    }
    if not target.exists():
        return entry
    stat = target.stat()
    entry["size_bytes"] = int(stat.st_size)
    entry["mtime_ns"] = int(stat.st_mtime_ns)
    if target.is_file() and stat.st_size <= int(max_hash_bytes):
        entry["sha256"] = _sha256_file(target)
        entry["sha256_status"] = "OK"
    elif target.is_file():
        entry["sha256_status"] = "SKIPPED_LARGE_FILE"
    else:
        entry["sha256_status"] = "SKIPPED_NOT_FILE"
    return entry


def _maybe_add_path(out: dict[str, str], key: str, value: Any) -> None:
    if value:
        out[key] = str(value)


def selected_input_paths(stage: str, cfg: dict[str, Any], config_path: str | Path) -> dict[str, str]:
    stage = stage.upper()
    inputs = cfg.get("inputs", {}) or {}
    folds = cfg.get("folds", {}) or {}
    teacher = cfg.get("teacher", {}) or {}
    paths: dict[str, str] = {}
    _maybe_add_path(paths, "config", config_path)
    if stage == "S0":
        _maybe_add_path(paths, "input_csv", inputs.get("input_csv"))
        _maybe_add_path(paths, "mni_template", inputs.get("mni_template"))
    if stage in {"S1", "S2", "S3"}:
        _maybe_add_path(paths, "pet_h5", inputs.get("pet_h5", DEFAULT_METHOD_Y_PET_CACHE))
        _maybe_add_path(paths, "pet_input_list", inputs.get("pet_input_list", DEFAULT_METHOD_Y_PET_INPUT_LIST))
        _maybe_add_path(paths, "fold_manifest", folds.get("manifest", "results/method_y_execution/fold_manifest.json"))
    if stage in {"S2", "S3"}:
        _maybe_add_path(paths, "mri_h5", inputs.get("mri_h5", "data/cached/archive/t1_volume_cache_128_real_20260712T203222Z_26fff22aa7de.h5"))
        _maybe_add_path(paths, "mri_ckpt", inputs.get("mri_ckpt", "models/method_x/best.pt"))
    if stage == "S2":
        pattern = str(inputs.get("s1_pet_ckpt_pattern", "models/method_y/pet_only_fold{fold}.pt"))
        for fold in range(int((folds.get("n_folds", 4) or 4))):
            _maybe_add_path(paths, f"s1_pet_ckpt_fold{fold}", pattern.format(fold=fold))
    if stage == "S3":
        _maybe_add_path(paths, "joint_ckpt", teacher.get("joint_ckpt", "models/method_y/joint_fold0.pt"))
        pattern = str(teacher.get("joint_ckpt_pattern", "models/method_y/joint_fold{fold}.pt"))
        for fold in range(int((folds.get("n_folds", 4) or 4))):
            _maybe_add_path(paths, f"joint_ckpt_fold{fold}", pattern.format(fold=fold))
    if stage == "S4":
        _maybe_add_path(paths, "joint_ckpt", inputs.get("joint_ckpt"))
        _maybe_add_path(paths, "distill_ckpt", inputs.get("distill_ckpt"))
        _maybe_add_path(paths, "validation_logits", inputs.get("validation_logits"))
    if stage == "S5":
        _maybe_add_path(paths, "calibration", inputs.get("calibration"))
        _maybe_add_path(paths, "validation_uncertainties", inputs.get("validation_uncertainties"))
        _maybe_add_path(paths, "validation_meta_features", inputs.get("validation_meta_features"))
    return paths


def write_frozen_input_manifest(
    *,
    stage: str,
    mode: str,
    cfg: dict[str, Any],
    config_path: str | Path,
    output_path: str | Path,
    extra_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    paths = selected_input_paths(stage, cfg, config_path)
    if extra_paths:
        paths.update({str(k): str(v) for k, v in extra_paths.items()})
    payload = {
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "stage": stage.upper(),
        "mode": mode,
        "project_root": _project_relative(PROJECT_ROOT),
        "hash_policy": {
            "max_hash_bytes": MAX_HASH_BYTES,
            "large_files": "size_mtime_only",
        },
        "inputs": {name: snapshot_file(path) for name, path in sorted(paths.items())},
    }
    atomic_write_json(output_path, payload)
    return payload


def verify_frozen_input_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(_project_path(manifest_path).read_text(encoding="utf-8"))
    changes: list[dict[str, Any]] = []
    for name, frozen in (manifest.get("inputs", {}) or {}).items():
        current = snapshot_file(frozen.get("path", ""))
        for field in ("exists", "size_bytes", "mtime_ns", "sha256"):
            if frozen.get(field) != current.get(field):
                changes.append(
                    {
                        "input": name,
                        "field": field,
                        "frozen": frozen.get(field),
                        "current": current.get(field),
                    }
                )
    return {
        "valid": not changes,
        "checked_at_utc": utc_now_iso(),
        "manifest": _project_relative(_project_path(manifest_path)),
        "changes": changes,
    }


def _short_id(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def create_isolated_run(
    *,
    stage: str,
    mode: str,
    cfg: dict[str, Any],
    config_path: str | Path,
    execution_dir: str | Path = EXECUTION_DIR,
    extra_paths: dict[str, str] | None = None,
) -> IsolatedRun:
    stage = stage.upper()
    base_dir = _project_path(execution_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stage.lower()}_{timestamp}_{_short_id()}"
    run_dir = base_dir / RUNS_DIRNAME / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = run_dir / "frozen_inputs.json"
    config_snapshot = run_dir / "config_snapshot.json"
    atomic_write_json(config_snapshot, cfg)
    write_frozen_input_manifest(
        stage=stage,
        mode=mode,
        cfg=cfg,
        config_path=config_path,
        output_path=frozen_path,
        extra_paths=extra_paths,
    )
    run = IsolatedRun(
        stage=stage,
        mode=mode,
        run_id=run_id,
        run_dir=run_dir,
        config_path=_project_path(config_path),
        frozen_input_manifest=frozen_path,
    )
    atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": utc_now_iso(),
            "stage": stage,
            "mode": mode,
            "run_id": run_id,
            "run_dir": _project_relative(run_dir),
            "config_path": _project_relative(_project_path(config_path)),
            "frozen_input_manifest": _project_relative(frozen_path),
        },
    )
    return run


def register_publish_artifact(
    run_dir: str | Path,
    *,
    source: str | Path,
    target: str | Path,
    required: bool = True,
    artifact_type: str | None = None,
) -> dict[str, Any]:
    run_path = _project_path(run_dir)
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.exists():
        raise ParallelRunError("REGISTER_PUBLISH_ARTIFACT_RUN_MANIFEST_MISSING", str(run_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = _project_path(source)
    target_path = _project_path(target)
    resolved_type = artifact_type or ("json" if target_path.suffix.lower() == ".json" else "text")
    artifact = {
        "source": _project_relative(source_path),
        "target": _project_relative(target_path),
        "required": bool(required),
        "type": resolved_type,
    }
    artifacts = [item for item in manifest.get("publish_artifacts", []) if item.get("target") != artifact["target"]]
    artifacts.append(artifact)
    manifest["publish_artifacts"] = artifacts
    atomic_write_json(manifest_path, manifest)
    return artifact


def isolated_config(cfg: dict[str, Any], run: IsolatedRun) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    try:
        frozen_inputs = json.loads(run.frozen_input_manifest.read_text(encoding="utf-8")).get("inputs", {})
    except Exception:
        frozen_inputs = {}

    def frozen_path(name: str) -> str | None:
        entry = frozen_inputs.get(name) or {}
        return str(entry.get("path")) if entry.get("path") else None

    out.setdefault("inputs", {})
    if frozen_path("pet_h5"):
        out["inputs"]["pet_h5"] = frozen_path("pet_h5")
    out["inputs"]["pet_input_list"] = frozen_path("pet_input_list") or "data/method_y/pet_input_list.csv"
    if frozen_path("mri_h5"):
        out["inputs"]["mri_h5"] = frozen_path("mri_h5")
    if frozen_path("mri_ckpt"):
        out["inputs"]["mri_ckpt"] = frozen_path("mri_ckpt")
    if "teacher" in out and frozen_path("joint_ckpt"):
        out.setdefault("teacher", {})["joint_ckpt"] = frozen_path("joint_ckpt")
    out.setdefault("outputs", {})
    out["outputs"]["plan_json"] = str(run.run_dir / f"{run.stage.lower()}_plan.json")
    out["outputs"]["validation_dir"] = str(run.run_dir / "validation_artifacts")
    out["outputs"]["checkpoint_dir"] = str(run.run_dir / "checkpoints")
    out.setdefault("folds", {})
    out["folds"]["manifest"] = str(run.run_dir / "fold_manifest.json")
    compute = out.setdefault("compute", {})
    compute["vram_budget_gb"] = 7.0
    compute["batch_size"] = 1
    compute["grad_accum"] = 4
    compute["amp"] = True
    compute["gradient_checkpointing"] = True
    compute["checkpointing"] = True
    compute["num_workers"] = 0
    if "memory_controls" in out:
        memory = out.setdefault("memory_controls", {})
        memory["batch_size"] = 1
        memory["grad_accum"] = 4
        memory["checkpointing"] = True
        memory["num_workers"] = 0
    out["parallel_execution"] = {
        "isolated_run": True,
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "frozen_input_manifest": str(run.frozen_input_manifest),
        "publish_required_for_official_state": True,
    }
    return out


def process_guard_snapshot(*, exclude_current: bool = True) -> list[dict[str, Any]]:
    if platform.system().lower().startswith("win"):
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process | "
                "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Depth 3"
            ),
        ]
        try:
            raw = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=15)
            payload = json.loads(raw or "[]")
        except Exception:
            return []
        rows = payload if isinstance(payload, list) else [payload]
    else:
        try:
            raw = subprocess.check_output(["ps", "-eo", "pid=,comm=,args="], text=True, timeout=15)
        except Exception:
            return []
        rows = []
        for line in raw.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) >= 2:
                rows.append({"ProcessId": parts[0], "Name": parts[1], "CommandLine": parts[2] if len(parts) > 2 else ""})
    current_pid = os.getpid()
    pid_to_parent: dict[int, int] = {}
    for row in rows:
        try:
            pid_to_parent[int(row.get("ProcessId") or row.get("PID") or 0)] = int(row.get("ParentProcessId") or 0)
        except Exception:
            continue
    excluded_pids = {current_pid}
    parent = pid_to_parent.get(current_pid)
    while parent and parent not in excluded_pids:
        excluded_pids.add(parent)
        parent = pid_to_parent.get(parent)
    out: list[dict[str, Any]] = []
    for row in rows:
        command_line = str(row.get("CommandLine") or "")
        try:
            pid = int(row.get("ProcessId") or row.get("PID") or 0)
        except Exception:
            pid = 0
        if exclude_current and pid in excluded_pids:
            continue
        if PROCESS_GUARD_PATTERN.search(command_line):
            out.append({"pid": pid, "name": row.get("Name"), "command_line": command_line})
    return out


def active_gpu_training_processes() -> list[dict[str, Any]]:
    return [row for row in process_guard_snapshot() if GPU_TRAINING_PATTERN.search(str(row.get("command_line") or ""))]


def assert_no_active_gpu_training() -> None:
    active = active_gpu_training_processes()
    if active:
        raise ParallelRunError("BLOCKED_BY_ACTIVE_GPU_TRAINING", str(active[0].get("pid")))


def assert_publish_process_guard_clear() -> None:
    active = process_guard_snapshot()
    if active:
        raise ParallelRunError("BLOCKED_BY_RUNNING_STAGE", str(active[0].get("pid")))


def capture_nvidia_smi_snapshot(run_dir: str | Path, label: str) -> dict[str, Any]:
    target = _project_path(run_dir) / f"nvidia_smi_{label}.json"
    payload: dict[str, Any] = {
        "captured_at_utc": utc_now_iso(),
        "label": label,
        "status": "NOT_RUN",
        "query": "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits",
    }
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        payload["status"] = "OK"
        payload["raw"] = raw.strip()
    except FileNotFoundError:
        payload["status"] = "NVIDIA_SMI_NOT_FOUND"
    except subprocess.CalledProcessError as exc:
        payload["status"] = "NVIDIA_SMI_FAILED"
        payload["raw"] = str(exc.output or "").strip()
    except Exception as exc:
        payload["status"] = "ERROR"
        payload["error"] = str(exc)
    atomic_write_json(target, payload)
    payload["path"] = _project_relative(target)
    return payload


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if platform.system().lower().startswith("win"):
        try:
            subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {int(pid)} -ErrorAction Stop | Out-Null"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return True
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _lock_event(execution_dir: Path, payload: dict[str, Any]) -> None:
    path = execution_dir / RUNS_DIRNAME / "writer_lock_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe({"at_utc": utc_now_iso(), **payload}), sort_keys=True) + "\n")


@contextmanager
def acquire_writer_lock(execution_dir: str | Path = EXECUTION_DIR) -> Iterator[None]:
    base_dir = _project_path(execution_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    lock_path = base_dir / WRITER_LOCK_NAME
    payload = {
        "pid": os.getpid(),
        "created_at_utc": utc_now_iso(),
        "host": platform.node(),
        "purpose": "method_y_atomic_publish",
    }
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        pid = int(existing.get("pid") or 0)
        if _process_alive(pid):
            raise ParallelRunError("BLOCKED_BY_ACTIVE_WRITER", str(pid))
        _lock_event(base_dir, {"event": "stale_lock_observed", "lock": existing})
        raise ParallelRunError("BLOCKED_BY_STALE_WRITER_LOCK", str(pid or "unknown"))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
        if int(current.get("pid") or -1) == os.getpid():
            lock_path.unlink(missing_ok=True)


def _validate_stage_report(payload: dict[str, Any], *, stage: str) -> None:
    missing = sorted(REQUIRED_STAGE_REPORT_KEYS - set(payload))
    if missing:
        raise ParallelRunError("PUBLISH_STAGE_REPORT_SCHEMA_INVALID", ",".join(missing))
    if str(payload.get("stage")).upper() != stage.upper():
        raise ParallelRunError("PUBLISH_STAGE_MISMATCH", str(payload.get("stage")))
    if payload.get("status") == "DONE" and payload.get("acceptance_gate_passed") is not True:
        raise ParallelRunError("PUBLISH_DONE_REQUIRES_ACCEPTANCE_GATE_PASS")
    if payload.get("acceptance_gate_passed") is True and payload.get("next_allowed_stage") == "STOP":
        raise ParallelRunError("PUBLISH_GATE_PASS_NEXT_STAGE_INVALID")
    isolated = payload.get("isolated_run")
    if not isinstance(isolated, dict) or not isolated.get("run_id") or not isolated.get("run_dir"):
        raise ParallelRunError("PUBLISH_ISOLATED_RUN_METADATA_MISSING")


def _stage_report_path(stage: str, execution_dir: Path) -> Path:
    return execution_dir / f"stage_{stage.lower()}_report.json"


def _stage_metrics_path(stage: str, execution_dir: Path) -> Path | None:
    return execution_dir / f"stage_{stage.lower()}_metrics.json" if stage.upper() in {"S1", "S2"} else None


def _delivery_stage_update(
    delivery_manifest: Path,
    *,
    stage: str,
    report: dict[str, Any],
    published_at_utc: str,
) -> None:
    if delivery_manifest.exists():
        try:
            manifest = json.loads(delivery_manifest.read_text(encoding="utf-8"))
        except Exception:
            manifest = {"bundle_root": "final_delivery", "manifest_parse_error": True}
    else:
        manifest = {"bundle_root": "final_delivery"}
    additions = manifest.setdefault("v3_method_y_additions", {})
    execution = additions.setdefault("execution_status", {})
    execution[stage.upper()] = {
        "mode": report.get("mode"),
        "status": report.get("status"),
        "blocker": report.get("blocker"),
        "updated_at_utc": report.get("ended_at_utc"),
        "isolated_run": report.get("isolated_run"),
        "published_at_utc": published_at_utc,
    }
    atomic_write_json(delivery_manifest, manifest)


def _publish_registered_artifacts(run_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    published: list[dict[str, Any]] = []
    for artifact in run_manifest.get("publish_artifacts", []) or []:
        source = _project_path(artifact.get("source", ""))
        target = _project_path(artifact.get("target", ""))
        required = bool(artifact.get("required", True))
        artifact_type = str(artifact.get("type") or "").lower()
        if not source.exists():
            if required:
                raise ParallelRunError("PUBLISH_REGISTERED_ARTIFACT_MISSING", str(source))
            continue
        if artifact_type == "json" or target.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            atomic_write_json(target, payload)
        elif artifact_type == "text" or target.suffix.lower() in {".md", ".txt"}:
            _atomic_write_text(target, source.read_text(encoding="utf-8"))
        elif artifact_type in {"binary", "file", "hdf5"}:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            shutil.copyfile(source, tmp)
            os.replace(tmp, target)
        else:
            raise ParallelRunError("PUBLISH_REGISTERED_ARTIFACT_TYPE_UNSUPPORTED", artifact_type or str(target))
        published.append({"source": _project_relative(source), "target": _project_relative(target)})
    return published


def _validate_registered_artifacts(run_manifest: dict[str, Any]) -> None:
    for artifact in run_manifest.get("publish_artifacts", []) or []:
        source = _project_path(artifact.get("source", ""))
        target = _project_path(artifact.get("target", ""))
        required = bool(artifact.get("required", True))
        artifact_type = str(artifact.get("type") or "").lower()
        if not source.exists():
            if required:
                raise ParallelRunError("PUBLISH_REGISTERED_ARTIFACT_MISSING", str(source))
            continue
        if artifact_type == "json" or target.suffix.lower() == ".json":
            json.loads(source.read_text(encoding="utf-8"))
        elif artifact_type == "text" or target.suffix.lower() in {".md", ".txt"}:
            source.read_text(encoding="utf-8")
        elif artifact_type in {"binary", "file", "hdf5"}:
            if not source.is_file():
                raise ParallelRunError("PUBLISH_REGISTERED_ARTIFACT_NOT_FILE", str(source))
        else:
            raise ParallelRunError("PUBLISH_REGISTERED_ARTIFACT_TYPE_UNSUPPORTED", artifact_type or str(target))


def _append_execution_log_atomic(execution_dir: Path, report: dict[str, Any]) -> None:
    path = execution_dir / "execution_log.md"
    header = (
        "| stage | mode | start_time_utc | end_time_utc | fold_level_metrics | acceptance_gate_verdict | next_allowed_stage |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else header
    metrics_text = json.dumps(report.get("metrics") or {}, sort_keys=True, allow_nan=False)
    line = (
        f"| {report.get('stage')} | {report.get('mode')} | {report.get('started_at_utc')} | "
        f"{report.get('ended_at_utc')} | `{metrics_text}` | {report.get('acceptance_gate_passed')} | "
        f"{report.get('next_allowed_stage')} |\n"
    )
    _atomic_write_text(path, existing.rstrip() + "\n" + line)


def publish_isolated_stage_run(
    run_dir: str | Path,
    *,
    stage: str | None = None,
    execution_dir: str | Path = EXECUTION_DIR,
    delivery_manifest: str | Path = DELIVERY_MANIFEST,
    require_process_guard_clear: bool = True,
) -> dict[str, Any]:
    if require_process_guard_clear:
        assert_publish_process_guard_clear()
    run_path = _project_path(run_dir)
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.exists():
        raise ParallelRunError("PUBLISH_RUN_MANIFEST_MISSING", str(run_path))
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_name = (stage or run_manifest.get("stage") or "").upper()
    if not stage_name:
        raise ParallelRunError("PUBLISH_STAGE_MISSING")
    report_path = run_path / f"stage_{stage_name.lower()}_report.json"
    if not report_path.exists():
        raise ParallelRunError("PUBLISH_STAGE_REPORT_MISSING", str(report_path))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_stage_report(report, stage=stage_name)
    frozen_path = run_path / "frozen_inputs.json"
    if not frozen_path.exists():
        raise ParallelRunError("PUBLISH_FROZEN_INPUT_MANIFEST_MISSING", str(frozen_path))
    verification = verify_frozen_input_manifest(frozen_path)
    atomic_write_json(run_path / "frozen_inputs_publish_check.json", verification)
    if not verification["valid"]:
        raise ParallelRunError("PUBLISH_FROZEN_INPUT_CHANGED")
    _validate_registered_artifacts(run_manifest)
    published_at = utc_now_iso()
    official_dir = _project_path(execution_dir)
    metrics_path = _stage_metrics_path(stage_name, official_dir)
    metrics_payload = {
        "stage": stage_name,
        "mode": report.get("mode"),
        "status": report.get("status"),
        "metrics": report.get("metrics") or {},
        "isolated_run": report.get("isolated_run"),
    }
    artifacts_published: list[dict[str, Any]] = []
    with acquire_writer_lock(official_dir):
        official_report = dict(report)
        official_report["published_from_isolated_run"] = True
        official_report["published_at_utc"] = published_at
        atomic_write_json(_stage_report_path(stage_name, official_dir), official_report)
        if metrics_path is not None:
            atomic_write_json(metrics_path, metrics_payload)
        _append_execution_log_atomic(official_dir, official_report)
        _delivery_stage_update(
            _project_path(delivery_manifest),
            stage=stage_name,
            report=official_report,
            published_at_utc=published_at,
        )
        artifacts_published = _publish_registered_artifacts(run_manifest)
    publish_result = {
        "status": "PUBLISHED",
        "stage": stage_name,
        "run_id": report.get("isolated_run", {}).get("run_id"),
        "published_at_utc": published_at,
        "official_report": _project_relative(_stage_report_path(stage_name, official_dir)),
        "artifacts_published": artifacts_published,
    }
    atomic_write_json(run_path / "publish_result.json", publish_result)
    return publish_result
