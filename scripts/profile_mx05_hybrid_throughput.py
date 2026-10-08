#!/usr/bin/env python3
"""Measure and lock the Hybrid-specific MX-05 execution contract.

The profile is disposable inner-only work: one fold, one epoch, seed 13 and a
fixed 64-row cap.  ``full_model`` is the prespecified worst-case Hybrid
variant.  It is measured before any ablation result is inspected, so loader or
checkpoint choices cannot become outcome-dependent.  Completed task receipts
are hash-bound and reused on an identical rerun.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.profile_mx04_throughput import operational_preflight  # noqa: E402
from scripts.run_method_x_finetune_orchestrator import (  # noqa: E402
    _exact_resume_storage_preflight,
)
from scripts.run_method_x_hybrid_orchestrator import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    EXPECTED_ABLATIONS,
    EXPECTED_FALLBACK,
    HYBRID_PROFILE_SOURCE_PATHS,
    _baseline_terminal,
    _canonical_sha,
    _protocol,
    _read,
    _sha,
    _trainer_flag_subset_blockers,
    _upstream_terminal,
    _validate_config,
    build_hybrid_command,
    validate_hybrid_smoke_report,
    validate_hybrid_throughput_profile,
)
from src.method_y_revival.strict_yaml import load_unique_yaml  # noqa: E402
from src.utils.io_helpers import atomic_write_json  # noqa: E402


PROFILE_VARIANT = "full_model"
PROFILE_FOLDS = 1
PROFILE_EPOCHS = 1
PROFILE_LIMIT = 64
PROFILE_SEED = 13
WORKER_CHOICES = (2, 4)
_STOP_REQUESTED = threading.Event()


def _request_stop(_signum: int, _frame: Any) -> None:
    _STOP_REQUESTED.set()


def _install_signal_handlers() -> None:
    _STOP_REQUESTED.clear()
    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, _request_stop)


def _raise_if_stopping() -> None:
    if _STOP_REQUESTED.is_set():
        raise RuntimeError("MX05_PROFILE_PAUSED_RESUMABLE:OPERATOR_STOP")


def _metric_vector(metrics: Mapping[str, Any]) -> dict[str, float]:
    rows = metrics.get("folds")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise RuntimeError("MX05_PROFILE_FOLD_METRICS_INVALID")
    result: dict[str, float] = {}
    for key in ("best_val_loss", "roc_auc", "brier"):
        try:
            value = float(rows[0][key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"MX05_PROFILE_METRIC_INVALID:{key}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"MX05_PROFILE_METRIC_NONFINITE:{key}")
        result[key] = value
    return result


def _metric_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if set(left) != set(right):
        return float("inf")
    return max(abs(float(left[key]) - float(right[key])) for key in left)


def _max_pairwise_metric_diff(rows: list[Mapping[str, Any]]) -> float:
    vectors = [row["metric_vector"] for row in rows]
    return max(
        (
            _metric_diff(vectors[left], vectors[right])
            for left in range(len(vectors))
            for right in range(left + 1, len(vectors))
        ),
        default=0.0,
    )


def _read_completed_metrics(path: Path) -> dict[str, Any]:
    metrics = _read(path)
    aggregate = metrics.get("aggregate")
    rows = metrics.get("folds")
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("status") != "COMPLETED"
        or aggregate.get("hpo_selection_only") is not True
        or aggregate.get("throughput_profile_only") is not True
        or aggregate.get("folds") != PROFILE_FOLDS
        or aggregate.get("completed_folds") != PROFILE_FOLDS
        or aggregate.get("outer_test_access_count") != 0
        or not isinstance(rows, list)
        or len(rows) != PROFILE_FOLDS
        or rows[0].get("outer_test_evaluation_count") != 0
    ):
        raise RuntimeError("MX05_PROFILE_METRICS_NOT_INNER_ONLY_COMPLETE")
    _metric_vector(metrics)
    return metrics


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in HYBRID_PROFILE_SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"MX05_PROFILE_SOURCE_MISSING:{relative}")
        hashes[relative] = _sha(path)
    return hashes


def _profile_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(protocol),
        "folds": PROFILE_FOLDS,
        "epochs": PROFILE_EPOCHS,
        "seed": PROFILE_SEED,
        "common": dict(protocol["common"]),
    }


def _execution_contract(
    config: Mapping[str, Any], *, workers: int, checkpoint: str
) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint,
        "loader": {
            "num_workers": int(workers),
            "persistent_workers": False,
            "prefetch_factor": 2,
            "pin_memory": True,
        },
        "exact_resume": dict(config["exact_resume"]),
        "yellow_gates": {
            "torch_compile": (
                "DROPPED:no Hybrid-specific determinism plus bitwise-resume receipt"
            ),
            "channels_last": (
                "DROPPED:no Hybrid-specific determinism plus bitwise-resume receipt"
            ),
        },
    }


def validate_reusable_task_receipt(
    receipt: Mapping[str, Any],
    *,
    metrics_path: Path,
    command_sha256: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if receipt.get("status") != "COMPLETED":
        raise RuntimeError("MX05_PROFILE_RECEIPT_NOT_COMPLETED")
    if receipt.get("command_sha256") != command_sha256:
        raise RuntimeError("MX05_PROFILE_RECEIPT_COMMAND_DRIFT")
    if receipt.get("source_sha256") != source_sha256:
        raise RuntimeError("MX05_PROFILE_RECEIPT_SOURCE_DRIFT")
    if not metrics_path.is_file() or receipt.get("metrics_sha256") != _sha(metrics_path):
        raise RuntimeError("MX05_PROFILE_RECEIPT_METRICS_DRIFT")
    metrics = _read_completed_metrics(metrics_path)
    if receipt.get("metric_vector") != _metric_vector(metrics):
        raise RuntimeError("MX05_PROFILE_RECEIPT_METRIC_VECTOR_DRIFT")
    return metrics


def _gpu_sample() -> tuple[float, float] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        line = (completed.stdout or "").strip().splitlines()[0]
        utilization, memory_mib = (
            float(value.strip()) for value in line.split(",")[:2]
        )
    except (IndexError, OSError, subprocess.SubprocessError, TypeError, ValueError):
        return None
    return utilization, memory_mib


def run_profile_task(
    *,
    profile_root: Path,
    task_name: str,
    checkpoint: str,
    workers: int,
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Run or reuse one exact-contract disposable Hybrid profile."""

    _raise_if_stopping()
    output_dir = profile_root / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    execution = _execution_contract(
        config, workers=int(workers), checkpoint=str(checkpoint)
    )
    command = build_hybrid_command(
        root=ROOT,
        output_dir=output_dir,
        variant=PROFILE_VARIANT,
        seed=PROFILE_SEED,
        selection_only=True,
        protocol=_profile_protocol(protocol),
        config=config,
        execution_contract=execution,
    )
    command[command.index("--phase") + 1] = "dev"
    command.extend(["--limit", str(PROFILE_LIMIT), "--throughput_profile_only"])
    if "--no_resume" in command:
        raise RuntimeError("MX05_PROFILE_NO_RESUME_FORBIDDEN")
    command_sha = _canonical_sha(command)
    receipt_path = output_dir / "profile_receipt.json"
    metrics_path = output_dir / "three_d_metrics.json"
    if receipt_path.is_file() and metrics_path.is_file():
        receipt = _read(receipt_path)
        try:
            metrics = validate_reusable_task_receipt(
                receipt,
                metrics_path=metrics_path,
                command_sha256=command_sha,
                source_sha256=source_sha256,
            )
        except RuntimeError:
            pass
        else:
            return {**receipt, "metrics": metrics, "reused": True}

    print(
        json.dumps(
            {
                "status": "PROFILE_TASK_START",
                "task": task_name,
                "variant": PROFILE_VARIANT,
                "checkpoint": checkpoint,
                "workers": workers,
            }
        ),
        flush=True,
    )
    samples: list[tuple[float, float]] = []
    sampler_stop = threading.Event()

    def sample_gpu() -> None:
        while not sampler_stop.is_set():
            sample = _gpu_sample()
            if sample is not None:
                samples.append(sample)
            sampler_stop.wait(1.0)

    thread = threading.Thread(target=sample_gpu, daemon=True)
    stdout_path = output_dir / "profile_stdout.log"
    stderr_path = output_dir / "profile_stderr.log"
    started = time.monotonic()
    thread.start()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            returncode = int(process.wait())
    finally:
        elapsed = time.monotonic() - started
        sampler_stop.set()
        thread.join(timeout=10.0)
    if returncode == 75 or _STOP_REQUESTED.is_set():
        raise RuntimeError(f"MX05_PROFILE_PAUSED_RESUMABLE:{task_name}")
    if returncode != 0:
        stderr_tail = ""
        try:
            stderr_tail = stderr_path.read_text(
                encoding="utf-8", errors="replace"
            )[-4000:]
        except OSError:
            pass
        return {
            "schema_version": 1,
            "status": "FAILED",
            "task": task_name,
            "variant": PROFILE_VARIANT,
            "checkpoint": checkpoint,
            "workers": int(workers),
            "returncode": returncode,
            "elapsed_s": float(elapsed),
            "command_sha256": command_sha,
            "source_sha256": dict(source_sha256),
            "stderr_tail": stderr_tail,
            "profile_only_never_selection_evidence": True,
        }

    metrics = _read_completed_metrics(metrics_path)
    row = metrics["folds"][0]
    try:
        epoch_s = float(row["average_epoch_seconds"])
        memory = (metrics.get("aggregate") or {})["memory_profile"]
        allocated = float(memory["cuda_max_memory_allocated_gb"])
        reserved = float(memory["cuda_max_memory_reserved_gb"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"MX05_PROFILE_RUNTIME_METRICS_INVALID:{task_name}") from exc
    if (
        not math.isfinite(epoch_s)
        or epoch_s <= 0
        or not math.isfinite(allocated)
        or not math.isfinite(reserved)
        or allocated < 0
        or reserved < allocated
    ):
        raise RuntimeError(f"MX05_PROFILE_RUNTIME_METRICS_INVALID:{task_name}")
    active_utilization = [value for value, memory_mib in samples if memory_mib >= 256]
    receipt = {
        "schema_version": 1,
        "status": "COMPLETED",
        "task": task_name,
        "variant": PROFILE_VARIANT,
        "checkpoint": checkpoint,
        "workers": int(workers),
        "command_sha256": command_sha,
        "source_sha256": dict(source_sha256),
        "metrics_sha256": _sha(metrics_path),
        "elapsed_s": float(elapsed),
        "epoch_wallclock_s": epoch_s,
        "sm_utilization_pct": (
            statistics.fmean(active_utilization) if active_utilization else 0.0
        ),
        "gpu_sample_count": len(active_utilization),
        "vram_allocated_peak_gb": allocated,
        "vram_reserved_peak_gb": reserved,
        "metric_vector": _metric_vector(metrics),
        "outer_test_access_count": 0,
        "profile_only_never_selection_evidence": True,
    }
    atomic_write_json(receipt_path, receipt)
    return {**receipt, "metrics": metrics, "reused": False}


def _validate_smoke(output_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    report = _read(output_root / "reports/amuse_x_hybrid_smoke.json")
    rows = report.get("variants")
    if (
        report.get("schema_version") != 2
        or report.get("status") != "PASS"
        or not isinstance(rows, list)
    ):
        raise RuntimeError("MX05_PROFILE_SMOKE_REPORT_INVALID")
    by_variant = {
        str(row.get("hybrid_variant")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    expected = {*EXPECTED_ABLATIONS, str(EXPECTED_FALLBACK["name"])}
    if set(by_variant) != expected:
        raise RuntimeError("MX05_PROFILE_SMOKE_COVERAGE_INVALID")
    for variant in sorted(expected):
        blockers = validate_hybrid_smoke_report(
            by_variant[variant],
            expected_variant=variant,
            parameter_budget_max=int(config["parameter_budget_max"]),
            vram_budget_gb=float(config["smoke_vram_budget_gb"]),
        )
        if blockers:
            raise RuntimeError(
                "MX05_PROFILE_SMOKE_EVIDENCE_INVALID:" + ",".join(blockers)
            )
    return report


def _existing_profile_reusable(
    *, artifact_path: Path, receipts_path: Path, config: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not artifact_path.is_file() or not receipts_path.is_file():
        return None
    try:
        profile = _read(artifact_path)
        receipts = _read(receipts_path)
        if validate_hybrid_throughput_profile(profile, config=config, root=ROOT):
            return None
        if (
            receipts.get("schema_version") != 1
            or receipts.get("status") != "COMPLETED"
            or receipts.get("profile_integrity_sha256")
            != profile.get("integrity_sha256")
            or receipts.get("source_sha256") != _source_hashes()
        ):
            return None
        tasks = receipts.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            return None
        for row in tasks:
            if not isinstance(row, Mapping) or row.get("status") != "COMPLETED":
                return None
            task_root = ROOT / str(receipts["profile_root"]) / str(row["task"])
            validate_reusable_task_receipt(
                row,
                metrics_path=task_root / "three_d_metrics.json",
                command_sha256=str(row["command_sha256"]),
                source_sha256=receipts["source_sha256"],
            )
        return profile
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


def build_profile(
    *, output_root: Path, config_path: Path, artifact_path: Path
) -> dict[str, Any]:
    loaded = load_unique_yaml(config_path)
    if not isinstance(loaded, Mapping):
        raise RuntimeError("MX05_PROFILE_CONFIG_INVALID")
    config = loaded
    configured = config.get("throughput_profile")
    if not isinstance(configured, Mapping):
        raise RuntimeError("MX05_PROFILE_CONFIG_MISSING")
    configured_artifact = (ROOT / str(configured.get("evidence_artifact") or "")).resolve()
    configured_receipts = (ROOT / str(configured.get("receipts_artifact") or "")).resolve()
    if artifact_path.resolve() != configured_artifact:
        raise RuntimeError("MX05_PROFILE_ARTIFACT_TARGET_MISMATCH")

    reusable = _existing_profile_reusable(
        artifact_path=artifact_path,
        receipts_path=configured_receipts,
        config=config,
    )
    if reusable is not None:
        return reusable

    blockers = _validate_config(config)
    blockers.extend(_trainer_flag_subset_blockers(root=ROOT))
    blockers.extend(_exact_resume_storage_preflight(config, root=ROOT))
    if not _baseline_terminal(output_root):
        blockers.append("HYBRID_BASELINE_BATTERY_NOT_FINALIZED")
    if not _upstream_terminal(output_root, "mx03_validity_orchestrator", {"COMPLETED"}):
        blockers.append("MX03_VALIDITY_NOT_COMPLETED")
    if not _upstream_terminal(output_root, "mx04_finetune_orchestrator", {"COMPLETED"}):
        blockers.append("MX04_FINETUNE_NOT_COMPLETED")
    operational, operational_blockers = operational_preflight()
    blockers.extend(
        str(value).replace("MX04_PROFILE", "MX05_PROFILE")
        for value in operational_blockers
    )
    try:
        _validate_smoke(output_root, config)
    except RuntimeError as exc:
        blockers.append(str(exc))
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda else None
    except (ImportError, RuntimeError):
        cuda, gpu_name = False, None
    if not cuda:
        blockers.append("MX05_PROFILE_CUDA_UNAVAILABLE")
    if blockers:
        raise RuntimeError(
            "MX05_PROFILE_PREFLIGHT_BLOCKED:" + ",".join(sorted(set(blockers)))
        )

    battery = _read(output_root / "battery_manifest.json")
    protocol = _protocol(battery)
    source_sha256 = _source_hashes()
    profile_id = _canonical_sha(
        {
            "source_sha256": source_sha256,
            "config_sha256": _sha(config_path),
            "gpu_name": gpu_name,
            "variant": PROFILE_VARIANT,
            "limit": PROFILE_LIMIT,
            "folds": PROFILE_FOLDS,
            "epochs": PROFILE_EPOCHS,
            "seed": PROFILE_SEED,
        }
    )[:16]
    profile_root = (
        ROOT / "results/method_x_registry/mx05_hybrid_throughput_smoke" / profile_id
    )
    trials = int(configured["loader_trials"])
    tolerance = float(configured["metric_equivalence_tolerance"])
    on_rows: dict[int, list[dict[str, Any]]] = {workers: [] for workers in WORKER_CHOICES}
    for workers in WORKER_CHOICES:
        for trial in range(1, trials + 1):
            row = run_profile_task(
                profile_root=profile_root,
                task_name=f"full_model_checkpoint_on_w{workers}_trial{trial}",
                checkpoint="on",
                workers=workers,
                protocol=protocol,
                config=config,
                source_sha256=source_sha256,
            )
            if row.get("status") != "COMPLETED":
                raise RuntimeError(
                    f"MX05_PROFILE_LOADER_TASK_FAILED:w{workers}:trial{trial}"
                )
            on_rows[workers].append(row)
    loader_diff = max(
        _max_pairwise_metric_diff(on_rows[2]),
        _max_pairwise_metric_diff(on_rows[4]),
        _metric_diff(on_rows[2][0]["metric_vector"], on_rows[4][0]["metric_vector"]),
    )
    if loader_diff > tolerance:
        raise RuntimeError(f"MX05_PROFILE_LOADER_METRIC_DRIFT:{loader_diff}")
    worker_medians = {
        workers: statistics.median(
            float(row["epoch_wallclock_s"]) for row in rows
        )
        for workers, rows in on_rows.items()
    }
    noise_s = max(
        *(
            max(float(row["epoch_wallclock_s"]) for row in rows)
            - min(float(row["epoch_wallclock_s"]) for row in rows)
            for rows in on_rows.values()
        ),
        0.03 * float(worker_medians[2]),
    )
    workers4_gain_s = float(worker_medians[2] - worker_medians[4])
    selected_workers = 4 if workers4_gain_s > noise_s else 2

    off_rows: list[dict[str, Any]] = []
    for trial in range(1, trials + 1):
        row = run_profile_task(
            profile_root=profile_root,
            task_name=f"full_model_checkpoint_off_w{selected_workers}_trial{trial}",
            checkpoint="off",
            workers=selected_workers,
            protocol=protocol,
            config=config,
            source_sha256=source_sha256,
        )
        if row.get("status") != "COMPLETED":
            off_rows = []
            break
        off_rows.append(row)
    selected_on = on_rows[selected_workers]
    cross_mode_diff = float("inf")
    off_gain_s = float("-inf")
    off_vram_safe = False
    off_equivalent = False
    off_materially_faster = False
    if len(off_rows) == trials:
        off_diff = _max_pairwise_metric_diff(off_rows)
        cross_mode_diff = max(
            off_diff,
            *(
                _metric_diff(off["metric_vector"], selected_on[0]["metric_vector"])
                for off in off_rows
            ),
        )
        off_equivalent = cross_mode_diff <= tolerance
        off_peak = max(
            max(
                float(row["vram_allocated_peak_gb"]),
                float(row["vram_reserved_peak_gb"]),
            )
            for row in off_rows
        )
        off_vram_safe = off_peak < float(configured["checkpoint_soft_cap_gb"])
        off_median = statistics.median(
            float(row["epoch_wallclock_s"]) for row in off_rows
        )
        on_median = statistics.median(
            float(row["epoch_wallclock_s"]) for row in selected_on
        )
        off_gain_s = float(on_median - off_median)
        off_materially_faster = off_gain_s > noise_s
    checkpoint = (
        "off"
        if off_rows and off_vram_safe and off_equivalent and off_materially_faster
        else "on"
    )
    selected_rows = off_rows if checkpoint == "off" else selected_on
    selected_allocated = max(float(row["vram_allocated_peak_gb"]) for row in selected_rows)
    selected_reserved = max(float(row["vram_reserved_peak_gb"]) for row in selected_rows)
    if selected_reserved >= float(configured["memory_hard_stop_gb"]):
        raise RuntimeError("MX05_PROFILE_SELECTED_VRAM_HARD_STOP")

    execution = _execution_contract(
        config, workers=selected_workers, checkpoint=checkpoint
    )
    profile: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETED",
        "artifact": "mx05_hybrid_throughput_profile",
        "family": "amuse_x_hybrid3d",
        "profile_variant": PROFILE_VARIANT,
        "profile_only_never_selection_evidence": True,
        "profile_id": profile_id,
        "profile_root": profile_root.relative_to(ROOT).as_posix(),
        "operational_preflight": operational,
        "source_binding": {
            "config_sha256": _sha(config_path),
            "source_sha256": source_sha256,
            "gpu_name": gpu_name,
        },
        "loader_decision": {
            "workers2_median_epoch_s": float(worker_medians[2]),
            "workers4_median_epoch_s": float(worker_medians[4]),
            "workers4_gain_s": workers4_gain_s,
            "observed_noise_s": float(noise_s),
            "selected_workers": selected_workers,
            "fallback_to_2_recorded": selected_workers == 2,
        },
        "checkpoint_decision": {
            "selected_setting": checkpoint,
            "off_trials_completed": len(off_rows),
            "off_under_7gb_soft_cap": off_vram_safe,
            "off_equivalent": off_equivalent,
            "off_materially_faster": off_materially_faster,
            "off_gain_s": off_gain_s if math.isfinite(off_gain_s) else None,
            "cross_mode_max_abs_metric_diff": (
                cross_mode_diff if math.isfinite(cross_mode_diff) else None
            ),
            "fail_closed_default": "on",
        },
        "metric_equivalence": {
            "max_abs_diff": float(loader_diff),
            "tolerance": tolerance,
            "loader_trial_equivalence": "PASS",
        },
        "selected_memory": {
            "allocated_peak_gb": selected_allocated,
            "reserved_peak_gb": selected_reserved,
            "hard_stop_gb": float(configured["memory_hard_stop_gb"]),
        },
        "execution_contract": execution,
        "yellow_gates": dict(execution["yellow_gates"]),
        "hybrid_application": {
            "family": "amuse_x_hybrid3d",
            "profile_variant": PROFILE_VARIANT,
            "uniform_across_all_seven_ablations": True,
            "execution_contract": execution,
        },
    }
    profile["integrity_sha256"] = _canonical_sha(profile)
    blockers = validate_hybrid_throughput_profile(profile, config=config, root=ROOT)
    if blockers:
        raise RuntimeError("MX05_PROFILE_SELF_VALIDATION_FAILED:" + ",".join(blockers))
    task_rows = [
        {key: value for key, value in row.items() if key not in {"metrics", "reused"}}
        for workers in WORKER_CHOICES
        for row in on_rows[workers]
    ]
    task_rows.extend(
        {key: value for key, value in row.items() if key not in {"metrics", "reused"}}
        for row in off_rows
    )
    receipts = {
        "schema_version": 1,
        "status": "COMPLETED",
        "profile_id": profile_id,
        "profile_root": profile_root.relative_to(ROOT).as_posix(),
        "profile_integrity_sha256": profile["integrity_sha256"],
        "source_sha256": source_sha256,
        "tasks": task_rows,
    }
    atomic_write_json(artifact_path, profile)
    atomic_write_json(configured_receipts, receipts)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    loaded = load_unique_yaml(config_path)
    configured = loaded.get("throughput_profile") if isinstance(loaded, Mapping) else {}
    default_artifact = ROOT / str((configured or {}).get("evidence_artifact") or "")
    artifact_path = args.artifact or default_artifact
    artifact_path = artifact_path if artifact_path.is_absolute() else ROOT / artifact_path
    _install_signal_handlers()
    try:
        profile = build_profile(
            output_root=output_root.resolve(),
            config_path=config_path.resolve(),
            artifact_path=artifact_path.resolve(),
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
    print(json.dumps({"status": "COMPLETED", "profile": profile}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
