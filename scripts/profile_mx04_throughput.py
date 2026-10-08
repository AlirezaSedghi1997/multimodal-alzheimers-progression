"""Measure and lock the MX-04 throughput contract before selection starts.

All child runs are one-fold, one-epoch, inner-only disposable profiles.  They
never become selection evidence.  The resulting plan-041 artifact is consumed
fail-closed by ``run_method_x_finetune_orchestrator.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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

from scripts.run_method_x_finetune_orchestrator import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    EXPECTED_FAMILIES,
    _baseline_terminal,
    _exact_resume_storage_preflight,
    _mx03_terminal,
    _protocol_args,
    _read,
    _sha,
    _trainer_flag_subset_blockers,
    _trainer_uses_legacy_amp,
    _validate_config,
    build_candidate_command,
    validate_throughput_profile,
)
from src.method_y_revival.strict_yaml import load_unique_yaml  # noqa: E402
from src.utils.io_helpers import atomic_write_json  # noqa: E402


PROFILE_LIMIT = 64
PROFILE_FOLDS = 1
PROFILE_EPOCHS = 1
PROFILE_SEED = 13
LOADER_TRIALS = 2
METRIC_EQUIVALENCE_TOLERANCE = 1e-6
CHECKPOINT_SOFT_CAP_GB = 7.0
VRAM_HARD_STOP_GB = 7.5
_PROFILE_STOP_REQUESTED = threading.Event()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _request_profile_stop(_signum: int, _frame: Any) -> None:
    """Record an operator stop without interrupting a trainer checkpoint write."""

    _PROFILE_STOP_REQUESTED.set()


def _install_profile_signal_handlers() -> None:
    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, _request_profile_stop)


def _raise_if_profile_stop_requested() -> None:
    if _PROFILE_STOP_REQUESTED.is_set():
        raise RuntimeError("MX04_PROFILE_PAUSED_RESUMABLE:OPERATOR_STOP")


def _run_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or "") + (completed.stderr or "")


def operational_preflight() -> tuple[dict[str, str], list[str]]:
    """Read-only Windows run-survival checks from plan 041 Step 1b."""

    blockers: list[str] = []
    power_rows: list[int] = []
    for alias in ("STANDBYIDLE", "HIBERNATEIDLE"):
        text = _run_text(["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", alias])
        power_rows.extend(
            int(value, 16)
            for value in re.findall(r"Current (?:AC|DC) Power Setting Index:\s*0x([0-9a-fA-F]+)", text)
        )
    sleep_ok = len(power_rows) == 4 and all(value == 0 or value >= 0x7FFFFFFF for value in power_rows)
    if not sleep_ok:
        blockers.append("MX04_PROFILE_SLEEP_OR_HIBERNATE_ENABLED")

    battery_text = _run_text(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "$b=Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue; if($null -eq $b){'NONE'}else{$b.BatteryStatus}",
        ]
    ).strip()
    if battery_text == "NONE" or not battery_text:
        stable_power = "N/A:no battery device; mains-only host"
    else:
        try:
            battery_status = int(battery_text.splitlines()[-1].strip())
        except (TypeError, ValueError):
            battery_status = -1
        if battery_status in {2, 3, 6, 7, 8, 9, 11}:
            stable_power = "PASS"
        else:
            stable_power = f"FAIL:battery_status_{battery_status}"
            blockers.append("MX04_PROFILE_STABLE_POWER_NOT_CONFIRMED")

    trainer_text = (ROOT / "scripts/train_3d_mri.py").read_text(encoding="utf-8")
    signal_ok = all(
        marker in trainer_text
        for marker in (
            "_install_training_signal_handlers",
            "SafeOptimizerBoundaryStop",
            "return 75",
        )
    )
    if not signal_ok:
        blockers.append("MX04_PROFILE_SIGNAL_HANDLING_NOT_INTACT")
    return (
        {
            "sleep_hibernate_off": "PASS" if sleep_ok else "FAIL:powercfg timeout is active",
            "cooling_secured": "N/A:physical cooling cannot be measured programmatically; keep vents clear",
            "stable_power": stable_power,
            "signal_handling_intact_no_no_resume": "PASS" if signal_ok else "FAIL",
        },
        blockers,
    )


def _gpu_sample() -> tuple[float, float] | None:
    text = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    if not text:
        return None
    try:
        utilization, memory_mib = (float(value.strip()) for value in text.splitlines()[0].split(",")[:2])
    except (TypeError, ValueError):
        return None
    return utilization, memory_mib


def _metric_vector(metrics: Mapping[str, Any]) -> dict[str, float]:
    rows = metrics.get("folds")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise RuntimeError("MX04_PROFILE_FOLD_METRICS_INVALID")
    row = rows[0]
    result: dict[str, float] = {}
    for key in ("best_val_loss", "roc_auc", "brier"):
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"MX04_PROFILE_METRIC_INVALID:{key}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"MX04_PROFILE_METRIC_NONFINITE:{key}")
        result[key] = value
    return result


def _metric_diff(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if set(left) != set(right):
        return float("inf")
    return max(abs(float(left[key]) - float(right[key])) for key in left)


def _read_completed_metrics(path: Path) -> dict[str, Any]:
    metrics = _read(path)
    aggregate = metrics.get("aggregate")
    rows = metrics.get("folds")
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("status") != "COMPLETED"
        or aggregate.get("hpo_selection_only") is not True
        or aggregate.get("throughput_profile_only") is not True
        or aggregate.get("folds") != 1
        or aggregate.get("completed_folds") != 1
        or aggregate.get("outer_test_access_count", 0) != 0
        or not isinstance(rows, list)
        or len(rows) != 1
        or rows[0].get("outer_test_evaluation_count") != 0
    ):
        raise RuntimeError("MX04_PROFILE_METRICS_NOT_INNER_ONLY_COMPLETE")
    _metric_vector(metrics)
    return metrics


def validate_reusable_profile_receipt(
    receipt: Mapping[str, Any],
    *,
    metrics_path: Path,
    command_sha256: str,
    trainer_sha256: str,
    profiler_sha256: str,
) -> dict[str, Any]:
    """Revalidate a cached profile receipt against live code and metric bytes."""

    if receipt.get("status") != "COMPLETED":
        raise RuntimeError("MX04_PROFILE_RECEIPT_NOT_COMPLETED")
    if receipt.get("command_sha256") != command_sha256:
        raise RuntimeError("MX04_PROFILE_RECEIPT_COMMAND_MISMATCH")
    if receipt.get("trainer_sha256") != trainer_sha256:
        raise RuntimeError("MX04_PROFILE_RECEIPT_TRAINER_MISMATCH")
    if receipt.get("profiler_sha256") != profiler_sha256:
        raise RuntimeError("MX04_PROFILE_RECEIPT_PROFILER_MISMATCH")
    if not metrics_path.is_file() or receipt.get("metrics_sha256") != _sha(metrics_path):
        raise RuntimeError("MX04_PROFILE_RECEIPT_METRICS_HASH_MISMATCH")
    metrics = _read_completed_metrics(metrics_path)
    stored_vector = receipt.get("metric_vector")
    if not isinstance(stored_vector, Mapping) or _metric_diff(
        stored_vector, _metric_vector(metrics)
    ) != 0.0:
        raise RuntimeError("MX04_PROFILE_RECEIPT_METRIC_VECTOR_MISMATCH")
    return metrics


def _profile_execution_contract(
    config: Mapping[str, Any],
    *,
    checkpoint: str,
    workers: int,
) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint,
        "loader": {
            "num_workers": int(workers),
            "pin_memory": True,
            "persistent_workers": False,
            "prefetch_factor": 2,
        },
        "exact_resume": dict(config["exact_resume"]),
        "yellow_gates": {
            "torch_compile": "DROPPED:gate not yet proven",
            "channels_last": "DROPPED:gate not yet proven",
        },
    }


def _profile_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(protocol),
        "folds": PROFILE_FOLDS,
        "epochs": PROFILE_EPOCHS,
        "seed": PROFILE_SEED,
        "common": dict(protocol["common"]),
    }


def run_profile_task(
    *,
    profile_root: Path,
    variant: str,
    family: str,
    checkpoint: str,
    workers: int,
    candidate: Mapping[str, Any],
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run or hash-verified reuse of one disposable, resume-safe profile task."""

    _raise_if_profile_stop_requested()
    output_dir = profile_root / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_contract = _profile_execution_contract(
        config,
        checkpoint=checkpoint,
        workers=workers,
    )
    command = build_candidate_command(
        output_root=output_dir,
        family=family,
        candidate=candidate,
        seed=PROFILE_SEED,
        selection_only=True,
        fold_manifest=None,
        protocol=_profile_protocol(protocol),
        execution_contract=execution_contract,
    )
    # Profile runs are intentionally one-epoch smoke measurements.  The
    # trainer's final-phase minimum-epoch guard protects claim-grade runs and
    # must remain intact, so disposable profiles use the dev phase while still
    # exercising the same 128^3 CUDA/model/loader/staging path.
    command[command.index("--phase") + 1] = "dev"
    command.extend(["--limit", str(PROFILE_LIMIT), "--throughput_profile_only"])
    if "--no_resume" in command:
        raise RuntimeError("MX04_PROFILE_NO_RESUME_FORBIDDEN")
    command_sha = _canonical_sha(command)
    trainer_sha = _sha(ROOT / "scripts/train_3d_mri.py")
    profiler_sha = _sha(Path(__file__).resolve())
    receipt_path = output_dir / "profile_receipt.json"
    metrics_path = output_dir / "three_d_metrics.json"
    if receipt_path.is_file() and metrics_path.is_file():
        receipt = _read(receipt_path)
        try:
            metrics = validate_reusable_profile_receipt(
                receipt,
                metrics_path=metrics_path,
                command_sha256=command_sha,
                trainer_sha256=trainer_sha,
                profiler_sha256=profiler_sha,
            )
        except RuntimeError:
            pass
        else:
            return {**receipt, "metrics": metrics, "reused": True}

    print(json.dumps({"status": "PROFILE_TASK_START", "variant": variant, "family": family, "checkpoint": checkpoint, "workers": workers}), flush=True)
    samples: list[tuple[float, float]] = []
    stop_sampling = threading.Event()

    def sampler() -> None:
        while not stop_sampling.is_set():
            sample = _gpu_sample()
            if sample is not None:
                samples.append(sample)
            stop_sampling.wait(1.0)

    sample_thread = threading.Thread(target=sampler, name=f"gpu-sampler-{variant}", daemon=True)
    stdout_path = output_dir / "profile_stdout.log"
    stderr_path = output_dir / "profile_stderr.log"
    started = time.monotonic()
    sample_thread.start()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
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
        stop_sampling.set()
        sample_thread.join(timeout=10.0)

    if returncode == 75:
        raise RuntimeError(f"MX04_PROFILE_PAUSED_RESUMABLE:{variant}")
    if _PROFILE_STOP_REQUESTED.is_set() and returncode != 0:
        raise RuntimeError(f"MX04_PROFILE_PAUSED_RESUMABLE:{variant}")
    if returncode != 0:
        stderr_tail = ""
        try:
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            pass
        return {
            "status": "FAILED",
            "variant": variant,
            "family": family,
            "checkpoint": checkpoint,
            "workers": workers,
            "returncode": returncode,
            "stderr_tail": stderr_tail,
            "elapsed_s": elapsed,
            "command_sha256": command_sha,
        }

    metrics = _read_completed_metrics(metrics_path)
    fold_row = metrics["folds"][0]
    try:
        epoch_wallclock_s = float(fold_row["average_epoch_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"MX04_PROFILE_EPOCH_TIMING_INVALID:{variant}") from exc
    if not math.isfinite(epoch_wallclock_s) or epoch_wallclock_s <= 0.0:
        raise RuntimeError(f"MX04_PROFILE_EPOCH_TIMING_INVALID:{variant}")
    memory = (metrics.get("aggregate") or {}).get("memory_profile") or {}
    allocated = float(memory.get("cuda_max_memory_allocated_gb"))
    reserved = float(memory.get("cuda_max_memory_reserved_gb"))
    if not all(math.isfinite(value) and value >= 0.0 for value in (allocated, reserved)):
        raise RuntimeError(f"MX04_PROFILE_VRAM_INVALID:{variant}")
    active_utilization = [util for util, memory_mib in samples if memory_mib >= 256.0]
    utilization = statistics.fmean(active_utilization) if active_utilization else 0.0
    receipt = {
        "schema_version": 1,
        "status": "COMPLETED",
        "variant": variant,
        "family": family,
        "checkpoint": checkpoint,
        "workers": workers,
        "command_sha256": command_sha,
        "trainer_sha256": trainer_sha,
        "profiler_sha256": profiler_sha,
        "metrics_sha256": _sha(metrics_path),
        "elapsed_s": float(elapsed),
        "epoch_wallclock_s": epoch_wallclock_s,
        "sm_utilization_pct": float(utilization),
        "gpu_sample_count": len(active_utilization),
        "vram_allocated_peak_gb": allocated,
        "vram_reserved_peak_gb": reserved,
        "metric_vector": _metric_vector(metrics),
        "outer_test_access_count": 0,
        "profile_only_never_selection_evidence": True,
    }
    atomic_write_json(receipt_path, receipt)
    print(json.dumps({"status": "PROFILE_TASK_COMPLETED", "variant": variant, "elapsed_s": elapsed, "utilization": utilization, "allocated_gb": allocated, "reserved_gb": reserved}), flush=True)
    return {**receipt, "metrics": metrics, "reused": False}


def _median_elapsed(rows: list[Mapping[str, Any]]) -> float:
    return float(statistics.median(float(row["epoch_wallclock_s"]) for row in rows))


def _max_pairwise_metric_diff(rows: list[Mapping[str, Any]]) -> float:
    vectors = [row["metric_vector"] for row in rows]
    return max(
        (_metric_diff(vectors[left], vectors[right]) for left in range(len(vectors)) for right in range(left + 1, len(vectors))),
        default=0.0,
    )


def build_profile(
    *,
    output_root: Path,
    config_path: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    config = load_unique_yaml(config_path)
    battery = _read(output_root / "battery_manifest.json")
    protocol = _protocol_args(battery)
    blockers = [
        blocker
        for blocker in _validate_config(config, root=ROOT, require_throughput=False)
        if blocker != "FINETUNE_RUNTIME_MODERNIZATION_EVIDENCE_MISSING_OR_INVALID"
    ]
    blockers.extend(_trainer_flag_subset_blockers(root=ROOT))
    blockers.extend(_exact_resume_storage_preflight(config, root=ROOT))
    if not _baseline_terminal(battery):
        blockers.append("BASELINE_BATTERY_NOT_FINALIZED")
    if not _mx03_terminal(output_root):
        blockers.append("MX03_VALIDITY_NOT_COMPLETED")
    if _trainer_uses_legacy_amp(ROOT / "scripts/train_3d_mri.py"):
        blockers.append("FINETUNE_RUNTIME_AMP_MIGRATION_MISSING")
    operational, operational_blockers = operational_preflight()
    blockers.extend(operational_blockers)
    try:
        import torch

        if not torch.cuda.is_available():
            blockers.append("MX04_PROFILE_CUDA_UNAVAILABLE")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except (ImportError, RuntimeError):
        gpu_name = None
        blockers.append("MX04_PROFILE_CUDA_UNAVAILABLE")
    if blockers:
        raise RuntimeError("MX04_PROFILE_PREFLIGHT_BLOCKED:" + ",".join(sorted(set(blockers))))

    trainer_sha = _sha(ROOT / "scripts/train_3d_mri.py")
    script_sha = _sha(Path(__file__).resolve())
    profile_id = _canonical_sha(
        {
            "trainer_sha256": trainer_sha,
            "profiler_sha256": script_sha,
            "limit": PROFILE_LIMIT,
            "folds": PROFILE_FOLDS,
            "epochs": PROFILE_EPOCHS,
            "seed": PROFILE_SEED,
            "gpu": gpu_name,
        }
    )[:16]
    profile_root = ROOT / "results/method_x_registry/mx04_throughput_smoke" / profile_id
    scratch = next(
        row for row in config["candidates"] if isinstance(row, Mapping) and row.get("id") == "scratch_baseline"
    )

    loader_rows: dict[int, list[dict[str, Any]]] = {2: [], 4: []}
    reference_family = "resnet34_3d"
    for workers in (2, 4):
        for trial in range(1, LOADER_TRIALS + 1):
            row = run_profile_task(
                profile_root=profile_root,
                variant=f"loader_w{workers}_trial{trial}",
                family=reference_family,
                checkpoint="on",
                workers=workers,
                candidate=scratch,
                protocol=protocol,
                config=config,
            )
            if row.get("status") != "COMPLETED":
                raise RuntimeError(f"MX04_PROFILE_LOADER_TASK_FAILED:w{workers}:trial{trial}")
            loader_rows[workers].append(row)
    loader_metric_diff = max(_max_pairwise_metric_diff(loader_rows[2]), _max_pairwise_metric_diff(loader_rows[4]))
    loader_metric_diff = max(
        loader_metric_diff,
        _metric_diff(loader_rows[2][0]["metric_vector"], loader_rows[4][0]["metric_vector"]),
    )
    if loader_metric_diff > METRIC_EQUIVALENCE_TOLERANCE:
        raise RuntimeError(f"MX04_PROFILE_LOADER_METRIC_DRIFT:{loader_metric_diff}")
    workers2_s = _median_elapsed(loader_rows[2])
    workers4_s = _median_elapsed(loader_rows[4])
    observed_noise_s = max(
        max(float(row["epoch_wallclock_s"]) for row in loader_rows[2]) - min(float(row["epoch_wallclock_s"]) for row in loader_rows[2]),
        max(float(row["epoch_wallclock_s"]) for row in loader_rows[4]) - min(float(row["epoch_wallclock_s"]) for row in loader_rows[4]),
        0.03 * workers2_s,
    )
    workers4_measured_gain_s = workers2_s - workers4_s
    selected_workers = 4 if workers4_measured_gain_s > observed_noise_s else 2
    fallback_to_2 = selected_workers == 2

    family_receipts: dict[str, dict[str, Any]] = {}
    selected_rows: dict[str, dict[str, Any]] = {}
    for family in EXPECTED_FAMILIES:
        off = run_profile_task(
            profile_root=profile_root,
            variant=f"family_{family}_checkpoint_off_w{selected_workers}",
            family=family,
            checkpoint="off",
            workers=selected_workers,
            candidate=scratch,
            protocol=protocol,
            config=config,
        )
        off_safe = bool(
            off.get("status") == "COMPLETED"
            and max(float(off["vram_allocated_peak_gb"]), float(off["vram_reserved_peak_gb"])) < CHECKPOINT_SOFT_CAP_GB
        )
        if off_safe:
            selected = off
            setting = "off"
        else:
            on = run_profile_task(
                profile_root=profile_root,
                variant=f"family_{family}_checkpoint_on_w{selected_workers}",
                family=family,
                checkpoint="on",
                workers=selected_workers,
                candidate=scratch,
                protocol=protocol,
                config=config,
            )
            if on.get("status") != "COMPLETED":
                raise RuntimeError(f"MX04_PROFILE_CHECKPOINT_ON_FAILED:{family}")
            if max(float(on["vram_allocated_peak_gb"]), float(on["vram_reserved_peak_gb"])) >= VRAM_HARD_STOP_GB:
                raise RuntimeError(f"MX04_PROFILE_VRAM_HARD_STOP:{family}")
            selected = on
            setting = "on"
        selected_rows[family] = selected
        family_receipts[family] = {
            "checkpoint_off": {key: value for key, value in off.items() if key != "metrics"},
            "selected_setting": setting,
            "selected_receipt": {key: value for key, value in selected.items() if key != "metrics"},
        }

    settings = {family: str(family_receipts[family]["selected_setting"]) for family in EXPECTED_FAMILIES}
    off_families = [family for family, setting in settings.items() if setting == "off"]
    mixed = len(set(settings.values())) > 1
    cross_diff = 0.0
    cross_family = off_families[0] if off_families else reference_family
    cross_verdict = "NOT_APPLICABLE(uniform-all-on)"
    if off_families:
        off_row = selected_rows[cross_family]
        on_row = run_profile_task(
            profile_root=profile_root,
            variant=f"cross_mode_{cross_family}_checkpoint_on_w{selected_workers}",
            family=cross_family,
            checkpoint="on",
            workers=selected_workers,
            candidate=scratch,
            protocol=protocol,
            config=config,
        )
        if on_row.get("status") != "COMPLETED":
            raise RuntimeError(f"MX04_PROFILE_CROSS_MODE_ON_FAILED:{cross_family}")
        cross_diff = _metric_diff(off_row["metric_vector"], on_row["metric_vector"])
        cross_verdict = "PASS" if cross_diff <= METRIC_EQUIVALENCE_TOLERANCE else "FAIL"
        if cross_verdict != "PASS":
            # Fail closed to uniform checkpoint-on; no mixed-mode comparison is
            # permitted when recomputation equivalence is not demonstrated.
            for family in EXPECTED_FAMILIES:
                on = run_profile_task(
                    profile_root=profile_root,
                    variant=f"family_{family}_checkpoint_on_w{selected_workers}",
                    family=family,
                    checkpoint="on",
                    workers=selected_workers,
                    candidate=scratch,
                    protocol=protocol,
                    config=config,
                )
                if on.get("status") != "COMPLETED":
                    raise RuntimeError(f"MX04_PROFILE_FAIL_CLOSED_ON_FAILED:{family}")
                selected_rows[family] = on
                settings[family] = "on"
                family_receipts[family]["selected_setting"] = "on"
                family_receipts[family]["selected_receipt"] = {
                    key: value for key, value in on.items() if key != "metrics"
                }
            mixed = False
            cross_verdict = "NOT_APPLICABLE(uniform-all-on)"

    reference_rows = loader_rows[2]
    reference_utilization = statistics.fmean(float(row["sm_utilization_pct"]) for row in reference_rows)
    reference_allocated = max(float(row["vram_allocated_peak_gb"]) for row in reference_rows)
    reference_reserved = max(float(row["vram_reserved_peak_gb"]) for row in reference_rows)
    decision = "loader_first" if reference_utilization < 70.0 else "compute_first"
    profile = {
        "schema_version": 1,
        "operational_preflight": operational,
        "profile_run": {
            "mode": "smoke_only_never_selection_evidence",
            "output_dir": str(profile_root.relative_to(ROOT)).replace("\\", "/"),
            "sm_utilization_pct": reference_utilization,
            "vram_allocated_peak_gb": reference_allocated,
            "vram_reserved_peak_gb": reference_reserved,
            "epoch_wallclock_s": workers2_s,
        },
        "decision": decision,
        "per_family_checkpoint": {
            family: {
                "setting": settings[family],
                "allocated_peak_gb": float(selected_rows[family]["vram_allocated_peak_gb"]),
                "reserved_peak_gb": float(selected_rows[family]["vram_reserved_peak_gb"]),
                "under_7p0_soft_cap": max(
                    float(selected_rows[family]["vram_allocated_peak_gb"]),
                    float(selected_rows[family]["vram_reserved_peak_gb"]),
                ) < CHECKPOINT_SOFT_CAP_GB,
            }
            for family in EXPECTED_FAMILIES
        },
        "cross_mode_receipt": {
            "family": cross_family,
            "seed": PROFILE_SEED,
            "max_abs_metric_diff": cross_diff,
            "verdict": cross_verdict,
        },
        "loader": {
            "num_workers": selected_workers,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "pin_memory": True,
            "epoch_before_s": workers2_s,
            "epoch_after_s": workers4_s if selected_workers == 4 else workers2_s,
            "fallback_to_2_recorded": fallback_to_2,
        },
        "yellow_gates": {
            "torch_compile": "DROPPED:no determinism plus bitwise-resume speed receipt on this GPU",
            "channels_last": "DROPPED:no determinism plus bitwise-resume speed receipt on this GPU",
        },
    }
    validate_throughput_profile(profile)
    receipts = {
        "schema_version": 1,
        "status": "COMPLETED",
        "profile_id": profile_id,
        "gpu_name": gpu_name,
        "trainer_sha256": trainer_sha,
        "profiler_sha256": script_sha,
        "profile_sha256": _canonical_sha(profile),
        "profile_limit": PROFILE_LIMIT,
        "loader_trials": {
            str(workers): [{key: value for key, value in row.items() if key != "metrics"} for row in rows]
            for workers, rows in loader_rows.items()
        },
        "loader_metric_max_abs_diff": loader_metric_diff,
        "observed_noise_s": observed_noise_s,
        "workers4_measured_gain_s": workers4_measured_gain_s,
        "mixed_checkpoint_settings_before_fail_closed": mixed,
        "family_receipts": family_receipts,
    }
    atomic_write_json(artifact_path, profile)
    atomic_write_json(artifact_path.with_name("MX04_THROUGHPUT_PROFILE_RECEIPTS.json"), receipts)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "results/method_x_registry/MX04_THROUGHPUT_PROFILE.json",
    )
    args = parser.parse_args()
    _install_profile_signal_handlers()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    artifact_path = args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    try:
        profile = build_profile(
            output_root=output_root.resolve(),
            config_path=config_path.resolve(),
            artifact_path=artifact_path.resolve(),
        )
    except (KeyboardInterrupt, RuntimeError) as exc:
        print(json.dumps({"status": "BLOCKED", "blocker": str(exc)}, indent=2), file=sys.stderr)
        return 75 if isinstance(exc, KeyboardInterrupt) or "PAUSED_RESUMABLE" in str(exc) else 3
    print(json.dumps({"status": "COMPLETED", "artifact": str(artifact_path), "profile": profile}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
