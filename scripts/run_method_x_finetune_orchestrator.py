"""Autonomous MX-04 fine-tuning graph.

The graph is deliberately data-dependent: audit mode writes contracts and
truthful blockers, while run mode is unlocked only after the matched battery
and MX-03 validity are terminal.  Pretrained weights are bound per fold; a
single fold-0 checkpoint can never be broadcast to the other folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.method_x_autosupervisor import (
    enforce_autonomous_supervision,
    evaluate_autonomous_supervision,
    run_supervised_process,
)
from src.training.method_x_loop_guard import record_failure, record_success
from src.utils.io_helpers import atomic_write_json


DEFAULT_OUTPUT_ROOT = ROOT / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729"
DEFAULT_CONFIG = ROOT / "configs/experiments/method_x_finetune_v1.yaml"
LOOP_STATE = "reports/method_x_loop_guard.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"INVALID_JSON:{path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"INVALID_JSON_OBJECT:{path.name}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def validate_fold_pretrained_manifest(manifest: Mapping[str, Any], *, folds: int = 5) -> dict[str, dict[str, str]]:
    """Validate the shape of a complete fold-local provenance map."""

    entries = manifest.get("folds") if isinstance(manifest, Mapping) else None
    expected = {str(index) for index in range(int(folds))}
    if not isinstance(entries, Mapping) or set(str(key) for key in entries) != expected:
        raise RuntimeError("PRETRAINED_FOLD_MANIFEST_INCOMPLETE")
    normalized: dict[str, dict[str, str]] = {}
    for fold in sorted(expected, key=int):
        value = entries[fold]
        if not isinstance(value, Mapping):
            raise RuntimeError(f"PRETRAINED_FOLD_MANIFEST_INVALID:{fold}")
        checkpoint = str(value.get("checkpoint", "")).strip()
        sidecar = str(value.get("manifest", "")).strip()
        sha256 = str(value.get("sha256", "")).strip().lower()
        if not checkpoint or not sidecar or len(sha256) != 64:
            raise RuntimeError(f"PRETRAINED_FOLD_MANIFEST_INVALID:{fold}")
        normalized[fold] = {"checkpoint": checkpoint, "manifest": sidecar, "sha256": sha256}
    return normalized


EFFECTIVE_FIVE = (
    "resnet10_3d",
    "resnet18_3d",
    "resnet34_3d",
    "densenet121_3d",
    "mednext_3d_s",
)
EXCLUDED_BACKBONES = ("resnet50_3d", "efficientnet3d_b0")
TERMINAL_TRAINING_STATUSES = frozenset(
    {
        "COMPLETED",
        "COMPLETED_VALIDATED_REPAIRED",
        "COMPLETED_VALIDATED_RECONCILED",
    }
)


def _baseline_terminal(manifest: Mapping[str, Any]) -> bool:
    if str(manifest.get("status", "")).upper() != "COMPLETED":
        return False
    rows = manifest.get("models")
    if not isinstance(rows, list) or len(rows) != 7:
        return False
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        name = str(row.get("backbone") or "")
        if not name or name in by_name:
            return False
        by_name[name] = row
    if set(by_name) != set(EFFECTIVE_FIVE) | set(EXCLUDED_BACKBONES):
        return False
    for name in EFFECTIVE_FIVE:
        row = by_name[name]
        training = row.get("training") or {}
        if str(row.get("status", "")).upper() != "COMPLETED":
            return False
        if str(training.get("status", "")).upper() not in TERMINAL_TRAINING_STATUSES:
            return False
    for name in EXCLUDED_BACKBONES:
        row = by_name[name]
        training = row.get("training") or {}
        if str(row.get("status", "")) != "PROTOCOL_EXCLUDED_NOT_STARTED":
            return False
        if str(training.get("status", "")) != "PROTOCOL_EXCLUDED_NOT_STARTED":
            return False
    return True


def _mx03_terminal(output_root: Path) -> bool:
    path = output_root / "reports/mx03_validity_orchestrator.json"
    if not path.is_file():
        return False
    try:
        return _read(path).get("status") == "COMPLETED"
    except RuntimeError:
        return False


def _validate_config(config: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if config.get("selection_scope") != "inner_validation_only":
        blockers.append("FINETUNE_SELECTION_SCOPE_NOT_INNER_VALIDATION_ONLY")
    if config.get("outer_test_access") != "exactly_once_after_lock":
        blockers.append("FINETUNE_OUTER_TEST_POLICY_MISMATCH")
    if config.get("inner_selection_only") is not True:
        blockers.append("FINETUNE_INNER_SELECTION_POLICY_MISSING")
    if config.get("fold_local_pretrained_required") is not True:
        blockers.append("FINETUNE_FOLD_LOCAL_PRETRAINED_POLICY_MISSING")
    if list(config.get("replication_seeds", [])) != [13, 29, 47]:
        blockers.append("FINETUNE_REPLICATION_SEEDS_MISMATCH")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        blockers.append("FINETUNE_CANDIDATE_GRID_MISSING")
    else:
        ids = [str(row.get("id")) for row in candidates if isinstance(row, Mapping)]
        if len(ids) != len(set(ids)):
            blockers.append("FINETUNE_CANDIDATE_IDS_INVALID")
        for row in candidates:
            if not isinstance(row, Mapping):
                blockers.append("FINETUNE_CANDIDATE_SCHEMA_INVALID")
                continue
            if bool(row.get("ema")) and bool(row.get("swa")):
                blockers.append(f"EMA_AND_SWA_SIMULTANEOUS:{row.get('id')}")
    return blockers


def build_pretrained_fold_manifest(*, output_root: Path, backbone: str, folds: int = 5) -> tuple[Path, dict[str, Any]]:
    """Create sidecars from the already-finalized baseline fold checkpoints."""

    baseline_dir = output_root / "models" / backbone
    contract_path = baseline_dir / "run_contract.json"
    contract = _read(contract_path).get("run_contract") if contract_path.is_file() else {}
    if not isinstance(contract, dict):
        raise RuntimeError(f"BASELINE_RUN_CONTRACT_MISSING:{backbone}")
    sidecar_dir = output_root / "pretrained_provenance" / backbone
    entries: dict[str, dict[str, str]] = {}
    for fold in range(int(folds)):
        checkpoint = baseline_dir / f"method_x_3d_fold_{fold}.pt"
        state_path = baseline_dir / f"method_x_3d_fold_{fold}_state.json"
        if not checkpoint.is_file() or not state_path.is_file():
            raise RuntimeError(f"BASELINE_CHECKPOINT_MISSING:{backbone}:FOLD_{fold}")
        state = _read(state_path)
        if str(state.get("stage", "")).upper() != "FINALIZED":
            raise RuntimeError(f"BASELINE_FOLD_NOT_FINALIZED:{backbone}:FOLD_{fold}")
        sidecar = sidecar_dir / f"fold_{fold}.json"
        checkpoint_sha = _sha(checkpoint)
        atomic_write_json(
            sidecar,
            {
                "schema_version": 1,
                "checkpoint_sha256": checkpoint_sha,
                "sha256": checkpoint_sha,
                "license": "PROJECT_GENERATED_INTERNAL_RESEARCH_WEIGHTS",
                "source_type": "FOLD_LOCAL_METHOD_X_BASELINE",
                "source_backbone": backbone,
                "source_fold": fold,
                "source_run_contract_digest": contract.get("digest"),
                "source_split_membership_digest": (state.get("details") or {}).get("split_membership_digest"),
                "raw_identifiers_stored": False,
                "external_download": False,
            },
        )
        entries[str(fold)] = {
            "checkpoint": _rel(checkpoint),
            "manifest": _rel(sidecar),
            "sha256": checkpoint_sha,
        }
    payload = {
        "schema_version": 1,
        "backbone": backbone,
        "folds": entries,
        "coverage_minimum": 0.95,
        "source_contract_digest": contract.get("digest"),
        "outer_test_used_for_selection": False,
        "created_at_utc": _now(),
    }
    manifest_path = sidecar_dir / "pretrained_fold_manifest.json"
    atomic_write_json(manifest_path, payload)
    validate_fold_pretrained_manifest(payload, folds=folds)
    return manifest_path, payload


def _protocol_args(battery: Mapping[str, Any]) -> dict[str, Any]:
    protocol = battery.get("locked_protocol") if isinstance(battery.get("locked_protocol"), Mapping) else {}
    common = protocol.get("common") if isinstance(protocol.get("common"), Mapping) else {}
    return {
        "folds": int(protocol.get("folds", 5)),
        "epochs": int(protocol.get("epochs", 100)),
        "resolution": int(protocol.get("resolution", 128)),
        "seed": int(protocol.get("seed", 13)),
        "cache_sha256": str((battery.get("locked_inputs") or {}).get("cache_sha256", "")),
        "spm_snapshot_sha256": str((battery.get("spm_gate_snapshot") or {}).get("snapshot_sha256", "")),
        "common": dict(common),
    }


def build_candidate_command(
    *,
    output_root: Path,
    family: str,
    candidate: Mapping[str, Any],
    seed: int,
    selection_only: bool,
    fold_manifest: Path | None,
    protocol: Mapping[str, Any],
) -> list[str]:
    common = protocol["common"]
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/train_3d_mri.py"),
        "--cache", "data/cached/t1_volume_cache_128.h5",
        "--cache_sha256", str(protocol["cache_sha256"]),
        "--spm_snapshot_sha256", str(protocol["spm_snapshot_sha256"]),
        "--require_exact_resume_contract",
        "--splits", "splits",
        "--out_dir", str(output_root),
        "--phase", "final",
        "--backbone", family,
        "--fusion", str(common.get("fusion", "cross_attention")),
        "--task_loss", str(candidate.get("task_loss", common.get("task_loss", "bce"))),
        "--lr", str(common.get("learning_rate", 1e-4)),
        "--scheduler", str(common.get("scheduler", "cosine")),
        "--dropout", str(common.get("dropout", 0.2)),
        "--resolution", str(protocol["resolution"]),
        "--folds", str(protocol["folds"]),
        "--epochs", str(protocol["epochs"]),
        "--seed", str(seed),
        "--batch_size", str(common.get("batch_size", 1)),
        "--grad_accum", str(common.get("grad_accum", 8)),
        "--device", "cuda",
        "--require_gpu",
        "--restore_best_at_end",
        "--amp",
        "--checkpoint",
        "--num_workers", str(common.get("num_workers", 2)),
        "--pin_memory",
        "--no-persistent_workers",
        "--prefetch_factor", str(common.get("prefetch_factor", 2)),
    ]
    if selection_only:
        command.append("--hpo_selection_only")
    if bool(candidate.get("pretrained")):
        if fold_manifest is None:
            raise RuntimeError("PRETRAINED_FOLD_MANIFEST_REQUIRED")
        command.extend(["--pretrained_fold_manifest", str(fold_manifest), "--require_pretrained_provenance"])
        command.extend(["--finetune_lr", str(candidate.get("finetune_lr", 5e-5))])
        command.extend(["--layerwise_lr_decay", str(candidate.get("layerwise_lr_decay", 0.75)), "--discriminative_lr"])
        schedule = str(candidate.get("freeze_schedule", "full_after_warmup"))
        if schedule == "head_8_then_last_stage":
            command.extend(["--freeze_encoder", "--unfreeze_schedule", schedule, "--unfreeze_epoch", "8"])
        elif schedule == "full_after_warmup":
            command.extend(["--freeze_encoder", "--unfreeze_schedule", schedule, "--unfreeze_epoch", "8"])
    if bool(candidate.get("ema")):
        command.extend(["--ema_decay", str(candidate.get("ema_decay", 0.999))])
    if bool(candidate.get("swa")):
        command.extend(["--swa", "--swa_start_fraction", "0.8"])
    if str(candidate.get("task_loss")) == "focal":
        command.extend(["--focal_gamma", str(candidate.get("focal_gamma", 2.0))])
    return command


def _inner_metric(path: Path, *, folds: int, allow_terminal_baseline: bool = False) -> float:
    payload = _read(path)
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), Mapping) else {}
    if aggregate.get("status") != "COMPLETED":
        raise RuntimeError("FINETUNE_INNER_METRICS_NOT_COMPLETED")
    if not bool(aggregate.get("hpo_selection_only")) and not allow_terminal_baseline:
        raise RuntimeError("FINETUNE_INNER_METRICS_NOT_INNER_ONLY")
    rows = payload.get("folds")
    if not isinstance(rows, list) or len(rows) != int(folds):
        raise RuntimeError("FINETUNE_INNER_FOLD_COUNT_INVALID")
    if not allow_terminal_baseline and any(
        int(row.get("outer_test_evaluation_count", 0)) != 0
        for row in rows
        if isinstance(row, Mapping)
    ):
        raise RuntimeError("FINETUNE_OUTER_TEST_USED_DURING_SELECTION")
    values = [float(row["best_val_loss"]) for row in rows if isinstance(row, Mapping) and row.get("best_val_loss") is not None]
    if len(values) != int(folds):
        raise RuntimeError("FINETUNE_INNER_VALIDATION_METRICS_MISSING")
    return float(statistics.fmean(values))


def _supervise(root: Path, output_root: Path, phase: str, protocol: Mapping[str, Any], pid: int | None = None) -> dict[str, Any]:
    return evaluate_autonomous_supervision(
        root=root,
        output_root=output_root,
        phase=phase,
        config={"resolution": protocol["resolution"], "folds": protocol["folds"], "seed": protocol["seed"], "hardware": {"hard_stop_vram_gb": 7.5}},
        expected_pid=pid,
    )


def run_finetune_orchestrator(*, root: Path, output_root: Path, config_path: Path, mode: str = "audit") -> dict[str, Any]:
    root = root.resolve()
    output_root = output_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    battery = _read(output_root / "battery_manifest.json")
    protocol = _protocol_args(battery)
    supervision = _supervise(root, output_root, "finetune_orchestrator", protocol)
    enforce_autonomous_supervision(root=root, output_root=output_root, report=supervision)
    blockers = _validate_config(config)
    if supervision.get("decision") == "BLOCK":
        blockers.append("AUTOSUPERVISOR_BLOCKED")
    if not _baseline_terminal(battery):
        blockers.append("BASELINE_BATTERY_NOT_FINALIZED")
    if not _mx03_terminal(output_root):
        blockers.append("MX03_VALIDITY_NOT_COMPLETED")
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "MX-04",
        "mode": mode,
        "status": "BLOCKED" if blockers else "READY",
        "training_started": False,
        "selection_scope": config.get("selection_scope"),
        "outer_test_used_for_selection": False,
        "blockers": sorted(set(blockers)),
        "candidate_tasks": [],
        "supervision_decision": supervision.get("decision"),
        "autonomous_supervision": {"mode": "AUTONOMOUS_MULTI_LAYER_FAIL_CLOSED", "human_approval_required": False, "manual_override_allowed": False, "fail_closed": True},
        "next_automatic_action": "RUN_OR_RESUME_MX02" if blockers else "RUN_MX04_INNER_SCREENING",
    }
    if blockers or mode == "audit":
        if blockers:
            atomic_write_json(output_root / "reports/BLOCKED_MX04_FINETUNE.json", report)
        atomic_write_json(output_root / "reports/mx04_finetune_readiness.json", report)
        atomic_write_json(output_root / "reports/mx04_finetune_orchestrator.json", report)
        return report

    candidates = [row for row in config.get("candidates", []) if isinstance(row, Mapping)]
    families = [str(value) for value in config.get("candidate_families", [])]
    state_path = output_root / "reports/mx04_task_state.json"
    state = _read(state_path) if state_path.is_file() else {"schema_version": 1, "tasks": {}}
    scores: list[dict[str, Any]] = []
    for family in families:
        fold_manifest_path, fold_manifest = build_pretrained_fold_manifest(output_root=output_root, backbone=family, folds=int(protocol["folds"]))
        for candidate in candidates:
            candidate_id = str(candidate.get("id"))
            task_id = f"{family}__{candidate_id}__seed13__inner"
            candidate_root = output_root / "finetune_v1" / family / candidate_id / "seed_13_inner"
            metrics_path = candidate_root / "three_d_metrics.json"
            if candidate_id == "scratch_baseline":
                baseline_metrics = output_root / "models" / family / "three_d_metrics.json"
                score = _inner_metric(
                    baseline_metrics,
                    folds=int(protocol["folds"]),
                    allow_terminal_baseline=True,
                ) if baseline_metrics.is_file() else None
                row = {"task_id": task_id, "family": family, "candidate": candidate_id, "status": "REUSED_BASELINE", "score": score, "training_started": False, "outer_test_used_for_selection": False}
                report["candidate_tasks"].append(row)
                if score is not None:
                    scores.append({**row, "score": score, "model_root": str((output_root / "models" / family).resolve())})
                continue
            task = state.get("tasks", {}).get(task_id, {}) if isinstance(state.get("tasks"), Mapping) else {}
            command = build_candidate_command(
                output_root=candidate_root,
                family=family,
                candidate=candidate,
                seed=13,
                selection_only=True,
                fold_manifest=fold_manifest_path,
                protocol=protocol,
            )
            contract_digest = hashlib.sha256(json.dumps(command, sort_keys=True).encode()).hexdigest()
            if task and task.get("contract") and str(task.get("contract")) != contract_digest:
                report["status"] = "BLOCKED"
                report["blockers"].append(f"TASK_CONTRACT_MISMATCH:{task_id}")
                break
            if task.get("status") == "COMPLETED" and metrics_path.is_file():
                score = _inner_metric(metrics_path, folds=int(protocol["folds"]))
                scores.append({"task_id": task_id, "family": family, "candidate": candidate_id, "score": score, "model_root": str(candidate_root.resolve()), "status": "COMPLETED", "training_started": True, "outer_test_used_for_selection": False})
                report["candidate_tasks"].append(scores[-1])
                continue
            candidate_root.mkdir(parents=True, exist_ok=True)
            state.setdefault("tasks", {})[task_id] = {"status": "RUNNING", "contract": contract_digest, "updated_at_utc": _now()}
            atomic_write_json(state_path, state)
            execution = run_supervised_process(
                command,
                cwd=root,
                output_root=candidate_root,
                interval_seconds=15.0,
                max_idle_seconds=3600.0,
                check=lambda pid, _protocol=protocol: _supervise(root, output_root, "finetune_watch", _protocol, pid),
            )
            report["training_started"] = True
            code = int(execution.get("returncode", 78))
            if code == 75:
                state["tasks"][task_id].update(status="PAUSED_RESUMABLE", execution=execution, updated_at_utc=_now())
                atomic_write_json(state_path, state)
                report["status"] = "PAUSED_RESUMABLE"
                report["blockers"].append(f"TASK_PAUSED_RESUMABLE:{task_id}")
                report["next_automatic_action"] = "RESUME_MX04_EXACT_CONTRACT"
                break
            if code != 0:
                failure = record_failure(output_root / LOOP_STATE, stage="MX-04", contract_digest=contract_digest, failure_signature=f"TASK_EXIT_{code}:{task_id}")
                state["tasks"][task_id].update(status="BLOCKED", execution=execution, loop_guard=failure, updated_at_utc=_now())
                atomic_write_json(state_path, state)
                report["status"] = "BLOCKED"
                report["blockers"].append(f"TASK_EXECUTION_BLOCKED:{task_id}")
                break
            try:
                score = _inner_metric(metrics_path, folds=int(protocol["folds"]))
            except RuntimeError as exc:
                report["status"] = "BLOCKED"
                report["blockers"].append(str(exc))
                break
            state["tasks"][task_id].update(status="COMPLETED", execution=execution, score=score, updated_at_utc=_now())
            atomic_write_json(state_path, state)
            record_success(output_root / LOOP_STATE, stage="MX-04", contract_digest=state["tasks"][task_id]["contract"])
            row = {"task_id": task_id, "family": family, "candidate": candidate_id, "status": "COMPLETED", "score": score, "model_root": str(candidate_root.resolve()), "training_started": True, "outer_test_used_for_selection": False}
            scores.append(row)
            report["candidate_tasks"].append(row)
        if report["status"] in {"BLOCKED", "PAUSED_RESUMABLE"}:
            break

    if report["status"] not in {"BLOCKED", "PAUSED_RESUMABLE"}:
        if not scores:
            report["status"] = "BLOCKED"
            report["blockers"].append("FINETUNE_NO_COMPLETED_INNER_CANDIDATE")
        else:
            winner = min(scores, key=lambda row: float(row["score"]))
            report["selected_backbone"] = winner["family"]
            report["selected_candidate"] = winner["candidate"]
            report["selected_model_root"] = winner["model_root"]
            report["status"] = "LOCKED_INNER_VALIDATION"
            report["next_automatic_action"] = "RUN_MX04_REPLICATION_SEEDS_THEN_OUTER_LOCK"
            atomic_write_json(
                output_root / "reports/finetune_selection_manifest.json",
                {
                    "schema_version": 1,
                    "status": "LOCKED_INNER_VALIDATION",
                    "selected_backbone": winner["family"],
                    "selected_candidate": winner["candidate"],
                    "selected_model_root": winner["model_root"],
                    "selection_scope": "inner_validation_only",
                    "outer_test_used_for_selection": False,
                    "scores": scores,
                    "replication_seeds": config.get("replication_seeds"),
                    "created_at_utc": _now(),
                },
            )
    report["blockers"] = sorted(set(str(value) for value in report["blockers"]))
    atomic_write_json(output_root / "reports/mx04_finetune_readiness.json", report)
    atomic_write_json(output_root / "reports/mx04_finetune_orchestrator.json", report)
    if report["blockers"]:
        atomic_write_json(output_root / "reports/BLOCKED_MX04_FINETUNE.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("audit", "run"), default="audit")
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    report = run_finetune_orchestrator(root=ROOT, output_root=output_root, config_path=config_path, mode=args.mode)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if report.get("status") == "PAUSED_RESUMABLE":
        return 75
    # An inner-validation lock is a scientifically valid intermediate state,
    # not a terminal fine-tuning result.  Fail closed so MX-05/XAI cannot be
    # unlocked until the prespecified replication stage is materialized.
    return 0 if report.get("status") in {"READY", "COMPLETED"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
