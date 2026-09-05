"""Autonomous MX-05 AMUSE-X-Hybrid3D ablation and replication graph."""

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

from scripts.smoke_method_x_backbone import smoke
from src.training.method_x_autosupervisor import enforce_autonomous_supervision, evaluate_autonomous_supervision, run_supervised_process
from src.training.method_x_loop_guard import record_failure, record_success
from src.utils.io_helpers import atomic_write_json


DEFAULT_OUTPUT_ROOT = ROOT / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729"
DEFAULT_CONFIG = ROOT / "configs/experiments/method_x_hybrid_v1.yaml"
LOOP_STATE = "reports/method_x_loop_guard.json"
EXPECTED_ABLATIONS = ["baseline", "plus_dense_skip", "plus_large_kernel", "plus_attention", "plus_a3", "plus_a4", "full_model"]


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


def _baseline_terminal(output_root: Path) -> bool:
    try:
        manifest = _read(output_root / "battery_manifest.json")
    except RuntimeError:
        return False
    rows = manifest.get("models")
    if manifest.get("status") != "COMPLETED":
        return False
    if not isinstance(rows, list) or len(rows) != 7:
        return False
    by_name: dict = {}
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
        if row.get("status") != "COMPLETED":
            return False
        if str(training.get("status") or "") not in TERMINAL_TRAINING_STATUSES:
            return False
    for name in EXCLUDED_BACKBONES:
        row = by_name[name]
        training = row.get("training") or {}
        if row.get("status") != "PROTOCOL_EXCLUDED_NOT_STARTED":
            return False
        if training.get("status") != "PROTOCOL_EXCLUDED_NOT_STARTED":
            return False
    return True


def _upstream_terminal(output_root: Path, stage: str, allowed: set[str]) -> bool:
    path = output_root / "reports" / f"{stage}.json"
    if not path.is_file():
        return False
    try:
        return _read(path).get("status") in allowed
    except RuntimeError:
        return False


def _protocol(battery: Mapping[str, Any]) -> dict[str, Any]:
    locked = battery.get("locked_protocol") if isinstance(battery.get("locked_protocol"), Mapping) else {}
    common = locked.get("common") if isinstance(locked.get("common"), Mapping) else {}
    return {
        "folds": int(locked.get("folds", 5)),
        "epochs": int(locked.get("epochs", 100)),
        "resolution": int(locked.get("resolution", 128)),
        "seed": int(locked.get("seed", 13)),
        "cache_sha256": str((battery.get("locked_inputs") or {}).get("cache_sha256", "")),
        "spm_snapshot_sha256": str((battery.get("spm_gate_snapshot") or {}).get("snapshot_sha256", "")),
        "common": dict(common),
    }


def _validate_config(config: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if config.get("backbone_name") != "amuse_x_hybrid3d":
        blockers.append("HYBRID_BACKBONE_NAME_MISMATCH")
    if list(config.get("tabular_features", [])) != ["age", "sex", "education_years", "apoe4_carrier", "mmse_base"]:
        blockers.append("HYBRID_TABULAR_FEATURE_ORDER_MISMATCH")
    if config.get("claim_boundary", {}).get("pet_input") is not False:
        blockers.append("HYBRID_PET_INPUT_BOUNDARY_FAILED")
    if list(config.get("ablation_order", [])) != EXPECTED_ABLATIONS:
        blockers.append("HYBRID_ABLATION_ORDER_INCOMPLETE")
    if config.get("inner_selection_only") is not True:
        blockers.append("HYBRID_INNER_SELECTION_POLICY_MISSING")
    if list(config.get("replication_seeds", [])) != [13, 29, 47]:
        blockers.append("HYBRID_REPLICATION_SEEDS_MISMATCH")
    definitions = config.get("variant_definitions")
    if not isinstance(definitions, Mapping) or set(str(key) for key in definitions) != set(EXPECTED_ABLATIONS):
        blockers.append("HYBRID_VARIANT_DEFINITIONS_INCOMPLETE")
    if int(config.get("parameter_budget_max", 0)) != 20_000_000:
        blockers.append("HYBRID_PARAMETER_BUDGET_INVALID")
    return blockers


def build_hybrid_command(*, root: Path, output_dir: Path, variant: str, seed: int, selection_only: bool, protocol: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    common = protocol["common"]
    command = [
        sys.executable, "-u", str(root / "scripts/train_3d_mri.py"),
        "--cache", "data/cached/t1_volume_cache_128.h5", "--cache_sha256", str(protocol["cache_sha256"]),
        "--spm_snapshot_sha256", str(protocol["spm_snapshot_sha256"]), "--require_exact_resume_contract",
        "--splits", "splits", "--out_dir", str(output_dir), "--phase", "final",
        "--backbone", "amuse_x_hybrid3d", "--hybrid_variant", variant,
        "--hybrid_attention_heads", str((config.get("attention") or {}).get("heads", 2)),
        "--hybrid_attention_window_size", str((config.get("attention") or {}).get("window_size", 4)),
        "--fusion", str(common.get("fusion", "cross_attention")), "--task_loss", str(common.get("task_loss", "bce")),
        "--lr", str(common.get("learning_rate", 1e-4)), "--scheduler", str(common.get("scheduler", "cosine")),
        "--dropout", str(common.get("dropout", 0.2)), "--resolution", str(protocol["resolution"]),
        "--folds", str(protocol["folds"]), "--epochs", str(protocol["epochs"]), "--seed", str(seed),
        "--batch_size", str(common.get("batch_size", 1)), "--grad_accum", str(common.get("grad_accum", 8)),
        "--device", "cuda", "--require_gpu", "--restore_best_at_end", "--amp", "--checkpoint",
        "--num_workers", str(common.get("num_workers", 2)), "--pin_memory", "--no-persistent_workers", "--prefetch_factor", str(common.get("prefetch_factor", 2)),
    ]
    if selection_only:
        command.append("--hpo_selection_only")
    if variant == "plus_a3":
        command.extend(["--latent_alignment_lambda", "0.05", "--alignment_head", "linear"])
    elif variant == "plus_a4":
        command.extend(["--q2_channel_gate", "phenotype"])
    elif variant == "full_model":
        command.extend(["--latent_alignment_lambda", "0.05", "--alignment_head", "linear", "--q2_channel_gate", "phenotype"])
    return command


def _inner_metric(path: Path, *, folds: int) -> float:
    payload = _read(path)
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), Mapping) else {}
    rows = payload.get("folds")
    if aggregate.get("status") != "COMPLETED" or not aggregate.get("hpo_selection_only") or not isinstance(rows, list) or len(rows) != folds:
        raise RuntimeError("HYBRID_INNER_METRICS_NOT_COMPLETED")
    if any(int(row.get("outer_test_evaluation_count", 0)) != 0 for row in rows if isinstance(row, Mapping)):
        raise RuntimeError("HYBRID_OUTER_TEST_USED_DURING_SELECTION")
    values = [float(row["best_val_loss"]) for row in rows if isinstance(row, Mapping) and row.get("best_val_loss") is not None]
    if len(values) != folds:
        raise RuntimeError("HYBRID_INNER_VALIDATION_METRICS_MISSING")
    return float(statistics.fmean(values))


def run_hybrid_orchestrator(*, root: Path, output_root: Path, config_path: Path, mode: str = "audit") -> dict[str, Any]:
    root, output_root = root.resolve(), output_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    battery = _read(output_root / "battery_manifest.json")
    protocol = _protocol(battery)
    supervision = evaluate_autonomous_supervision(root=root, output_root=output_root, phase="hybrid_orchestrator", config={"resolution": protocol["resolution"], "folds": protocol["folds"], "seed": protocol["seed"], "hardware": {"hard_stop_vram_gb": 7.5}})
    enforce_autonomous_supervision(root=root, output_root=output_root, report=supervision)
    blockers = _validate_config(config)
    if supervision.get("decision") == "BLOCK":
        blockers.append("AUTOSUPERVISOR_BLOCKED")
    if not _baseline_terminal(output_root):
        blockers.append("HYBRID_BASELINE_BATTERY_NOT_FINALIZED")
    if not _upstream_terminal(output_root, "mx03_validity_orchestrator", {"COMPLETED"}):
        blockers.append("MX03_VALIDITY_NOT_COMPLETED")
    if not _upstream_terminal(output_root, "mx04_finetune_orchestrator", {"COMPLETED"}):
        blockers.append("MX04_FINETUNE_NOT_COMPLETED")
    report: dict[str, Any] = {
        "schema_version": 1, "stage": "MX-05", "mode": mode,
        "status": "BLOCKED" if blockers else "READY", "training_started": False,
        "outer_test_used_for_selection": False, "ablation_order": list(config.get("ablation_order", [])),
        "candidate_tasks": [], "blockers": sorted(set(blockers)), "supervision_decision": supervision.get("decision"),
        "autonomous_supervision": {"mode": "AUTONOMOUS_MULTI_LAYER_FAIL_CLOSED", "human_approval_required": False, "manual_override_allowed": False, "fail_closed": True},
        "next_automatic_action": "RUN_OR_RESUME_MX02" if blockers else "RUN_MX05_ABLATIONS",
    }
    if blockers or mode == "audit":
        if blockers:
            atomic_write_json(output_root / "reports/BLOCKED_MX05_HYBRID.json", report)
        atomic_write_json(output_root / "reports/mx05_hybrid_readiness.json", report)
        atomic_write_json(output_root / "reports/mx05_hybrid_orchestrator.json", report)
        return report

    state_path = output_root / "reports/mx05_task_state.json"
    state = _read(state_path) if state_path.is_file() else {"schema_version": 1, "tasks": {}}
    scores: list[dict[str, Any]] = []
    for variant in EXPECTED_ABLATIONS:
        smoke_report = smoke("amuse_x_hybrid3d", "cuda", 128, float(config.get("smoke_vram_budget_gb", 6.5)), hybrid_variant=variant)
        if smoke_report.get("status") != "PASS":
            if bool((config.get("fallback") or {}).get("remove_attention_if_smoke_fails")) and variant in {"plus_attention", "full_model"}:
                smoke_report = smoke("amuse_x_hybrid3d", "cuda", 128, float(config.get("smoke_vram_budget_gb", 6.5)), hybrid_variant="amuse_x_hybrid_lite")
                if smoke_report.get("status") == "PASS":
                    variant = "amuse_x_hybrid_lite"
            if smoke_report.get("status") != "PASS":
                report["blockers"].append(f"HYBRID_SMOKE_FAILED:{variant}")
                break
        task_id = f"{variant}__seed13__inner"
        candidate_root = output_root / "hybrid_v1" / variant / "seed_13_inner"
        metrics_path = candidate_root / "three_d_metrics.json"
        task = state.get("tasks", {}).get(task_id, {}) if isinstance(state.get("tasks"), Mapping) else {}
        command = build_hybrid_command(
            root=root,
            output_dir=candidate_root,
            variant=variant,
            seed=13,
            selection_only=True,
            protocol=protocol,
            config=config,
        )
        contract_digest = hashlib.sha256(json.dumps(command, sort_keys=True).encode()).hexdigest()
        if task and task.get("contract") and str(task.get("contract")) != contract_digest:
            report["status"] = "BLOCKED"
            report["blockers"].append(f"TASK_CONTRACT_MISMATCH:{task_id}")
            break
        if task.get("status") == "COMPLETED" and metrics_path.is_file():
            score = _inner_metric(metrics_path, folds=protocol["folds"])
            row = {"task_id": task_id, "variant": variant, "score": score, "status": "COMPLETED", "training_started": True, "outer_test_used_for_selection": False, "smoke": smoke_report}
            scores.append(row); report["candidate_tasks"].append(row); continue
        candidate_root.mkdir(parents=True, exist_ok=True)
        state.setdefault("tasks", {})[task_id] = {"status": "RUNNING", "contract": contract_digest, "updated_at_utc": _now()}
        atomic_write_json(state_path, state)
        execution = run_supervised_process(command, cwd=root, output_root=candidate_root, interval_seconds=15.0, max_idle_seconds=3600.0, check=lambda pid: evaluate_autonomous_supervision(root=root, output_root=output_root, phase="hybrid_watch", config={"resolution": 128, "folds": 5, "seed": 13, "hardware": {"hard_stop_vram_gb": 7.5}}, expected_pid=pid))
        report["training_started"] = True
        code = int(execution.get("returncode", 78))
        if code == 75:
            state["tasks"][task_id].update(status="PAUSED_RESUMABLE", execution=execution, updated_at_utc=_now()); atomic_write_json(state_path, state)
            report["status"] = "PAUSED_RESUMABLE"; report["blockers"].append(f"TASK_PAUSED_RESUMABLE:{task_id}"); report["next_automatic_action"] = "RESUME_MX05_EXACT_CONTRACT"; break
        if code != 0:
            guard = record_failure(output_root / LOOP_STATE, stage="MX-05", contract_digest=contract_digest, failure_signature=f"TASK_EXIT_{code}:{task_id}")
            state["tasks"][task_id].update(status="BLOCKED", execution=execution, loop_guard=guard, updated_at_utc=_now()); atomic_write_json(state_path, state)
            report["status"] = "BLOCKED"; report["blockers"].append(f"TASK_EXECUTION_BLOCKED:{task_id}"); break
        try:
            score = _inner_metric(metrics_path, folds=protocol["folds"])
        except RuntimeError as exc:
            report["status"] = "BLOCKED"; report["blockers"].append(str(exc)); break
        state["tasks"][task_id].update(status="COMPLETED", score=score, execution=execution, updated_at_utc=_now()); atomic_write_json(state_path, state)
        record_success(output_root / LOOP_STATE, stage="MX-05", contract_digest=contract_digest)
        row = {"task_id": task_id, "variant": variant, "score": score, "status": "COMPLETED", "training_started": True, "outer_test_used_for_selection": False, "smoke": smoke_report}
        scores.append(row); report["candidate_tasks"].append(row)

    if report["status"] not in {"BLOCKED", "PAUSED_RESUMABLE"}:
        if len(scores) != len(EXPECTED_ABLATIONS):
            report["status"] = "BLOCKED"; report["blockers"].append("HYBRID_ABLATION_SET_INCOMPLETE")
        else:
            winner = min(scores, key=lambda row: float(row["score"]))
            report.update(status="LOCKED_INNER_VALIDATION", selected_variant=winner["variant"], selected_model_root=str((output_root / "hybrid_v1" / winner["variant"] / "seed_13_inner").resolve()), next_automatic_action="RUN_MX05_REPLICATION_SEEDS_THEN_OUTER_LOCK")
            atomic_write_json(output_root / "reports/hybrid_selection_manifest.json", {
                "schema_version": 1, "status": "LOCKED_INNER_VALIDATION", "selected_variant": winner["variant"],
                "selected_model_root": report["selected_model_root"], "ablation_order": EXPECTED_ABLATIONS,
                "scores": scores, "outer_test_used_for_selection": False, "replication_seeds": [13, 29, 47], "created_at_utc": _now(),
            })
            atomic_write_json(output_root / "reports/amuse_x_hybrid_architecture_manifest.json", {
                "schema_version": 1, "status": "LOCKED_INNER_VALIDATION", "single_model": True,
                "variant": winner["variant"], "parameter_budget_max": config.get("parameter_budget_max"),
                "smoke_vram_budget_gb": config.get("smoke_vram_budget_gb"), "components": config.get("components"),
                "claim_boundary": config.get("claim_boundary"), "outer_test_used_for_selection": False,
            })
    report["blockers"] = sorted(set(str(value) for value in report["blockers"]))
    atomic_write_json(output_root / "reports/mx05_hybrid_readiness.json", report)
    atomic_write_json(output_root / "reports/mx05_hybrid_orchestrator.json", report)
    if report["blockers"]:
        atomic_write_json(output_root / "reports/BLOCKED_MX05_HYBRID.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("audit", "run"), default="audit")
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    report = run_hybrid_orchestrator(root=ROOT, output_root=output_root, config_path=config_path, mode=args.mode)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if report.get("status") == "PAUSED_RESUMABLE":
        return 75
    # Inner-only ablation selection cannot unlock XAI or publication.  The
    # selected variant must first complete its three-seed replication.
    return 0 if report.get("status") in {"READY", "COMPLETED"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
