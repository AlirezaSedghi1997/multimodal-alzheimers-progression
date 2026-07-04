#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

try:
    from ._stage_common import (
        add_mode_args,
        compute_config_diff,
        DEFAULT_METHOD_Y_PET_CACHE,
        load_cfg,
        print_mode_banner,
        project_path,
        resolve_stage_epochs,
        utc_now_iso,
        validate_method_y_pet_cache_protocol,
        validate_train_authorization,
        write_json,
        write_stage_report,
    )
    from ._parallel_run import (
        ParallelRunError,
        assert_no_active_gpu_training,
        capture_nvidia_smi_snapshot,
        create_isolated_run,
        isolated_config,
        process_guard_snapshot,
        publish_isolated_stage_run,
    )
except ImportError:  # pragma: no cover - direct script execution
    from _stage_common import (
        add_mode_args,
        compute_config_diff,
        DEFAULT_METHOD_Y_PET_CACHE,
        load_cfg,
        print_mode_banner,
        project_path,
        resolve_stage_epochs,
        utc_now_iso,
        validate_method_y_pet_cache_protocol,
        validate_train_authorization,
        write_json,
        write_stage_report,
    )
    from _parallel_run import (
        ParallelRunError,
        assert_no_active_gpu_training,
        capture_nvidia_smi_snapshot,
        create_isolated_run,
        isolated_config,
        process_guard_snapshot,
        publish_isolated_stage_run,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Method Y stage S1 PET-only guarded trainer")
    add_mode_args(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grad_accum", type=int, default=None)
    args = parser.parse_args()
    if args.publish_run:
        try:
            publish_result = publish_isolated_stage_run(args.publish_run, stage="S1")
        except ParallelRunError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(publish_result, indent=2))
        return 0
    print_mode_banner(args.mode)
    validate_train_authorization(args)
    start = utc_now_iso()
    cfg = load_cfg(args.config)
    validate_method_y_pet_cache_protocol(cfg, stage="S1")
    run = None
    if args.isolated_run:
        run = create_isolated_run(stage="S1", mode=args.mode, cfg=cfg, config_path=args.config)
        cfg = isolated_config(cfg, run)
    elif process_guard_snapshot():
        raise SystemExit("BLOCKED_BY_RUNNING_STAGE")
    if args.mode == "train":
        try:
            assert_no_active_gpu_training()
        except ParallelRunError as exc:
            raise SystemExit(str(exc)) from exc
    epochs = resolve_stage_epochs(args, cfg, default_train=60)
    active_grad_accum = int(args.grad_accum or cfg.get("compute", {}).get("grad_accum", 4))
    configured_report_root = (cfg.get("outputs", {}) or {}).get("stage_report_root")
    plan = {
        "stage": "S1",
        "status": "PLAN_READY",
        "mode": args.mode,
        "dry_run": bool(cfg.get("dry_run", True)),
        "logic": "PET-only baseline with evidential binary head, Centiloid regression, and tracer-adversarial regularization.",
        "inputs": cfg.get("inputs", {}),
        "model": cfg.get("model", {}),
        "losses": cfg.get("losses", {}),
        "augmentation": cfg.get("augmentation", {}),
        "acceptance_gate": cfg.get("acceptance_gate", {}),
        "training_defaults": {
            "folds": 4,
            "optimizer": "AdamW",
            "grad_accum": active_grad_accum,
            "early_stopping": "validation evidential loss patience 8",
        },
    }
    output_path = cfg.get("outputs", {}).get("plan_json")
    if output_path:
        write_json(output_path, plan)
    report_kwargs = {"execution_dir": run.run_dir, "update_official": False} if run else (
        {"execution_dir": configured_report_root, "update_official": False} if configured_report_root else {}
    )
    isolated_extra = (
        {
            "isolated_run": run.public_dict(),
            "frozen_input_manifest": run.public_dict()["frozen_input_manifest"],
            "publish_required_for_official_state": True,
        }
        if run
        else {}
    )
    pet_h5 = project_path(cfg.get("inputs", {}).get("pet_h5", DEFAULT_METHOD_Y_PET_CACHE))
    if not pet_h5.exists():
        report = write_stage_report(
            stage="S1",
            mode=args.mode,
            status="DRY_RUN",
            blocker="BLOCKED_S0_PET_CACHE_MISSING",
            acceptance_gate_passed=False,
            next_allowed_stage="STOP",
            start_time_utc=start,
            metrics={},
            extra={"plan": plan, "model_files_written": [],
                   "config_diff_vs_canonical": compute_config_diff(cfg),
                   "compute_profile": {**(cfg.get("compute", {}) or {})},
                   **isolated_extra},
            **report_kwargs,
        )
        print(json.dumps({"status": report["status"], "blocker": report["blocker"]}, indent=2))
        return 0
    if args.mode == "plan":
        write_stage_report(
            stage="S1",
            mode=args.mode,
            status="PLAN_READY",
            blocker=None,
            acceptance_gate_passed=False,
            next_allowed_stage="S2_PLAN_ONLY",
            start_time_utc=start,
            metrics={},
            extra={"plan": plan, "model_files_written": [],
                   "config_diff_vs_canonical": compute_config_diff(cfg),
                   **isolated_extra},
            **report_kwargs,
        )
        return 0
    # dry_run / train: execute real PET-only training loop (gated; heavy only in train mode).
    from src.training.method_y_stages import run_stage_s1

    nvidia_before = capture_nvidia_smi_snapshot(run.run_dir, "before") if run else None
    result = run_stage_s1(cfg, mode=args.mode, limit=int(args.limit), epochs=epochs)
    nvidia_after = capture_nvidia_smi_snapshot(run.run_dir, "after") if run else None
    status = "DONE" if result["acceptance_gate_passed"] else ("BLOCKED" if result.get("vram_blocker") or result.get("blocker") else "GATE_FAILED")
    blocker = result.get("vram_blocker") or result.get("blocker") or (None if result["acceptance_gate_passed"] else "ACCEPTANCE_GATE_FAILED")
    write_stage_report(
        stage="S1",
        mode=args.mode,
        status=status,
        blocker=blocker,
        acceptance_gate_passed=result["acceptance_gate_passed"],
        next_allowed_stage="S2" if result["acceptance_gate_passed"] else "STOP",
        start_time_utc=start,
        metrics={"folds_auroc_ge_0.85": result["folds_auroc_ge_0.85"],
                 "min_spearman": result["min_spearman"],
                 "max_tracer_probe_auc": result.get("max_tracer_probe_auc"),
                 "tracer_probe_status": result.get("tracer_probe_status"),
                 "gate_metric_subjects": result.get("gate_metric_subjects"),
                 "max_cohort_probe_auc": result.get("max_cohort_probe_auc")},
        extra={"plan": plan, "fold_metrics": result["fold_metrics"],
               "fold_manifest": result.get("fold_manifest"),
               "tracer_probe_gate": result.get("tracer_probe_gate"),
               "cohort_probe_gate": result.get("cohort_probe_gate"),
               "blocker_reasons": result.get("blocker_reasons", []),
               "deferred_risks": result.get("deferred_risks", []),
               "model_files_written": [m.get("checkpoint") for m in result["fold_metrics"]],
               "config_diff_vs_canonical": result.get("config_diff_vs_canonical"),
               "compute_profile": result.get("compute_profile"),
               "nvidia_smi": {"before": nvidia_before, "after": nvidia_after},
               **isolated_extra},
        **report_kwargs,
    )
    print(json.dumps({"acceptance_gate_passed": result["acceptance_gate_passed"], "vram_blocker": result.get("vram_blocker")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
