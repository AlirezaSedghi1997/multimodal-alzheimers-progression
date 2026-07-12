#!/usr/bin/env python3
"""Stage 0 baseline and sanity workflow for Method X 3D MRI.

Default mode writes a NOT_RUN plan. Pass --execute for explicit smoke runs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import atomic_write_json


def output_name(name: str, suffix: str = "") -> str:
    suffix = suffix.strip()
    if not suffix:
        return f"stage0_{name}"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in suffix)
    return f"stage0_{name}_{safe}"


def command(args: argparse.Namespace, name: str, extra: list[str]) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_3d_mri.py"),
        "--cache",
        args.cache,
        "--splits",
        args.splits,
        "--out_dir",
        f"results/redesign_stage_runs/{output_name(name, args.out_suffix)}",
        "--phase",
        "dev",
        "--resolution",
        str(args.resolution),
        "--folds",
        str(args.folds),
        "--epochs",
        str(args.epochs),
        "--limit",
        str(args.limit),
        "--backbone",
        args.backbone,
        "--fusion",
        args.fusion,
        "--scheduler",
        "cosine",
        "--label_smoothing",
        "0.0",
        "--latent_alignment_lambda",
        "0.0",
        "--synib_lambda",
        "0.0",
        "--no_supplementary_train",
        "--device",
        args.device,
        *extra,
    ]


def run_one(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60 * 30,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def parse_metrics(out_dir: Path) -> dict[str, Any]:
    metrics_path = out_dir / "three_d_metrics.json"
    if not metrics_path.exists():
        return {"status": "NOT_FOUND", "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT))}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "INVALID_JSON",
            "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
            "error": str(exc),
        }
    aggregate = payload.get("aggregate", {})
    return {
        "status": aggregate.get("status", "UNKNOWN"),
        "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
        "mode": aggregate.get("mode"),
        "roc_auc": aggregate.get("roc_auc"),
        "pr_auc": aggregate.get("pr_auc"),
        "brier": aggregate.get("brier"),
        "temperature_scaling_status": (aggregate.get("temperature_scaling") or {}).get("status"),
        "final_claim_allowed": aggregate.get("final_claim_allowed"),
        "uses_pet_as_input": aggregate.get("uses_pet_as_input"),
        "input_mode": aggregate.get("input_mode"),
        "blocker": aggregate.get("blocker"),
    }


def execution_status(result: dict[str, Any], metrics: dict[str, Any]) -> str:
    if int(result.get("returncode", 1)) != 0:
        return "FAILED_OR_BLOCKED"
    if metrics.get("status") in {"BLOCKED", "NOT_FOUND", "INVALID_JSON"}:
        return "FAILED_OR_BLOCKED"
    return "COMPLETED_REAL_SMOKE"


def sanity_outcome(
    name: str,
    metrics: dict[str, Any],
    *,
    auc_threshold: float = 0.55,
    brier_threshold: float | None = None,
) -> dict[str, Any]:
    if name == "baseline":
        return {"status": "BASELINE_SMOKE_ONLY"}
    if metrics.get("roc_auc") is None and metrics.get("pr_auc") is None:
        return {"status": "NOT_ASSESSABLE", "reason": "ranking_metrics_missing_or_single_class"}
    suspicious = False
    reasons: list[str] = []
    roc = metrics.get("roc_auc")
    if isinstance(roc, (int, float)) and float(roc) > auc_threshold:
        suspicious = True
        reasons.append(f"roc_auc_above_{auc_threshold:.2f}")
    brier_value = metrics.get("brier")
    if (
        brier_threshold is not None
        and isinstance(brier_value, (int, float))
        and float(brier_value) <= brier_threshold
    ):
        suspicious = True
        reasons.append(f"brier_at_or_below_{brier_threshold:.2f}")
    if suspicious:
        return {"status": "SANITY_FAILURE_POSSIBLE_LEAKAGE_OR_TABULAR_SIGNAL", "reasons": reasons}
    return {"status": "SANITY_BEHAVIOR_NOT_OBVIOUSLY_SUSPICIOUS"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_128.h5")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--backbone", default="tiny")
    ap.add_argument("--fusion", default="concat")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--baseline_input_mode", default="mri_plus_tabular")
    ap.add_argument("--random_labels_input_mode", default="mri_plus_tabular")
    ap.add_argument("--random_noise_input_mode", default="mri_only")
    ap.add_argument("--sanity_auc_threshold", type=float, default=0.55)
    ap.add_argument("--sanity_brier_threshold", type=float, default=None)
    ap.add_argument(
        "--out_suffix",
        default="",
        help="Optional suffix for stage0 output directories and summary file to avoid overwriting prior smoke artifacts.",
    )
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    runs = [
        ("baseline", ["--input_mode", args.baseline_input_mode]),
        ("random_labels", ["--sanity_random_labels", "--input_mode", args.random_labels_input_mode]),
        ("random_noise_images", ["--sanity_random_noise_images", "--input_mode", args.random_noise_input_mode]),
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "stage0_baseline_reproducibility_and_sanity",
        "config": {
            "folds": args.folds,
            "epochs": args.epochs,
            "limit": args.limit,
            "backbone": args.backbone,
            "fusion": args.fusion,
            "sanity_auc_threshold": args.sanity_auc_threshold,
            "sanity_brier_threshold": args.sanity_brier_threshold,
            "out_suffix": args.out_suffix,
        },
        "method_x_pet_free": True,
        "claim_target": "amyloid_biomarker_status_preclinical_risk_only",
        "final_reporting_mode": False,
        "runs": {},
    }
    for name, extra in runs:
        cmd = command(args, name, extra)
        summary["runs"][name] = {
            "status": "NOT_RUN",
            "reason": "default_planning_mode_requires_explicit_execute",
            "command": cmd,
            "expected_behavior": "near_chance_or_failure" if extra else "baseline_smoke_only_no_final_claim",
            "train_labels_randomized_only": name == "random_labels",
            "validation_and_test_labels_preserved": name == "random_labels",
            "tabular_signal_disabled": name == "random_noise_images",
            "test_labels_used_for_selection": False,
            "pet_input_allowed": False,
            "out_dir": f"results/redesign_stage_runs/{output_name(name, args.out_suffix)}",
        }
        if args.execute:
            out_dir = PROJECT_ROOT / "results" / "redesign_stage_runs" / output_name(name, args.out_suffix)
            result = run_one(cmd)
            metrics = parse_metrics(out_dir)
            summary["runs"][name].update(
                {
                    "status": execution_status(result, metrics),
                    "reason": None,
                    "metrics": metrics,
                    "sanity_outcome": sanity_outcome(
                        name,
                        metrics,
                        auc_threshold=float(args.sanity_auc_threshold),
                        brier_threshold=args.sanity_brier_threshold,
                    ),
                    "result": result,
                }
            )
    if not args.execute:
        summary["status"] = "NOT_RUN"
        summary["blocker"] = "Execution requires explicit --execute to avoid accidental heavy training."
    else:
        if any(row["status"] == "FAILED_OR_BLOCKED" for row in summary["runs"].values()):
            summary["status"] = "FAILED_OR_BLOCKED"
        else:
            summary["status"] = "COMPLETED_REAL_SMOKE"
    summary_name = "stage0_baseline_sanity_summary"
    if args.out_suffix.strip():
        summary_name = f"{summary_name}_{output_name('', args.out_suffix).removeprefix('stage0__').removeprefix('stage0_')}"
    atomic_write_json(PROJECT_ROOT / "results" / "redesign_stage_runs" / f"{summary_name}.json", summary)
    print(json.dumps({"status": summary["status"], "execute": args.execute}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
