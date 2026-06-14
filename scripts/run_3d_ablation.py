from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


VARIANTS = [
    {"name": "concat", "fusion": "concat", "latent_alignment_lambda": 0.0},
    {"name": "cross_attention", "fusion": "cross_attention", "latent_alignment_lambda": 0.0},
    {
        "name": "cross_attention_latent_alignment_0_15",
        "fusion": "cross_attention",
        "latent_alignment_lambda": 0.15,
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def variant_metrics_path(base_out_dir: Path, name: str) -> Path:
    return base_out_dir / name / "three_d_metrics.json"


def run_variant(args: argparse.Namespace, variant: dict[str, object]) -> dict[str, Any]:
    name = str(variant["name"])
    metrics_path = variant_metrics_path(Path(args.base_out_dir), name)
    if metrics_path.exists() and not args.force:
        obj = load_json(metrics_path)
        obj["resumed_from_existing"] = True
        return obj

    out_dir = metrics_path.parent
    cmd = [
        sys.executable,
        "scripts/train_3d_mri.py",
        "--cache",
        args.cache,
        "--splits",
        args.splits,
        "--out_dir",
        str(out_dir),
        "--phase",
        "final",
        "--resolution",
        "96",
        "--amp",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--folds",
        str(args.folds),
        "--fusion",
        str(variant["fusion"]),
        "--latent_alignment_lambda",
        str(variant["latent_alignment_lambda"]),
    ]
    if args.force:
        cmd.append("--no_resume")
    code = subprocess.call(cmd)
    if code != 0:
        raise RuntimeError(f"3D ablation variant failed: {name}")
    obj = load_json(metrics_path)
    obj["resumed_from_existing"] = False
    return obj


def _variant_label(agg: dict[str, Any]) -> str:
    if agg.get("ablation_variant"):
        return str(agg["ablation_variant"])
    fusion = agg.get("fusion")
    lam = float(agg.get("latent_alignment_lambda", 0.0) or 0.0)
    for spec in VARIANTS:
        if spec["fusion"] == fusion and float(spec["latent_alignment_lambda"]) == lam:
            return str(spec["name"])
    checkpoints = agg.get("checkpoints") or []
    if checkpoints:
        return Path(str(checkpoints[0])).parent.name
    return f"{fusion or 'unknown'}_{lam}"


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for obj in results:
        agg = obj["aggregate"]
        rows.append(
            {
                "variant": _variant_label(agg),
                "mode": agg["mode"],
                "folds": agg["folds"],
                "cache_resolution": agg.get("cache_resolution"),
                "roc_auc": agg.get("roc_auc"),
                "pr_auc": agg.get("pr_auc"),
                "brier": agg.get("brier"),
                "external_like_roc_auc": (
                    agg["external_like"].get("roc_auc") if agg.get("external_like") else None
                ),
                "uses_pet_as_input": agg["uses_pet_as_input"],
                "uses_raw_identifiers": agg["uses_raw_identifiers"],
                "final_claim_allowed": agg["final_claim_allowed"],
                "phase": agg.get("phase"),
                "requested_resolution": agg.get("requested_resolution"),
            }
        )
    best = max(
        (row for row in rows if row["roc_auc"] is not None),
        key=lambda row: float(row["roc_auc"]),
        default=None,
    )
    phase_mismatch = []
    for obj in results:
        agg = obj.get("aggregate", {})
        variant_name = _variant_label(agg)
        if agg.get("phase") != "final" or agg.get("requested_resolution") != 96:
            if variant_name:
                phase_mismatch.append(variant_name)
    return {
        "schema_version": 1,
        "status": "COMPLETED",
        "variants": rows,
        "best_by_cv_roc_auc": best,
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "public_output_granularity": "aggregate_variant_level",
        "training_contract": {
            "required_phase": "final",
            "required_resolution": 96,
            "variants_with_dev_mode_artifact": sorted(set(phase_mismatch)),
            "note": "Re-run with --force if any variant was trained under dev/64 metadata.",
        },
    }


def write_summary(summary: dict[str, Any]) -> None:
    out_json = Path("results/final_results/three_d_ablation/three_d_ablation_summary.json")
    out_md = Path("results/final_results/three_d_ablation/three_d_ablation_summary.md")
    atomic_write_json(out_json, summary)
    lines = [
        "# 3D Fusion Ablation Summary\n\n",
        "| variant | mode | folds | resolution | roc_auc | pr_auc | brier | external_like_roc_auc |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary["variants"]:
        lines.append(
            f"| {row['variant']} | {row['mode']} | {row['folds']} | {row['cache_resolution']} | "
            f"{row['roc_auc']} | {row['pr_auc']} | {row['brier']} | {row['external_like_roc_auc']} |\n"
        )
    lines.extend(
        [
            "\n",
            f"- best_by_cv_roc_auc: `{summary['best_by_cv_roc_auc']}`\n",
            "- contains_pet_input: `False`\n",
            "- contains_raw_identifiers: `False`\n",
            "- public_output_granularity: `aggregate_variant_level`\n",
        ]
    )
    atomic_write_text(out_md, "".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_96.h5")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--base_out_dir", default="results/final_results/three_d_ablation")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--force", action="store_true", help="Re-run all selected variants even if metrics exist.")
    ap.add_argument(
        "--variants",
        default="",
        help="Comma-separated subset of variant folder names to run (default: all).",
    )
    args = ap.parse_args()

    selected = {v.strip() for v in args.variants.split(",") if v.strip()}
    variants = [v for v in VARIANTS if not selected or str(v["name"]) in selected]
    if not variants:
        raise SystemExit("No ablation variants selected.")
    results = [run_variant(args, variant) for variant in variants]
    if not args.variants:
        summary = summarize(results)
    else:
        merged: dict[str, dict[str, Any]] = {}
        base = Path(args.base_out_dir)
        for spec in VARIANTS:
            name = str(spec["name"])
            path = variant_metrics_path(base, name)
            if path.exists():
                merged[name] = load_json(path)
        for obj in results:
            merged[_variant_label(obj["aggregate"])] = obj
        summary = summarize([merged[str(spec["name"])] for spec in VARIANTS if str(spec["name"]) in merged])
    write_summary(summary)
    write_run_manifest("results/logs/three_d_ablation_manifest.json", stage="three_d_ablation", data_root=None, extra=summary)
    StateManager().append_event("three_d_ablation", "COMPLETED", "3D fusion ablation summary generated.")
    print(json.dumps({"three_d_ablation": "COMPLETED", "best": summary["best_by_cv_roc_auc"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
