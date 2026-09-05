"""Validate and smoke-test MX-05 AMUSE-X-Hybrid3D without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.smoke_method_x_backbone import smoke
from src.training.method_x_autosupervisor import enforce_autonomous_supervision, evaluate_autonomous_supervision
from src.utils.io_helpers import atomic_write_json


EXPECTED_FEATURES = ["age", "sex", "education_years", "apoe4_carrier", "mmse_base"]


def validate_hybrid_readiness(*, root: Path, output_root: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    manifest = json.loads((output_root / "battery_manifest.json").read_text(encoding="utf-8"))
    protocol = manifest.get("locked_protocol") if isinstance(manifest.get("locked_protocol"), dict) else {}
    supervision = evaluate_autonomous_supervision(
        root=root,
        output_root=output_root,
        phase="hybrid",
        config={"resolution": protocol.get("resolution", 128), "folds": protocol.get("folds", 5), "seed": protocol.get("seed", 13), "hardware": {"hard_stop_vram_gb": 7.5}},
    )
    enforce_autonomous_supervision(root=root, output_root=output_root, report=supervision)
    blockers: list[str] = []
    if config.get("backbone_name") != "amuse_x_hybrid3d":
        blockers.append("HYBRID_BACKBONE_NAME_MISMATCH")
    if int(config.get("resolution", -1)) != 128 or int(config.get("folds", -1)) != 5 or int(config.get("seed", -1)) != 13:
        blockers.append("HYBRID_LOCKED_PROTOCOL_MISMATCH")
    if list(config.get("tabular_features", [])) != EXPECTED_FEATURES:
        blockers.append("HYBRID_TABULAR_FEATURE_ORDER_MISMATCH")
    if config.get("claim_boundary", {}).get("pet_input") is not False:
        blockers.append("HYBRID_PET_INPUT_BOUNDARY_FAILED")
    if len(config.get("ablation_order", [])) != 7:
        blockers.append("HYBRID_ABLATION_ORDER_INCOMPLETE")
    smoke_report = smoke("amuse_x_hybrid3d", "cuda", 128, float(config.get("smoke_vram_budget_gb", 6.5)))
    if smoke_report.get("status") != "PASS":
        blockers.append(f"HYBRID_SMOKE_{smoke_report.get('status', 'UNKNOWN')}")
    rows = manifest.get("models") if isinstance(manifest.get("models"), list) else []
    by_name = {str(row.get("backbone") or ""): row for row in rows if isinstance(row, dict)}
    effective = ["resnet10_3d", "resnet18_3d", "resnet34_3d", "densenet121_3d", "mednext_3d_s"]
    excluded = ["resnet50_3d", "efficientnet3d_b0"]
    terminal_training = {"COMPLETED", "COMPLETED_VALIDATED_REPAIRED", "COMPLETED_VALIDATED_RECONCILED"}
    baseline_ok = (
        set(by_name) == set(effective) | set(excluded)
        and all(str(by_name[name].get("status")) == "COMPLETED" for name in effective)
        and all(str((by_name[name].get("training") or {}).get("status") or "") in terminal_training for name in effective)
        and all(str(by_name[name].get("status")) == "PROTOCOL_EXCLUDED_NOT_STARTED" for name in excluded)
    )
    if not rows or not baseline_ok:
        blockers.append("HYBRID_BASELINE_BATTERY_NOT_FINALIZED")
    report = {
        "schema_version": 1,
        "stage": "MX-05",
        "status": "READY" if not blockers and supervision.get("decision") != "BLOCK" else "BLOCKED",
        "training_started": False,
        "smoke": smoke_report,
        "parameter_budget_max": config.get("parameter_budget_max"),
        "ablation_order": config.get("ablation_order"),
        "supervision_decision": supervision.get("decision"),
        "blockers": blockers + (["AUTOSUPERVISOR_BLOCKED"] if supervision.get("decision") == "BLOCK" else []),
        "next_automatic_action": "FINALIZE_MATCHED_BASELINE_THEN_RUN_LOCKED_ABLATIONS" if "HYBRID_BASELINE_BATTERY_NOT_FINALIZED" in blockers else "RUN_LOCKED_HYBRID_ABLATIONS",
    }
    atomic_write_json(output_root / "reports/mx05_hybrid_readiness.json", report)
    if report["status"] == "BLOCKED":
        atomic_write_json(output_root / "reports/BLOCKED_MX05_HYBRID.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/method_x_hybrid_v1.yaml")
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    report = validate_hybrid_readiness(root=ROOT, output_root=output_root, config_path=config_path)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
