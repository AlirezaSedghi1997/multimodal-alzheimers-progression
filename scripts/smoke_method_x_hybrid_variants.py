"""Run the non-training AMUSE-X-Hybrid3D variant gate.

This command only constructs each pre-registered architecture and performs a
single CUDA forward at the locked resolution.  It never reads the MRI cache,
starts a trainer, or creates a checkpoint.  A missing/failed variant is a
fail-closed pre-training blocker; the report is bound to the versioned battery
root so a later run cannot be mistaken for this gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.smoke_method_x_backbone import smoke
from src.utils.io_helpers import atomic_write_json

VARIANTS = (
    "baseline",
    "plus_dense_skip",
    "plus_large_kernel",
    "plus_attention",
    "plus_a3",
    "plus_a4",
    "full_model",
    "amuse_x_hybrid_lite",
)
HYBRID_CONFIG = ROOT / "configs/experiments/method_x_hybrid_v1.yaml"
SMOKE_SOURCE_PATHS = (
    ROOT / "scripts/smoke_method_x_backbone.py",
    Path(__file__).resolve(),
    ROOT / "src/models/backbones/lightweight_3d.py",
    ROOT / "src/models/backbones/backbone_factory.py",
    ROOT / "src/models/backbones/amuse_x_hybrid3d.py",
    ROOT / "src/models/fusion/bottleneck_cross_attention.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _smoke_binding() -> dict[str, Any]:
    sources = {
        path.resolve().relative_to(ROOT.resolve()).as_posix(): _sha256_file(path)
        for path in SMOKE_SOURCE_PATHS
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "hybrid_config_path": HYBRID_CONFIG.relative_to(ROOT).as_posix(),
        "hybrid_config_sha256": _sha256_file(HYBRID_CONFIG),
        "implementation_sources": sources,
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _validated_cached_row(
    row: Any,
    *,
    variant: str,
    hard_stop_vram_gb: float,
) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    try:
        from scripts.run_method_x_hybrid_orchestrator import (
            validate_hybrid_smoke_report,
        )

        blockers = validate_hybrid_smoke_report(
            row,
            expected_variant=variant,
            parameter_budget_max=20_000_000,
            vram_budget_gb=float(hard_stop_vram_gb),
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        return None
    return dict(row) if not blockers else None


def _read_state(path: Path, *, binding: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "smoke_binding": binding, "rows": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        value = None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("smoke_binding") != binding
        or not isinstance(value.get("rows"), dict)
    ):
        return {"schema_version": 1, "smoke_binding": binding, "rows": {}}
    return value


def run_gate(
    *,
    output_root: Path,
    device: str = "cuda",
    resolution: int = 128,
    hard_stop_vram_gb: float = 6.5,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    binding = _smoke_binding()
    report_path = output_root / "reports" / "amuse_x_hybrid_smoke.json"
    state_path = output_root / "reports" / "amuse_x_hybrid_smoke_state.json"
    if report_path.is_file():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("schema_version") == 2
            and existing.get("status") == "PASS"
            and existing.get("smoke_binding") == binding
            and existing.get("output_root") == str(output_root)
            and existing.get("device") == str(device)
            and existing.get("resolution") == int(resolution)
            and float(existing.get("hard_stop_vram_gb", -1))
            == float(hard_stop_vram_gb)
            and isinstance(existing.get("variants"), list)
            and len(existing["variants"]) == len(VARIANTS)
            and all(
                _validated_cached_row(
                    row,
                    variant=variant,
                    hard_stop_vram_gb=hard_stop_vram_gb,
                )
                is not None
                for variant, row in zip(VARIANTS, existing["variants"], strict=True)
            )
        ):
            return {**existing, "reused": True}
    state = _read_state(state_path, binding=binding)
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        result = _validated_cached_row(
            state["rows"].get(variant),
            variant=variant,
            hard_stop_vram_gb=hard_stop_vram_gb,
        )
        if result is None:
            try:
                result = smoke(
                    "amuse_x_hybrid3d",
                    device,
                    int(resolution),
                    float(hard_stop_vram_gb),
                    hybrid_variant=variant,
                )
            except (ImportError, RuntimeError, TypeError, ValueError) as exc:
                result = {
                    "status": "BLOCKED_SMOKE_FAILED",
                    "backbone": "amuse_x_hybrid3d",
                    "hybrid_variant": variant,
                    "blocker": f"{type(exc).__name__}:{exc}",
                }
            state["rows"][variant] = result
            atomic_write_json(state_path, state)
        rows.append(result)
    blockers = [
        f"{row.get('hybrid_variant', 'unknown')}:{row.get('status', 'UNKNOWN')}"
        for row in rows
        if row.get("status") != "PASS"
    ]
    report = {
        "schema_version": 2,
        "stage": "MX05_HYBRID_FULL_CLASSIFIER_SMOKE",
        "status": "PASS" if not blockers else "BLOCKED",
        "training_started": False,
        "device": str(device),
        "resolution": int(resolution),
        "hard_stop_vram_gb": float(hard_stop_vram_gb),
        "variants": rows,
        "expected_variants": list(VARIANTS),
        "smoke_binding": binding,
        "blockers": blockers,
        "autonomous_supervision": {
            "mode": "AUTONOMOUS_MULTI_LAYER_FAIL_CLOSED",
            "human_approval_required": False,
            "manual_override_allowed": False,
            "fail_closed": True,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "next_action": (
            "RUN_MX05_HYBRID_THROUGHPUT_PROFILE"
            if not blockers
            else "REPAIR_HYBRID_SMOKE_BLOCKER"
        ),
    }
    atomic_write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--hard-stop-vram-gb", type=float, default=6.5)
    args = parser.parse_args()
    try:
        report = run_gate(
            output_root=args.output_root if args.output_root.is_absolute() else ROOT / args.output_root,
            device=args.device,
            resolution=args.resolution,
            hard_stop_vram_gb=args.hard_stop_vram_gb,
        )
    except KeyboardInterrupt:
        print(json.dumps({"status": "PAUSED_RESUMABLE"}, indent=2), file=sys.stderr)
        return 75
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "PASS" else 78


if __name__ == "__main__":
    raise SystemExit(main())
