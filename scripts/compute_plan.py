from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text


def main() -> int:
    gpu_plan_path = Path("configs/compute/gpu_execution_plan.yaml")
    gpu_plan = {}
    if gpu_plan_path.exists():
        try:
            import yaml

            gpu_plan = yaml.safe_load(gpu_plan_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            gpu_plan = {"load_error": exc.__class__.__name__}
    gpu = {"available": False, "name": None}
    try:
        import torch

        gpu["available"] = bool(torch.cuda.is_available())
        gpu["name"] = torch.cuda.get_device_name(0) if gpu["available"] else None
    except Exception as exc:
        gpu["error"] = exc.__class__.__name__
    plan = {
        "preferred_device": "cuda",
        "gpu": gpu,
        "cpu_fallback_allowed": True,
        "vram_policy": {"dryrun_guard_gb": 6.5, "training_soft_cap_gb": 7.0, "hard_stop_gb": 7.5},
        "decision": "GPU_READY" if gpu["available"] else "CPU_ONLY_NON_TRAINING_CHECKS",
        "execution_plan_config": str(gpu_plan_path),
        "method_x_mri": gpu_plan.get("method_x_mri", {}),
        "method_y_pet": gpu_plan.get("method_y_pet", {}),
        "next_gpu_actions": [
            "Build the official 96^3 MRI cache with --device auto after SPM_COMPLETE.",
            "Run final Method X 3D training with --device auto --amp and record peak_vram_gb.",
            "Keep Method Y PET execute/cache/training gated by dry-run review, MNI_TEMPLATE, and readiness status.",
        ],
    }
    atomic_write_json("results/tables/compute_plan.json", plan)
    atomic_write_text("results/tables/compute_plan.md", "# Compute Plan\n\n```json\n" + json.dumps(plan, indent=2) + "\n```\n")
    print(plan["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

