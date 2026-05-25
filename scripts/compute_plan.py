from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text


def main() -> int:
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
    }
    atomic_write_json("results/tables/compute_plan.json", plan)
    atomic_write_text("results/tables/compute_plan.md", "# Compute Plan\n\n```json\n" + json.dumps(plan, indent=2) + "\n```\n")
    print(plan["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

