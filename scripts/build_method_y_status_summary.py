from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.state_manager import StateManager


def load(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"status": "MISSING"}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    readiness = load("results/tables/method_y_training_readiness.json")
    design = load("results/tables/method_y_multimodal_design.json")
    inventory = load("results/tables/method_y_pet_inventory.json")
    phase_y = load("results/final_results/method_y/phase_y_summary.json")

    summary = {
        "schema_version": 1,
        "branch": "method_y_pet_assisted",
        "status": readiness.get("status", "MISSING"),
        "design_status": design.get("status", "MISSING"),
        "inventory_status": inventory.get("status", "MISSING"),
        "phase_y_status": phase_y.get("status", "MISSING"),
        "method_x_claim_unchanged": readiness.get("method_x_claim_unchanged", True),
        "pet_as_method_x_input": False,
        "clinical_alzheimer_diagnosis_allowed": readiness.get("clinical_alzheimer_diagnosis_allowed", False),
        "blockers": readiness.get("blockers", []),
        "warnings": readiness.get("warnings", []),
        "required_before_training": readiness.get("required_before_training", []),
        "claim_boundary": (
            "Exploratory PET-assisted branch (Method Y). PET never enters Method X-Strict. "
            "Training has not started; design and inventory artifacts only."
        ),
    }

    out_json = Path("docs/method_y_status_summary.json")
    out_md = Path("docs/method_y_status_summary.md")
    atomic_write_json(out_json, summary)

    lines = [
        "# Method Y PET-Assisted Branch — Status Summary\n\n",
        f"- training readiness: `{summary['status']}`\n",
        f"- multimodal design: `{summary['design_status']}`\n",
        f"- PET inventory: `{summary['inventory_status']}`\n",
        f"- phase Y summary: `{summary['phase_y_status']}`\n",
        f"- method_x_claim_unchanged: `{summary['method_x_claim_unchanged']}`\n",
        f"- pet_as_method_x_input: `{summary['pet_as_method_x_input']}`\n\n",
        "## Claim boundary\n\n",
        f"{summary['claim_boundary']}\n\n",
        "## Blockers\n\n",
    ]
    for blocker in summary["blockers"]:
        lines.append(f"- `{blocker}`\n")
    lines.extend(["\n", "## Warnings\n\n"])
    for warning in summary["warnings"]:
        lines.append(f"- `{warning}`\n")
    lines.extend(["\n", "## Required before training\n\n"])
    for item in summary["required_before_training"]:
        lines.append(f"- {item}\n")
    atomic_write_text(out_md, "".join(lines))
    StateManager().append_event("method_y_status_summary", "COMPLETED", "Method Y status summary written to docs/.")
    print(json.dumps({"method_y_status_summary": "COMPLETED", "status": summary["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
