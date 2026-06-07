from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATE = Path("results/governance/label_freeze_template.json")
DEFAULT_OUT = Path("results/governance/label_freeze.json")
PLACEHOLDER = "[INSERT PRESPECIFIED CUTOFF BEFORE MODELING]"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("cutoff must be positive")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the primary centiloid threshold before modeling.")
    parser.add_argument("--cutoff", type=positive_float, required=True)
    parser.add_argument("--source_rationale", required=True)
    parser.add_argument("--approved_by", required=True)
    parser.add_argument("--date_frozen", default=date.today().isoformat())
    parser.add_argument("--sensitivity_label", default="VA / visual-read consensus label")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    template_path = Path(args.template)
    if not template_path.exists():
        raise FileNotFoundError(f"Label-freeze template not found: {template_path}")
    template = read_json(template_path)
    existing_path = Path(args.out)
    if existing_path.exists():
        existing = read_json(existing_path)
        existing_cutoff = existing.get("centiloid_threshold")
        if existing_cutoff != args.cutoff:
            raise RuntimeError(
                "Refusing to change frozen centiloid cutoff. "
                f"Existing={existing_cutoff!r}, requested={args.cutoff!r}."
            )

    freeze = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_MODELING",
        "primary_binary_label": "centiloid_threshold",
        "centiloid_threshold": args.cutoff,
        "cutoff_value": args.cutoff,
        "source_rationale": args.source_rationale,
        "date_frozen": args.date_frozen,
        "approved_by": args.approved_by,
        "sensitivity_label": args.sensitivity_label,
        "change_policy": template.get(
            "change_policy",
            "No change after modeling starts, model selection, performance inspection, calibration, or threshold selection.",
        ),
        "previous_placeholder": PLACEHOLDER,
        "modeling_started_before_freeze": False,
        "pet_as_input": False,
        "raw_identifiers_written": False,
        "raw_paths_written": False,
    }
    write_json(existing_path, freeze)
    md = [
        "# Label Freeze\n\n",
        f"- status: `{freeze['status']}`\n",
        f"- primary_binary_label: `{freeze['primary_binary_label']}`\n",
        f"- centiloid_threshold: `{freeze['centiloid_threshold']}`\n",
        f"- sensitivity_label: `{freeze['sensitivity_label']}`\n",
        f"- date_frozen: `{freeze['date_frozen']}`\n",
        f"- approved_by: `{freeze['approved_by']}`\n",
        f"- change_policy: {freeze['change_policy']}\n",
        "- PET as input: `False`\n",
        "- raw identifiers written: `False`\n",
        "- raw paths written: `False`\n",
    ]
    write_text(existing_path.with_suffix(".md"), "".join(md))
    print(json.dumps({"status": freeze["status"], "centiloid_threshold": args.cutoff}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
