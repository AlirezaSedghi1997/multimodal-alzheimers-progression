#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.pet_spatial_qc import atomic_write_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "results/tables/method_y_gap_filter_report.json",
    "results/tables/method_y_pet_manifest_summary.json",
    "results/tables/method_y_pet_preprocess_summary.json",
    "results/tables/method_y_pet_cache_summary.json",
    "results/tables/method_y_training_readiness.json",
]


def read(path):
    p = PROJECT_ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    lines = ["# Method Y Phase 2 Summary\n\n"]
    for rel in FILES:
        obj = read(rel)
        lines.append(f"## {rel}\n\n")
        if obj is None:
            lines.append("- Missing\n\n")
        else:
            status = obj.get("status") or obj.get("verdict") or obj.get("stage")
            lines.append(f"- Status/verdict: `{status}`\n")
            for key in ["rows", "n_candidates", "selected_work_items", "candidate_count", "processed_items", "tracer"]:
                if key in obj:
                    lines.append(f"- {key}: `{obj[key]}`\n")
            if "blockers" in obj:
                lines.append(f"- blockers: `{obj['blockers']}`\n")
            lines.append("\n")
    out = PROJECT_ROOT / "results" / "final_results" / "method_y" / "phase_y_summary.md"
    atomic_write_text(out, "".join(lines))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
