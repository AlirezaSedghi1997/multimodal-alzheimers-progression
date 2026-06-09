from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_pet_preprocess_resume(tracer: str, spatial_mode: str, project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or Path(".")
    path = root / "results" / "logs" / f"method_y_pet_preprocess_resume_{tracer}_{spatial_mode}.json"
    if not path.exists():
        return {"path": str(path), "exists": False, "items": {}, "status_counts": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    status_counts = Counter(v.get("status", "UNKNOWN") for v in data.get("items", {}).values())
    return {
        "path": str(path),
        "exists": True,
        "config": data.get("config", {}),
        "items": data.get("items", {}),
        "status_counts": dict(status_counts),
        "done_count": int(status_counts.get("PASS", 0) + status_counts.get("SKIPPED_EXISTS", 0)),
        "fail_count": int(status_counts.get("FAIL", 0) + status_counts.get("QC_FAIL", 0)),
    }


def summarize_pet_preprocess_progress(
    *,
    linked_work_items: int,
    tracer: str = "fbp",
    spatial_mode: str = "spm_pet_t1_mni",
    project_root: Path | None = None,
) -> dict[str, Any]:
    resume = load_pet_preprocess_resume(tracer, spatial_mode, project_root)
    done = resume["done_count"]
    pending = max(0, linked_work_items - done)
    return {
        "tracer": tracer,
        "spatial_mode": spatial_mode,
        "linked_work_items": linked_work_items,
        "resume_done": done,
        "resume_fail": resume["fail_count"],
        "resume_pending": pending,
        "resume_path": resume["path"],
        "resume_exists": resume["exists"],
        "status_counts": resume["status_counts"],
        "preprocessing_started": done > 0 or resume["fail_count"] > 0,
        "preprocessing_complete": linked_work_items > 0 and pending == 0 and resume["fail_count"] == 0,
    }
