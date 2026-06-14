#!/usr/bin/env python3
"""Read-only Method Y PET preprocess progress snapshot (does not modify resume log)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.pet_preprocess_progress import summarize_pet_preprocess_progress


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume_log", default="results/logs/method_y_pet_preprocess_resume_fbp_spm_pet_t1_mni.json")
    ap.add_argument("--linked_n", type=int, default=266)
    ap.add_argument("--out_json", default="results/tables/method_y_pet_preprocess_progress_snapshot.json")
    ap.add_argument("--out_md", default="results/tables/method_y_pet_preprocess_progress_snapshot.md")
    args = ap.parse_args()

    resume_path = PROJECT_ROOT / args.resume_log
    resume = json.loads(resume_path.read_text(encoding="utf-8")) if resume_path.exists() else {}
    progress = summarize_pet_preprocess_progress(linked_work_items=args.linked_n)

    report = {
        "stage": "method_y_pet_preprocess_progress_snapshot",
        "status": "COMPLETED" if progress.get("preprocessing_complete") else "IN_PROGRESS",
        "read_only": True,
        "linked_work_items": args.linked_n,
        **progress,
        "note": "Snapshot only; does not alter resume log or re-run preprocess.",
    }
    atomic_write_json(PROJECT_ROOT / args.out_json, report)
    atomic_write_text(
        PROJECT_ROOT / args.out_md,
        f"# PET Preprocess Progress Snapshot\n\n- Done: {progress.get('resume_done', 0)}/{args.linked_n}\n"
        f"- Pending: {progress.get('resume_pending', 0)}\n- FAIL: {progress.get('resume_fail', 0)}\n",
    )
    print(json.dumps({"status": report["status"], "done": progress.get("resume_done"), "pending": progress.get("resume_pending")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
