#!/usr/bin/env python3
"""Orchestrate Method Y training battery Y1–Y10 with readiness and QC gates."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import atomic_write_json, atomic_write_text


GATE_BYPASS_IDS = frozenset({"Y7", "Y8", "Y10"})


def _gate_blockers_for_row(row_id: str, gate_blockers: list[str]) -> list[str]:
    if row_id in GATE_BYPASS_IDS:
        return [b for b in gate_blockers if not b.startswith("READINESS_")]
    return gate_blockers


BATTERY_ROWS = [
    {
        "id": "Y1",
        "title": "PET-only baseline",
        "script": "scripts/train_pet_method_y.py",
        "args": ["--tracer", "fbp", "--device", "auto", "--amp"],
        "requires_gpu": True,
        "reporting": "Supplementary or Methods; do not headline PET-only AUC (tautology caveat).",
    },
    {
        "id": "Y2",
        "title": "MRI-only on paired subset",
        "script": "scripts/train_mri_method_y_paired.py",
        "args": ["--device", "auto", "--amp"],
        "requires_gpu": True,
        "note": "Same grouped splits as Y1; paired hash list required.",
    },
    {
        "id": "Y3",
        "title": "Late fusion MRI+PET",
        "script": None,
        "note": "Implement late-fusion trainer on paired PET+MRI cache.",
    },
    {
        "id": "Y4",
        "title": "Gated feature fusion MRI+PET",
        "script": None,
        "note": "Implement gated fusion trainer (Method Y branch).",
    },
    {
        "id": "Y5",
        "title": "Early channel concat MRI+PET (ablation)",
        "script": None,
        "note": "Ablation row only; never primary claim path.",
    },
    {
        "id": "Y6",
        "title": "Cross-attention + modality dropout",
        "script": None,
        "note": "Requires multimodal trainer with cross-attention head.",
    },
    {
        "id": "Y7",
        "title": "Temporal-gap sensitivity (120/90/60-day)",
        "script": "scripts/run_headline_temporal_gap_evaluation.py",
        "args": [],
        "requires_gpu": False,
    },
    {
        "id": "Y8",
        "title": "Tracer sensitivity (FBP primary; FTP subset)",
        "script": "scripts/preprocess_pet_spatial.py",
        "args": ["--tracer", "ftp", "--tier", "primary_feasible_90", "--dry_run"],
        "requires_gpu": False,
        "note": "FTP preprocess/training extension; start with dry-run inventory.",
    },
    {
        "id": "Y9",
        "title": "Missing-PET simulation / modality dropout at inference",
        "script": None,
        "note": "Evaluate fusion model with PET encoder dropped at inference.",
    },
    {
        "id": "Y10",
        "title": "Label reproducibility (centiloid vs visual-read discordance)",
        "script": "scripts/run_cohort_label_coupling_audit.py",
        "args": [],
        "requires_gpu": False,
    },
]


def _run_step(python: str, script: str, args: list[str]) -> dict:
    cmd = [python, script, *args]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "status": "PASS" if proc.returncode == 0 else "FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="Run available battery scripts (respects readiness).")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--ids", default="", help="Comma-separated battery ids to run (default all available).")
    ap.add_argument("--out_json", default="results/tables/method_y_battery_status.json")
    ap.add_argument("--out_md", default="results/tables/method_y_battery_status.md")
    args = ap.parse_args()

    readiness_path = PROJECT_ROOT / "results/tables/method_y_training_readiness.json"
    audit_path = PROJECT_ROOT / "results/tables/preprocess_quality_audit.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8")) if readiness_path.exists() else {}
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    visual_manifest_path = PROJECT_ROOT / "results/qc/visual_review_manifest.json"
    visual = json.loads(visual_manifest_path.read_text(encoding="utf-8")) if visual_manifest_path.exists() else {}

    gate_blockers: list[str] = []
    if readiness.get("status") != "READY_FOR_EXPLORATORY_METHOD_Y_TRAINING":
        gate_blockers.append(f"READINESS_{readiness.get('status', 'UNKNOWN')}")
    if audit.get("gate_verdict", {}).get("training_ready_after_audit") is False:
        gate_blockers.extend(audit.get("gate_verdict", {}).get("blockers") or [])
    if visual and not visual.get("all_signed_off", False):
        gate_blockers.append("VISUAL_QC_SIGNOFF_INCOMPLETE")

    selected = {x.strip() for x in args.ids.split(",") if x.strip()} if args.ids else None
    rows_out = []
    for row in BATTERY_ROWS:
        if selected and row["id"] not in selected:
            continue
        entry = dict(row)
        script = row.get("script")
        if script is None:
            entry["status"] = "SCRIPT_PENDING"
        elif not (PROJECT_ROOT / script).exists():
            entry["status"] = "SCRIPT_MISSING"
        elif not args.execute:
            entry["status"] = "READY_TO_RUN"
        elif gate_blockers:
            row_blockers = _gate_blockers_for_row(row["id"], gate_blockers)
            if row_blockers:
                entry["status"] = "BLOCKED_BY_GATE"
                entry["gate_blockers"] = row_blockers
            else:
                entry["run"] = _run_step(args.python, script, list(row.get("args") or []))
                entry["status"] = entry["run"]["status"]
        else:
            entry["run"] = _run_step(args.python, script, list(row.get("args") or []))
            entry["status"] = entry["run"]["status"]
        rows_out.append(entry)

    report = {
        "stage": "method_y_battery",
        "version": "v1.0-orchestrator",
        "execute_mode": bool(args.execute),
        "readiness_status": readiness.get("status"),
        "preprocess_audit_ready": audit.get("gate_verdict", {}).get("training_ready_after_audit"),
        "visual_qc_all_signed": visual.get("all_signed_off"),
        "gate_blockers": gate_blockers,
        "battery": rows_out,
        "scientific_reporting_rules": {
            "headline_incremental_value_metric": "MRI incremental value over PET-only on paired subset (ΔAUC, DCA net benefit)",
            "do_not_headline": ["Y1 PET-only AUC as primary finding", "co-primary without N=266 and 232/34 imbalance caveats"],
            "abstract_method_y_framing": "co-primary exploratory Results subsection with explicit small-N and class-imbalance limitations",
        },
    }
    out_json = PROJECT_ROOT / args.out_json
    out_md = PROJECT_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_json, report)
    md_lines = [
        "# Method Y Battery Status",
        "",
        f"- Execute mode: `{report['execute_mode']}`",
        f"- Readiness: `{report['readiness_status']}`",
        f"- Gate blockers: `{gate_blockers}`",
        "",
        "| id | title | status | script |",
        "|---|---|---|---|",
    ]
    for row in rows_out:
        md_lines.append(f"| {row['id']} | {row['title']} | {row['status']} | `{row.get('script')}` |")
    atomic_write_text(out_md, "\n".join(md_lines) + "\n")
    executed = [r for r in rows_out if r.get("run")]
    failed = [r for r in executed if r.get("status") != "PASS"]
    print(json.dumps({"status": "OK", "gate_blockers": gate_blockers, "battery_rows": len(rows_out), "failed": len(failed)}, indent=2))
    if args.execute and failed:
        return 1
    if args.execute and gate_blockers and not executed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
