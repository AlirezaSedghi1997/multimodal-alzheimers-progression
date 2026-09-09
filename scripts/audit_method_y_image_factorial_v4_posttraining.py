from __future__ import annotations

"""Read-only, evidence-first audit of the completed Method Y image run.

The audit never resumes training and never changes a locked result.  It writes
only a versioned audit record/document, while all source artefacts are read and
hash-checked.  A technically finalized run is intentionally reported
separately from its scientific claim gate.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival.atomic_io import atomic_write_bytes, atomic_write_json
from src.method_y_revival.privacy import scan_public_bundle


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _active_method_y_processes() -> list[dict[str, Any]]:
    # Command-line filtering is intentional: an unrelated Python process must
    # never be treated as a writer for this project.
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -and "
        "($_.CommandLine -match 'method_y_image_factorial|method_y_revival_v4') -and "
        "($_.CommandLine -notmatch 'audit_method_y_image_factorial_v4_posttraining') } | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def _run_verifier(config_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts/verify_method_y_image_factorial_v4.py"),
        "--level",
        "final",
        "--config",
        str(config_path),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        parsed = {"status": "FAIL", "failures": ["VERIFIER_NON_JSON_OUTPUT"]}
    if not isinstance(parsed, dict):
        parsed = {"status": "FAIL", "failures": ["VERIFIER_INVALID_OUTPUT"]}
    parsed["process_returncode"] = result.returncode
    return parsed


def _selected_arm_summary(root: Path, selected: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for branch in ("amyloid", "tau"):
        arm = str(selected.get(branch, ""))
        rows = []
        for fold in range(5):
            path = root / "reports" / f"{branch}_{arm}_fold_{fold}_public.json"
            payload = _read(path, {})
            aggregate = payload.get("aggregate", {}) if isinstance(payload, dict) else {}
            if isinstance(aggregate, dict):
                rows.append(aggregate)
        if len(rows) != 5:
            summary[branch] = {"arm": arm, "status": "INCOMPLETE", "folds": len(rows)}
            continue
        def mean(key: str) -> float | None:
            values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
            return sum(values) / len(values) if len(values) == 5 else None
        summary[branch] = {
            "arm": arm,
            "folds": 5,
            "n_test": sum(int(row.get("n_test", 0)) for row in rows),
            "test_events": sum(int(row.get("test_events", 0)) for row in rows),
            "mean_integrated_brier_score": mean("integrated_brier_score"),
            "mean_uno_c_index": mean("uno_c_index"),
            "status": "COMPLETE",
        }
    return summary


def _scanner_summary(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for branch in ("amyloid", "tau"):
        path = root / "analysis_pairs" / "A3_vs_A4" / "reports" / f"{branch}_scanner_comparison.json"
        payload = _read(path, {})
        reference = payload.get("reference", {}) if isinstance(payload, dict) else {}
        eligibility = reference.get("leave_manufacturer_out_eligibility", {}) if isinstance(reference, dict) else {}
        lmo = reference.get("leave_manufacturer_out_performance", {}) if isinstance(reference, dict) else {}
        result[branch] = {
            "status": payload.get("status"),
            "scanner_probe_not_worse": payload.get("scanner_probe_not_worse"),
            "leave_manufacturer_out_status": lmo.get("status") if isinstance(lmo, dict) else None,
            "eligible_manufacturers": eligibility.get("eligible_manufacturers", []) if isinstance(eligibility, dict) else [],
            "not_estimable_manufacturers": eligibility.get("not_estimable", []) if isinstance(eligibility, dict) else [],
        }
    return result


def audit(root: Path, config_path: Path) -> dict[str, Any]:
    state_path = root / "orchestration_state.json"
    contract_path = root / "run_contract.json"
    state = _read(state_path, {})
    contract = _read(contract_path, {})
    xai = _read(root / "attribution/xai_science_gate_v4.json", {})
    post_eval = _read(root / "post_eval/post_eval_report.json", {})
    bundle = root / "public_bundle"
    privacy_findings = scan_public_bundle(bundle) if bundle.is_dir() else [
        {"rule": "PUBLIC_BUNDLE_MISSING", "path": "public_bundle"}
    ]
    verifier = _run_verifier(config_path)
    active = _active_method_y_processes()

    outer_files = sorted((root / "private/outer_evaluation").rglob("outer_result.json"))
    valid_outer_hashes = 0
    for path in outer_files:
        payload = _read(path, {})
        if isinstance(payload, dict) and _is_sha256(payload.get("result_record_sha256")) and _is_sha256(payload.get("metrics_sha256")):
            valid_outer_hashes += 1

    selected = state.get("inner_only_selected_arms", {})
    selected_summary = _selected_arm_summary(root, selected)
    scanner_summary = _scanner_summary(root)
    technical_complete = verifier.get("status") == "PASS" and not active
    science_status = xai.get("science_gate_status", "NOT_EVALUATED")
    post_science = post_eval.get("science_gate_status", "NOT_EVALUATED")
    claim_status = post_eval.get("claim_status", state.get("claim_status", "NO_CLAIM"))
    blockers: list[str] = []
    for source in (xai.get("claim_blockers", []), post_eval.get("claim_blockers", [])):
        for item in source if isinstance(source, list) else []:
            if isinstance(item, dict):
                label = ":".join(
                    str(item.get(key))
                    for key in ("leg", "status")
                    if item.get(key) is not None
                ) or "STRUCTURED_BLOCKER"
            else:
                label = str(item)
            if label not in blockers:
                blockers.append(label)
    if privacy_findings:
        blockers.append("PUBLIC_BUNDLE_PRIVACY_OR_COMPLETENESS")
    if active:
        blockers.append("ACTIVE_METHOD_Y_WRITER")

    payload: dict[str, Any] = {
        "schema_version": "4.0-posttraining-audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_mode": "READ_ONLY_AGENT_MULTI_PASS",
        "human_signoff_required": False,
        "root_name": root.name,
        "technical_execution": {
            "status": "FINALIZED" if technical_complete else "INCOMPLETE_OR_BLOCKED",
            "verifier": verifier,
            "state_stage": state.get("stage"),
            "outer_cells_finalized": state.get("outer_cells_finalized"),
            "outer_result_files": len(outer_files),
            "outer_result_hash_fields": valid_outer_hashes,
            "expected_outer_cells": 50,
        },
        "scientific_gates": {
            "xai_science_gate_status": science_status,
            "post_eval_science_gate_status": post_science,
            "claim_status": claim_status,
            "positive_claim_allowed": False,
            "external_claim_allowed": False,
            "claim_blockers": blockers,
        },
        "selection": {
            "inner_only_selected_arms": selected,
            "selection_is_not_outer_test_selection": True,
            "selected_arm_outer_summary": selected_summary,
        },
        "diagnostics": {
            "scanner": scanner_summary,
            "xai_agreement": (
                xai.get("details", {}).get("agreement", {})
                if isinstance(xai.get("details"), dict)
                else {}
            ),
            "xai_fidelity": (
                xai.get("details", {}).get("fidelity", {})
                if isinstance(xai.get("details"), dict)
                else {}
            ),
            "post_eval_omnibus_p": (
                post_eval.get("omnibus", {}).get("p")
                if isinstance(post_eval.get("omnibus"), dict)
                else None
            ),
            "post_eval_contrast_count": len(post_eval.get("contrasts", []))
            if isinstance(post_eval.get("contrasts"), list)
            else 0,
        },
        "privacy": {
            "public_bundle_present": bundle.is_dir(),
            "finding_count": len(privacy_findings),
            "findings": privacy_findings,
        },
        "active_writers": active,
        "provenance": {
            "run_contract_sha256": contract.get("digest"),
            "state_sha256": _sha256(state_path),
            "run_contract_file_sha256": _sha256(contract_path),
            "xai_report_sha256": _sha256(root / "attribution/xai_science_gate_v4.json"),
            "post_eval_report_sha256": _sha256(root / "post_eval/post_eval_report.json"),
        },
        "supersession": {
            "historical_pass_reports_are_not_current_science_status": True,
            "current_policy": "POST_HOC_INTERNAL_EXPLORATORY",
        },
    }
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    execution = payload["technical_execution"]
    science = payload["scientific_gates"]
    privacy = payload["privacy"]
    selected = payload["selection"]["selected_arm_outer_summary"]
    diagnostics = payload["diagnostics"]
    lines = [
        "# Method Y image v4 post-training audit (2026-09-09)",
        "",
        "این سند ممیزی خواندنی و چندلایه است و هیچ آموزش یا artifact قفل‌شده‌ای را تغییر نمی‌دهد.",
        "",
        f"- وضعیت اجرای فنی: `{execution['status']}` (verifier: `{execution['verifier'].get('status')}`)",
        f"- سلول‌های outer ثبت‌شده: `{execution.get('outer_cells_finalized')}/50`",
        f"- وضعیت گیت علمی XAI: `{science['xai_science_gate_status']}`",
        f"- وضعیت گیت علمی پساآموزش: `{science['post_eval_science_gate_status']}`",
        f"- claim status: `{science['claim_status']}`",
        "- ادعای مثبت و ادعای خارجی: `false`",
        f"- privacy findings در bundle: `{privacy['finding_count']}`",
        f"- آزمودنی/رویداد outer در arm انتخابی amyloid: `{selected.get('amyloid', {}).get('n_test')}/{selected.get('amyloid', {}).get('test_events')}`",
        f"- آزمودنی/رویداد outer در arm انتخابی tau: `{selected.get('tau', {}).get('n_test')}/{selected.get('tau', {}).get('test_events')}`",
        f"- تعداد contrastهای post-eval: `{diagnostics.get('post_eval_contrast_count')}`",
        "",
        "## تفسیر",
        "",
        "تکمیل فنی و تکمیل علمی دو وضعیت مستقل‌اند. نتیجه فعلی فقط داخلی/اکتشافی است؛ "
        "مردودی XAI، کمّی‌سازی PET، robustness اسکنر یا external validation با برچسب تکمیل فنی رفع نمی‌شود.",
        "",
        "## موانع ثبت‌شده",
        "",
    ]
    blockers = science.get("claim_blockers", [])
    lines.extend([f"- `{item}`" for item in blockers] or ["- بدون مورد جدید؛ محدودیت claim policy همچنان برقرار است."])
    lines.extend([
        "",
        "گزارش‌های تاریخی که پیش از reconciliation وضعیت PASS داشته‌اند، شاهد تاریخی‌اند و بر وضعیت canonical فعلی غلبه نمی‌کنند.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results/method_y_revival_v4_image_factorial_v1",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/method_y_image_factorial_v4.yaml",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args(argv)
    payload = audit(args.root, args.config)
    json_out = args.json_out or args.root / "provenance/posttraining_audit_20260909.json"
    markdown_out = args.markdown_out or ROOT / "docs/method_y_posttraining_audit_20260909.md"
    atomic_write_json(json_out, payload)
    atomic_write_bytes(markdown_out, _markdown(payload).encode("utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["technical_execution"]["status"] == "FINALIZED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
