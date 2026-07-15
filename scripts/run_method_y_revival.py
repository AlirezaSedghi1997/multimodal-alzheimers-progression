from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.method_y_revival.atomic_io import atomic_write_json, atomic_write_bytes
from src.method_y_revival.orchestrator import initialize_run
from src.method_y_revival.process_guard import WriterLock


def _git(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def repository_snapshot() -> dict[str, Any]:
    status = _git("status", "--porcelain") or ""
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_entry_count": len(status.splitlines()),
        "dirty_tree_digest": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def dependency_snapshot() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    for module_name in ("numpy", "pandas", "sklearn", "sksurv", "nibabel", "torch", "monai"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "UNKNOWN")
        except Exception as exc:
            versions[module_name] = f"UNAVAILABLE:{type(exc).__name__}"
    return versions


def _source_checks(config: dict[str, Any]) -> dict[str, bool]:
    raw_root = Path(config["paths"]["raw_root"])
    checks = {"raw_root": raw_root.exists() and raw_root.is_dir()}
    for name, relative in config["clinical_sources"].items():
        checks[f"clinical:{name}"] = (raw_root / relative).is_file()
    for name in (
        "amyloid_input_manifest",
        "tau_image_manifest",
        "tau_endpoint_table",
        "tau_regional_features",
    ):
        checks[f"project:{name}"] = (PROJECT_ROOT / config["paths"][name]).is_file()
    return checks


def run_preflight(config: dict[str, Any], output_root: Path, *, preflight_only: bool) -> dict[str, Any]:
    repo = repository_snapshot()
    initialized = initialize_run(config=config, output_root=output_root, repo_snapshot=repo)
    checks = _source_checks(config)
    source_ready = all(checks.values())
    template_value = config["paths"].get("reference_template")
    template_ready = bool(template_value) and (PROJECT_ROOT / str(template_value)).is_file()
    report = {
        "status": "PREFLIGHT_DONE" if source_ready else "PREFLIGHT_BLOCKED",
        "run_contract_sha256": initialized["run_contract_sha256"],
        "source_checks": checks,
        "regional_path_status": (
            "READY_FOR_MANIFEST_CONSTRUCTION" if source_ready else "BLOCKED_MISSING_LOCAL_SOURCE"
        ),
        "image_path_status": (
            "READY_FOR_AFFINE_PREPROCESSING"
            if template_ready
            else "BLOCKED_REFERENCE_TEMPLATE_NOT_LOCKED"
        ),
        "external_validation_status": "EXTERNAL_PENDING",
        "visual_qc_status": "PENDING_HUMAN_REVIEW",
        "raw_root_mutated": False,
        "execution_mode": "PREFLIGHT_ONLY" if preflight_only else "GUARDED_AUTO",
        "next_required_stage": "MANIFEST_CONSTRUCTION_AND_LOCK",
    }
    atomic_write_json(output_root / "provenance" / "preflight_report.json", report)
    atomic_write_json(output_root / "provenance" / "dependency_snapshot.json", dependency_snapshot())
    atomic_write_json(
        output_root / "provenance" / "process_snapshot.json",
        {"current_pid": __import__("os").getpid(), "active_writer_for_output": True},
    )
    status_text = (
        "# Method Y Revival Implementation Status\n\n"
        f"- pipeline_stage: `{report['status']}`\n"
        f"- regional_path: `{report['regional_path_status']}`\n"
        f"- image_path: `{report['image_path_status']}`\n"
        f"- external_validation: `{report['external_validation_status']}`\n"
        f"- visual_qc: `{report['visual_qc_status']}`\n"
        "- raw_root_mutated: `false`\n"
        "- positive_claim_allowed: `false`\n"
    )
    atomic_write_bytes(output_root / "reports" / "IMPLEMENTATION_STATUS.md", status_text.encode("utf-8"))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable fail-closed Method Y revival launcher")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "method_y_revival_v1.yaml"))
    parser.add_argument("--output-root")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root) if args.output_root else PROJECT_ROOT / config["paths"]["output_root"]
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    lock = WriterLock(output_root / ".writer.lock", output_root=output_root)
    with lock:
        report = run_preflight(config, output_root, preflight_only=bool(args.preflight_only))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["status"] == "PREFLIGHT_DONE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
