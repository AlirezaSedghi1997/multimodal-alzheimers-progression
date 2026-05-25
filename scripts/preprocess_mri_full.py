from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_mri_volume_cache import build_t1_hash_index
from src.data.a4_schema import resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.resumable import ResumableManifest, config_digest
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


TOOL_GROUPS = {
    "bias_correction": ["N4BiasFieldCorrection"],
    "registration": ["antsRegistrationSyNQuick.sh"],
    "brain_extraction": ["mri_synthstrip", "hd-bet", "bet"],
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def first_available(names: list[str]) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def check_toolchain(template: Path | None) -> dict[str, object]:
    available: dict[str, str | None] = {group: first_available(names) for group, names in TOOL_GROUPS.items()}
    missing = [group for group, tool in available.items() if tool is None]
    if template is None or not template.exists():
        missing.append("template")
    status = "READY" if not missing else "BLOCKED"
    return {
        "status": status,
        "available": available,
        "missing": sorted(set(missing)),
        "required": TOOL_GROUPS,
        "template": str(template) if template else None,
        "template_exists": bool(template and template.exists()),
        "how_to_unblock": [
            "Run inside the same WSL2 Conda environment used for training.",
            "Put N4BiasFieldCorrection and antsRegistrationSyNQuick.sh on PATH.",
            "Put one brain extraction command on PATH: mri_synthstrip, hd-bet, or bet.",
            "Pass a real template path with --template or METHOD_X_MNI_TEMPLATE.",
        ],
    }


def write_toolchain_report(report: dict[str, object]) -> None:
    toolchain = report.get("toolchain", {})
    missing = report.get("missing")
    if missing is None and isinstance(toolchain, dict):
        missing = toolchain.get("missing", [])
    atomic_write_json("results/tables/full_mri_preprocess_toolchain_report.json", report)
    lines = [
        "# Full MRI Preprocessing Toolchain Report\n\n",
        f"- status: `{report['status']}`\n",
        f"- missing: `{missing}`\n",
        f"- required: `{toolchain.get('required', TOOL_GROUPS) if isinstance(toolchain, dict) else TOOL_GROUPS}`\n",
        f"- template: `{toolchain.get('template') if isinstance(toolchain, dict) else None}`\n",
        "- contains_pet_input: `False`\n",
        "- contains_raw_identifiers: `False`\n",
        "- output_granularity: `hashed_sample_level`\n",
    ]
    atomic_write_text("results/tables/full_mri_preprocess_toolchain_report.md", "".join(lines))


def brain_extraction_command(tool_path: str, n4_path: Path, brain_path: Path, mask_path: Path) -> list[str]:
    name = Path(tool_path).name.lower()
    if "synthstrip" in name:
        return [tool_path, "-i", str(n4_path), "-o", str(brain_path), "-m", str(mask_path)]
    if "hd-bet" in name:
        return [tool_path, "-i", str(n4_path), "-o", str(brain_path)]
    if name == "bet" or name == "bet.exe":
        return [tool_path, str(n4_path), str(brain_path.with_suffix("")), "-R", "-f", "0.25", "-m"]
    raise ValueError(f"Unsupported brain extraction tool: {tool_path}")


def run_quiet(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return int(proc.returncode)


def preprocess_one(
    *,
    sample_hash: str,
    source_t1: Path,
    template: Path,
    out_dir: Path,
    tools: dict[str, str],
) -> tuple[bool, str]:
    scratch = out_dir / "_scratch" / sample_hash
    scratch.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{sample_hash}.nii.gz"
    n4_path = scratch / "n4.nii.gz"
    brain_path = scratch / "brain.nii.gz"
    mask_path = scratch / "brain_mask.nii.gz"
    reg_prefix = scratch / "reg_"
    warped_path = scratch / "reg_Warped.nii.gz"
    tmp_final = final_path.with_suffix(final_path.suffix + ".tmp")

    steps = [
        [tools["bias_correction"], "-d", "3", "-i", str(source_t1), "-o", str(n4_path)],
        brain_extraction_command(tools["brain_extraction"], n4_path, brain_path, mask_path),
        [
            tools["registration"],
            "-d",
            "3",
            "-f",
            str(template),
            "-m",
            str(brain_path),
            "-o",
            str(reg_prefix),
            "-t",
            "r",
        ],
    ]
    for i, cmd in enumerate(steps):
        code = run_quiet(cmd)
        if code != 0:
            return False, f"step_{i}_returncode_{code}"
    if not warped_path.exists():
        return False, "registration_warped_output_missing"
    tmp_final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(warped_path, tmp_final)
    os.replace(tmp_final, final_path)
    return True, str(final_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--template", default=os.environ.get("METHOD_X_MNI_TEMPLATE"))
    ap.add_argument("--out_dir", default="data/preprocessed_mri_96")
    ap.add_argument("--manifest", default="results/logs/full_mri_preprocess_resume.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    template = Path(args.template) if args.template else None
    toolchain = check_toolchain(template)
    report = {
        "status": toolchain["status"],
        "toolchain": toolchain,
        "missing": toolchain["missing"],
        "execute_requested": bool(args.execute),
        "contains_pet_input": False,
        "contains_raw_identifiers": False,
        "output_resolution": 96,
    }
    if toolchain["status"] != "READY":
        write_toolchain_report(report)
        StateManager().append_event("preprocess_full_mri", "BLOCKED", "Full MRI preprocessing blocked by missing specialized toolchain or template.")
        print(json.dumps({"status": "BLOCKED", "missing": toolchain["missing"]}, indent=2))
        return 2
    if not args.execute:
        report["status"] = "READY_FOR_EXECUTION"
        write_toolchain_report(report)
        print(json.dumps({"status": "READY_FOR_EXECUTION", "message": "rerun with --execute for the long preprocessing stage"}, indent=2))
        return 0

    data_root = resolve_data_root(args.data_root)
    rows = read_rows(Path(args.catalog))
    if args.limit > 0:
        rows = rows[: args.limit]
    t1_index = build_t1_hash_index(data_root)
    config = {"catalog": args.catalog, "out_dir": args.out_dir, "template": str(template), "resolution": 96}
    resume = ResumableManifest(args.manifest, stage="preprocess_full_mri", config=config)
    tools = {group: str(path) for group, path in toolchain["available"].items() if isinstance(path, str)}
    out_dir = Path(args.out_dir)
    ok = failed = missing = skipped = 0
    for row in rows:
        sample_hash = row["sample_hash"]
        if resume.item_done(sample_hash):
            skipped += 1
            continue
        source = t1_index.get(row["t1_file_hash"])
        if source is None:
            missing += 1
            resume.set_item(sample_hash, "FAILED", message="matching_t1_not_visible")
            continue
        resume.set_item(sample_hash, "RUNNING")
        success, detail = preprocess_one(
            sample_hash=sample_hash,
            source_t1=source,
            template=template,  # type: ignore[arg-type]
            out_dir=out_dir,
            tools=tools,
        )
        if success:
            ok += 1
            resume.set_item(sample_hash, "DONE", artifact=detail)
        else:
            failed += 1
            resume.set_item(sample_hash, "FAILED", message=detail)
    summary = {
        "ok": ok,
        "failed": failed,
        "missing": missing,
        "skipped_done": skipped,
        "total": len(rows),
        "limit": args.limit,
        "is_smoke_test": args.limit > 0,
    }
    if args.limit > 0 and failed == 0 and missing == 0:
        status = "SMOKE_TEST_COMPLETED"
    elif failed == 0 and missing == 0:
        status = "COMPLETED"
    else:
        status = "PARTIAL"
    report.update({"status": status, "summary": summary, "config_hash": config_digest(config)})
    write_toolchain_report(report)
    write_run_manifest("results/logs/full_mri_preprocess_manifest.json", stage="preprocess_full_mri", data_root=str(data_root), extra=summary)
    StateManager().append_event("preprocess_full_mri", str(report["status"]), "Full MRI preprocessing stage finished or paused with resumable manifest.")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"COMPLETED", "SMOKE_TEST_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
