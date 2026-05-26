from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_CONFIG_HASH = "4324429c60083666"
EXPECTED_TOTAL = 1700
MANIFEST_PATH = Path("results/logs/spm_mri_preprocess_v2_resume_wsl.json")
PREPROCESSED_DIR = Path("data/preprocessed_mri_96")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def manifest_counts(obj: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not obj:
        return None, None
    items = obj.get("items")
    if isinstance(items, dict):
        done = sum(1 for item in items.values() if isinstance(item, dict) and item.get("status") == "DONE")
        return done, len(items)
    summary = obj.get("summary")
    if isinstance(summary, dict):
        done = summary.get("done_total")
        total = summary.get("total")
        return int(done) if isinstance(done, int) else None, int(total) if isinstance(total, int) else None
    return None, None


def count_preprocessed_files(preprocessed_dir: Path) -> int:
    if not preprocessed_dir.exists():
        return 0
    return sum(1 for path in preprocessed_dir.glob("*.nii.gz") if path.is_file())


def spm_completion_status(
    *,
    manifest_path: Path = MANIFEST_PATH,
    preprocessed_dir: Path = PREPROCESSED_DIR,
    expected_config_hash: str = EXPECTED_CONFIG_HASH,
    expected_total: int = EXPECTED_TOTAL,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    done_count, manifest_total = manifest_counts(manifest)
    file_count = count_preprocessed_files(preprocessed_dir)
    config_hash = manifest.get("config_hash") if manifest else None

    checks = {
        "manifest_exists": manifest_path.exists(),
        "manifest_readable": manifest is not None,
        "config_hash_ok": config_hash == expected_config_hash,
        "done_count_ok": done_count == expected_total,
        "manifest_total_ok": manifest_total == expected_total,
        "preprocessed_file_count_ok": file_count == expected_total,
    }
    ready = all(checks.values())
    return {
        "ready": ready,
        "status": "SPM_COMPLETE" if ready else "SPM_INCOMPLETE",
        "manifest_path": str(manifest_path),
        "preprocessed_dir": str(preprocessed_dir),
        "expected_config_hash": expected_config_hash,
        "config_hash": config_hash,
        "expected_total": expected_total,
        "done_count": done_count,
        "manifest_total": manifest_total,
        "preprocessed_file_count": file_count,
        "checks": checks,
        "privacy": {
            "raw_identifiers_exported": False,
            "raw_paths_exported": False,
            "subject_level_outputs_exported": False,
        },
    }


def spm_completion_ready(status: dict[str, Any]) -> bool:
    return bool(status.get("ready"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--preprocessed_dir", default=str(PREPROCESSED_DIR))
    ap.add_argument("--expected_config_hash", default=EXPECTED_CONFIG_HASH)
    ap.add_argument("--expected_total", type=int, default=EXPECTED_TOTAL)
    args = ap.parse_args()

    status = spm_completion_status(
        manifest_path=Path(args.manifest),
        preprocessed_dir=Path(args.preprocessed_dir),
        expected_config_hash=args.expected_config_hash,
        expected_total=args.expected_total,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if spm_completion_ready(status) else 4


if __name__ == "__main__":
    raise SystemExit(main())
