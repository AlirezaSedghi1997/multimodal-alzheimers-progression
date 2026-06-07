from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPECTED_CONFIG_HASH = "4324429c60083666"
EXPECTED_TOTAL = 1700
MANIFEST_PATH = Path("results/logs/spm_mri_preprocess_v2_resume_wsl.json")
PREPROCESSED_DIR = Path("data/preprocessed_mri_96")
CATALOG_PATH = Path("data/processed/master_catalog.csv")
BLOCKED_POST_SPM_STAGES = [
    "mri-volume-cache",
    "train-3d",
    "optuna",
    "fusion",
    "external",
    "calibration",
    "final-analysis",
]
SAFE_BEFORE_SPM_COMPLETE = [
    "spm-completion-guard",
    "mri-volume-cache-smoke",
    "clinical-ad-feasibility",
    "public-bundle",
    "privacy-leakage-tests",
    "docs-config-script-tests",
]


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_catalog_sample_hashes(path: Path | None) -> list[str] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        return [row["sample_hash"] for row in csv.DictReader(f) if row.get("sample_hash")]


def manifest_counts(
    obj: dict[str, Any] | None,
    expected_samples: set[str] | None = None,
) -> tuple[int | None, int | None]:
    if not obj:
        return None, None
    items = obj.get("items")
    if isinstance(items, dict):
        if expected_samples is not None:
            done = sum(
                1
                for sample_hash in expected_samples
                if isinstance(items.get(sample_hash), dict) and items[sample_hash].get("status") == "DONE"
            )
            return done, len(expected_samples)
        done = sum(1 for item in items.values() if isinstance(item, dict) and item.get("status") == "DONE")
        return done, len(items)
    summary = obj.get("summary")
    if isinstance(summary, dict):
        done = summary.get("done_total")
        total = summary.get("total")
        return int(done) if isinstance(done, int) else None, int(total) if isinstance(total, int) else None
    return None, None


def count_preprocessed_files(preprocessed_dir: Path, expected_samples: set[str] | None = None) -> int:
    if not preprocessed_dir.exists():
        return 0
    if expected_samples is not None:
        return sum(1 for sample_hash in expected_samples if (preprocessed_dir / f"{sample_hash}.nii.gz").is_file())
    return sum(1 for path in preprocessed_dir.glob("*.nii.gz") if path.is_file())


def qc_counts(preprocessed_dir: Path, expected_samples: set[str] | None = None) -> tuple[int, int, int]:
    qc_dir = preprocessed_dir / "qc"
    if not qc_dir.exists():
        return 0, 0, 0
    if expected_samples is None:
        qc_files = list(qc_dir.glob("*.json"))
    else:
        qc_files = [qc_dir / f"{sample_hash}.json" for sample_hash in sorted(expected_samples)]
    invalid = 0
    for path in qc_files:
        obj = read_json(path)
        if not obj:
            invalid += 1
            continue
        nonzero_fraction = float(obj.get("normalized_nonzero_fraction") or 0.0)
        normalized_std = float(obj.get("normalized_brain_std") or 0.0)
        if nonzero_fraction <= 0.0 or normalized_std <= 0.0:
            invalid += 1
    return len(qc_files), invalid, len(qc_files) - invalid


def spm_completion_status(
    *,
    manifest_path: Path = MANIFEST_PATH,
    preprocessed_dir: Path = PREPROCESSED_DIR,
    expected_config_hash: str = EXPECTED_CONFIG_HASH,
    expected_total: int | None = None,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    if catalog_path is None and manifest_path == MANIFEST_PATH and preprocessed_dir == PREPROCESSED_DIR:
        catalog_path = CATALOG_PATH
    catalog_samples = read_catalog_sample_hashes(catalog_path)
    expected_sample_set = set(catalog_samples) if catalog_samples is not None else None
    expected_total_source = "catalog" if catalog_samples is not None else "constant"
    if expected_total is None:
        expected_total = len(catalog_samples) if catalog_samples is not None else EXPECTED_TOTAL
    manifest = read_json(manifest_path)
    done_count, manifest_total = manifest_counts(manifest, expected_sample_set)
    file_count = count_preprocessed_files(preprocessed_dir, expected_sample_set)
    qc_file_count, invalid_qc_count, valid_qc_count = qc_counts(preprocessed_dir, expected_sample_set)
    config_hash = manifest.get("config_hash") if manifest else None
    qc_validation_available = qc_file_count > 0
    manifest_item_count = len(manifest.get("items", {})) if isinstance(manifest, dict) and isinstance(manifest.get("items"), dict) else manifest_total

    checks = {
        "manifest_exists": manifest_path.exists(),
        "manifest_readable": manifest is not None,
        "config_hash_ok": config_hash == expected_config_hash,
        "done_count_ok": done_count == expected_total,
        "manifest_total_ok": manifest_total == expected_total,
        "preprocessed_file_count_ok": file_count == expected_total,
        "preprocessed_qc_ok": (invalid_qc_count == 0 and valid_qc_count == expected_total) if qc_validation_available else True,
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
        "expected_total_source": expected_total_source,
        "catalog_path": str(catalog_path) if catalog_path else None,
        "done_count": done_count,
        "manifest_total": manifest_total,
        "manifest_item_count": manifest_item_count,
        "preprocessed_file_count": file_count,
        "preprocessed_qc_file_count": qc_file_count,
        "preprocessed_qc_invalid_count": invalid_qc_count,
        "preprocessed_qc_valid_count": valid_qc_count,
        "active_catalog_sample_count": len(catalog_samples) if catalog_samples is not None else None,
        "checks": checks,
        "blocked_post_spm_stages": BLOCKED_POST_SPM_STAGES,
        "safe_before_spm_complete": SAFE_BEFORE_SPM_COMPLETE,
        "source_of_truth": {
            "manifest_path": str(manifest_path),
            "preprocessed_dir": str(preprocessed_dir),
            "expected_config_hash": expected_config_hash,
            "expected_total": expected_total,
            "expected_total_source": expected_total_source,
            "catalog_path": str(catalog_path) if catalog_path else None,
        },
        "next_action": "post_spm_stages_allowed" if ready else "continue_spm_preprocessing",
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
    ap.add_argument("--expected_total", type=int, default=None)
    ap.add_argument("--catalog", default=str(CATALOG_PATH))
    args = ap.parse_args()

    status = spm_completion_status(
        manifest_path=Path(args.manifest),
        preprocessed_dir=Path(args.preprocessed_dir),
        expected_config_hash=args.expected_config_hash,
        expected_total=args.expected_total,
        catalog_path=Path(args.catalog) if args.catalog else None,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if spm_completion_ready(status) else 4


if __name__ == "__main__":
    raise SystemExit(main())
