from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXECUTION_DIR = PROJECT_ROOT / "results" / "method_y_execution"
DELIVERY_MANIFEST = PROJECT_ROOT / "final_delivery" / "results" / "tables" / "delivery_manifest.json"
TRAINING_AUTHORIZATION = EXECUTION_DIR / "training_authorization.json"
MODE_BANNER_GATE = "heavy training is gated."
MODE_BANNER_TRAIN = "heavy training enabled, see training_authorization.json."
LEGACY_METHOD_Y_PET_CACHE = "caches/method_y/pet_mni_128.h5"
DEFAULT_METHOD_Y_PET_CACHE = "caches/method_y/pet_mni_128_broad_s1_fbp_only_hq_v1.h5"
DEFAULT_METHOD_Y_PET_INPUT_LIST = "data/method_y/pet_input_list_broad_s1_fbp_only.csv"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_cfg(path: str | Path) -> dict[str, Any]:
    text = project_path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data or {}
    except Exception:
        return json.loads(text)


def validate_method_y_pet_cache_protocol(cfg: dict[str, Any], *, stage: str) -> None:
    """Fail loudly when a Method Y stage points at the legacy per-scan z-score PET cache."""
    inputs = cfg.get("inputs", {}) or {}
    pet_h5 = str(inputs.get("pet_h5") or DEFAULT_METHOD_Y_PET_CACHE).replace("\\", "/").lower()
    legacy_cache = pet_h5 == LEGACY_METHOD_Y_PET_CACHE or pet_h5.endswith(f"/{LEGACY_METHOD_Y_PET_CACHE}")
    allow_legacy = bool(
        inputs.get("allow_legacy_zscore_cache_for_exploratory_only")
        or (cfg.get("claim_boundary", {}) or {}).get("legacy_zscore_cache_allowed_for_exploratory_only")
    )
    if legacy_cache and not allow_legacy:
        raise SystemExit(
            "BLOCKED_LEGACY_PET_ZSCORE_CACHE_NOT_PROTOCOL_COMPLIANT:"
            f"{stage}:{LEGACY_METHOD_Y_PET_CACHE}:"
            "use a SUVR-preserving FBP cache such as "
            "caches/method_y/pet_mni_128_suvr_preserved_v1.h5 or "
            f"{DEFAULT_METHOD_Y_PET_CACHE}"
        )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=False, allow_nan=False), encoding="utf-8")
    tmp.replace(target)


def _json_safe(value: Any) -> Any:
    """Convert non-finite Python/numpy scalars to strict JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _json_safe(value.item())
        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
    except Exception:
        pass
    return value


def add_mode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("plan", "dry_run", "train"), default="plan")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--i_understand_compute_cost", action="store_true")
    parser.add_argument("--isolated-run", action="store_true", help="Write Method Y artifacts under a run directory first.")
    parser.add_argument("--publish-run", default=None, help="Publish a completed isolated run directory atomically.")


def print_mode_banner(mode: str) -> None:
    if mode == "train":
        print(f"MODE: train \u2014 {MODE_BANNER_TRAIN}")
    else:
        print(f"MODE: {mode} \u2014 {MODE_BANNER_GATE}")


def validate_train_authorization(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.mode != "train":
        return None
    if not args.i_understand_compute_cost:
        raise SystemExit("BLOCKED_TRAIN_MODE_REQUIRES_I_UNDERSTAND_COMPUTE_COST")
    if not TRAINING_AUTHORIZATION.exists():
        raise SystemExit("BLOCKED_TRAINING_AUTHORIZATION_JSON_MISSING")
    payload = json.loads(TRAINING_AUTHORIZATION.read_text(encoding="utf-8"))
    required = ["authorized_by", "authorized_at_utc", "expected_gpu_hours", "target_gpu_vram_gb"]
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise SystemExit(f"BLOCKED_TRAINING_AUTHORIZATION_FIELDS_MISSING:{missing}")
    try:
        target_vram = float(payload.get("target_gpu_vram_gb"))
    except Exception:
        raise SystemExit("BLOCKED_TRAINING_AUTHORIZATION_TARGET_VRAM_INVALID")
    if target_vram > 8.0:
        raise SystemExit(f"BLOCKED_TRAINING_AUTHORIZATION_TARGET_VRAM_EXCEEDS_OPERATOR_BUDGET:{target_vram:.2f}GB>8.00GB")
    try:
        import torch

        if torch.cuda.is_available():
            actual_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if target_vram > actual_vram + 0.25:
                raise SystemExit(
                    f"BLOCKED_TRAINING_AUTHORIZATION_TARGET_VRAM_EXCEEDS_DEVICE:{target_vram:.2f}GB>{actual_vram:.2f}GB"
                )
    except SystemExit:
        raise
    except Exception:
        pass
    return payload


def resolve_stage_epochs(args: argparse.Namespace, cfg: dict[str, Any], *, default_dry_run: int = 1, default_train: int) -> int:
    """Resolve epochs so full train does not silently inherit the dry-run smoke default."""
    if args.epochs is not None:
        return int(args.epochs)
    training = cfg.get("training", {}) or {}
    if args.mode == "train":
        return int(training.get("train_epochs", training.get("epochs", default_train)))
    if args.mode == "dry_run":
        return int(training.get("dry_run_epochs", default_dry_run))
    return int(default_dry_run)


def execution_status(status: str, blocker: str | None = None) -> str:
    return f"{status}:{blocker}" if blocker else status


def append_execution_log(
    *,
    stage: str,
    mode: str,
    start_time_utc: str,
    end_time_utc: str,
    metrics: dict[str, Any] | None,
    acceptance_gate_verdict: str,
    next_allowed_stage: str,
    execution_dir: str | Path | None = None,
) -> None:
    target_dir = project_path(execution_dir) if execution_dir is not None else EXECUTION_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "execution_log.md"
    if not path.exists():
        path.write_text(
            "| stage | mode | start_time_utc | end_time_utc | fold_level_metrics | acceptance_gate_verdict | next_allowed_stage |\n"
            "|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    metrics_text = json.dumps(_json_safe(metrics or {}), sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            f"| {stage} | {mode} | {start_time_utc} | {end_time_utc} | `{metrics_text}` | "
            f"{acceptance_gate_verdict} | {next_allowed_stage} |\n"
        )


def update_delivery_manifest(stage: str, status_payload: dict[str, Any]) -> None:
    if DELIVERY_MANIFEST.exists():
        try:
            manifest = json.loads(DELIVERY_MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {"bundle_root": "final_delivery", "manifest_parse_error": True}
    else:
        manifest = {
            "bundle_root": "final_delivery",
            "notes": [
                "Created by Method Y guarded stage scripts because final_delivery was missing in the project repo."
            ],
        }
    additions = manifest.setdefault("v3_method_y_additions", {})
    execution = additions.setdefault("execution_status", {})
    execution[stage] = status_payload
    write_json(DELIVERY_MANIFEST, manifest)


def stage_paths(stage: str) -> tuple[Path, Path | None]:
    report = EXECUTION_DIR / f"stage_{stage.lower()}_report.json"
    metrics = EXECUTION_DIR / f"stage_{stage.lower()}_metrics.json" if stage in {"S1", "S2"} else None
    return report, metrics


def stage_paths_in(stage: str, execution_dir: str | Path | None = None) -> tuple[Path, Path | None]:
    target_dir = project_path(execution_dir) if execution_dir is not None else EXECUTION_DIR
    report = target_dir / f"stage_{stage.lower()}_report.json"
    metrics = target_dir / f"stage_{stage.lower()}_metrics.json" if stage in {"S1", "S2"} else None
    return report, metrics


def write_stage_report(
    *,
    stage: str,
    mode: str,
    status: str,
    acceptance_gate_passed: bool | None,
    blocker: str | None = None,
    next_allowed_stage: str = "STOP",
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    start_time_utc: str | None = None,
    execution_dir: str | Path | None = None,
    update_official: bool = True,
) -> dict[str, Any]:
    start = start_time_utc or utc_now_iso()
    end = utc_now_iso()
    payload: dict[str, Any] = {
        "stage": stage,
        "mode": mode,
        "status": status,
        "acceptance_gate_passed": acceptance_gate_passed,
        "blocker": blocker,
        "started_at_utc": start,
        "ended_at_utc": end,
        "next_allowed_stage": next_allowed_stage,
        "metrics": metrics or {},
    }
    if extra:
        payload.update(extra)
    report_path, metrics_path = stage_paths_in(stage, execution_dir)
    write_json(report_path, payload)
    if metrics_path is not None:
        metrics_payload: dict[str, Any] = {"stage": stage, "mode": mode, "status": status, "metrics": metrics or {}}
        if extra and isinstance(extra.get("isolated_run"), dict):
            metrics_payload["isolated_run"] = extra["isolated_run"]
        write_json(metrics_path, metrics_payload)
    append_execution_log(
        stage=stage,
        mode=mode,
        start_time_utc=start,
        end_time_utc=end,
        metrics=metrics,
        acceptance_gate_verdict=str(acceptance_gate_passed),
        next_allowed_stage=next_allowed_stage,
        execution_dir=execution_dir,
    )
    if update_official:
        update_delivery_manifest(stage, {"mode": mode, "status": status, "blocker": blocker, "updated_at_utc": end})
    return payload


def write_human_decision_required(
    reason: str,
    options: Iterable[str],
    *,
    execution_dir: str | Path | None = None,
) -> None:
    text = [
        "# HUMAN DECISION REQUIRED",
        "",
        f"Reason: {reason}",
        "",
        "Options:",
    ]
    for idx, option in enumerate(options, 1):
        text.append(f"{idx}. {option}")
    target_dir = project_path(execution_dir) if execution_dir is not None else EXECUTION_DIR
    (target_dir / "HUMAN_DECISION_REQUIRED.md").parent.mkdir(parents=True, exist_ok=True)
    (target_dir / "HUMAN_DECISION_REQUIRED.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def csv_rows(path: str | Path) -> list[dict[str, str]]:
    target = project_path(path)
    with target.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_auc(y_true: list[int], scores: list[float]) -> float | None:
    pos = [score for label, score in zip(y_true, scores, strict=True) if int(label) == 1]
    neg = [score for label, score in zip(y_true, scores, strict=True) if int(label) == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def cohort_probe_auc(features: Any, cohorts: Any) -> float | None:
    import numpy as np

    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    labels_text = [str(v).upper() for v in list(cohorts)]
    if not {"A4", "LEARN"}.issubset(set(labels_text)):
        return None
    y = np.asarray([1 if value == "A4" else 0 for value in labels_text], dtype=float)
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=float)], axis=1)
    try:
        weights, *_ = np.linalg.lstsq(design, y, rcond=None)
        scores = design @ weights
    except Exception:
        scores = np.full_like(y, float(y.mean()))
    auc = safe_auc([int(v) for v in y.tolist()], [float(v) for v in scores.tolist()])
    return None if auc is None or math.isnan(float(auc)) else float(auc)


CANONICAL_16GB_DEFAULTS = {
    "base_channels": 8, "batch_size": 2, "grad_accum": 2, "gradient_checkpointing": False,
    "amp": True, "hidden_dim": 256, "bottleneck_dim": 32, "num_heads": 2,
    "mri_backbone": "resnet18_3d", "vram_budget_gb": 14.0,
}
VRAM8_DEFAULTS = {
    "base_channels": 4, "batch_size": 1, "grad_accum": 4, "gradient_checkpointing": True,
    "amp": True, "hidden_dim": 128, "bottleneck_dim": 16, "num_heads": 2,
    "mri_backbone": "resnet10_3d", "vram_budget_gb": 7.0,
}


def compute_config_diff(cfg: dict[str, Any]) -> dict[str, Any]:
    """Diff of the active compute profile vs the canonical 16 GB defaults.

    Torch-free so it can be used in plan mode. Single source of truth for the
    operator-approved VRAM=8 shrink (logged in every stage report per prompt Section 4).
    """
    profile = dict(VRAM8_DEFAULTS)
    profile.update(cfg.get("compute", {}) or {})
    profile.update(cfg.get("memory_controls", {}) or {})
    return {
        key: {"canonical_16gb": CANONICAL_16GB_DEFAULTS[key], "active_vram8": profile[key]}
        for key in CANONICAL_16GB_DEFAULTS
        if profile.get(key) != CANONICAL_16GB_DEFAULTS[key]
    }


def current_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "cache_paths_present": {
            "caches/method_y": (PROJECT_ROOT / "caches" / "method_y").exists(),
            "data/method_y": (PROJECT_ROOT / "data" / "method_y").exists(),
            "data/cached": (PROJECT_ROOT / "data" / "cached").exists(),
            "data/preprocessed_pet": (PROJECT_ROOT / "data" / "preprocessed_pet").exists(),
        },
        "splits_paths_present": {
            "splits": (PROJECT_ROOT / "splits").exists(),
            "data/splits": (PROJECT_ROOT / "data" / "splits").exists(),
        },
    }
    for name in ("torch", "monai", "nibabel", "h5py"):
        try:
            module = __import__(name)
            info[f"{name}_version"] = getattr(module, "__version__", "installed")
        except Exception:
            info[f"{name}_version"] = None
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["gpu_total_vram_gb"] = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if torch.cuda.is_available()
            else None
        )
    except Exception as exc:
        info["cuda_available"] = False
        info["gpu_name"] = None
        info["gpu_total_vram_gb"] = None
        info["torch_error"] = str(exc)
    return info
