from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival.atomic_io import atomic_write_json, read_json  # noqa: E402
from src.method_y_revival.contracts import sha256_mapping  # noqa: E402
from src.method_y_revival.image_factorial_orchestrator_v4 import (  # noqa: E402
    ImageFactorialOrchestratorV4,
    resolved_arm_model_config,
)
from src.method_y_revival.image_factorial_v4 import (  # noqa: E402
    select_inner_only_factorial_arm,
)
from src.method_y_revival.image_outer_eval_v4 import evaluate_locked_outer_v4  # noqa: E402
from src.method_y_revival.image_factorial_reporting_v4 import (  # noqa: E402
    build_factorial_pair_reports,
)
from src.method_y_revival.image_reporting_v3 import privacy_scan_reports  # noqa: E402
from src.method_y_revival.process_guard import WriterLock  # noqa: E402
from src.method_y_revival.software_contract_v4 import (  # noqa: E402
    bind_software_runtime_contract_v4,
)


class FactorialLauncherV4Error(RuntimeError):
    pass


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _apply_deterministic_environment(config: dict[str, Any], output_root: Path) -> None:
    temporary = output_root / "tmp"
    pycache = output_root / "runtime" / "pycache"
    temporary.mkdir(parents=True, exist_ok=True)
    pycache.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temporary)
    os.environ["TMP"] = str(temporary)
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(
        config.get("runtime", {}).get("cublas_workspace_config", ":4096:8")
    )
    os.environ["PYTHONHASHSEED"] = str(int(config.get("seed", 0)))


def _load_bound_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if Path(raw.get("project_root", "")).resolve() != ROOT.resolve():
        raise FactorialLauncherV4Error("PROJECT_ROOT_MUST_BE_H_NEW_PROJECT")
    output_root = _path(ROOT, raw["output_root"])
    if output_root.drive.upper() != "H:":
        raise FactorialLauncherV4Error("OUTPUT_ROOT_MUST_BE_ON_H")
    _apply_deterministic_environment(raw, output_root)
    return bind_software_runtime_contract_v4(raw, ROOT), output_root


def _input_fingerprints(config: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    shared = config["shared"]
    common = {
        "foreground_mask": _path(ROOT, shared["foreground_mask"]),
        "augmentation_qc_contract": _path(ROOT, shared["augmentation_qc_contract"]),
        "reference_template": _path(ROOT, shared["reference_template"]),
    }
    for name, path in common.items():
        if not path.is_file():
            raise FactorialLauncherV4Error(f"INPUT_MISSING:{name}:{path}")
        records[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size": int(path.stat().st_size),
        }
    for branch in ("amyloid", "tau"):
        branch_cfg = config["branches"][branch]
        for key in ("membership", "splits", "pattern_cache"):
            path = _path(ROOT, branch_cfg[key])
            if not path.is_file():
                raise FactorialLauncherV4Error(f"INPUT_MISSING:{branch}:{key}:{path}")
            records[f"{branch}_{key}"] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size": int(path.stat().st_size),
            }
    return records


def _validate_development_split_contract(
    development, split: dict[str, Any], *, branch: str
) -> dict[str, int]:
    """Fail closed unless the effective image cohort has valid nested CV support.

    The locked regional split files contain subjects that can be ineligible for the
    image branch.  Validation is therefore performed after intersecting each split
    with the immutable image-development membership, without regenerating folds.
    """

    required = {"patient_hash", "event", "time_days"}
    missing = sorted(required.difference(development.columns))
    if missing:
        raise FactorialLauncherV4Error(
            f"DEVELOPMENT_COLUMNS_MISSING:{branch}:{missing}"
        )
    subject_series = development["patient_hash"].astype(str)
    if subject_series.duplicated().any():
        raise FactorialLauncherV4Error(f"DUPLICATE_DEVELOPMENT_SUBJECT:{branch}")
    event_values = development["event"]
    if event_values.isna().any() or not set(event_values.astype(int).tolist()).issubset(
        {0, 1}
    ):
        raise FactorialLauncherV4Error(f"NONBINARY_DEVELOPMENT_EVENT:{branch}")
    eligible = set(subject_series)
    if not eligible:
        raise FactorialLauncherV4Error(f"DEVELOPMENT_ENDPOINT_EMPTY:{branch}")
    event_by_subject = dict(zip(subject_series, event_values.astype(int), strict=True))

    outer_rows = split.get("outer_folds", [])
    if {int(row.get("fold", -1)) for row in outer_rows} != set(range(5)):
        raise FactorialLauncherV4Error(f"EXPECTED_FIVE_OUTER_FOLDS:{branch}")
    test_exposure_count = {key: 0 for key in eligible}

    def require_both_classes(keys: set[str], role: str) -> None:
        classes = {event_by_subject[key] for key in keys}
        if classes != {0, 1}:
            raise FactorialLauncherV4Error(
                f"BOTH_EVENT_CLASSES_REQUIRED:{branch}:{role}"
            )

    for outer in outer_rows:
        fold = int(outer["fold"])
        train = set(
            outer.get("train_subjects") or outer.get("train_subject_keys") or []
        ).intersection(eligible)
        test = set(
            outer.get("test_subjects") or outer.get("test_subject_keys") or []
        ).intersection(eligible)
        if train & test:
            raise FactorialLauncherV4Error(
                f"OUTER_PARTITION_LEAKAGE:{branch}:fold_{fold}"
            )
        if not train or not test or train | test != eligible:
            raise FactorialLauncherV4Error(
                f"OUTER_PARTITION_INCOMPLETE:{branch}:fold_{fold}"
            )
        require_both_classes(train, f"outer_train_{fold}")
        require_both_classes(test, f"outer_test_{fold}")
        for key in test:
            test_exposure_count[key] += 1

        inner_rows = _restrict_inner_splits(outer.get("inner_folds", []), train)
        if {int(row["fold"]) for row in inner_rows} != set(range(3)):
            raise FactorialLauncherV4Error(
                f"EXPECTED_THREE_INNER_FOLDS:{branch}:outer_{fold}"
            )
        for inner in inner_rows:
            inner_fold = int(inner["fold"])
            inner_train = set(inner["train_subjects"])
            validation = set(inner["validation_subjects"])
            require_both_classes(
                inner_train, f"inner_train_{fold}_{inner_fold}"
            )
            require_both_classes(
                validation, f"inner_validation_{fold}_{inner_fold}"
            )

    if any(count != 1 for count in test_exposure_count.values()):
        raise FactorialLauncherV4Error(
            f"OUTER_TEST_COVERAGE_NOT_EXACTLY_ONCE:{branch}"
        )
    return {
        "subjects": int(len(development)),
        "events": int(event_values.astype(int).sum()),
        "outer_test_coverage": int(len(test_exposure_count)),
        "outer_folds": 5,
        "inner_folds": 3,
    }


def _preflight(config_path: Path) -> dict[str, Any]:
    import pandas as pd
    import torch

    config, output_root = _load_bound_config(config_path)
    fingerprints = _input_fingerprints(config)
    qc = read_json(_path(ROOT, config["shared"]["augmentation_qc_contract"]))
    expected_qc = config["shared"]["augmentation_qc_expected_contract_sha256"]
    if qc.get("contract_sha256") != expected_qc or qc.get("status") != "PASS":
        raise FactorialLauncherV4Error("AUGMENTATION_QC_CONTRACT_MISMATCH")
    selected = qc.get("selected_envelope", {})
    for key in (
        "max_rotation_degrees",
        "max_translation_voxels",
        "scale_min",
        "scale_max",
    ):
        if float(selected.get(key, float("nan"))) != float(
            config["augmentation"].get(key, float("nan"))
        ):
            raise FactorialLauncherV4Error(f"AUGMENTATION_ENVELOPE_MISMATCH:{key}")
    branch_summary: dict[str, Any] = {}
    for branch in ("amyloid", "tau"):
        membership = pd.read_parquet(_path(ROOT, config["branches"][branch]["membership"]))
        development = membership[membership["role"] == "development"].copy()
        split = read_json(_path(ROOT, config["branches"][branch]["splits"]))
        branch_summary[branch] = _validate_development_split_contract(
            development, split, branch=branch
        )
    contract_body = {
        "schema_version": "4.0",
        "config_sha256": _sha256_file(config_path),
        "input_fingerprints": fingerprints,
        "augmentation_qc_contract_sha256": expected_qc,
        "software_runtime_contract": config["software_runtime_contract"],
        "factorial_arms": config["factorial_arms"],
        "runtime": config["runtime"],
        "seed": int(config["seed"]),
    }
    digest = sha256_mapping(contract_body)
    return {
        "status": "READY",
        "output_root": str(output_root),
        "config": config,
        "input_fingerprints": fingerprints,
        "branch_summary": branch_summary,
        "gpu": {
            "available": bool(torch.cuda.is_available()),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory)
            if torch.cuda.is_available()
            else 0,
        },
        "run_contract_body": contract_body,
        "run_contract_sha256": digest,
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
    }


def _ensure_run_contract(output_root: Path, preflight: dict[str, Any]) -> None:
    path = output_root / "run_contract.json"
    payload = {
        **preflight["run_contract_body"],
        "digest": preflight["run_contract_sha256"],
    }
    if path.is_file():
        existing = read_json(path)
        if existing != payload:
            raise FactorialLauncherV4Error("RUN_CONTRACT_MISMATCH")
    else:
        atomic_write_json(path, payload)


def _restrict_inner_splits(
    inner_splits: list[dict[str, Any]], eligible_keys: set[str]
) -> list[dict[str, Any]]:
    restricted: list[dict[str, Any]] = []
    for row in inner_splits:
        train = set(row.get("train_subjects") or row.get("train_subject_keys") or [])
        validation = set(
            row.get("validation_subjects")
            or row.get("validation_subject_keys")
            or row.get("val_subjects")
            or []
        )
        if train & validation:
            raise FactorialLauncherV4Error("INNER_PARTITION_LEAKAGE")
        train &= eligible_keys
        validation &= eligible_keys
        if not train or not validation or train | validation != eligible_keys:
            raise FactorialLauncherV4Error("INNER_PARTITION_INCOMPLETE_AFTER_FILTER")
        restricted.append(
            {
                "fold": int(row.get("fold", len(restricted))),
                "train_subjects": sorted(train),
                "validation_subjects": sorted(validation),
            }
        )
    if len(restricted) != 3:
        raise FactorialLauncherV4Error("EXPECTED_THREE_INNER_FOLDS")
    return restricted


def _gpu_smoke(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    import time

    import nibabel as nib
    import numpy as np
    import torch

    from src.method_y_revival.image_factorial_v4 import build_factorial_image_model
    from src.method_y_revival.image_survival_v3 import discrete_time_survival_nll
    from src.method_y_revival.pet_augmentation_v4 import (
        PetSafeAugmentationConfig,
        apply_pet_safe_augmentation_batch,
    )

    if not torch.cuda.is_available():
        raise FactorialLauncherV4Error("CUDA_REQUIRED_FOR_PRODUCTION_FACTORIAL_BENCHMARK")
    device = torch.device("cuda")
    results: dict[str, Any] = {}
    total = int(torch.cuda.get_device_properties(0).total_memory)
    for label, arm_id, width in (
        ("tiny", "A0", None),
        ("resnet10_w025", "A2", 0.25),
        ("resnet18_w025", "A4", 0.25),
        ("resnet18_w050_resource_only", "A4", 0.5),
    ):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model_config = resolved_arm_model_config(config, arm_id)
        if width is not None:
            model_config["resnet_width_multiplier"] = width
        model = build_factorial_image_model(
            arm_id,
            model_config,
            tabular_dim=12,
            hazard_bins=5,
        ).to(device)
        image = torch.zeros((2, 1, 128, 128, 128), device=device)
        tabular = torch.zeros((2, 12), device=device)
        event = torch.tensor([0.0, 1.0], device=device)
        time_days = torch.tensor([700.0, 900.0], device=device)
        edges = torch.tensor([0.0, 365.0, 730.0, 1095.0, 1460.0, 1640.0], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(image, tabular)
        loss = discrete_time_survival_nll(logits.float(), event, time_days, edges)
        loss.backward()
        peak = int(torch.cuda.max_memory_allocated())
        results[label] = {
            "status": "PASS" if torch.isfinite(loss) else "FAIL",
            "loss": float(loss.detach().cpu()),
            "peak_memory_bytes": peak,
            "peak_vram_fraction": peak / max(1, total),
            "eligible_for_primary": peak / max(1, total)
            <= float(config["runtime"]["max_vram_fraction_for_smoke"]),
        }
        del model, image, tabular, logits, loss
    mask_array = (
        np.asarray(
            nib.load(_path(ROOT, config["shared"]["foreground_mask"])).dataobj
        )
        > 0
    ).astype(np.float32)
    augmentation_config = PetSafeAugmentationConfig(
        enabled=True,
        probability=float(config["augmentation"]["probability"]),
        max_rotation_degrees=float(config["augmentation"]["max_rotation_degrees"]),
        max_translation_voxels=float(config["augmentation"]["max_translation_voxels"]),
        scale_min=float(config["augmentation"]["scale_min"]),
        scale_max=float(config["augmentation"]["scale_max"]),
        qc_contract_sha256=config["shared"][
            "augmentation_qc_expected_contract_sha256"
        ],
    )
    smoke_images = torch.zeros((2, 1, 128, 128, 128), device=device)
    smoke_masks = torch.from_numpy(mask_array)[None, None].repeat(2, 1, 1, 1, 1).to(device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    augmented = apply_pet_safe_augmentation_batch(
        images=smoke_images,
        masks=smoke_masks,
        config=augmentation_config,
        split_role="outer_train",
        branch="tau",
        subject_keys=("smoke_a", "smoke_b"),
        outer_fold=0,
        inner_fold=None,
        epoch=1,
        base_seed=int(config["seed"]),
    )
    torch.cuda.synchronize()
    augmentation_seconds = time.perf_counter() - started
    results["pet_safe_vectorized_augmentation"] = {
        "status": "PASS"
        if torch.isfinite(augmented.images).all() and all(augmented.applied)
        else "FAIL",
        "batch_size": 2,
        "shape": [128, 128, 128],
        "seconds": float(augmentation_seconds),
        "execution_device": "cuda",
    }
    payload = {
        "schema_version": "4.0",
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "total_memory_bytes": total,
        "results": results,
        "a4_primary_width": 0.25,
        "a4_width_050_role": "RESOURCE_FEASIBILITY_ONLY",
        "software_runtime_contract_sha256": config["software_runtime_contract"][
            "contract_sha256"
        ],
    }
    atomic_write_json(output_root / "provenance" / "gpu_smoke_v4.json", payload)
    return payload


def _load_arm_payload(output_root: Path, branch: str, arm_id: str) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    selected_epoch_nll_gaps: list[float] = []
    for fold in range(5):
        path = (
            output_root
            / "private"
            / "outer_evaluation"
            / branch
            / arm_id
            / f"fold_{fold}_private.json"
        )
        payload = read_json(path)
        body = {key: value for key, value in payload.items() if key != "private_record_sha256"}
        if payload.get("private_record_sha256") != sha256_mapping(body):
            raise FactorialLauncherV4Error("PRIVATE_OUTER_RECORD_HASH_MISMATCH")
        folds.append({"fold": fold, **body["metrics"]})
        winner_path = (
            output_root
            / "orchestration"
            / branch
            / f"winner_{arm_id}_outer_{fold}.json"
        )
        winner_record = read_json(winner_path)
        winner_body = {
            key: value
            for key, value in winner_record.items()
            if key != "winner_record_sha256"
        }
        if winner_record.get("winner_record_sha256") != sha256_mapping(winner_body):
            raise FactorialLauncherV4Error("WINNER_RECORD_HASH_MISMATCH")
        metric_records = winner_body.get("winner", {}).get(
            "inner_fold_metric_records", []
        )
        if len(metric_records) != 3:
            raise FactorialLauncherV4Error("WINNER_INNER_METRICS_INCOMPLETE")
        selected_epoch_nll_gaps.extend(
            float(row["train_validation_nll_gap_at_selected_epoch"])
            for row in metric_records
        )
    return {
        "folds": folds,
        "inner_selected_epoch_nll_gaps": selected_epoch_nll_gaps,
    }


def _build_post_training_reports(
    *, output_root: Path, config: dict[str, Any], memberships: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    pair_reports: dict[str, Any] = {}
    inner_only_selections: dict[str, Any] = {}
    all_pairs = [
        ("TINY_AUGMENTATION_EFFECT", "A0", "A1"),
        ("RESNET10_AUGMENTATION_EFFECT", "A2", "A3"),
        ("ARCHITECTURE_EFFECT_WITHOUT_AUGMENTATION", "A0", "A2"),
        ("ARCHITECTURE_EFFECT_WITH_PET_SAFE_AUGMENTATION", "A1", "A3"),
        ("DEPTH_EFFECT_WITH_PET_SAFE_AUGMENTATION", "A3", "A4"),
    ]
    stability: dict[str, Any] = {}
    for branch in ("amyloid", "tau"):
        selection_record = read_json(
            output_root
            / "orchestration"
            / branch
            / "inner_only_arm_selection.json"
        )
        inner_only_selections[branch] = {
            "selection": selection_record["selection"],
            "locked_before_outer_evaluation": selection_record[
                "locked_before_outer_evaluation"
            ],
            "outer_test_consulted": selection_record["outer_test_consulted"],
        }
        metadata = memberships[branch].copy()
        metadata["age_group"] = pd.cut(
            metadata["age"], bins=[-np.inf, 70, 75, 80, np.inf], labels=False
        ).astype(str)
        arm_payloads = {arm: _load_arm_payload(output_root, branch, arm) for arm in config["execution"]["arms"]}
        stability[branch] = {}
        for arm, payload in arm_payloads.items():
            ibs = [float(row["integrated_brier_score"]) for row in payload["folds"]]
            uno = [float(row["uno_c_index"]) for row in payload["folds"]]
            stability[branch][arm] = {
                "outer_folds": 5,
                "ibs_mean": float(np.mean(ibs)),
                "ibs_sd": float(np.std(ibs, ddof=1)),
                "ibs_range": [float(min(ibs)), float(max(ibs))],
                "uno_mean": float(np.mean(uno)),
                "uno_sd": float(np.std(uno, ddof=1)),
                "uno_range": [float(min(uno)), float(max(uno))],
            }
        for pair_kind, reference_arm, challenger_arm in all_pairs:
            pair_id = f"{reference_arm}_vs_{challenger_arm}"
            pair_root = output_root / "analysis_pairs" / pair_id
            result = build_factorial_pair_reports(
                output_root=pair_root,
                branch=branch,
                pair_kind=pair_kind,
                reference_arm=reference_arm,
                challenger_arm=challenger_arm,
                reference=arm_payloads[reference_arm],
                challenger=arm_payloads[challenger_arm],
                metadata=metadata,
                horizons_days=[float(value) for value in config["shared"]["requested_horizons_days"]],
                bootstrap_iterations=int(config["robustness"]["bootstrap_iterations"]),
                seed=int(config["seed"]),
                margins=config["factorial_comparison"],
                expected_outer_folds=5,
            )
            pair_reports[f"{branch}:{pair_id}"] = result
    payload = {
        "schema_version": "4.0",
        "status": "COMPLETED",
        "analysis_role": "POST_HOC_INTERNAL_EXPLORATORY",
        "real_subject_count_unchanged": True,
        "augmentation_creates_new_subjects": False,
        "inner_only_arm_selections": inner_only_selections,
        "stability": stability,
        "pair_reports": pair_reports,
        "positive_claim_allowed": False,
        "external_claim_allowed": False,
        "science_gate_blockers": [
            "QUANTITATIVE_EQUIVALENCE_NOT_ESTABLISHED",
            "SCANNER_ROBUSTNESS_NOT_ESTABLISHED",
            "LOCKED_EXTERNAL_VALIDATION_MISSING",
        ],
    }
    atomic_write_json(output_root / "reports" / "factorial_benchmark_summary.json", payload)
    payload["privacy_scan"] = privacy_scan_reports(output_root / "reports")
    return payload


def _lock_inner_only_arm_selection(
    *,
    output_root: Path,
    branch: str,
    winners: dict[tuple[int, str], dict[str, Any]],
    config: dict[str, Any],
    master_contract_sha256: str,
) -> dict[str, Any]:
    """Lock the architecture preference before any outer result is materialized."""

    path = output_root / "orchestration" / branch / "inner_only_arm_selection.json"
    outer_root = output_root / "private" / "outer_evaluation" / branch
    if not path.is_file() and any(outer_root.glob("**/outer_result.json")):
        raise FactorialLauncherV4Error(
            f"OUTER_RESULT_EXISTS_BEFORE_ARM_SELECTION_LOCK:{branch}"
        )
    evidence: dict[str, list[float]] = {}
    evidence_records: dict[str, list[dict[str, Any]]] = {}
    for arm_id in config["execution"]["arms"]:
        outer_partition_means: list[float] = []
        records: list[dict[str, Any]] = []
        for fold in range(5):
            winner = winners.get((fold, arm_id))
            if not isinstance(winner, dict):
                raise FactorialLauncherV4Error(
                    f"INNER_WINNER_MISSING_FOR_ARM_LOCK:{branch}:{arm_id}:{fold}"
                )
            fold_records = list(winner.get("inner_fold_metric_records", []))
            if {int(row.get("inner_fold", -1)) for row in fold_records} != set(range(3)):
                raise FactorialLauncherV4Error(
                    f"INNER_METRIC_FOLDS_INCOMPLETE:{branch}:{arm_id}:{fold}"
                )
            fold_values: list[float] = []
            for row in fold_records:
                value = float(row.get("metric_value", float("nan")))
                if row.get("metric_name") != "IPCW_INTEGRATED_BRIER_SCORE":
                    raise FactorialLauncherV4Error(
                        f"INNER_SELECTION_METRIC_INVALID:{branch}:{arm_id}:{fold}"
                    )
                fold_values.append(value)
                records.append(
                    {
                        "outer_fold": fold,
                        "inner_fold": int(row["inner_fold"]),
                        "metric_name": row["metric_name"],
                        "metric_value": value,
                        "locked_checkpoint_sha256": row.get(
                            "locked_checkpoint_sha256"
                        ),
                    }
                )
            outer_partition_means.append(float(sum(fold_values) / len(fold_values)))
        if len(records) != 15 or len(outer_partition_means) != 5:
            raise FactorialLauncherV4Error(
                f"EXPECTED_FIFTEEN_INNER_METRICS:{branch}:{arm_id}"
            )
        evidence[arm_id] = outer_partition_means
        evidence_records[arm_id] = records
    selection = select_inner_only_factorial_arm(
        evidence,
        complexity_ranks={
            str(key): int(value)
            for key, value in config["execution"][
                "inner_only_arm_complexity_ranks"
            ].items()
        },
    )
    body = {
        "schema_version": "4.0",
        "branch": branch,
        "run_contract_sha256": master_contract_sha256,
        "selection": selection,
        "inner_evidence": evidence_records,
        "inner_evidence_sha256": sha256_mapping(evidence_records),
        "locked_before_outer_evaluation": True,
        "outer_test_consulted": False,
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
    }
    payload = {**body, "selection_record_sha256": sha256_mapping(body)}
    if path.is_file():
        existing = read_json(path)
        if existing != payload:
            raise FactorialLauncherV4Error(
                f"INNER_ARM_SELECTION_LOCK_MISMATCH:{branch}"
            )
        return existing
    atomic_write_json(path, payload)
    return payload


def _run(config: dict[str, Any], output_root: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd
    import torch

    _ensure_run_contract(output_root, preflight)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise FactorialLauncherV4Error("CUDA_REQUIRED_FOR_PRODUCTION_FACTORIAL_BENCHMARK")
    smoke_path = output_root / "provenance" / "gpu_smoke_v4.json"
    if (
        not smoke_path.is_file()
        or read_json(smoke_path).get("software_runtime_contract_sha256")
        != config["software_runtime_contract"]["contract_sha256"]
    ):
        _gpu_smoke(config, output_root)
    qc = read_json(_path(ROOT, config["shared"]["augmentation_qc_contract"]))
    mask_path = _path(ROOT, config["shared"]["foreground_mask"])
    fingerprints = preflight["input_fingerprints"]
    memberships: dict[str, pd.DataFrame] = {}
    selected_arms: dict[str, str] = {}
    failures: list[str] = []
    atomic_write_json(
        output_root / "orchestration_state.json",
        {
            "schema_version": "4.0",
            "stage": "TRAINING_RUNNING",
            "run_contract_sha256": preflight["run_contract_sha256"],
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        },
    )
    for branch in config["execution"]["branch_order"]:
        branch_cfg = config["branches"][branch]
        membership = pd.read_parquet(_path(ROOT, branch_cfg["membership"]))
        development = membership[membership["role"] == "development"].copy()
        memberships[branch] = development
        splits = read_json(_path(ROOT, branch_cfg["splits"]))
        cache_path = _path(ROOT, branch_cfg["pattern_cache"])
        orchestrator = ImageFactorialOrchestratorV4(
            config=config,
            output_root=output_root,
            branch=branch,
            membership_sha256=fingerprints[f"{branch}_membership"]["sha256"],
            cache_sha256=fingerprints[f"{branch}_pattern_cache"]["sha256"],
            mask_sha256=fingerprints["foreground_mask"]["sha256"],
            split_sha256=fingerprints[f"{branch}_splits"]["sha256"],
            augmentation_qc_sha256=qc["contract_sha256"],
            h5_cache_path=cache_path,
            mask_path=mask_path,
            augmentation_qc=qc,
        )
        orchestrator.verify_inputs()
        outer_rows = sorted(splits["outer_folds"], key=lambda row: int(row["fold"]))
        winners: dict[tuple[int, str], dict[str, Any]] = {}
        # Phase 1: complete all inner-only selection evidence before any outer access.
        for outer in outer_rows:
            fold = int(outer["fold"])
            train_keys = set(
                outer.get("train_subjects") or outer.get("train_subject_keys") or []
            )
            train_frame = development[
                development["patient_hash"].astype(str).isin(train_keys)
            ].copy()
            inner = _restrict_inner_splits(
                outer.get("inner_folds", []),
                set(train_frame["patient_hash"].astype(str)),
            )
            for arm_id in config["execution"]["arms"]:
                if orchestrator.winner_path(arm_id, fold).is_file():
                    winner = orchestrator.read_winner(arm_id, fold)
                    print(
                        json.dumps(
                            {
                                "event": "INNER_WINNER_RECOVERED",
                                "branch": branch,
                                "arm_id": arm_id,
                                "outer_fold": fold,
                            }
                        ),
                        flush=True,
                    )
                else:
                    winner = orchestrator.run_inner_hpo(
                        outer_fold=fold,
                        arm_id=arm_id,
                        outer_train_frame=train_frame,
                        inner_splits=inner,
                        tabular_columns=branch_cfg["model_features"],
                        hazard_bin_edges=config["shared"]["hazard_bin_edges_days"],
                        device=device,
                    )["winner"]
                winners[(fold, arm_id)] = winner
        selection_lock = _lock_inner_only_arm_selection(
            output_root=output_root,
            branch=branch,
            winners=winners,
            config=config,
            master_contract_sha256=orchestrator.master_contract_sha256,
        )
        selected_arms[branch] = str(selection_lock["selection"]["selected_arm"])
        print(
            json.dumps(
                {
                    "event": "INNER_ONLY_ARM_SELECTION_LOCKED",
                    "branch": branch,
                    "selected_arm": selection_lock["selection"]["selected_arm"],
                }
            ),
            flush=True,
        )

        # Phase 2: train locked candidates and expose each matched outer cell once.
        for outer in outer_rows:
            fold = int(outer["fold"])
            train_keys = set(outer.get("train_subjects") or outer.get("train_subject_keys") or [])
            test_keys = set(outer.get("test_subjects") or outer.get("test_subject_keys") or [])
            train_frame = development[
                development["patient_hash"].astype(str).isin(train_keys)
            ].copy()
            for arm_id in config["execution"]["arms"]:
                try:
                    winner = winners[(fold, arm_id)]
                    final = orchestrator.run_final_training(
                        outer_fold=fold,
                        arm_id=arm_id,
                        outer_train_frame=train_frame,
                        tabular_columns=branch_cfg["model_features"],
                        hazard_bin_edges=config["shared"]["hazard_bin_edges_days"],
                        winner=winner,
                        device=device,
                    )
                    metrics = orchestrator.outer_evaluate(
                        outer_fold=fold,
                        arm_id=arm_id,
                        run_contract_sha256=final["run_contract_sha256"],
                        evaluator=lambda keys=frozenset(test_keys), cfg=final["resolved_model_config"], locked=final["locked_checkpoint_path"], contract=final["run_contract_sha256"]: evaluate_locked_outer_v4(
                            locked_checkpoint_path=locked,
                            run_contract_digest=contract,
                            train_frame=train_frame,
                            test_frame=development[
                                development["patient_hash"].astype(str).isin(keys)
                            ].copy(),
                            h5_cache_path=cache_path,
                            tabular_columns=branch_cfg["model_features"],
                            arm_id=arm_id,
                            hazard_bin_edges=config["shared"]["hazard_bin_edges_days"],
                            model_config=cfg,
                            device=device,
                            amp=bool(config["model"]["amp"]),
                            batch_size=int(config["model"]["batch_size"]),
                            requested_horizons_days=config["shared"]["requested_horizons_days"],
                        ),
                    )
                    print(json.dumps({"event": "OUTER_CELL_FINALIZED", "branch": branch, "arm_id": arm_id, "outer_fold": fold, "ibs": metrics["integrated_brier_score"], "uno": metrics["uno_c_index"]}), flush=True)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    failures.append(f"{branch}:{arm_id}:fold_{fold}:{type(exc).__name__}:{exc}")
                    raise
        orchestrator.finalize_branch(expected_folds=5)
    if failures:
        raise FactorialLauncherV4Error("FAILED_INCOMPLETE_OUTER_CV:" + "|".join(failures))
    reports = _build_post_training_reports(
        output_root=output_root, config=config, memberships=memberships
    )
    atomic_write_json(
        output_root / "orchestration_state.json",
        {
            "schema_version": "4.0",
            "stage": "FINALIZED",
            "run_contract_sha256": preflight["run_contract_sha256"],
            "outer_cells_finalized": 50,
            "inner_only_selected_arms": selected_arms,
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
            "positive_claim_allowed": False,
            "external_claim_allowed": False,
        },
    )
    return {"status": "FINALIZED", "outer_cells": 50, "reports": reports}


def main() -> int:
    parser = argparse.ArgumentParser(description="Method Y PET-safe factorial image benchmark v4")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "method_y_image_factorial_v4.yaml")
    )
    parser.add_argument(
        "--mode", choices=("preflight", "smoke", "run", "status", "verify"), default="run"
    )
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()
    try:
        config_path = Path(args.config)
        preflight = _preflight(config_path)
        config = preflight["config"]
        output_root = Path(preflight["output_root"])
        if args.mode == "preflight":
            print(json.dumps({key: value for key, value in preflight.items() if key not in {"config", "run_contract_body", "input_fingerprints"}}, indent=2), flush=True)
            return 0
        if args.mode == "smoke":
            print(json.dumps(_gpu_smoke(config, output_root), indent=2), flush=True)
            return 0
        if args.mode == "status":
            state = output_root / "orchestration_state.json"
            print(json.dumps(read_json(state) if state.is_file() else {"status": "NOT_STARTED"}, indent=2), flush=True)
            return 0
        if args.mode == "verify":
            from scripts.verify_method_y_image_factorial_v4 import verify

            result = verify(config_path=config_path, level="final")
            print(json.dumps(result, indent=2), flush=True)
            return 0 if result["status"] == "PASS" else 2
        lock_path = output_root / ".factorial_v4.writer.lock"
        with WriterLock(
            lock_path,
            output_root=output_root,
            run_contract_digest=preflight["run_contract_sha256"],
        ):
            result = _run(config, output_root, preflight)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except KeyboardInterrupt:
        print(json.dumps({"status": "INTERRUPTED", "resume": "RERUN_SAME_COMMAND"}), flush=True)
        return 130
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": f"{type(exc).__name__}:{exc}"}, indent=2), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
