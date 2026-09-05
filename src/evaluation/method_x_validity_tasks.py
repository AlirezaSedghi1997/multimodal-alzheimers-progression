"""Machine-controlled MX-03 task contracts and private artifact materializers.

This module contains only deterministic orchestration helpers.  It never
selects a model from an outer-test score, never writes subject identifiers,
and never marks a missing or malformed result as completed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.utils.io_helpers import atomic_write_json
from src.utils.reproducibility import canonical_digest


REFERENCE_BACKBONE = "resnet34_3d"
ALLOWED_FEATURES: tuple[str, ...] = (
    "age",
    "sex",
    "education_years",
    "apoe4_carrier",
    "mmse_base",
)
EXECUTABLE_TASK_IDS: tuple[str, ...] = (
    "random_labels",
    "random_noise",
    "mri_only",
    "tabular_only",
    *(f"feature_drop_{name}" for name in ALLOWED_FEATURES),
)

# Five-model protocol (append-only amendment
# `amendments/method_x_battery_v3_five_model_amendment.v1.json`): the battery
# manifest keeps 7 structural rows, but only these five backbones are effective
# members. The remaining two are protocol exclusions, never training members.
EFFECTIVE_FIVE: tuple[str, ...] = (
    "resnet10_3d",
    "resnet18_3d",
    "resnet34_3d",
    "densenet121_3d",
    "mednext_3d_s",
)
EXCLUDED_BACKBONES: tuple[str, ...] = (
    "resnet50_3d",
    "efficientnet3d_b0",
)
# Reconciler-issued terminal training states for effective members. The live
# manifest records COMPLETED_VALIDATED_REPAIRED / COMPLETED_VALIDATED_RECONCILED
# for rows repaired by the transactional reconciler; these are terminal and must
# not be confused with non-terminal RUNNING/PAUSED states.
TERMINAL_TRAINING_STATUSES: frozenset[str] = frozenset(
    {
        "COMPLETED",
        "COMPLETED_VALIDATED_REPAIRED",
        "COMPLETED_VALIDATED_RECONCILED",
    }
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _relative_or_string(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _protocol_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    protocol = manifest.get("locked_protocol")
    if not isinstance(protocol, Mapping):
        protocol = {}
    return {
        "resolution": int(protocol.get("resolution", 128)),
        "folds": int(protocol.get("folds", 5)),
        "epochs": int(protocol.get("epochs", 100)),
        "seed": int(protocol.get("seed", 13)),
        "common": dict(protocol.get("common") or {}),
    }


def _task_arguments(task_id: str) -> list[str]:
    if task_id == "random_labels":
        return ["--sanity_random_labels", "--input_mode", "mri_plus_tabular"]
    if task_id == "random_noise":
        return ["--sanity_random_noise_images", "--input_mode", "mri_only"]
    if task_id == "mri_only":
        return ["--input_mode", "mri_only"]
    if task_id == "tabular_only":
        return ["--input_mode", "tabular_only"]
    if task_id.startswith("feature_drop_"):
        feature = task_id.removeprefix("feature_drop_")
        if feature not in ALLOWED_FEATURES:
            raise ValueError(f"UNKNOWN_METHOD_X_FEATURE_DROP:{feature}")
        return [
            "--input_mode",
            "mri_plus_tabular",
            "--drop_tabular_feature",
            feature,
        ]
    raise ValueError(f"UNKNOWN_METHOD_X_VALIDITY_TASK:{task_id}")


def _replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def build_task_contracts(
    *,
    root: Path,
    output_root: Path,
    manifest: Mapping[str, Any],
    python: str,
) -> list[dict[str, Any]]:
    """Build immutable, exact commands for all MX-03 executable arms."""

    from scripts.run_method_x_claim_benchmark import build_training_command

    root = root.resolve()
    output_root = output_root.resolve()
    locked = manifest.get("locked_inputs") if isinstance(manifest.get("locked_inputs"), Mapping) else {}
    spm = manifest.get("spm_gate_snapshot") if isinstance(manifest.get("spm_gate_snapshot"), Mapping) else {}
    cache_sha = str(locked.get("cache_sha256") or "")
    spm_sha = str(spm.get("snapshot_sha256") or "")
    protocol = _protocol_config(manifest)
    config = {**protocol, "common": protocol["common"]}
    if not valid_sha256(cache_sha) or not valid_sha256(spm_sha):
        raise RuntimeError("MX03_TASK_CONTRACT_REQUIRES_LOCKED_CACHE_AND_SPM_DIGESTS")

    contracts: list[dict[str, Any]] = []
    for sequence, task_id in enumerate(EXECUTABLE_TASK_IDS):
        task_output = output_root / "mx03_runs" / task_id
        command = build_training_command(
            python=python,
            config=config,
            backbone=REFERENCE_BACKBONE,
            output_dir=task_output,
            cache_sha256=cache_sha,
            spm_snapshot_sha256=spm_sha,
            phase="dev",
            extra_args=_task_arguments(task_id),
        )
        body = {
            "schema_version": 1,
            "task_id": task_id,
            "sequence": sequence,
            "stage": "MX-03",
            "kind": "sanity" if task_id in {"random_labels", "random_noise"} else "ablation",
            "reference_backbone": REFERENCE_BACKBONE,
            "output_dir": _relative_or_string(task_output, root),
            "artifact": _relative_or_string(
                output_root
                / "private"
                / (
                    "sanity_random_labels.json"
                    if task_id == "random_labels"
                    else "sanity_random_noise.json"
                    if task_id == "random_noise"
                    else "ablation_mri_only.json"
                    if task_id == "mri_only"
                    else "ablation_tabular_only.json"
                    if task_id == "tabular_only"
                    else f"ablation_{task_id}.json"
                ),
                root,
            ),
            "locked_cache_sha256": cache_sha,
            "locked_spm_snapshot_sha256": spm_sha,
            "locked_resolution": protocol["resolution"],
            "locked_folds": protocol["folds"],
            "locked_seed": protocol["seed"],
            "outer_test_used_for_selection": False,
            "training_started": False,
            "resume_policy": "exact_run_contract_and_last_committed_optimizer_step",
            "command": [
                str(value).replace(str(root), "PROJECT_ROOT") for value in command
            ],
        }
        body["command_sha256"] = canonical_digest(body["command"])
        body["contract_sha256"] = canonical_digest(body)
        contracts.append(body)
    return contracts


def baseline_is_terminal(manifest: Mapping[str, Any]) -> bool:
    """True only for the terminal five-model battery (7 structural rows).

    Effective members must be COMPLETED with a reconciler-recognized terminal
    training status; excluded backbones must remain PROTOCOL_EXCLUDED_NOT_STARTED
    in both row and training status. Any other shape is NOT terminal.
    """
    rows = manifest.get("models")
    if not isinstance(rows, list) or len(rows) != 7:
        return False
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        name = str(row.get("backbone") or "")
        if not name or name in by_name:
            return False
        by_name[name] = row
    if set(by_name) != set(EFFECTIVE_FIVE) | set(EXCLUDED_BACKBONES):
        return False
    for name in EFFECTIVE_FIVE:
        row = by_name[name]
        training = row.get("training")
        if row.get("status") != "COMPLETED":
            return False
        if not isinstance(training, Mapping):
            return False
        if str(training.get("status") or "") not in TERMINAL_TRAINING_STATUSES:
            return False
    for name in EXCLUDED_BACKBONES:
        row = by_name[name]
        training = row.get("training")
        if row.get("status") != "PROTOCOL_EXCLUDED_NOT_STARTED":
            return False
        if not isinstance(training, Mapping):
            return False
        if training.get("status") != "PROTOCOL_EXCLUDED_NOT_STARTED":
            return False
    return True


def _outer_result(path: Path) -> dict[str, Any] | None:
    payload = read_json(path)
    if payload is None:
        return None
    labels = payload.get("y_true")
    probabilities = payload.get("y_prob")
    if (
        payload.get("evaluation_count") != 1
        or payload.get("raw_identifiers_stored") is not False
        or not isinstance(labels, list)
        or not isinstance(probabilities, list)
        or not labels
        or len(labels) != len(probabilities)
        or not isinstance(payload.get("split_membership_digest"), str)
        or not valid_sha256(payload.get("split_membership_digest"))
    ):
        return None
    if any(int(value) not in {0, 1} for value in labels):
        return None
    if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in probabilities):
        return None
    return payload


def materialize_private_oof(
    *, root: Path, output_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive paired private OOF probabilities from locked outer results."""

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "BLOCKED",
        "subject_level_vectors_exported": False,
        "blockers": [],
        "models": {},
        "source_artifacts": [],
    }
    if not baseline_is_terminal(manifest):
        report["blockers"].append("BASELINE_BATTERY_NOT_FINALIZED")
        return report
    expected_spm = str((manifest.get("spm_gate_snapshot") or {}).get("snapshot_sha256") or "")
    fold_labels: list[list[int]] | None = None
    fold_memberships: dict[str, str] = {}
    model_rows = manifest.get("models") or []
    for row in sorted(model_rows, key=lambda item: int(item.get("sequence", 0))):
        name = str(row.get("backbone"))
        if name in EXCLUDED_BACKBONES:
            # Protocol exclusions are smoke-only and carry no outer results.
            continue
        model_dir = output_root / str(row.get("output_dir"))
        metrics_path = model_dir / "three_d_metrics.json"
        metrics = read_json(metrics_path)
        contract = read_json(model_dir / "run_contract.json")
        if not metrics or not contract:
            report["blockers"].append(f"BASELINE_ARTIFACT_MISSING:{name}")
            continue
        contract_body = contract.get("run_contract") if isinstance(contract.get("run_contract"), Mapping) else {}
        if contract_body.get("spm_snapshot_sha256") != expected_spm:
            report["blockers"].append(f"SPM_SNAPSHOT_DIGEST_MISMATCH:{name}")
        aggregate = metrics.get("aggregate") if isinstance(metrics.get("aggregate"), Mapping) else {}
        if aggregate.get("status") != "COMPLETED" or int(aggregate.get("completed_folds", 0)) != 5:
            report["blockers"].append(f"BASELINE_METRICS_NOT_FINALIZED:{name}")
        all_labels: list[int] = []
        all_probabilities: list[float] = []
        for fold in range(5):
            outer_path = model_dir / f"method_x_3d_fold_{fold}_outer_eval.json"
            outer = _outer_result(outer_path)
            if outer is None:
                report["blockers"].append(f"OUTER_RESULT_INVALID:{name}:fold_{fold}")
                continue
            labels = [int(value) for value in outer["y_true"]]
            probabilities = [float(value) for value in outer["y_prob"]]
            if fold_labels is None:
                fold_labels = []
            if len(fold_labels) <= fold:
                fold_labels.append(labels)
            elif fold_labels[fold] != labels:
                report["blockers"].append(f"PAIRED_LABEL_ORDER_MISMATCH:{name}:fold_{fold}")
            fold_key = f"fold_{fold}"
            digest = str(outer["split_membership_digest"])
            if fold_key in fold_memberships and fold_memberships[fold_key] != digest:
                report["blockers"].append(f"PAIRED_MEMBERSHIP_MISMATCH:{name}:fold_{fold}")
            fold_memberships[fold_key] = digest
            all_labels.extend(labels)
            all_probabilities.extend(probabilities)
            source_digest = sha256_file(outer_path)
            if source_digest:
                report["source_artifacts"].append(
                    {"path": _relative_or_string(outer_path, root), "sha256": source_digest}
                )
        report["models"][name] = {
            "probabilities": all_probabilities,
            "fold_count": 5,
            "prediction_count": len(all_probabilities),
            "run_contract_digest": contract.get("run_contract_digest"),
        }
    if fold_labels is None or len(fold_labels) != 5:
        report["blockers"].append("PAIRED_OOF_FOLD_COVERAGE_INCOMPLETE")
    else:
        y_true = [value for fold in fold_labels for value in fold]
        counts = {len(row.get("probabilities", [])) for row in report["models"].values()}
        if len(counts) != 1 or (counts and next(iter(counts)) != len(y_true)):
            report["blockers"].append("PAIRED_OOF_VECTOR_LENGTH_MISMATCH")
        report["y_true"] = y_true
    report["membership_digest"] = canonical_digest(fold_memberships)
    report["expected_spm_snapshot_sha256"] = expected_spm
    report["source_artifacts"] = sorted(
        report["source_artifacts"], key=lambda item: str(item["path"])
    )
    report["blockers"] = sorted(set(str(value) for value in report["blockers"]))
    report["status"] = "FINALIZED" if not report["blockers"] else "BLOCKED"
    if report["status"] == "FINALIZED":
        atomic_write_json(output_root / "private/oof_predictions.json", report)
    return report


def materialize_baseline_diagnostics(
    *, root: Path, output_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Materialize calibration/cohort summaries from fold-local reports only."""

    result: dict[str, Any] = {"status": "BLOCKED", "blockers": [], "calibration": [], "cohort_probe": []}
    if not baseline_is_terminal(manifest):
        result["blockers"].append("BASELINE_BATTERY_NOT_FINALIZED")
        return result
    for row in manifest.get("models", []):
        name = str(row.get("backbone"))
        if name in EXCLUDED_BACKBONES:
            # Protocol exclusions carry no calibration/cohort artifacts.
            continue
        metrics = read_json(output_root / str(row.get("output_dir")) / "three_d_metrics.json") or {}
        aggregate = metrics.get("aggregate") if isinstance(metrics.get("aggregate"), Mapping) else {}
        temperature = aggregate.get("temperature_scaling")
        if not isinstance(temperature, Mapping) or temperature.get("status") != "FITTED_ON_INNER_VALIDATION_ONLY":
            result["blockers"].append(f"CALIBRATION_INNER_VALIDATION_ARTIFACT_MISSING_OR_INCOMPLETE:{name}")
        else:
            result["calibration"].append(
                {
                    "backbone": name,
                    "status": "COMPLETED",
                    "fit_scope": temperature.get("fit_scope"),
                    "test_labels_used_for_calibration": temperature.get("test_labels_used_for_calibration"),
                    "aggregate_test_calibrated": temperature.get("aggregate_test_calibrated"),
                }
            )
        probe = aggregate.get("cohort_probe")
        if not isinstance(probe, Mapping) or probe.get("status") != "COMPLETED":
            result["blockers"].append(f"COHORT_PROBE_ARTIFACT_MISSING_OR_INCOMPLETE:{name}")
        else:
            result["cohort_probe"].append(
                {
                    "backbone": name,
                    "status": "COMPLETED",
                    "auc": aggregate.get("cohort_probe_auc"),
                    "folds": probe.get("folds", []),
                    "fit_scope": "inner_validation_only",
                }
            )
    result["blockers"] = sorted(set(str(value) for value in result["blockers"]))
    result["status"] = "COMPLETED" if not result["blockers"] else "BLOCKED"
    if result["status"] == "COMPLETED":
        atomic_write_json(output_root / "private/calibration_summary.json", {
            "schema_version": 1,
            "status": "COMPLETED",
            "fit_scope": "inner_validation_only_by_fold",
            "test_labels_used_for_calibration": False,
            "models": result["calibration"],
        })
        atomic_write_json(output_root / "private/cohort_probe_summary.json", {
            "schema_version": 1,
            "status": "COMPLETED",
            "fit_scope": "inner_validation_only_by_fold",
            "models": result["cohort_probe"],
        })
    return result


def read_task_metrics(
    *, root: Path, output_root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    configured = Path(str(contract.get("output_dir", "")))
    task_dir = configured if configured.is_absolute() else root / configured
    if not task_dir.resolve().is_relative_to(output_root.resolve()):
        return {"status": "BLOCKED", "blocker": "TASK_OUTPUT_OUTSIDE_BATTERY_ROOT"}
    metrics = read_json(task_dir / "three_d_metrics.json")
    if not metrics or not isinstance(metrics.get("aggregate"), Mapping):
        return {"status": "BLOCKED", "blocker": "TASK_METRICS_MISSING_OR_INVALID"}
    aggregate = dict(metrics["aggregate"])
    if aggregate.get("status") != "COMPLETED":
        return {"status": "BLOCKED", "blocker": "TASK_METRICS_NOT_COMPLETED", "aggregate": aggregate}
    if int(aggregate.get("folds", 0)) != 5:
        return {"status": "BLOCKED", "blocker": "TASK_FOLD_COUNT_INVALID", "aggregate": aggregate}
    return {
        "status": "COMPLETED",
        "aggregate": {
            key: aggregate.get(key)
            for key in (
                "status", "mode", "phase", "backbone", "input_mode", "drop_tabular_feature",
                "roc_auc", "pr_auc", "brier", "ece", "completed_folds", "requested_resolution",
                "outer_test_evaluation_policy", "outer_test_evaluation_count",
            )
        },
        "metrics_sha256": sha256_file(task_dir / "three_d_metrics.json"),
        "outer_test_used_for_selection": False,
        "training_started": True,
    }


def sanity_artifact(task_id: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = metrics.get("aggregate") if isinstance(metrics.get("aggregate"), Mapping) else {}
    auc = aggregate.get("roc_auc")
    near_chance = isinstance(auc, (int, float)) and math.isfinite(float(auc)) and abs(float(auc) - 0.5) <= 0.10
    return {
        "schema_version": 1,
        "status": "COMPLETED" if metrics.get("status") == "COMPLETED" else "BLOCKED",
        "task_id": task_id,
        "roc_auc": auc,
        "near_chance_gate": near_chance,
        "gate": "PASS" if near_chance else "FAIL",
        "outer_test_used_for_selection": False,
        "subject_level_vectors_exported": False,
        "metrics_sha256": metrics.get("metrics_sha256"),
    }


def ablation_artifact(task_id: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = metrics.get("aggregate") if isinstance(metrics.get("aggregate"), Mapping) else {}
    return {
        "schema_version": 1,
        "status": metrics.get("status", "BLOCKED"),
        "task_id": task_id,
        "input_mode": aggregate.get("input_mode"),
        "drop_tabular_feature": aggregate.get("drop_tabular_feature", "none"),
        "roc_auc": aggregate.get("roc_auc"),
        "pr_auc": aggregate.get("pr_auc"),
        "brier": aggregate.get("brier"),
        "ece": aggregate.get("ece"),
        "outer_test_used_for_selection": False,
        "subject_level_vectors_exported": False,
        "metrics_sha256": metrics.get("metrics_sha256"),
    }


def aggregate_feature_drop_artifact(artifacts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(item) for item in artifacts]
    blockers = [str(item.get("task_id")) for item in rows if item.get("status") != "COMPLETED"]
    return {
        "schema_version": 1,
        "status": "COMPLETED" if not blockers and len(rows) == len(ALLOWED_FEATURES) else "BLOCKED",
        "features": rows,
        "blockers": blockers,
        "outer_test_used_for_selection": False,
        "subject_level_vectors_exported": False,
    }


def _oriented_delta(summary: Mapping[str, Any], *, invert: bool) -> dict[str, Any] | None:
    try:
        mean = float(summary["delta_mean"])
        low, high = (float(value) for value in summary["ci95"])
        resamples = int(summary["resamples_completed"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (mean, low, high)) or low > high:
        return None
    if invert:
        mean, low, high = -mean, -high, -low
    return {
        "delta": mean,
        "ci95": [low, high],
        "resamples_completed": resamples,
        "valid_paired_summary": bool(
            summary.get("status") == "COMPLETED"
            and summary.get("delta_definition") == "metric_b_minus_metric_a"
            and summary.get("subject_level_vectors_exported") is False
            and resamples >= 1000
        ),
    }


def _finite_metric(value: Any, *, probability: bool = False) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if probability and not 0.0 <= numeric <= 1.0:
        return None
    return numeric


def build_promotion_decision(
    *,
    pairwise: Mapping[str, Any],
    model_names: Sequence[str],
    calibration_ece: Mapping[str, float],
    cohort_probe_auc: Mapping[str, float],
    sanity_passed: bool,
    reference_backbone: str = REFERENCE_BACKBONE,
) -> dict[str, Any]:
    """Apply the pre-specified reporting promotion rule without tuning models."""

    decisions: list[dict[str, Any]] = []
    blockers: list[str] = []
    if reference_backbone not in model_names:
        blockers.append("REFERENCE_BACKBONE_MISSING")
    reference_ece = _finite_metric(calibration_ece.get(reference_backbone))
    reference_cohort = _finite_metric(
        cohort_probe_auc.get(reference_backbone), probability=True
    )
    for candidate in sorted(name for name in model_names if name != reference_backbone):
        left, right = sorted((reference_backbone, candidate))
        pair = pairwise.get(f"{left}__vs__{right}")
        if not isinstance(pair, Mapping):
            decisions.append(
                {"backbone": candidate, "status": "NOT_PROMOTED", "reason": "PAIRED_COMPARISON_MISSING"}
            )
            blockers.append(f"PAIRED_COMPARISON_MISSING:{candidate}")
            continue
        invert = right == reference_backbone
        roc = _oriented_delta(pair.get("roc_auc", {}), invert=invert)
        pr = _oriented_delta(pair.get("pr_auc", {}), invert=invert)
        brier = _oriented_delta(pair.get("brier", {}), invert=invert)
        candidate_ece = _finite_metric(calibration_ece.get(candidate))
        candidate_cohort = _finite_metric(
            cohort_probe_auc.get(candidate), probability=True
        )
        required = all(
            value is not None
            for value in (
                roc,
                pr,
                brier,
                reference_ece,
                candidate_ece,
                reference_cohort,
                candidate_cohort,
            )
        )
        paired_bootstrap_ready = bool(
            required
            and roc["valid_paired_summary"]
            and pr["valid_paired_summary"]
            and brier["valid_paired_summary"]
        )
        reference_cohort_strength = (
            abs(float(reference_cohort) - 0.5) if reference_cohort is not None else None
        )
        candidate_cohort_strength = (
            abs(float(candidate_cohort) - 0.5) if candidate_cohort is not None else None
        )
        criteria = {
            "sanity_passed": bool(sanity_passed),
            "paired_bootstrap_min_1000_completed": paired_bootstrap_ready,
            "roc_auc_delta_gte_0_02": bool(roc and roc["delta"] >= 0.02),
            "roc_auc_lower_ci_positive": bool(roc and roc["ci95"][0] > 0.0),
            "brier_not_worse_by_more_than_0_01": bool(brier and brier["delta"] <= 0.01),
            "ece_not_worse_by_more_than_0_02": bool(
                reference_ece is not None
                and candidate_ece is not None
                and float(candidate_ece) - float(reference_ece) <= 0.02
            ),
            "cohort_probe_not_worse": bool(
                reference_cohort_strength is not None
                and candidate_cohort_strength is not None
                and candidate_cohort_strength <= reference_cohort_strength + 1e-12
            ),
            "all_required_metrics_present": required,
        }
        if not required or not paired_bootstrap_ready:
            blockers.append(f"PROMOTION_EVIDENCE_INCOMPLETE:{candidate}")
        promoted = all(criteria.values())
        decisions.append(
            {
                "backbone": candidate,
                "status": "PROMOTED_FOR_REPORTING" if promoted else "NOT_PROMOTED",
                "criteria": criteria,
                "paired_delta_candidate_minus_reference": {
                    "roc_auc": roc,
                    "pr_auc": pr,
                    "brier": brier,
                },
                "calibration_ece": {"reference": reference_ece, "candidate": candidate_ece},
                "cohort_probe_auc": {
                    "reference": reference_cohort,
                    "candidate": candidate_cohort,
                    "reference_distance_from_chance": reference_cohort_strength,
                    "candidate_distance_from_chance": candidate_cohort_strength,
                },
            }
        )
    promoted = [row for row in decisions if row.get("status") == "PROMOTED_FOR_REPORTING"]
    promoted.sort(
        key=lambda row: float(
            ((row.get("paired_delta_candidate_minus_reference") or {}).get("roc_auc") or {}).get(
                "delta", float("-inf")
            )
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "status": "COMPLETED" if not blockers else "BLOCKED",
        "reference_backbone": reference_backbone,
        "reporting_backbone": promoted[0]["backbone"] if promoted else reference_backbone,
        "promotion_occurred": bool(promoted),
        "decisions": decisions,
        "blockers": blockers,
        "rule": {
            "roc_auc_min_delta": 0.02,
            "roc_auc_lower_ci_must_be_positive": True,
            "brier_max_worsening": 0.01,
            "ece_max_worsening": 0.02,
            "cohort_probe_must_not_worsen": True,
            "paired_subject_bootstrap_min_resamples": 1000,
        },
        "outer_test_used_for_training_or_hyperparameter_selection": False,
        "selection_scope": "final_reporting_comparison_only",
        "final_claim_allowed": False,
    }


def materialize_promotion_report(
    *, output_root: Path, pairwise: Mapping[str, Any]
) -> dict[str, Any]:
    oof = read_json(output_root / "private/oof_predictions.json") or {}
    calibration = read_json(output_root / "private/calibration_summary.json") or {}
    cohort = read_json(output_root / "private/cohort_probe_summary.json") or {}
    sanity_rows = [
        read_json(output_root / "private/sanity_random_labels.json") or {},
        read_json(output_root / "private/sanity_random_noise.json") or {},
    ]
    calibration_ece: dict[str, float] = {}
    for row in calibration.get("models", []) if isinstance(calibration.get("models"), list) else []:
        metrics = row.get("aggregate_test_calibrated") if isinstance(row, Mapping) else None
        value = metrics.get("ece") if isinstance(metrics, Mapping) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            calibration_ece[str(row.get("backbone"))] = float(value)
    cohort_auc: dict[str, float] = {}
    for row in cohort.get("models", []) if isinstance(cohort.get("models"), list) else []:
        value = row.get("auc") if isinstance(row, Mapping) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            cohort_auc[str(row.get("backbone"))] = float(value)
    report = build_promotion_decision(
        pairwise=pairwise,
        model_names=sorted(str(name) for name in (oof.get("models") or {})),
        calibration_ece=calibration_ece,
        cohort_probe_auc=cohort_auc,
        sanity_passed=all(row.get("near_chance_gate") is True for row in sanity_rows),
    )
    atomic_write_json(output_root / "reports/mx03_promotion_report.json", report)
    return report
