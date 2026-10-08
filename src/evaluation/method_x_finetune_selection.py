"""Private inner-validation evidence and promotion calculations for MX-04."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.reproducibility import canonical_digest


def _read_object(path: Path, error: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(error) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(error)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any, error: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(error) from exc
    if not math.isfinite(number):
        raise RuntimeError(error)
    return number


def roc_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Rank-based binary ROC-AUC with average ranks for ties."""

    if len(y_true) != len(y_prob) or not y_true:
        raise RuntimeError("MX04_ROC_INPUT_INVALID")
    pairs = sorted((float(score), int(label)) for label, score in zip(y_true, y_prob))
    positives = sum(label == 1 for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError("MX04_ROC_REQUIRES_BOTH_CLASSES")
    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            pairs[position][1] == 1 for position in range(index, end)
        )
        index = end
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    if len(y_true) != len(y_prob) or not y_true:
        raise RuntimeError("MX04_BRIER_INPUT_INVALID")
    return float(
        sum((float(label) - float(probability)) ** 2 for label, probability in zip(y_true, y_prob))
        / len(y_true)
    )


def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    if len(y_true) != len(y_prob) or not y_true or int(bins) < 2:
        raise RuntimeError("MX04_ECE_INPUT_INVALID")
    total = len(y_true)
    error = 0.0
    for bin_index in range(int(bins)):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, probability in enumerate(y_prob)
            if lower <= float(probability) < upper
            or (bin_index == bins - 1 and float(probability) == 1.0)
        ]
        if not members:
            continue
        confidence = sum(float(y_prob[index]) for index in members) / len(members)
        accuracy = sum(int(y_true[index]) for index in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return float(error)


def validate_selection_run(
    run_root: str | Path,
    *,
    expected_backbone: str,
    expected_seed: int,
    folds: int = 5,
) -> dict[str, Any]:
    """Validate a terminal private selection run without reading outer data."""

    root = Path(run_root).resolve()
    metrics_path = root / "three_d_metrics.json"
    evidence_path = root / "selection_evidence.json"
    contract_path = root / "run_contract.json"
    metrics_payload = _read_object(metrics_path, "MX04_SELECTION_METRICS_INVALID")
    aggregate = metrics_payload.get("aggregate")
    metric_rows = metrics_payload.get("folds")
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("status") != "COMPLETED"
        or aggregate.get("hpo_selection_only") is not True
        or int(aggregate.get("outer_test_access_count", -1)) != 0
        or aggregate.get("outer_test_used_for_selection") is not False
        or int(aggregate.get("folds", -1)) != int(folds)
        or int(aggregate.get("completed_folds", -1)) != int(folds)
        or not isinstance(metric_rows, list)
        or len(metric_rows) != int(folds)
    ):
        raise RuntimeError("MX04_SELECTION_METRICS_CONTRACT_INVALID")
    if any(
        not isinstance(row, Mapping)
        or row.get("hpo_selection_only") is not True
        or int(row.get("outer_test_evaluation_count", -1)) != 0
        for row in metric_rows
    ):
        raise RuntimeError("MX04_SELECTION_FOLD_METRICS_INVALID")
    if {int(row.get("fold", -1)) for row in metric_rows} != set(range(int(folds))):
        raise RuntimeError("MX04_SELECTION_FOLD_IDENTITIES_INVALID")

    evidence = _read_object(evidence_path, "MX04_SELECTION_EVIDENCE_INVALID")
    evidence_body = {key: value for key, value in evidence.items() if key != "integrity_sha256"}
    descriptor = aggregate.get("selection_evidence")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("artifact") != "method_x_private_selection_evidence"
        or evidence.get("integrity_sha256") != canonical_digest(evidence_body)
        or evidence.get("backbone") != expected_backbone
        or evidence.get("seed") != int(expected_seed)
        or evidence.get("raw_identifiers_stored") is not False
        or evidence.get("publication_allowed") is not False
        or evidence.get("outer_test_used_for_selection") is not False
        or int(evidence.get("outer_test_access_count", -1)) != 0
        or not isinstance(descriptor, Mapping)
        or descriptor.get("sha256") != _sha256(evidence_path)
        or descriptor.get("integrity_sha256") != evidence.get("integrity_sha256")
    ):
        raise RuntimeError("MX04_SELECTION_EVIDENCE_CONTRACT_INVALID")
    contract_payload = _read_object(contract_path, "MX04_SELECTION_RUN_CONTRACT_INVALID")
    contract = contract_payload.get("run_contract")
    parameters = contract.get("parameters") if isinstance(contract, Mapping) else None
    if (
        not isinstance(contract, Mapping)
        or not isinstance(parameters, Mapping)
        or contract.get("digest") != evidence.get("run_contract_digest")
        or int(contract.get("seed", -1)) != int(expected_seed)
        or parameters.get("backbone") != expected_backbone
        or parameters.get("hpo_selection_only") is not True
    ):
        raise RuntimeError("MX04_SELECTION_RUN_CONTRACT_INVALID")

    rows = evidence.get("folds")
    if not isinstance(rows, list) or len(rows) != int(folds):
        raise RuntimeError("MX04_SELECTION_EVIDENCE_FOLDS_INCOMPLETE")
    normalized_rows: list[dict[str, Any]] = []
    for fold in range(int(folds)):
        matches = [row for row in rows if isinstance(row, Mapping) and row.get("fold") == fold]
        if len(matches) != 1:
            raise RuntimeError("MX04_SELECTION_EVIDENCE_FOLD_IDENTITIES_INVALID")
        row = matches[0]
        labels = row.get("y_true")
        probabilities = row.get("y_prob")
        logits = row.get("logits")
        digest = str(row.get("split_membership_digest") or "")
        if (
            not isinstance(labels, list)
            or not isinstance(probabilities, list)
            or not isinstance(logits, list)
            or not labels
            or len(labels) != len(probabilities)
            or len(labels) != len(logits)
            or any(int(value) not in {0, 1} for value in labels)
            or any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in probabilities)
            or any(not math.isfinite(float(value)) for value in logits)
            or len(digest) != 64
        ):
            raise RuntimeError(f"MX04_SELECTION_EVIDENCE_FOLD_INVALID:{fold}")
        state = _read_object(
            root / f"method_x_3d_fold_{fold}_state.json",
            f"MX04_SELECTION_FOLD_STATE_INVALID:{fold}",
        )
        if (
            state.get("stage") != "FINALIZED"
            or int((state.get("details") or {}).get("outer_test_access_count", -1)) != 0
            or (root / f"method_x_3d_fold_{fold}_outer_eval.json").exists()
            or not (root / f"method_x_3d_fold_{fold}.pt").is_file()
        ):
            raise RuntimeError(f"MX04_SELECTION_FOLD_NOT_FINALIZED:{fold}")
        normalized_rows.append(
            {
                "fold": fold,
                "split_membership_digest": digest,
                "y_true": [int(value) for value in labels],
                "y_prob": [float(value) for value in probabilities],
            }
        )

    allocated = _finite_float(
        aggregate.get("vram_peak_allocated_gb"),
        "MX04_SELECTION_VRAM_ALLOCATED_MISSING",
    )
    reserved = _finite_float(
        aggregate.get("vram_peak_reserved_gb"),
        "MX04_SELECTION_VRAM_RESERVED_MISSING",
    )
    cohort_auc = _finite_float(
        aggregate.get("cohort_probe_auc"),
        "MX04_SELECTION_COHORT_PROBE_MISSING",
    )
    labels = [label for row in normalized_rows for label in row["y_true"]]
    probabilities = [probability for row in normalized_rows for probability in row["y_prob"]]
    return {
        "model_root": str(root),
        "backbone": expected_backbone,
        "seed": int(expected_seed),
        "run_contract_digest": str(contract["digest"]),
        "folds": normalized_rows,
        "all_five_folds_finalized": True,
        "outer_test_access_count": 0,
        "roc_auc": roc_auc(labels, probabilities),
        "brier": brier_score(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities),
        "cohort_probe_auc": cohort_auc,
        "peak_vram_allocated_gb": allocated,
        "peak_vram_reserved_gb": reserved,
        "selection_evidence_sha256": _sha256(evidence_path),
    }


def _aligned_vectors(
    candidate_runs: Sequence[Mapping[str, Any]],
    comparator_runs: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[float], list[float]]:
    candidate_by_seed = {int(run["seed"]): run for run in candidate_runs}
    comparator_by_seed = {int(run["seed"]): run for run in comparator_runs}
    if set(candidate_by_seed) != {13, 29, 47} or set(comparator_by_seed) != {13, 29, 47}:
        raise RuntimeError("MX04_REPLICATION_SEEDS_INCOMPLETE")
    labels: list[int] = []
    candidate_probabilities: list[float] = []
    comparator_probabilities: list[float] = []
    for seed in (13, 29, 47):
        candidate_folds = {int(row["fold"]): row for row in candidate_by_seed[seed]["folds"]}
        comparator_folds = {int(row["fold"]): row for row in comparator_by_seed[seed]["folds"]}
        if set(candidate_folds) != set(comparator_folds) or set(candidate_folds) != {0, 1, 2, 3, 4}:
            raise RuntimeError("MX04_REPLICATION_FOLD_ALIGNMENT_INVALID")
        for fold in range(5):
            candidate = candidate_folds[fold]
            comparator = comparator_folds[fold]
            if (
                candidate["split_membership_digest"] != comparator["split_membership_digest"]
                or candidate["y_true"] != comparator["y_true"]
                or len(candidate["y_prob"]) != len(comparator["y_prob"])
            ):
                raise RuntimeError(f"MX04_REPLICATION_MEMBERSHIP_MISMATCH:{seed}:{fold}")
            labels.extend(int(value) for value in candidate["y_true"])
            candidate_probabilities.extend(float(value) for value in candidate["y_prob"])
            comparator_probabilities.extend(float(value) for value in comparator["y_prob"])
    return labels, candidate_probabilities, comparator_probabilities


def build_promotion_evidence(
    *,
    candidate_runs: Sequence[Mapping[str, Any]],
    comparator_runs: Sequence[Mapping[str, Any]],
    sanity_checks_passed: bool,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 13,
) -> dict[str, Any]:
    """Build paired three-seed evidence against the fixed scratch comparator."""

    if int(bootstrap_resamples) < 1000:
        raise RuntimeError("MX04_BOOTSTRAP_RESAMPLES_BELOW_PROTOCOL")
    labels, candidate_probabilities, comparator_probabilities = _aligned_vectors(
        candidate_runs,
        comparator_runs,
    )
    positive_indices = [index for index, label in enumerate(labels) if label == 1]
    negative_indices = [index for index, label in enumerate(labels) if label == 0]
    if not positive_indices or not negative_indices:
        raise RuntimeError("MX04_BOOTSTRAP_REQUIRES_BOTH_CLASSES")
    observed_candidate_auc = roc_auc(labels, candidate_probabilities)
    observed_comparator_auc = roc_auc(labels, comparator_probabilities)
    rng = random.Random(int(bootstrap_seed))
    deltas: list[float] = []
    for _ in range(int(bootstrap_resamples)):
        sampled = [rng.choice(positive_indices) for _ in positive_indices]
        sampled.extend(rng.choice(negative_indices) for _ in negative_indices)
        sampled_labels = [labels[index] for index in sampled]
        candidate_sample = [candidate_probabilities[index] for index in sampled]
        comparator_sample = [comparator_probabilities[index] for index in sampled]
        deltas.append(
            roc_auc(sampled_labels, candidate_sample)
            - roc_auc(sampled_labels, comparator_sample)
        )
    deltas.sort()
    lower_index = max(0, math.floor(0.025 * (len(deltas) - 1)))
    upper_index = min(len(deltas) - 1, math.ceil(0.975 * (len(deltas) - 1)))
    candidate_brier = brier_score(labels, candidate_probabilities)
    comparator_brier = brier_score(labels, comparator_probabilities)
    candidate_ece = expected_calibration_error(labels, candidate_probabilities)
    comparator_ece = expected_calibration_error(labels, comparator_probabilities)
    candidate_cohort = max(float(run["cohort_probe_auc"]) for run in candidate_runs)
    comparator_cohort = max(float(run["cohort_probe_auc"]) for run in comparator_runs)
    allocated = max(float(run["peak_vram_allocated_gb"]) for run in candidate_runs)
    reserved = max(float(run["peak_vram_reserved_gb"]) for run in candidate_runs)
    return {
        "all_five_folds_finalized": all(
            run.get("all_five_folds_finalized") is True for run in candidate_runs
        ),
        "sanity_checks_passed": bool(sanity_checks_passed),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_unit": "paired_subject_row_stratified_by_label",
        "bootstrap_seed": int(bootstrap_seed),
        "roc_auc_candidate": observed_candidate_auc,
        "roc_auc_comparator": observed_comparator_auc,
        "roc_auc_delta": observed_candidate_auc - observed_comparator_auc,
        "roc_auc_delta_lower_ci": float(deltas[lower_index]),
        "roc_auc_delta_upper_ci": float(deltas[upper_index]),
        "brier_candidate": candidate_brier,
        "brier_comparator": comparator_brier,
        "brier_degradation": candidate_brier - comparator_brier,
        "ece_candidate": candidate_ece,
        "ece_comparator": comparator_ece,
        "ece_degradation": candidate_ece - comparator_ece,
        "cohort_probe_auc_candidate_max": candidate_cohort,
        "cohort_probe_auc_comparator_max": comparator_cohort,
        "cohort_probe_auc_increase": candidate_cohort - comparator_cohort,
        "peak_vram_allocated_gb": allocated,
        "peak_vram_reserved_gb": reserved,
        "replication_seeds": [13, 29, 47],
        "outer_test_access_count": 0,
    }


def choose_handoff(
    family_rows: Sequence[Mapping[str, Any]],
    *,
    family_order: Sequence[str],
    comparator: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose only among qualifying promotions; otherwise retain comparator."""

    order = {family: index for index, family in enumerate(family_order)}
    qualifying = [row for row in family_rows if row.get("would_promote") is True]
    if not qualifying:
        return {
            "decision": "NO_PROMOTION",
            "selected_family": str(comparator["family"]),
            "selected_candidate": str(comparator["candidate"]),
            "selected_model_root": str(comparator["model_root"]),
            "canonical_seed": 13,
        }
    winner = min(
        qualifying,
        key=lambda row: (
            -float(row["mean_inner_roc_auc"]),
            float(row["mean_inner_brier"]),
            float(row["mean_inner_ece"]),
            order[str(row["family"])],
        ),
    )
    return {
        "decision": "PROMOTE",
        "selected_family": str(winner["family"]),
        "selected_candidate": str(winner["candidate"]),
        "selected_model_root": str(winner["canonical_model_root"]),
        "canonical_seed": 13,
    }


def build_replication_spec(
    family_winners: Mapping[str, Mapping[str, Any]],
    *,
    family_order: Sequence[str],
    seeds: Sequence[int],
    comparator_family: str = "resnet34_3d",
    comparator_candidate: str = "scratch_baseline",
) -> list[dict[str, Any]]:
    """Return the deduplicated prespecified five-winner + comparator grid."""

    normalized_seeds = [int(seed) for seed in seeds]
    if normalized_seeds != [13, 29, 47]:
        raise RuntimeError("MX04_REPLICATION_SEEDS_INCOMPLETE")
    specs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for family in family_order:
        winner = family_winners.get(str(family))
        if not isinstance(winner, Mapping) or not str(winner.get("candidate") or ""):
            raise RuntimeError(f"MX04_REPLICATION_FAMILY_WINNER_MISSING:{family}")
        candidate = str(winner["candidate"])
        for seed in normalized_seeds:
            specs[(str(family), candidate, seed)] = {
                "family": str(family),
                "candidate": candidate,
                "seed": seed,
                "role": "family_winner",
            }
    for seed in normalized_seeds:
        key = (str(comparator_family), str(comparator_candidate), seed)
        if key in specs:
            specs[key]["role"] = "family_winner_and_fixed_comparator"
        else:
            specs[key] = {
                "family": str(comparator_family),
                "candidate": str(comparator_candidate),
                "seed": seed,
                "role": "fixed_comparator",
            }
    return list(specs.values())
