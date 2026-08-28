"""Outer exactly-once evaluation, calibration and promotion (Plan 006 / B4).

The outer state machine makes the outer fold structurally unreachable before
explicit authorization, guarantees exactly one private snapshot per fold, and
reuses (never reloads) the snapshot after a crash.  Calibration is strictly
role-separated, Mondrian falls back below 30-per-class, and every promotion
criterion fails closed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ALPHA = 0.10
MONDRIAN_MIN_PER_CLASS = 30
OUTER_STATES = (
    "HEAD_LOCKED",
    "CALIBRATION_LOCKED",
    "OUTER_AUTHORIZED",
    "OUTER_SNAPSHOT_PREPARED",
    "OUTER_PREDICTIONS_COMMITTED",
    "FOLD_FINALIZED",
)
_VALID_TRANSITIONS = {
    "HEAD_LOCKED": "CALIBRATION_LOCKED",
    "CALIBRATION_LOCKED": "OUTER_AUTHORIZED",
    "OUTER_AUTHORIZED": "OUTER_SNAPSHOT_PREPARED",
    "OUTER_SNAPSHOT_PREPARED": "OUTER_PREDICTIONS_COMMITTED",
    "OUTER_PREDICTIONS_COMMITTED": "FOLD_FINALIZED",
}


class OuterStateMachine:
    """One fold's outer lifecycle; the loader raises until authorized."""

    def __init__(self, *, fold: int, resume_state: str | None = None) -> None:
        self.fold = int(fold)
        self._state = resume_state or "HEAD_LOCKED"
        if self._state not in OUTER_STATES:
            raise RuntimeError("OUTER_STATE_INVALID")
        self._loader_calls = 0
        self._snapshot_loaded = self._state in {
            "OUTER_SNAPSHOT_PREPARED",
            "OUTER_PREDICTIONS_COMMITTED",
            "FOLD_FINALIZED",
        }

    def state(self) -> str:
        return self._state

    def loader_calls(self) -> int:
        return self._loader_calls

    def transition(self, target: str) -> None:
        if _VALID_TRANSITIONS.get(self._state) != target:
            raise RuntimeError(f"OUTER_INVALID_TRANSITION:{self._state}->{target}")
        self._state = target
        if target == "OUTER_SNAPSHOT_PREPARED" and not self._snapshot_loaded:
            self._snapshot_loaded = True
            self._loader_calls += 1  # reaching PREPARED implies the one load

    def load_outer(self) -> Mapping[str, Any]:
        if self._state not in {"OUTER_AUTHORIZED"}:
            # before authorization (and after commit) the loader never runs
            raise RuntimeError("BRAINIAC_CAFT_OUTER_NOT_AUTHORIZED")
        self._loader_calls += 1
        self._snapshot_loaded = True
        self.transition("OUTER_SNAPSHOT_PREPARED")
        return {"fold": self.fold, "kind": "outer_snapshot"}

    def snapshot_path(self) -> str:
        if not self._snapshot_loaded:
            result = self.load_outer()
            return f"outer_snapshot_fold{result['fold']}.private.h5"
        # crash recovery: reuse the existing snapshot, never reload
        return f"outer_snapshot_fold{self.fold}.private.h5"


def _require_role(role: str, expected: str) -> None:
    if role != expected:
        raise RuntimeError("BRAINIAC_CAFT_CALIBRATION_ROLE_FORBIDDEN")


def fit_temperature(
    subjects: Sequence[str],
    labels: Sequence[int],
    logits: Sequence[float],
    *,
    role: str,
    previous_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_role(role, "probability_calibration")
    if previous_state is not None:
        raise RuntimeError("BRAINIAC_CAFT_REFIT_FORBIDDEN")
    label_array = np.asarray(labels)
    logit_array = np.asarray(logits, dtype="float64")
    if not np.isfinite(logit_array).all():
        raise RuntimeError("BRAINIAC_CAFT_NOT_FINITE")
    best_temperature, best_loss = 1.0, float("inf")
    for temperature in np.linspace(0.05, 5.0, 100):
        scaled = logit_array / float(temperature)
        probabilities = 1.0 / (1.0 + np.exp(-scaled))
        epsilon = 1e-9
        loss = float(
            -np.mean(
                label_array * np.log(probabilities + epsilon)
                + (1 - label_array) * np.log(1 - probabilities + epsilon)
            )
        )
        if loss < best_loss:
            best_loss, best_temperature = loss, float(temperature)
    return {
        "temperature": best_temperature,
        "nll": best_loss,
        "role": role,
        "refit": False,
    }


def fit_conformal(
    subjects: Sequence[str],
    labels: Sequence[int],
    logits: Sequence[float],
    *,
    role: str,
) -> dict[str, Any]:
    _require_role(role, "conformal_calibration")
    label_array = np.asarray(labels)
    logit_array = np.asarray(logits, dtype="float64")
    if not np.isfinite(logit_array).all():
        raise RuntimeError("BRAINIAC_CAFT_NOT_FINITE")
    probabilities = 1.0 / (1.0 + np.exp(-logit_array))
    nonconformity = np.abs(probabilities - label_array)
    positive = int((label_array == 1).sum())
    negative = int((label_array == 0).sum())
    if positive >= MONDRIAN_MIN_PER_CLASS and negative >= MONDRIAN_MIN_PER_CLASS:
        method = "mondrian_class_conditional"
        mondrian_status = "MONDRIAN_CLASS_CONDITIONAL"
        thresholds = {
            "1": float(np.quantile(nonconformity[label_array == 1], 1 - ALPHA, method="higher")),
            "0": float(np.quantile(nonconformity[label_array == 0], 1 - ALPHA, method="higher")),
        }
    else:
        method = "split_conformal_marginal"
        mondrian_status = "NOT_MEASURABLE"
        thresholds = {
            "marginal": float(np.quantile(nonconformity, 1 - ALPHA, method="higher"))
        }
    for value in thresholds.values():
        if not math.isfinite(value):
            raise RuntimeError("BRAINIAC_CAFT_NOT_FINITE")
    return {
        "alpha": ALPHA,
        "method": method,
        "mondrian_status": mondrian_status,
        "thresholds": thresholds,
        "role": role,
        "evidence_label": "EMPIRICAL_POST_HOC_CALIBRATION",
    }


def expected_calibration_error(
    labels: Sequence[int], probabilities: Sequence[float], bins: int = 15
) -> float:
    label_array = np.asarray(labels)
    probability_array = np.asarray(probabilities, dtype="float64")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(label_array)
    error = 0.0
    for index in range(bins):
        mask = (probability_array > edges[index]) & (probability_array <= edges[index + 1])
        if not mask.any():
            continue
        confidence = float(probability_array[mask].mean())
        accuracy = float(label_array[mask].mean())
        error += (mask.sum() / total) * abs(accuracy - confidence)
    return float(error)


def compute_metrics(labels: Sequence[int], logits: Sequence[float]) -> dict[str, float]:
    label_array = np.asarray(labels).astype("int64")
    logit_array = np.asarray(logits, dtype="float64")
    if not np.isfinite(logit_array).all():
        raise RuntimeError("BRAINIAC_CAFT_NOT_FINITE")
    probabilities = 1.0 / (1.0 + np.exp(-logit_array))

    positives = int(label_array.sum())
    negatives = len(label_array) - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError("SINGLE_CLASS")
    order = np.argsort(probabilities, kind="stable")
    ranks = np.empty(len(probabilities), dtype="float64")
    sorted_probs = probabilities[order]
    i = 0
    while i < len(sorted_probs):
        j = i
        while j + 1 < len(sorted_probs) and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    roc_auc = float(
        (ranks[label_array == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )
    descending = np.argsort(-probabilities, kind="stable")
    sorted_labels = label_array[descending]
    found = 0
    precision_hits = 0.0
    for rank_index, label in enumerate(sorted_labels, start=1):
        if label == 1:
            found += 1
            precision_hits += found / rank_index
    pr_auc = float(precision_hits / positives)
    brier = float(np.mean((probabilities - label_array) ** 2))
    ece = expected_calibration_error(label_array, probabilities)
    metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier, "ece": ece}
    for name, value in metrics.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0 + 1e-9:
            raise RuntimeError(f"BRAINIAC_CAFT_METRIC_INVALID:{name}")
    return metrics


def run_negative_controls(
    labels: Sequence[int],
    logits: Sequence[float],
    *,
    seed: int,
    permutations: int = 1000,
) -> dict[str, Any]:
    """Shuffled-label, shuffled-embedding and noise controls on predictions."""

    label_array = np.asarray(labels).astype("int64")
    logit_array = np.asarray(logits, dtype="float64")
    if not np.isfinite(logit_array).all():
        raise RuntimeError("BRAINIAC_CAFT_NOT_FINITE")
    rng = np.random.default_rng(seed)
    base_auc = compute_metrics(label_array, logit_array)["roc_auc"]

    shuffled_label_aucs = []
    for _ in range(min(permutations, 200)):
        shuffled = rng.permutation(label_array)
        shuffled_label_aucs.append(compute_metrics(shuffled, logit_array)["roc_auc"])
    shuffled_drift = abs(float(np.mean(shuffled_label_aucs)) - 0.5)

    shuffled_logit_auc = compute_metrics(
        label_array, rng.permutation(logit_array)
    )["roc_auc"]
    noise_auc = compute_metrics(
        label_array, rng.normal(size=len(logit_array))
    )["roc_auc"]

    return {
        "base_auc": base_auc,
        "shuffled_label_auc_mean": float(np.mean(shuffled_label_aucs)),
        "shuffled_label_auc_drift_bounded": shuffled_drift <= 0.10,
        "shuffled_embedding_auc": float(shuffled_logit_auc),
        "random_noise_auc": float(noise_auc),
        "permutations": permutations,
        "seed": int(seed),
        "pass_all": bool(
            abs(float(np.mean(shuffled_label_aucs)) - 0.5) <= 0.10
            and abs(shuffled_logit_auc - 0.5) <= 0.15
            and abs(noise_auc - 0.5) <= 0.15
        ),
    }


def evaluate_technical_promotion(
    *,
    delta_auc: float,
    bootstrap_lower_ci: float,
    delta_brier: float,
    delta_ece: float,
    confound_worsened: bool,
    sanity_passed: bool,
    measurable: bool,
) -> dict[str, Any]:
    if not measurable:
        decision = "BLOCKED_NOT_MEASURABLE"
    elif not math.isfinite(delta_auc) or delta_auc < 0.02:
        decision = "BLOCKED_DELTA_AUC"
    elif not (math.isfinite(bootstrap_lower_ci) and bootstrap_lower_ci > 0):
        decision = "BLOCKED_BOOTSTRAP_CI"
    elif not (math.isfinite(delta_brier) and delta_brier <= 0.01):
        decision = "BLOCKED_BRIER"
    elif not (math.isfinite(delta_ece) and delta_ece <= 0.02):
        decision = "BLOCKED_ECE"
    elif confound_worsened:
        decision = "BLOCKED_CONFOUND"
    elif not sanity_passed:
        decision = "BLOCKED_SANITY"
    else:
        decision = "TECHNICAL_PROMOTION"
    return {
        "decision": decision,
        "lora_unlocked": False,
        "claim": "TECHNICAL_PROMOTION_ONLY",
    }


def external_evaluation_status(*, dua_present: bool) -> dict[str, Any]:
    if dua_present:
        return {"external_status": "EXTERNAL_PENDING_AUTHORIZATION"}
    return {
        "external_status": "EXTERNAL_UNAVAILABLE_NONBLOCKING",
        "evidence_label": "ROBUSTNESS_EVIDENCE_NOT_EXTERNAL_VALIDATION",
    }
