"""BrainIAC-CAFT: residual gated fusion, conformal abstention, promotion gate.

The design intent is that the locked ResNet34 baseline is the floor. The
foundation branch may only *correct* it:

    final_logit = locked_logit + gate(local, global, tabular) * residual

The actual gated correction is regularized toward zero, so if BrainIAC adds
nothing the fusion decays to the baseline rather than to noise. Cohort, site,
scanner, manufacturer and field strength never enter the prediction or the
gate -- they are audit-only.

Uncertainty is split-conformal, and a two-label set is reported as an explicit
indeterminate research output rather than being forced to a class.
"""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.data.combined_mri_cache import METHOD_X_PHENOTYPE_FEATURES
from src.research.brainiac_adapter import validate_locked_split_registry

CAFT_SCHEMA_VERSION = 2

INDETERMINATE = "INDETERMINATE_RESEARCH_OUTPUT"
EVIDENCE_LABEL = "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE"

#: Promotion thresholds. Every one must hold; there is no aggregate score.
MIN_AUC_GAIN = 0.02
MAX_BRIER_DEGRADATION = 0.01
MAX_ECE_DEGRADATION = 0.02
MIN_BOOTSTRAP_ITERATIONS = 1000

#: Recipe retained for a separately versioned later experiment.  R1c cannot
#: enable it, even after a frozen-stage promotion.
LORA_CONFIG = {
    "target_blocks": "last_two_transformer_blocks",
    "target_projections": ("q", "v"),
    "rank": 4,
    "alpha": 8,
    "dropout": 0.05,
    "learning_rate": 5e-5,
    "layerwise_lr_decay": 0.8,
    "full_fine_tune_permitted": False,
    "minimum_run_version": "r2_lora",
    "selection_evidence_scope": "inner_oof_only",
}

#: Grad-CAM on a ViT is not a valid attribution route; these are.
PERMITTED_XAI_METHODS = frozenset(
    {
        "integrated_gradients",
        "token_attribution",
        "occlusion",
        "region_additive_sensitivity",
    }
)
FORBIDDEN_XAI_METHODS = frozenset({"grad_cam", "grad_cam_plus_plus", "raw_cnn_gradcam"})
REQUIRED_XAI_SANITY = frozenset({"randomization", "occlusion_fidelity"})

ABLATION_ARMS = (
    "locked_resnet34_plus_tabular",
    "brainiac_only",
    "naive_concatenation",
    "caft_residual_gated",
)

CAFT_FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        list(METHOD_X_PHENOTYPE_FEATURES), separators=(",", ":")
    ).encode("utf-8")
).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text.lower())


# --- Fusion -------------------------------------------------------------------


@dataclass
class CaftConfig:
    # Locked ResNet34 representation contract: the local latent is exactly
    # 128-d (Plan 004 representation contract); the historical 512 default
    # could never match the real checkpoint and is now structurally invalid.
    local_embedding_dim: int = 128
    global_embedding_dim: int = 768
    projection_dim: int = 128
    tabular_projection_dim: int = 32
    correction_l2_weight: float = 1e-2
    gate_bias_init: float = -4.0
    residual_bound: float = 2.0
    dropout: float = 0.2
    freeze_local_encoder: bool = True
    freeze_global_encoder: bool = True
    tabular_features: int = 5
    audit_fields_used: tuple[str, ...] = field(default_factory=tuple)
    gate_l2_weight: float | None = None

    def validate(self) -> None:
        if self.audit_fields_used:
            raise RuntimeError("AUDIT_FIELD_IN_PREDICTORS")
        if not (self.freeze_local_encoder and self.freeze_global_encoder):
            raise RuntimeError("CAFT_STAGE_REQUIRES_FROZEN_ENCODERS")
        if (
            not math.isfinite(float(self.correction_l2_weight))
            or self.correction_l2_weight <= 0.0
        ):
            raise RuntimeError("CORRECTION_MUST_BE_REGULARIZED_TOWARD_ZERO")
        if self.gate_l2_weight is not None:
            raise RuntimeError("GATE_ONLY_REGULARIZATION_FORBIDDEN")
        if self.tabular_features != 5:
            raise RuntimeError("ONLY_FIVE_ALLOWED_FEATURES")
        if self.global_embedding_dim != 768:
            raise RuntimeError("BRAINIAC_EMBEDDING_MUST_BE_768")
        if self.projection_dim != 128 or self.tabular_projection_dim != 32:
            raise RuntimeError("CAFT_PROJECTION_CONTRACT_VIOLATION")
        if self.local_embedding_dim != 128:
            raise RuntimeError("CAFT_LOCAL_EMBEDDING_MUST_BE_EXACTLY_128")
        if not (math.isfinite(float(self.dropout)) and 0.0 <= self.dropout < 1.0):
            raise RuntimeError("CAFT_DROPOUT_INVALID")
        if not math.isfinite(float(self.residual_bound)) or self.residual_bound <= 0.0:
            raise RuntimeError("CAFT_DIMENSION_OR_BOUND_INVALID")
        if not math.isfinite(self.gate_bias_init) or self.gate_bias_init > -3.0:
            raise RuntimeError("CAFT_GATE_BIAS_NOT_NEAR_ZERO")


class BrainIACCAFT(nn.Module):
    """Frozen-output CAFT residual head.

    The module receives outputs from the locked ResNet34 and frozen BrainIAC
    encoders.  All three encoder-derived tensors are detached at the boundary,
    making it impossible for an R1c optimizer to update either encoder.  Only
    the projections, gate, and bounded residual head are trainable.
    """

    def __init__(self, config: CaftConfig | None = None) -> None:
        super().__init__()
        self.config = config or CaftConfig()
        self.config.validate()
        self.local_projection = nn.Linear(
            self.config.local_embedding_dim, self.config.projection_dim
        )
        self.global_projection = nn.Linear(
            self.config.global_embedding_dim, self.config.projection_dim
        )
        self.tabular_projection = nn.Linear(
            self.config.tabular_features, self.config.tabular_projection_dim
        )
        fusion_dim = 2 * self.config.projection_dim + self.config.tabular_projection_dim
        self.local_norm = nn.LayerNorm(self.config.local_embedding_dim)
        self.global_norm = nn.LayerNorm(self.config.global_embedding_dim)
        self.tabular_norm = nn.LayerNorm(self.config.tabular_features)
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(self.config.dropout)
        self.gate_head = nn.Linear(fusion_dim, 1)
        self.residual_head = nn.Linear(fusion_dim, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, self.config.gate_bias_init)
        # Exact baseline reproduction at initialization is an intentional safe
        # starting point; the residual head can move only from inner-fit loss.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @staticmethod
    def _matrix(name: str, value: torch.Tensor, width: int) -> torch.Tensor:
        if value.ndim != 2 or value.shape[1] != width:
            raise RuntimeError(f"CAFT_{name.upper()}_SHAPE_INVALID:{tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"CAFT_{name.upper()}_NOT_FINITE")
        return value

    def forward(
        self,
        locked_logit: torch.Tensor,
        local_embedding: torch.Tensor,
        brainiac_embedding: torch.Tensor,
        tabular_features: torch.Tensor,
        *,
        feature_names: Sequence[str] | None = None,
        feature_schema_sha256: str | None = None,
        tabular_preprocessing_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        if (
            feature_names is None
            or feature_schema_sha256 is None
            or tabular_preprocessing_receipt is None
        ):
            raise RuntimeError("CAFT_TABULAR_CONTRACT_REQUIRED")
        if tuple(feature_names) != tuple(METHOD_X_PHENOTYPE_FEATURES):
            raise RuntimeError(f"FEATURE_ORDER_VIOLATION:{tuple(feature_names)}")
        if feature_schema_sha256 != CAFT_FEATURE_SCHEMA_SHA256:
            raise RuntimeError("CAFT_FEATURE_SCHEMA_DIGEST_MISMATCH")
        receipt = dict(tabular_preprocessing_receipt)
        if (
            receipt.get("locked") is not True
            or receipt.get("fit_role") != "head_fit"
            or receipt.get("imputation") != "median"
            or receipt.get("scaling") != "standard"
            or receipt.get("feature_schema_sha256") != CAFT_FEATURE_SCHEMA_SHA256
            or not _is_sha256(receipt.get("artifact_sha256"))
            or receipt.get("finite_output") is not True
            or receipt.get("outer_test_touched") is not False
        ):
            raise RuntimeError("CAFT_TABULAR_PREPROCESSING_RECEIPT_INVALID")
        locked = self._matrix("locked_logit", locked_logit, 1).detach()
        local = self._matrix(
            "local_embedding", local_embedding, self.config.local_embedding_dim
        ).detach()
        global_embedding = self._matrix(
            "brainiac_embedding", brainiac_embedding, self.config.global_embedding_dim
        ).detach()
        tabular = self._matrix(
            "tabular", tabular_features, self.config.tabular_features
        )
        batch_sizes = {locked.shape[0], local.shape[0], global_embedding.shape[0], tabular.shape[0]}
        if len(batch_sizes) != 1:
            raise RuntimeError("CAFT_BATCH_MISALIGNED")

        local_projected = self.activation(self.local_projection(self.local_norm(local)))
        global_projected = self.activation(
            self.global_projection(self.global_norm(global_embedding))
        )
        tabular_projected = self.activation(
            self.tabular_projection(self.tabular_norm(tabular))
        )
        fused = torch.cat((local_projected, global_projected, tabular_projected), dim=1)
        fused = self.dropout_layer(fused)
        gate = torch.sigmoid(self.gate_head(fused))
        bounded_residual = self.config.residual_bound * torch.tanh(
            self.residual_head(fused)
        )
        correction = gate * bounded_residual
        regularization = self.config.correction_l2_weight * correction.square().mean()
        final_logit = locked + correction
        if not all(
            torch.isfinite(value).all()
            for value in (gate, bounded_residual, correction, regularization, final_logit)
        ):
            raise RuntimeError("CAFT_OUTPUT_NOT_FINITE")
        return {
            "final_logit": final_logit,
            "locked_logit": locked,
            "gate": gate,
            "bounded_residual": bounded_residual,
            "correction": correction,
            "regularization": regularization,
        }


def residual_fusion_logit(
    locked_logit: np.ndarray, gate: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    """final = locked + gate * residual, with gate in [0, 1]."""

    locked = np.asarray(locked_logit, dtype=np.float64)
    g = np.asarray(gate, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    if not np.all(np.isfinite(g)):
        raise RuntimeError("GATE_NOT_FINITE")
    if not np.all(np.isfinite(locked)) or not np.all(np.isfinite(correction)):
        raise RuntimeError("FUSION_INPUT_NOT_FINITE")
    if np.any(g < 0.0) or np.any(g > 1.0):
        raise RuntimeError("GATE_OUT_OF_RANGE")
    fused = locked + g * correction
    if not np.all(np.isfinite(fused)):
        raise RuntimeError("FUSION_OUTPUT_NOT_FINITE")
    return fused


def gate_penalty(
    gate: np.ndarray, residual: np.ndarray | None = None, *, weight: float
) -> float:
    """Penalty on the actual gated correction, retained as a NumPy helper."""

    if residual is None:
        raise RuntimeError("CORRECTION_REGULARIZATION_REQUIRES_RESIDUAL")
    if not math.isfinite(float(weight)) or float(weight) <= 0.0:
        raise RuntimeError("CORRECTION_WEIGHT_INVALID")
    correction = np.asarray(gate, dtype=np.float64) * np.asarray(
        residual, dtype=np.float64
    )
    if not np.all(np.isfinite(correction)):
        raise RuntimeError("CORRECTION_NOT_FINITE")
    return float(weight) * float(np.mean(correction**2))


def decays_to_baseline(
    locked_logit: np.ndarray, residual: np.ndarray, *, tolerance: float = 1e-12
) -> bool:
    """With a closed gate the fusion must reproduce the baseline exactly."""

    zero = np.zeros_like(np.asarray(locked_logit, dtype=np.float64))
    fused = residual_fusion_logit(locked_logit, zero, residual)
    return bool(np.all(np.abs(fused - np.asarray(locked_logit)) <= tolerance))


# --- Complementarity ----------------------------------------------------------


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA. 1.0 means redundant representations, 0.0 means orthogonal."""

    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    cross = float(np.linalg.norm(a.T @ b, ord="fro") ** 2)
    scale = float(np.linalg.norm(a.T @ a, ord="fro")) * float(
        np.linalg.norm(b.T @ b, ord="fro")
    )
    return cross / scale if scale > 0.0 else 0.0


def residual_predictive_value(
    baseline_prob: Sequence[float],
    fused_prob: Sequence[float],
    labels: Sequence[int],
) -> dict[str, Any]:
    """Does the fused score beat the baseline on the same subjects?"""

    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels).astype(int)
    base = float(roc_auc_score(y, np.asarray(baseline_prob, dtype=np.float64)))
    fused = float(roc_auc_score(y, np.asarray(fused_prob, dtype=np.float64)))
    return {
        "baseline_auc": base,
        "fused_auc": fused,
        "delta_auc": fused - base,
        "adds_information": fused - base >= MIN_AUC_GAIN,
    }


# --- Split-conformal abstention -----------------------------------------------


def split_conformal_calibrate(
    calibration_prob: Sequence[float],
    calibration_labels: Sequence[int],
    *,
    alpha: float = 0.1,
    method: str = "marginal",
    real_flags: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Fit a split-conformal threshold on a held-out calibration split.

    Scores are ``1 - p(true label)`` pooled across both classes, so the
    guarantee is **marginal**: coverage holds on average over the calibration
    distribution, not within each class. Class-conditional (Mondrian) coverage
    would require separate per-class thresholds; it is not what this computes,
    and the returned ``method`` says so.
    """

    if not 0.0 < alpha < 1.0:
        raise RuntimeError("INVALID_ALPHA")
    probs = np.asarray(calibration_prob, dtype=np.float64)
    labels_raw = np.asarray(calibration_labels)
    if probs.size == 0 or probs.size != labels_raw.size:
        raise RuntimeError("CONFORMAL_CALIBRATION_EMPTY_OR_MISALIGNED")
    if not np.all(np.isfinite(probs)) or probs.min() < 0.0 or probs.max() > 1.0:
        raise RuntimeError("CONFORMAL_CALIBRATION_PROBABILITY_INVALID")
    if not np.all(np.isin(labels_raw, (0, 1))):
        raise RuntimeError("CONFORMAL_CALIBRATION_LABEL_INVALID")
    labels = labels_raw.astype(int)
    if real_flags is not None:
        if len(real_flags) != probs.size:
            raise RuntimeError("CONFORMAL_REAL_FLAGS_MISALIGNED")
        if not all(bool(flag) for flag in real_flags):
            raise RuntimeError("CONFORMAL_CALIBRATION_MUST_BE_REAL_ONLY")
    scores = np.where(labels == 1, 1.0 - probs, probs)
    n = int(scores.size)
    class_counts = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    if method == "marginal":
        # Finite-sample correction guarantees >= 1 - alpha marginal coverage.
        rank = min(n, math.ceil((n + 1) * (1.0 - alpha)))
        threshold: float | dict[int, float] = float(np.sort(scores)[rank - 1])
        method_name = "split_conformal_marginal"
        guarantee = "MARGINAL_NOT_CLASS_CONDITIONAL"
    elif method == "mondrian":
        if class_counts.get(0, 0) < 30 or class_counts.get(1, 0) < 30:
            raise RuntimeError("MONDRIAN_MIN_CLASS_COUNT_30_NOT_MET")
        thresholds: dict[int, float] = {}
        for label in (0, 1):
            class_scores = scores[labels == label]
            class_n = int(class_scores.size)
            rank = min(class_n, math.ceil((class_n + 1) * (1.0 - alpha)))
            thresholds[label] = float(np.sort(class_scores)[rank - 1])
        threshold = thresholds
        method_name = "split_conformal_mondrian_class_conditional"
        guarantee = "CLASS_CONDITIONAL_WITHIN_EACH_LABEL"
    else:
        raise RuntimeError(f"UNKNOWN_CONFORMAL_METHOD:{method}")
    return {
        "schema_version": CAFT_SCHEMA_VERSION,
        "alpha": float(alpha),
        "target_coverage": 1.0 - float(alpha),
        "threshold": threshold,
        "calibration_n": n,
        "method": method_name,
        "coverage_guarantee": guarantee,
        "class_counts": class_counts,
        "real_only": (
            True if real_flags is not None and all(bool(flag) for flag in real_flags) else "UNKNOWN"
        ),
    }


def conformal_prediction_set(
    probability: float, *, threshold: float | Mapping[int, float]
) -> dict[str, Any]:
    """Return the label set. A two-label set is an explicit abstention."""

    p = float(probability)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise RuntimeError("CONFORMAL_PROBABILITY_INVALID")
    if isinstance(threshold, Mapping):
        if set(threshold) != {0, 1}:
            raise RuntimeError("MONDRIAN_THRESHOLDS_INVALID")
        negative_threshold = float(threshold[0])
        positive_threshold = float(threshold[1])
    else:
        negative_threshold = positive_threshold = float(threshold)
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (negative_threshold, positive_threshold)
    ):
        raise RuntimeError("CONFORMAL_THRESHOLD_INVALID")
    labels: list[str] = []
    if 1.0 - p <= positive_threshold:
        labels.append("AB_POSITIVE")
    if p <= negative_threshold:
        labels.append("AB_NEGATIVE")
    if len(labels) == 2:
        return {
            "labels": ["AB_NEGATIVE", "AB_POSITIVE"],
            "decision": INDETERMINATE,
            "abstained": True,
            "clinical_triage_claim": False,
        }
    if not labels:
        # An empty set is still uncertainty, not a confident answer.
        return {
            "labels": [],
            "decision": INDETERMINATE,
            "abstained": True,
            "empty_set": True,
            "clinical_triage_claim": False,
        }
    return {
        "labels": labels,
        "decision": labels[0],
        "abstained": False,
        "clinical_triage_claim": False,
    }


def conformal_coverage_report(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    threshold: float | Mapping[int, float],
    target_coverage: float,
    cohorts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Empirical coverage and abstention rate on an evaluation split."""

    probs = np.asarray(probabilities, dtype=np.float64)
    truth_raw = np.asarray(labels)
    # `zip` would silently truncate to the shorter sequence and report coverage
    # over a subset, so a length mismatch fails closed instead.
    if probs.size != truth_raw.size:
        raise RuntimeError("CONFORMAL_EVALUATION_MISALIGNED")
    if probs.size == 0:
        raise RuntimeError("CONFORMAL_EVALUATION_EMPTY")
    if not np.all(np.isfinite(probs)) or probs.min() < 0.0 or probs.max() > 1.0:
        raise RuntimeError("CONFORMAL_EVALUATION_PROBABILITY_INVALID")
    if not np.all(np.isin(truth_raw, (0, 1))):
        raise RuntimeError("CONFORMAL_EVALUATION_LABEL_INVALID")
    truth = truth_raw.astype(int)
    if not math.isfinite(float(target_coverage)) or not 0.0 < float(target_coverage) <= 1.0:
        raise RuntimeError("CONFORMAL_TARGET_COVERAGE_INVALID")
    if cohorts is not None and len(cohorts) != probs.size:
        raise RuntimeError("CONFORMAL_COHORTS_MISALIGNED")

    covered_flags: list[bool] = []
    abstained_flags: list[bool] = []
    for probability, label in zip(probs, truth):
        result = conformal_prediction_set(probability, threshold=threshold)
        abstained_flags.append(bool(result["abstained"]))
        true_label = "AB_POSITIVE" if int(label) == 1 else "AB_NEGATIVE"
        covered_flags.append(true_label in result["labels"])
    total = int(probs.size)
    covered = int(sum(covered_flags))
    abstained = int(sum(abstained_flags))
    coverage = covered / total

    def _wilson(successes: int, count: int) -> dict[str, float]:
        if count <= 0:
            return {"lower": float("nan"), "upper": float("nan")}
        z = 1.959963984540054
        proportion = successes / count
        denominator = 1.0 + z * z / count
        centre = (proportion + z * z / (2.0 * count)) / denominator
        radius = (
            z
            * math.sqrt(
                proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
            )
            / denominator
        )
        return {"lower": max(0.0, centre - radius), "upper": min(1.0, centre + radius)}

    class_coverage: dict[int, dict[str, Any]] = {}
    for label in (0, 1):
        indices = np.flatnonzero(truth == label)
        successes = sum(covered_flags[int(index)] for index in indices)
        count = int(indices.size)
        class_coverage[label] = {
            "coverage": successes / count if count else None,
            "n": count,
            "ci_95": _wilson(successes, count),
        }

    cohort_coverage: dict[str, dict[str, Any]] | dict[str, str]
    if cohorts is None:
        cohort_coverage = {"status": "NOT_MEASURABLE"}
        cohort_status = "NOT_MEASURABLE"
    else:
        cohort_array = np.asarray([str(item) for item in cohorts], dtype=object)
        cohort_coverage = {}
        for cohort in sorted(set(cohort_array.tolist())):
            indices = np.flatnonzero(cohort_array == cohort)
            successes = sum(covered_flags[int(index)] for index in indices)
            count = int(indices.size)
            cohort_coverage[cohort] = {
                "coverage": successes / count,
                "n": count,
                "ci_95": _wilson(successes, count),
            }
        cohort_status = (
            "MEASURABLE_PASS"
            if all(
                float(record["coverage"]) >= float(target_coverage)
                for record in cohort_coverage.values()
            )
            else "MEASURABLE_FAIL"
        )

    confidence = np.abs(probs - 0.5) * 2.0
    predicted = (probs >= 0.5).astype(int)
    risk_curve: list[dict[str, Any]] = []
    for minimum_confidence in (0.0, 0.25, 0.5, 0.75, 0.9):
        selected = confidence >= minimum_confidence
        selected_n = int(selected.sum())
        risk_curve.append(
            {
                "minimum_confidence": minimum_confidence,
                "coverage": selected_n / total,
                "risk": (
                    float(np.mean(predicted[selected] != truth[selected]))
                    if selected_n
                    else None
                ),
                "n": selected_n,
            }
        )
    return {
        "empirical_coverage": coverage,
        "coverage_ci_95": _wilson(covered, total),
        "target_coverage": float(target_coverage),
        "coverage_met": coverage >= float(target_coverage),
        "class_coverage": class_coverage,
        "class_coverage_met": all(
            record["coverage"] is not None
            and float(record["coverage"]) >= float(target_coverage)
            for record in class_coverage.values()
        ),
        "cohort_coverage": cohort_coverage,
        "cohort_coverage_status": cohort_status,
        "abstention_rate": abstained / total,
        "indeterminate_count": abstained,
        "risk_coverage_curve": risk_curve,
        "n": total,
    }


def validate_calibration_partition_contract(
    *,
    probability_calibration_hashes: Sequence[str],
    conformal_calibration_hashes: Sequence[str],
    outer_training_hashes: Sequence[str],
    probability_real_flags: Sequence[bool],
    conformal_real_flags: Sequence[bool],
) -> dict[str, Any]:
    """Require two real, mutually disjoint calibration subsets.

    Both calibration subsets must also be disjoint from outer-training.  The
    actual outer test is intentionally not an accepted argument to this API.
    """

    probability = tuple(str(item) for item in probability_calibration_hashes)
    conformal = tuple(str(item) for item in conformal_calibration_hashes)
    training = tuple(str(item) for item in outer_training_hashes)
    if not probability or not conformal or not training:
        raise RuntimeError("CALIBRATION_PARTITION_EMPTY")
    if len(probability_real_flags) != len(probability) or len(conformal_real_flags) != len(
        conformal
    ):
        raise RuntimeError("CALIBRATION_REAL_FLAGS_MISALIGNED")
    if not all(bool(flag) for flag in (*probability_real_flags, *conformal_real_flags)):
        raise RuntimeError("CALIBRATION_PARTITIONS_MUST_BE_REAL_ONLY")
    if any(
        len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower())
        for value in (*probability, *conformal, *training)
    ):
        raise RuntimeError("CALIBRATION_SUBJECT_HASH_INVALID")
    if (
        set(probability).intersection(conformal)
        or set(probability).intersection(training)
        or set(conformal).intersection(training)
    ):
        raise RuntimeError("CALIBRATION_PARTITION_OVERLAP")
    return {
        "schema_version": CAFT_SCHEMA_VERSION,
        "status": "PASS",
        "probability_calibration_n": len(probability),
        "conformal_calibration_n": len(conformal),
        "outer_training_n": len(training),
        "real_only": True,
        "outer_test_touched": False,
    }


# --- Calibration --------------------------------------------------------------


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], *, bins: int = 10
) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (p > lower) & (p <= upper) if lower > 0 else (p >= lower) & (p <= upper)
        if not mask.any():
            continue
        error += (mask.sum() / p.size) * abs(
            float(y[mask].mean()) - float(p[mask].mean())
        )
    return float(error)


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels).astype(float)
    return float(np.mean((p - y) ** 2))


# --- Paired bootstrap ---------------------------------------------------------


def paired_bootstrap_auc_delta(
    baseline_prob: Sequence[float],
    candidate_prob: Sequence[float],
    labels: Sequence[int],
    *,
    iterations: int = MIN_BOOTSTRAP_ITERATIONS,
    seed: int = 13,
) -> dict[str, Any]:
    """Subject-level paired bootstrap CI on the AUC difference."""

    from sklearn.metrics import roc_auc_score

    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or int(iterations) < MIN_BOOTSTRAP_ITERATIONS
    ):
        raise RuntimeError("INSUFFICIENT_BOOTSTRAP_ITERATIONS")
    base = np.asarray(baseline_prob, dtype=np.float64)
    cand = np.asarray(candidate_prob, dtype=np.float64)
    labels_raw = np.asarray(labels)
    if (
        base.ndim != 1
        or cand.ndim != 1
        or labels_raw.ndim != 1
        or base.size == 0
        or base.size != cand.size
        or base.size != labels_raw.size
    ):
        raise RuntimeError("BOOTSTRAP_INPUT_MISALIGNED")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(cand)):
        raise RuntimeError("BOOTSTRAP_PROBABILITY_NOT_FINITE")
    if base.min() < 0.0 or base.max() > 1.0 or cand.min() < 0.0 or cand.max() > 1.0:
        raise RuntimeError("BOOTSTRAP_PROBABILITY_OUT_OF_RANGE")
    if not np.all(np.isin(labels_raw, (0, 1))):
        raise RuntimeError("BOOTSTRAP_LABEL_INVALID")
    y = labels_raw.astype(int)
    if len(set(y.tolist())) != 2:
        raise RuntimeError("BOOTSTRAP_REQUIRES_BOTH_CLASSES")
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    n = y.size
    attempts = 0
    maximum_attempts = int(iterations) * 100
    while len(deltas) < int(iterations) and attempts < maximum_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        if len(set(y[idx].tolist())) < 2:
            continue
        deltas.append(
            float(roc_auc_score(y[idx], cand[idx]))
            - float(roc_auc_score(y[idx], base[idx]))
        )
    if len(deltas) != int(iterations):
        raise RuntimeError("BOOTSTRAP_DEGENERATE")
    array = np.asarray(deltas)
    lower = float(np.percentile(array, 2.5))
    return {
        "delta_auc_point": float(roc_auc_score(y, cand)) - float(roc_auc_score(y, base)),
        "ci_lower": lower,
        "ci_upper": float(np.percentile(array, 97.5)),
        "iterations": len(deltas),
        "requested_iterations": int(iterations),
        "sampling_attempts": attempts,
        "lower_ci_positive": lower > 0.0,
        "paired": True,
    }


def confound_permutation_audit(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    iterations: int = MIN_BOOTSTRAP_ITERATIONS,
    seed: int = 13,
) -> dict[str, Any]:
    """Audit confound decodability with bootstrap CI and permutation null."""

    from sklearn.metrics import roc_auc_score

    if int(iterations) < MIN_BOOTSTRAP_ITERATIONS:
        raise RuntimeError("INSUFFICIENT_CONFOUND_PERMUTATIONS")
    score = np.asarray(scores, dtype=np.float64)
    raw_labels = np.asarray(labels)
    if score.ndim != 1 or raw_labels.ndim != 1 or score.size != raw_labels.size:
        raise RuntimeError("CONFOUND_AUDIT_MISALIGNED")
    if score.size == 0 or not np.all(np.isfinite(score)):
        raise RuntimeError("CONFOUND_SCORE_INVALID")
    if not np.all(np.isin(raw_labels, (0, 1))):
        raise RuntimeError("CONFOUND_LABEL_INVALID")
    y = raw_labels.astype(int)
    if len(set(y.tolist())) != 2:
        raise RuntimeError("CONFOUND_AUDIT_REQUIRES_BOTH_CLASSES")

    raw_auc = float(roc_auc_score(y, score))
    observed = confound_decodability(raw_auc)
    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []
    null_values: list[float] = []
    n = y.size
    for _ in range(int(iterations)):
        sample = rng.integers(0, n, size=n)
        if len(set(y[sample].tolist())) == 2:
            bootstrap_values.append(
                confound_decodability(float(roc_auc_score(y[sample], score[sample])))
            )
        permuted = rng.permutation(y)
        null_values.append(
            confound_decodability(float(roc_auc_score(permuted, score)))
        )
    if len(bootstrap_values) < int(iterations * 0.9):
        raise RuntimeError("CONFOUND_BOOTSTRAP_DEGENERATE")
    bootstrap_array = np.asarray(bootstrap_values, dtype=np.float64)
    null_array = np.asarray(null_values, dtype=np.float64)
    return {
        "status": "MEASURABLE",
        "raw_auc": raw_auc,
        "decodability": observed,
        "ci_lower": float(np.percentile(bootstrap_array, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_array, 97.5)),
        "null_mean": float(null_array.mean()),
        "null_ci_lower": float(np.percentile(null_array, 2.5)),
        "null_ci_upper": float(np.percentile(null_array, 97.5)),
        "permutation_p_value": float(
            (1 + np.count_nonzero(null_array >= observed)) / (null_array.size + 1)
        ),
        "permutation_iterations": int(null_array.size),
        "definition": "max(AUC,1-AUC)",
    }


_BUDGET_FIELDS = (
    "split_hash",
    "fit_subject_hash",
    "max_steps",
    "batch_size",
    "optimizer",
    "scheduler",
    "early_stopping",
    "selection_role",
    "outer_test_touched",
    "encoders_frozen",
)

_ARM_BINDING_FIELDS = (
    "arm_id",
    "model_class",
    "model_artifact_sha256",
    "prediction_artifact_sha256",
    "step_history_sha256",
    "trainable_parameter_policy",
)


def validate_equal_budget_contract(
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate equal data/optimization budget for the four R1c arms."""

    if set(arms) != set(ABLATION_ARMS):
        raise RuntimeError("BASELINE_ARMS_INCOMPLETE")
    reference = dict(arms[ABLATION_ARMS[0]])
    for field in _BUDGET_FIELDS:
        if field not in reference:
            raise RuntimeError(f"BASELINE_BUDGET_FIELD_MISSING:{field}")
    if reference["selection_role"] != "inner_validation":
        raise RuntimeError("BASELINE_SELECTION_ROLE_INVALID")
    if reference["outer_test_touched"] is not False:
        raise RuntimeError("BASELINE_OUTER_TEST_EXCLUSION_UNPROVEN")
    if reference["encoders_frozen"] is not True:
        raise RuntimeError("BASELINE_ENCODERS_MUST_BE_FROZEN")
    optimizer = reference.get("optimizer")
    if (
        not isinstance(optimizer, Mapping)
        or optimizer.get("name") != "adamw"
        or not math.isfinite(float(optimizer.get("learning_rate", float("nan"))))
        or float(optimizer.get("learning_rate", 0.0)) <= 0.0
        or not math.isfinite(float(optimizer.get("weight_decay", float("nan"))))
        or float(optimizer.get("weight_decay", -1.0)) < 0.0
    ):
        raise RuntimeError("BASELINE_OPTIMIZER_INVALID")
    scheduler = reference.get("scheduler")
    if (
        not isinstance(scheduler, Mapping)
        or scheduler.get("name") != "cosine"
        or not isinstance(scheduler.get("warmup_steps"), int)
        or isinstance(scheduler.get("warmup_steps"), bool)
        or int(scheduler.get("warmup_steps", -1)) < 0
    ):
        raise RuntimeError("BASELINE_SCHEDULER_INVALID")
    early_stopping = reference.get("early_stopping")
    if (
        not isinstance(early_stopping, Mapping)
        or early_stopping.get("monitor") != "inner_validation_auc"
        or not isinstance(early_stopping.get("patience"), int)
        or isinstance(early_stopping.get("patience"), bool)
        or int(early_stopping.get("patience", 0)) <= 0
    ):
        raise RuntimeError("BASELINE_EARLY_STOPPING_INVALID")
    for field in ("split_hash", "fit_subject_hash"):
        value = str(reference[field])
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
            raise RuntimeError(f"BASELINE_BUDGET_HASH_INVALID:{field}")
    for field in ("max_steps", "batch_size"):
        value = reference[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"BASELINE_BUDGET_VALUE_INVALID:{field}")
    for arm in ABLATION_ARMS:
        record = dict(arms[arm])
        for field in _ARM_BINDING_FIELDS:
            if field not in record:
                raise RuntimeError(f"BASELINE_ARM_BINDING_MISSING:{arm}.{field}")
        if record["arm_id"] != arm:
            raise RuntimeError(f"BASELINE_ARM_ID_MISMATCH:{arm}")
        if not str(record["model_class"] or "").strip() or not str(
            record["trainable_parameter_policy"] or ""
        ).strip():
            raise RuntimeError(f"BASELINE_ARM_BINDING_INVALID:{arm}")
        for field in (
            "model_artifact_sha256",
            "prediction_artifact_sha256",
            "step_history_sha256",
        ):
            if not _is_sha256(record[field]):
                raise RuntimeError(f"BASELINE_ARM_BINDING_INVALID:{arm}.{field}")
    for arm in ABLATION_ARMS[1:]:
        record = dict(arms[arm])
        for field in _BUDGET_FIELDS:
            if field not in record:
                raise RuntimeError(f"BASELINE_BUDGET_FIELD_MISSING:{arm}.{field}")
            if record[field] != reference[field]:
                raise RuntimeError(f"BASELINE_BUDGET_MISMATCH:{arm}.{field}")
    return {
        "schema_version": CAFT_SCHEMA_VERSION,
        "status": "PASS",
        "arms": list(ABLATION_ARMS),
        "equal_budget_fields": list(_BUDGET_FIELDS),
        "outer_test_touched": False,
    }


# --- Promotion gate -----------------------------------------------------------


def confound_decodability(auc: float) -> float:
    """Fold an audit AUC onto [0.5, 1.0].

    A probe AUC of 0.10 is not "better than chance in reverse" -- it means the
    cohort label is decodable with 90% accuracy after inverting the sign, i.e.
    severe confounding. Comparing raw AUC would read 0.10 as an improvement over
    0.75, so decodability is always ``max(auc, 1 - auc)``.
    """

    value = float(auc)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError(f"INVALID_PROBE_AUC:{auc}")
    return max(value, 1.0 - value)


def prediction_artifact_digest(artifact: Mapping[str, Any]) -> str:
    """Canonical digest for the selection-safe portion of a prediction artifact."""

    payload = {
        "locked": artifact.get("locked"),
        "split_role": artifact.get("split_role"),
        "outer_test_touched": artifact.get("outer_test_touched"),
        "split_hash": artifact.get("split_hash"),
        "group_hash": artifact.get("group_hash"),
        "split_registry_sha256": artifact.get("split_registry_sha256"),
        "subject_order_sha256": artifact.get("subject_order_sha256"),
        "subject_hashes": [str(item) for item in artifact.get("subject_hashes", ())],
        "labels": [int(item) for item in artifact.get("labels", ())],
        "baseline_probabilities": [
            float(item) for item in artifact.get("baseline_probabilities", ())
        ],
        "candidate_probabilities": [
            float(item) for item in artifact.get("candidate_probabilities", ())
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _subject_order_sha256(subject_hashes: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps([str(item) for item in subject_hashes], separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _validate_promotion_split_registry(
    artifact: Mapping[str, Any] | None,
    registry: Mapping[str, Any] | None,
    *,
    expected_sha256: str | None,
) -> list[str]:
    record = dict(artifact or {})
    if registry is None or expected_sha256 is None:
        return ["PROMOTION_SPLIT_REGISTRY_REQUIRED"]
    try:
        evidence = validate_locked_split_registry(
            registry, expected_sha256=expected_sha256
        )
    except RuntimeError as exc:
        if "DIGEST_MISMATCH" in str(exc):
            return ["PROMOTION_SPLIT_REGISTRY_DIGEST_MISMATCH"]
        return [f"PROMOTION_SPLIT_REGISTRY_INVALID:{exc}"]
    blockers: list[str] = []
    hashes = tuple(str(item) for item in record.get("subject_hashes", ()))
    if (
        record.get("split_registry_sha256") != evidence["integrity_sha256"]
        or record.get("split_hash") != evidence["split_hash"]
        or record.get("group_hash") != evidence["group_hash"]
        or record.get("subject_order_sha256") != _subject_order_sha256(hashes)
    ):
        blockers.append("PROMOTION_SPLIT_REGISTRY_ARTIFACT_BINDING_MISMATCH")
    expected_role = record.get("split_role")
    if expected_role not in {"inner_validation", "oof_inner_training"}:
        blockers.append("PROMOTION_SPLIT_ROLE_INVALID")
    for subject_hash in hashes:
        row = evidence["subjects"].get(subject_hash)
        if row is None:
            blockers.append("PROMOTION_SUBJECT_NOT_IN_SPLIT_REGISTRY")
            continue
        if row.get("role") != expected_role or row.get("real") is not True:
            blockers.append("PROMOTION_SPLIT_REGISTRY_ROW_MISMATCH")
    return list(dict.fromkeys(blockers))


def _recompute_locked_prediction_metrics(
    artifact: Mapping[str, Any] | None,
) -> tuple[dict[str, float] | None, dict[str, float] | None, list[str]]:
    from sklearn.metrics import roc_auc_score

    blockers: list[str] = []
    record = dict(artifact or {})
    if not record:
        return None, None, ["LOCKED_PREDICTION_ARTIFACT_MISSING"]
    if record.get("locked") is not True:
        blockers.append("PREDICTION_ARTIFACT_NOT_LOCKED")
    if record.get("split_role") not in {"inner_validation", "oof_inner_training"}:
        blockers.append("PROMOTION_SPLIT_ROLE_INVALID")
    if record.get("outer_test_touched") is not False:
        blockers.append("OUTER_TEST_SELECTION_FORBIDDEN")
    for field in ("split_hash", "group_hash", "split_registry_sha256"):
        if not _is_sha256(record.get(field)):
            blockers.append(f"PREDICTION_PROVENANCE_HASH_INVALID:{field}")
    expected_digest = prediction_artifact_digest(record)
    if record.get("sha256") != expected_digest:
        blockers.append("PREDICTION_ARTIFACT_DIGEST_MISMATCH")
    hashes = tuple(str(item) for item in record.get("subject_hashes", ()))
    if record.get("subject_order_sha256") != _subject_order_sha256(hashes):
        blockers.append("PREDICTION_SUBJECT_ORDER_DIGEST_MISMATCH")
    labels_raw = np.asarray(record.get("labels", ()))
    baseline = np.asarray(record.get("baseline_probabilities", ()), dtype=np.float64)
    candidate = np.asarray(record.get("candidate_probabilities", ()), dtype=np.float64)
    if (
        not hashes
        or len(set(hashes)) != len(hashes)
        or any(
            len(item) != 64
            or any(ch not in "0123456789abcdef" for ch in item.lower())
            for item in hashes
        )
    ):
        blockers.append("PREDICTION_SUBJECT_LINEAGE_INVALID")
    if labels_raw.size != len(hashes) or baseline.size != len(hashes) or candidate.size != len(
        hashes
    ):
        blockers.append("PREDICTION_ARTIFACT_MISALIGNED")
        return None, None, blockers
    if not np.all(np.isin(labels_raw, (0, 1))) or len(set(labels_raw.tolist())) != 2:
        blockers.append("PREDICTION_LABEL_INVALID")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidate)):
        blockers.append("PROMOTION_METRIC_NOT_FINITE:PREDICTIONS")
    elif (
        baseline.min() < 0.0
        or baseline.max() > 1.0
        or candidate.min() < 0.0
        or candidate.max() > 1.0
    ):
        blockers.append("PROMOTION_METRIC_OUT_OF_RANGE:PREDICTIONS")
    if blockers:
        return None, None, blockers
    labels = labels_raw.astype(int)
    base_metrics = {
        "roc_auc": float(roc_auc_score(labels, baseline)),
        "brier": brier_score(baseline, labels),
        "ece": expected_calibration_error(baseline, labels),
    }
    candidate_metrics = {
        "roc_auc": float(roc_auc_score(labels, candidate)),
        "brier": brier_score(candidate, labels),
        "ece": expected_calibration_error(candidate, labels),
    }
    return base_metrics, candidate_metrics, []


def confound_artifact_digest(artifact: Mapping[str, Any]) -> str:
    payload = {
        "locked": artifact.get("locked"),
        "selection_role": artifact.get("selection_role"),
        "outer_test_touched": artifact.get("outer_test_touched"),
        "prediction_artifact_sha256": artifact.get("prediction_artifact_sha256"),
        "split_registry_sha256": artifact.get("split_registry_sha256"),
        "subject_order_sha256": artifact.get("subject_order_sha256"),
        "subject_hashes": [str(item) for item in artifact.get("subject_hashes", ())],
        "labels": [int(item) for item in artifact.get("labels", ())],
        "scores": [float(item) for item in artifact.get("scores", ())],
        "iterations": artifact.get("iterations"),
        "seed": artifact.get("seed"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_locked_confound_artifact(
    *,
    scores: Sequence[float],
    labels: Sequence[int],
    subject_hashes: Sequence[str],
    prediction_artifact_sha256: str,
    split_registry_sha256: str,
    iterations: int = MIN_BOOTSTRAP_ITERATIONS,
    seed: int = 13,
) -> dict[str, Any]:
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels)
    hashes = tuple(str(item) for item in subject_hashes)
    if (
        score_array.ndim != 1
        or label_array.ndim != 1
        or score_array.size == 0
        or score_array.size != label_array.size
        or score_array.size != len(hashes)
        or not np.all(np.isfinite(score_array))
        or not np.all(np.isin(label_array, (0, 1)))
        or len(set(label_array.tolist())) != 2
        or len(set(hashes)) != len(hashes)
        or any(not _is_sha256(item) for item in hashes)
    ):
        raise RuntimeError("CONFOUND_ARTIFACT_INPUT_INVALID")
    if not _is_sha256(prediction_artifact_sha256) or not _is_sha256(
        split_registry_sha256
    ):
        raise RuntimeError("CONFOUND_ARTIFACT_BINDING_INVALID")
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations < MIN_BOOTSTRAP_ITERATIONS
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise RuntimeError("CONFOUND_ARTIFACT_RESAMPLING_INVALID")
    artifact: dict[str, Any] = {
        "schema_version": CAFT_SCHEMA_VERSION,
        "locked": True,
        "selection_role": "inner_oof_audit_only",
        "outer_test_touched": False,
        "prediction_artifact_sha256": str(prediction_artifact_sha256),
        "split_registry_sha256": str(split_registry_sha256),
        "subject_order_sha256": _subject_order_sha256(hashes),
        "subject_hashes": list(hashes),
        "labels": label_array.astype(int).tolist(),
        "scores": score_array.tolist(),
        "iterations": int(iterations),
        "seed": int(seed),
    }
    return {**artifact, "sha256": confound_artifact_digest(artifact)}


def _validate_confound_evidence(
    evidence: Mapping[str, Any] | None,
    *,
    role: str,
    expected_subject_hashes: Sequence[str],
    prediction_artifact_sha256: str | None,
    split_registry_sha256: str | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    record = dict(evidence or {})
    if not record:
        return None, [f"CONFOUND_EVIDENCE_{role.upper()}_MISSING_OR_UNKNOWN"]
    blockers: list[str] = []
    hashes = tuple(str(item) for item in record.get("subject_hashes", ()))
    if record.get("locked") is not True or record.get("outer_test_touched") is not False:
        blockers.append(f"CONFOUND_ARTIFACT_NOT_LOCKED:{role}")
    if record.get("selection_role") != "inner_oof_audit_only":
        blockers.append(f"CONFOUND_SELECTION_ROLE_INVALID:{role}")
    if record.get("sha256") != confound_artifact_digest(record):
        blockers.append(f"CONFOUND_ARTIFACT_DIGEST_MISMATCH:{role}")
    if (
        record.get("prediction_artifact_sha256") != prediction_artifact_sha256
        or record.get("split_registry_sha256") != split_registry_sha256
        or hashes != tuple(str(item) for item in expected_subject_hashes)
        or record.get("subject_order_sha256") != _subject_order_sha256(hashes)
    ):
        blockers.append(f"CONFOUND_ARTIFACT_BINDING_MISMATCH:{role}")
    if blockers:
        return None, blockers
    try:
        report = confound_permutation_audit(
            record.get("scores", ()),
            record.get("labels", ()),
            iterations=record.get("iterations", 0),
            seed=record.get("seed", 0),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        return None, [f"CONFOUND_RECOMPUTE_FAILED:{role}:{exc}"]
    return report, []


def evaluate_promotion(
    *,
    locked_predictions: Mapping[str, Any] | None = None,
    locked_split_registry: Mapping[str, Any] | None = None,
    expected_split_registry_sha256: str | None = None,
    baseline_confound: Mapping[str, Any] | None = None,
    candidate_confound: Mapping[str, Any] | None = None,
    bootstrap: Mapping[str, Any],
    coverage: Mapping[str, Any],
    sanity_passed: bool | None,
    baseline: Mapping[str, float] | None = None,
    candidate: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Recompute metrics from locked inner/OOF predictions and fail closed."""

    blockers: list[str] = []
    blockers.extend(
        _validate_promotion_split_registry(
            locked_predictions,
            locked_split_registry,
            expected_sha256=expected_split_registry_sha256,
        )
    )
    base, cand, prediction_blockers = _recompute_locked_prediction_metrics(
        locked_predictions
    )
    blockers.extend(prediction_blockers)
    prediction_record = dict(locked_predictions or {})
    split_sha256 = None
    if isinstance(locked_split_registry, Mapping):
        split_sha256 = locked_split_registry.get("integrity_sha256")
    if split_sha256 is None:
        split_sha256 = expected_split_registry_sha256
    if split_sha256 is None:
        split_sha256 = prediction_record.get("split_registry_sha256")
    prediction_sha256 = prediction_record.get("sha256")
    subject_hashes = tuple(
        str(item) for item in prediction_record.get("subject_hashes", ())
    )
    baseline_probe, confound_blockers = _validate_confound_evidence(
        baseline_confound,
        role="baseline",
        expected_subject_hashes=subject_hashes,
        prediction_artifact_sha256=prediction_sha256
        if isinstance(prediction_sha256, str)
        else None,
        split_registry_sha256=split_sha256 if isinstance(split_sha256, str) else None,
    )
    blockers.extend(confound_blockers)
    candidate_probe, confound_blockers = _validate_confound_evidence(
        candidate_confound,
        role="candidate",
        expected_subject_hashes=subject_hashes,
        prediction_artifact_sha256=prediction_sha256
        if isinstance(prediction_sha256, str)
        else None,
        split_registry_sha256=split_sha256 if isinstance(split_sha256, str) else None,
    )
    blockers.extend(confound_blockers)

    delta_auc: float | None = None
    if base is not None and cand is not None:
        delta_auc = cand["roc_auc"] - base["roc_auc"]
        if delta_auc < MIN_AUC_GAIN:
            blockers.append("AUC_GAIN_BELOW_THRESHOLD")
        if cand["brier"] - base["brier"] > MAX_BRIER_DEGRADATION:
            blockers.append("BRIER_DEGRADED")
        if cand["ece"] - base["ece"] > MAX_ECE_DEGRADATION:
            blockers.append("ECE_DEGRADED")
    confound_source = "NOT_RECOMPUTED"
    if baseline_probe is not None and candidate_probe is not None:
        confound_source = "RECOMPUTED_FROM_LOCKED_SUBJECT_ARTIFACTS"
        try:
            baseline_decodability = float(baseline_probe["decodability"])
            candidate_decodability = float(candidate_probe["decodability"])
        except (KeyError, TypeError, ValueError):
            blockers.append("COHORT_PROBE_RECOMPUTE_INVALID")
        else:
            if candidate_decodability > baseline_decodability:
                blockers.append("COHORT_PROBE_WORSENED")

    recomputed_bootstrap: dict[str, Any] | None = None
    prediction_record = dict(locked_predictions or {})
    if not all(
        field in bootstrap
        for field in ("prediction_artifact_sha256", "iterations", "seed")
    ):
        blockers.append("PAIRED_BOOTSTRAP_EVIDENCE_MISSING")
    elif bootstrap.get("prediction_artifact_sha256") != prediction_record.get("sha256"):
        blockers.append("PAIRED_BOOTSTRAP_ARTIFACT_BINDING_INVALID")
    elif (
        not isinstance(bootstrap.get("iterations"), int)
        or isinstance(bootstrap.get("iterations"), bool)
        or int(bootstrap["iterations"]) < MIN_BOOTSTRAP_ITERATIONS
    ):
        blockers.append("INSUFFICIENT_BOOTSTRAP_ITERATIONS")
    elif not isinstance(bootstrap.get("seed"), int) or isinstance(
        bootstrap.get("seed"), bool
    ):
        blockers.append("PAIRED_BOOTSTRAP_SEED_INVALID")
    elif base is not None and cand is not None:
        try:
            recomputed_bootstrap = paired_bootstrap_auc_delta(
                prediction_record.get("baseline_probabilities", ()),
                prediction_record.get("candidate_probabilities", ()),
                prediction_record.get("labels", ()),
                iterations=int(bootstrap["iterations"]),
                seed=int(bootstrap["seed"]),
            )
        except RuntimeError as exc:
            blockers.append(f"PAIRED_BOOTSTRAP_RECOMPUTE_FAILED:{exc}")
        else:
            if not recomputed_bootstrap["lower_ci_positive"]:
                blockers.append("PAIRED_CI_LOWER_NOT_POSITIVE")
    else:
        blockers.append("PAIRED_BOOTSTRAP_PREDICTIONS_INVALID")

    if coverage.get("coverage_met") is None:
        blockers.append("CONFORMAL_COVERAGE_EVIDENCE_MISSING")
    elif not isinstance(coverage.get("coverage_met"), (bool, np.bool_)):
        blockers.append("CONFORMAL_COVERAGE_EVIDENCE_INVALID")
    elif not bool(coverage.get("coverage_met")):
        blockers.append("CONFORMAL_COVERAGE_NOT_MET")
    for field in ("empirical_coverage", "target_coverage"):
        try:
            value = float(coverage[field])
        except (KeyError, TypeError, ValueError):
            blockers.append(f"CONFORMAL_COVERAGE_FIELD_INVALID:{field}")
            continue
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            blockers.append(f"CONFORMAL_COVERAGE_FIELD_INVALID:{field}")
    if coverage.get("method") not in {
        "split_conformal_marginal",
        "split_conformal_mondrian_class_conditional",
    }:
        blockers.append("CONFORMAL_METHOD_INVALID_OR_UNKNOWN")
    if coverage.get("calibration_partitions_validated") is not True:
        blockers.append("CALIBRATION_PARTITION_EVIDENCE_MISSING")
    if coverage.get("real_only") is not True:
        blockers.append("CALIBRATION_REAL_ONLY_EVIDENCE_MISSING")
    if coverage.get("class_coverage_met") is not True:
        blockers.append("CONFORMAL_CLASS_COVERAGE_NOT_MET_OR_UNKNOWN")
    if coverage.get("cohort_coverage_status") != "MEASURABLE_PASS":
        blockers.append("CONFORMAL_COHORT_COVERAGE_NOT_MET_OR_UNKNOWN")
    if sanity_passed is None:
        blockers.append("SANITY_EVIDENCE_MISSING")
    elif not isinstance(sanity_passed, (bool, np.bool_)):
        blockers.append("SANITY_EVIDENCE_INVALID")
    elif not bool(sanity_passed):
        blockers.append("SANITY_FAILED")

    blockers = list(dict.fromkeys(blockers))
    promoted = not blockers
    return {
        "schema_version": CAFT_SCHEMA_VERSION,
        "promoted": promoted,
        "decision": "PROMOTE" if promoted else "NO_IMPROVEMENT_DEMONSTRATED",
        "delta_auc": delta_auc,
        "blockers": blockers,
        "lora_unlocked": False,
        "lora_status": "HARD_DISABLED_DEFERRED_TO_LATER_VERSION",
        "metrics_source": "LOCKED_PREDICTION_ARTIFACT_RECOMPUTED",
        "baseline_metrics": base,
        "candidate_metrics": cand,
        "paired_bootstrap": recomputed_bootstrap,
        "bootstrap_source": "RECOMPUTED_FROM_LOCKED_PREDICTION_ARTIFACT",
        "confound_source": confound_source,
        "baseline_confound": baseline_probe,
        "candidate_confound": candidate_probe,
        "evidence_label": EVIDENCE_LABEL,
    }


def lora_config(
    *,
    frozen_stage_promoted: bool,
    evidence_scope: str = "",
    run_version: str = "r1c",
    protocol_manifest: Mapping[str, Any] | None = None,
    promotion_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the recipe only for a later-version, inner/OOF-selected run."""

    if str(run_version).lower() != "r2_lora":
        raise RuntimeError(
            "LORA_PROTOCOL_NOT_REGISTERED:"
            "LORA_HARD_DISABLED_R1C_DEFERRED_TO_LATER_VERSION"
        )
    if not isinstance(protocol_manifest, Mapping):
        raise RuntimeError("LORA_PROTOCOL_MANIFEST_REQUIRED")
    manifest = dict(protocol_manifest)
    manifest_body = {
        key: value for key, value in manifest.items() if key != "integrity_sha256"
    }
    if (
        manifest.get("protocol_id") != "method_x_brainiac_r2_lora"
        or manifest.get("run_version") != "r2_lora"
        or manifest.get("status") != "REGISTERED_LOCKED"
        or manifest.get("selection_evidence_scope") != "inner_oof_only"
        or manifest.get("outer_test_touched") is not False
        or manifest.get("integrity_sha256")
        != hashlib.sha256(
            json.dumps(manifest_body, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    ):
        raise RuntimeError("LORA_PROTOCOL_MANIFEST_INVALID")
    if evidence_scope != "inner_oof":
        raise RuntimeError("LORA_REQUIRES_INNER_OOF_EVIDENCE_ONLY")
    if not frozen_stage_promoted:
        raise RuntimeError("LORA_REQUIRES_FROZEN_STAGE_PROMOTION")
    if not isinstance(promotion_receipt, Mapping):
        raise RuntimeError("LORA_PROMOTION_RECEIPT_REQUIRED")
    receipt = dict(promotion_receipt)
    receipt_body = {
        key: value for key, value in receipt.items() if key != "integrity_sha256"
    }
    if (
        receipt.get("status") != "PASS"
        or receipt.get("promoted") is not True
        or receipt.get("evidence_scope") != "inner_oof"
        or receipt.get("outer_test_touched") is not False
        or receipt.get("protocol_sha256") != manifest["integrity_sha256"]
        or not _is_sha256(receipt.get("prediction_artifact_sha256"))
        or receipt.get("integrity_sha256")
        != hashlib.sha256(
            json.dumps(receipt_body, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    ):
        raise RuntimeError("LORA_PROMOTION_RECEIPT_INVALID")
    return {**LORA_CONFIG, "run_version": str(run_version), "enabled": True}


# --- XAI contract -------------------------------------------------------------


def validate_xai_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Reject CNN Grad-CAM on a ViT and require the sanity checks."""

    methods = {str(m) for m in plan.get("methods", ())}
    if not methods:
        raise RuntimeError("XAI_PLAN_EMPTY")
    forbidden = sorted(methods & FORBIDDEN_XAI_METHODS)
    if forbidden:
        raise RuntimeError(f"FORBIDDEN_XAI_METHOD:{','.join(forbidden)}")
    unknown = sorted(methods - PERMITTED_XAI_METHODS)
    if unknown:
        raise RuntimeError(f"UNKNOWN_XAI_METHOD:{','.join(unknown)}")
    missing = sorted(REQUIRED_XAI_SANITY - {str(s) for s in plan.get("sanity", ())})
    if missing:
        raise RuntimeError(f"MISSING_XAI_SANITY:{','.join(missing)}")
    if plan.get("aggregate_only") is not True:
        raise RuntimeError("XAI_MUST_BE_AGGREGATE_ONLY")
    if plan.get("raw_identifiers_stored") is not False:
        raise RuntimeError("XAI_RAW_IDENTIFIERS_FORBIDDEN")
    return {
        "methods": sorted(methods),
        "sanity": sorted(REQUIRED_XAI_SANITY),
        "aggregate_only": True,
        "permitted_phrase": "Regions with higher attribution in the model decision",
        "evidence_label": EVIDENCE_LABEL,
    }
