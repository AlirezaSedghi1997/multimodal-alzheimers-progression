"""Train and evaluate BrainIAC-CAFT residual gated fusion against ablation ladder."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"H:\New project")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.brainiac_caft import (
    CaftConfig,
    BrainIACCAFT,
    CAFT_FEATURE_SCHEMA_SHA256,
    ABLATION_ARMS,
    MIN_AUC_GAIN,
    MAX_BRIER_DEGRADATION,
    MAX_ECE_DEGRADATION,
    MIN_BOOTSTRAP_ITERATIONS,
    INDETERMINATE,
    EVIDENCE_LABEL,
)
from src.data.combined_mri_cache import METHOD_X_PHENOTYPE_FEATURES
from src.utils.io_helpers import atomic_write_json

OUT_DIR = ROOT / "results" / "final_results" / "method_x_brainiac_caft_r1"
RESULTS_PATH = OUT_DIR / "caft_benchmark_results.json"


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def compute_conformal_sets(
    cal_probs: np.ndarray, cal_labels: np.ndarray, test_probs: np.ndarray, alpha: float = 0.10
) -> dict[str, Any]:
    """Split-conformal prediction sets with explicit indeterminate classification."""
    # Non-conformity score: 1 - prob(true_class)
    cal_scores = np.where(cal_labels == 1, 1.0 - cal_probs, cal_probs)
    n = len(cal_scores)
    q_level = math.ceil((n + 1) * (1.0 - alpha)) / n
    q_hat = float(np.quantile(cal_scores, min(1.0, q_level)))

    # Construct prediction sets for test
    sets = []
    indeterminate_count = 0
    correct_coverage_count = 0

    for p in test_probs:
        include_neg = (p <= q_hat)
        include_pos = ((1.0 - p) <= q_hat)

        if include_neg and include_pos:
            sets.append(["A_beta_neg", "A_beta_pos"])
            indeterminate_count += 1
        elif include_pos:
            sets.append(["A_beta_pos"])
        elif include_neg:
            sets.append(["A_beta_neg"])
        else:
            # Fallback to argmax if set is empty
            sets.append(["A_beta_pos"] if p >= 0.5 else ["A_beta_neg"])

    return {
        "quantile_threshold": q_hat,
        "alpha": alpha,
        "indeterminate_rate": float(indeterminate_count / len(test_probs)) if len(test_probs) > 0 else 0.0,
        "prediction_sets_head": sets[:10]
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    probs_baseline: np.ndarray,
    probs_candidate: np.ndarray,
    n_iterations: int = 1000,
    seed: int = 42
) -> dict[str, Any]:
    """Paired bootstrap test comparing candidate model against locked baseline."""
    np.random.seed(seed)
    n = len(y_true)
    auc_diffs = []
    brier_diffs = []
    ece_diffs = []

    base_auc = roc_auc_score(y_true, probs_baseline)
    cand_auc = roc_auc_score(y_true, probs_candidate)

    base_brier = brier_score_loss(y_true, probs_baseline)
    cand_brier = brier_score_loss(y_true, probs_candidate)

    base_ece = compute_ece(y_true, probs_baseline)
    cand_ece = compute_ece(y_true, probs_candidate)

    for _ in range(n_iterations):
        boot_idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[boot_idx])) < 2:
            continue
        b_true = y_true[boot_idx]
        b_base = probs_baseline[boot_idx]
        b_cand = probs_candidate[boot_idx]

        auc_diffs.append(roc_auc_score(b_true, b_cand) - roc_auc_score(b_true, b_base))
        brier_diffs.append(brier_score_loss(b_true, b_cand) - brier_score_loss(b_true, b_base))
        ece_diffs.append(compute_ece(b_true, b_cand) - compute_ece(b_true, b_base))

    auc_diffs = np.array(auc_diffs)
    brier_diffs = np.array(brier_diffs)
    ece_diffs = np.array(ece_diffs)

    ci_lower_auc = float(np.percentile(auc_diffs, 2.5))
    ci_upper_auc = float(np.percentile(auc_diffs, 97.5))
    p_value_auc = float(np.mean(auc_diffs <= 0.0))

    promoted = (
        (cand_auc - base_auc >= MIN_AUC_GAIN)
        and (ci_lower_auc > 0.0)
        and (cand_brier - base_brier <= MAX_BRIER_DEGRADATION)
        and (cand_ece - base_ece <= MAX_ECE_DEGRADATION)
    )

    return {
        "baseline_auc": float(base_auc),
        "candidate_auc": float(cand_auc),
        "delta_auc": float(cand_auc - base_auc),
        "ci_95_delta_auc": [ci_lower_auc, ci_upper_auc],
        "p_value_auc": p_value_auc,
        "delta_brier": float(cand_brier - base_brier),
        "delta_ece": float(cand_ece - base_ece),
        "promotion_criteria_met": bool(promoted),
        "promotion_status": "PROMOTED_TO_R2" if promoted else "RETAIN_FROZEN_TECHNICAL_EVIDENCE"
    }


def run_caft_ablation_benchmark() -> dict[str, Any]:
    """Reject the historical simulated benchmark before any data/output I/O.

    This legacy routine generated random representations and evaluated fitted
    models on the fitting cohort. It is retained only so its historical output
    can be traced; it is not a scientific execution path.
    """
    raise RuntimeError("BRAINIAC_SIMULATED_BENCHMARK_FORBIDDEN")

    # Unreachable historical implementation retained for provenance review.
    print("Loading datasets and representations...")
    # Load cache and representations
    cache_path = ROOT / "data" / "cached" / "t1_volume_cache_128.h5"
    if not cache_path.exists():
        raise RuntimeError("CACHE_NOT_FOUND")

    with h5py.File(cache_path, "r") as h5:
        is_synth = h5["is_synthetic"][()]
        real_idx = np.where(~is_synth)[0]
        y_real = h5["label"][()][real_idx]
        tab_real = h5["tabular"][()][real_idx]

    # Standardize tabular features
    scaler = StandardScaler()
    tab_scaled = scaler.fit_transform(tab_real)

    # Generate reproducible synthetic feature representations for benchmark simulation
    # (matching exact dims: 512 for local, 768 for global)
    np.random.seed(42)
    n_real = len(real_idx)

    # Load ResNet-34 metrics for baseline calibration
    m34_metrics_path = ROOT / "results" / "final_results" / "method_x_model_battery_v3_depth_and_diversity_20260729" / "models" / "resnet34_3d" / "three_d_metrics.json"
    if m34_metrics_path.exists():
        m34_data = json.loads(m34_metrics_path.read_text(encoding="utf-8"))
        m34_reported_auc = float(m34_data.get("aggregate", {}).get("val_auc_mean", 0.65))
    else:
        m34_reported_auc = 0.65

    print(f"ResNet-34 locked baseline reference AUC: {m34_reported_auc:.4f}")

    # Construct calibrated logits matching ResNet34 AUC
    signal = (y_real - 0.5) * 1.2
    noise = np.random.randn(n_real) * 1.5
    locked_logits = signal + noise
    locked_probs = 1.0 / (1.0 + np.exp(-locked_logits))

    # Arm 1: Locked ResNet-34 + Tabular
    clf_tab = LogisticRegression(C=1.0, max_iter=500)
    clf_tab.fit(tab_scaled, y_real)
    tab_logits = clf_tab.decision_function(tab_scaled)
    arm1_logits = locked_logits + 0.3 * tab_logits
    arm1_probs = 1.0 / (1.0 + np.exp(-arm1_logits))

    # Arm 2: BrainIAC Foundation Only (768-dim simulated global embedding)
    global_emb = np.random.randn(n_real, 768).astype(np.float32)
    # Give BrainIAC modest representation power
    global_signal = (y_real[:, None] - 0.5) * 0.05
    global_emb[:, :50] += global_signal
    clf_biac = LogisticRegression(C=0.1, max_iter=500)
    clf_biac.fit(np.hstack([global_emb, tab_scaled]), y_real)
    arm2_probs = clf_biac.predict_proba(np.hstack([global_emb, tab_scaled]))[:, 1]

    # Arm 3: Naive Concatenation
    arm3_logits = 0.6 * locked_logits + 0.4 * clf_biac.decision_function(np.hstack([global_emb, tab_scaled]))
    arm3_probs = 1.0 / (1.0 + np.exp(-arm3_logits))

    # Arm 4: BrainIAC-CAFT Residual Gated Fusion
    # Gate activates selectively where global embedding adds coherent signal
    gate_values = 1.0 / (1.0 + np.exp(-(-2.0 + 0.5 * np.abs(tab_scaled[:, 0]))))  # Regularized toward 0
    bounded_residual = 2.0 * np.tanh(0.3 * (arm2_probs - 0.5))
    arm4_logits = locked_logits + gate_values * bounded_residual
    arm4_probs = 1.0 / (1.0 + np.exp(-arm4_logits))

    # Conformal Uncertainty Sets
    n_cal = n_real // 2
    cal_probs = arm4_probs[:n_cal]
    cal_labels = y_real[:n_cal]
    test_probs = arm4_probs[n_cal:]
    conformal_report = compute_conformal_sets(cal_probs, cal_labels, test_probs, alpha=0.10)

    # Paired Bootstrap against ResNet-34
    bootstrap_results = paired_bootstrap_test(y_real, locked_probs, arm4_probs, n_iterations=MIN_BOOTSTRAP_ITERATIONS)

    results = {
        "schema_version": 2,
        "evidence_label": EVIDENCE_LABEL,
        "total_real_subjects": n_real,
        "ablation_ladder": {
            "locked_resnet34": {
                "roc_auc": float(roc_auc_score(y_real, locked_probs)),
                "pr_auc": float(average_precision_score(y_real, locked_probs)),
                "brier_score": float(brier_score_loss(y_real, locked_probs)),
                "ece": compute_ece(y_real, locked_probs)
            },
            "arm1_locked_resnet34_plus_tabular": {
                "roc_auc": float(roc_auc_score(y_real, arm1_probs)),
                "pr_auc": float(average_precision_score(y_real, arm1_probs)),
                "brier_score": float(brier_score_loss(y_real, arm1_probs)),
                "ece": compute_ece(y_real, arm1_probs)
            },
            "arm2_brainiac_only": {
                "roc_auc": float(roc_auc_score(y_real, arm2_probs)),
                "pr_auc": float(average_precision_score(y_real, arm2_probs)),
                "brier_score": float(brier_score_loss(y_real, arm2_probs)),
                "ece": compute_ece(y_real, arm2_probs)
            },
            "arm3_naive_concatenation": {
                "roc_auc": float(roc_auc_score(y_real, arm3_probs)),
                "pr_auc": float(average_precision_score(y_real, arm3_probs)),
                "brier_score": float(brier_score_loss(y_real, arm3_probs)),
                "ece": compute_ece(y_real, arm3_probs)
            },
            "arm4_brainiac_caft_residual_gated": {
                "roc_auc": float(roc_auc_score(y_real, arm4_probs)),
                "pr_auc": float(average_precision_score(y_real, arm4_probs)),
                "brier_score": float(brier_score_loss(y_real, arm4_probs)),
                "ece": compute_ece(y_real, arm4_probs),
                "conformal_calibration": conformal_report
            }
        },
        "promotion_audit": bootstrap_results,
        "conclusions": (
            "BrainIAC-CAFT residual gated fusion strictly preserves the baseline floor "
            "while incorporating foundation representations via regularized gating. "
            "Split-conformal prediction sets identify ambiguous cases as explicit "
            "INDETERMINATE_RESEARCH_OUTPUT without forced classification."
        )
    }

    atomic_write_json(RESULTS_PATH, results)
    print(f"CAFT Benchmark results saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    res = run_caft_ablation_benchmark()
    print("Locked ResNet34 AUC:", res["ablation_ladder"]["locked_resnet34"]["roc_auc"])
    print("BrainIAC-CAFT AUC:", res["ablation_ladder"]["arm4_brainiac_caft_residual_gated"]["roc_auc"])
    print("Delta AUC:", res["promotion_audit"]["delta_auc"])
    print("Promotion Status:", res["promotion_audit"]["promotion_status"])
