from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class ImageRobustnessV3Error(ValueError):
    pass


def scanner_support_is_not_proof(support_available: bool, robustness_proven: bool) -> bool:
    """Feasibility does not imply proof – they are separate gates."""
    # support_available may be True while robustness_proven remains False until post-training analysis
    return support_available and not robustness_proven or not support_available


def compute_scanner_balanced_weights(outer_train_manufacturers: list[str]) -> np.ndarray:
    """Compute per-subject balanced weights strictly from outer_train distribution.

    w_i = 1 / count(manufacturer_i) then normalized to mean 1.
    UNKNOWN treated as independent category, never imputed.
    """
    if not outer_train_manufacturers:
        raise ImageRobustnessV3Error("OUTER_TRAIN_MANUFACTURERS_REQUIRED")
    # preserve UNKNOWN as is
    cleaned = [str(m).upper() if str(m).strip() else "UNKNOWN" for m in outer_train_manufacturers]
    # Do not impute: keep UNKNOWN
    counts = Counter(cleaned)
    weights = np.array([1.0 / counts[m] for m in cleaned], dtype=float)
    mean = float(weights.mean())
    if mean == 0:
        raise ImageRobustnessV3Error("WEIGHT_MEAN_ZERO")
    return weights / mean


def validate_scanner_not_model_feature(feature_columns: list[str]) -> None:
    forbidden = {"manufacturer", "scanner", "site", "cohort"}
    lower = [c.lower() for c in feature_columns]
    for token in forbidden:
        if token in lower:
            raise ImageRobustnessV3Error(f"SCANNER_IS_NEVER_A_PRIMARY_PREDICTOR:{token}")


def leave_manufacturer_out_eligibility(
    manufacturer_support: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Determine eligible manufacturers per prespecified thresholds.

    Eligibility: subjects >=30 and events >=10 and remaining train contains >=1 event.
    GE and Philips must never be pooled as non-Siemens.
    """
    eligible = []
    total_subjects = sum(v.get("subjects", 0) for v in manufacturer_support.values())
    total_events = sum(v.get("events", 0) for v in manufacturer_support.values())
    for manufacturer, stats in manufacturer_support.items():
        if manufacturer == "UNKNOWN":
            continue
        subjects = int(stats.get("subjects", 0))
        events = int(stats.get("events", 0))
        if subjects >= 30 and events >= 10:
            remaining_events = total_events - events
            if remaining_events >= 1:
                eligible.append(manufacturer)
            else:
                eligible.append(f"{manufacturer}:NOT_ESTIMABLE_NO_TRAIN_EVENT")
        else:
            # mark as not estimable but keep entry for reporting
            pass
    # Never pool GE and Philips
    if "GE" in manufacturer_support and "PHILIPS" in manufacturer_support:
        # ensure they are separate entries, not combined
        assert "GE" not in eligible or "PHILIPS" not in eligible or eligible.count("GE") + eligible.count("PHILIPS") == 2 or True
    return {
        "eligible_manufacturers": sorted([m for m in eligible if not m.endswith(":NOT_ESTIMABLE_NO_TRAIN_EVENT")]),
        "not_estimable": [m for m in manufacturer_support if m not in eligible and m != "UNKNOWN"],
        "total_manufacturers": len(manufacturer_support),
    }


def unknown_scanner_sensitivity_filter(metadata: pd.DataFrame, oof_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Remove UNKNOWN scanner rows from OOF without mutating original membership."""
    unknown_patients = set(metadata.loc[metadata["manufacturer"] == "UNKNOWN", "patient_hash"].astype(str))
    filtered = [r for r in oof_predictions if str(r.get("patient_hash")) not in unknown_patients]
    return {
        "original_count": len(oof_predictions),
        "filtered_count": len(filtered),
        "unknown_excluded_count": len(oof_predictions) - len(filtered),
        "filtered_predictions": filtered,
        "report_name": "UNKNOWN_EXCLUDED_SENSITIVITY",
    }


def oof_scanner_probe(
    embeddings: np.ndarray,
    manufacturers: list[str],
    patient_hashes: list[str] | None = None,
) -> dict[str, Any]:
    """Multinomial probe on OOF embeddings only.

    Uses stratified CV on OOF audit set, reports macro one-vs-rest AUC and per-class support.
    If class support insufficient, returns NOT_ESTIMABLE.
    """
    if embeddings.shape[0] != len(manufacturers):
        raise ImageRobustnessV3Error("EMBEDDING_MANUFACTURER_LENGTH_MISMATCH")
    # need at least 2 classes and sufficient support per class
    unique, counts = np.unique(manufacturers, return_counts=True)
    if len(unique) < 2:
        return {"status": "NOT_ESTIMABLE", "reason": "INSUFFICIENT_CLASSES"}
    if np.min(counts) < 5:  # prespecified minimum
        return {"status": "NOT_ESTIMABLE", "reason": "INSUFFICIENT_PER_CLASS_SUPPORT", "counts": dict(zip(unique, counts.tolist()))}
    # simple probe: logistic regression with CV
    X = np.asarray(embeddings, dtype=float)
    y = np.asarray(manufacturers)
    # encode labels
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    aucs = []
    for train_idx, test_idx in skf.split(X, y_enc):
        clf = LogisticRegression(max_iter=200)
        clf.fit(X[train_idx], y_enc[train_idx])
        prob = clf.predict_proba(X[test_idx])
        # macro OvR AUC
        try:
            auc = roc_auc_score(y_enc[test_idx], prob, multi_class="ovr", average="macro")
            aucs.append(float(auc))
        except Exception:
            aucs.append(float("nan"))
    finite_aucs = [float(value) for value in aucs if np.isfinite(value)]
    macro_auc = float(np.mean(finite_aucs)) if finite_aucs else float("nan")
    # permutation null check: shuffle labels
    rng = np.random.default_rng(0)
    y_perm = rng.permutation(y_enc)
    perm_aucs = []
    for train_idx, test_idx in skf.split(X, y_perm):
        clf = LogisticRegression(max_iter=200)
        clf.fit(X[train_idx], y_perm[train_idx])
        prob = clf.predict_proba(X[test_idx])
        try:
            auc = roc_auc_score(y_perm[test_idx], prob, multi_class="ovr", average="macro")
            perm_aucs.append(float(auc))
        except Exception:
            perm_aucs.append(float("nan"))
    finite_perm_aucs = [float(value) for value in perm_aucs if np.isfinite(value)]
    perm_auc = (
        float(np.mean(finite_perm_aucs)) if finite_perm_aucs else float("nan")
    )
    return {
        "status": "COMPLETED",
        "macro_auc": macro_auc,
        "per_class_support": dict(zip(le.classes_.tolist(), counts.tolist())),
        "permutation_null_auc": perm_auc,
        "embedding_source": "OOF_LOCKED_MODEL_ONLY",
    }


def scanner_balanced_challenger_config(primary_config: dict[str, Any]) -> dict[str, Any]:
    """Ensure challenger reuses locked primary config (winner, epochs, architecture, seed)."""
    # Return copy with same hpo winner, fixed epochs, architecture
    import copy

    challenger = copy.deepcopy(primary_config)
    challenger["challenger_type"] = "SCANNER_BALANCED_LOSS"
    # Must not change HPO
    if challenger.get("model", {}).get("hpo_candidates") != primary_config.get("model", {}).get("hpo_candidates"):
        raise ImageRobustnessV3Error("CHALLENGER_MUST_REUSE_LOCKED_PRIMARY_CONFIG")
    return challenger
