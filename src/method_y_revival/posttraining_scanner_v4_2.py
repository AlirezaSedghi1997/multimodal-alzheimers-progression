"""Scanner probe in each locked model's own embedding space.

Embeddings emitted by independently trained outer-fold models have unrelated
coordinates.  The v4 pooled OOF probe crossed those spaces.  Here each probe
is fitted and cross-validated within a single outer fold/model, using scanner
labels only.  This remains a diagnostic; leave-manufacturer-out progression
performance is a separate, unmet gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .atomic_io import atomic_write_json
from .contracts import sha256_mapping
from .posttraining_correction_v4_2 import BRANCHES, CorrectionV42Error, _verified_outer


def _cross_validated_auc(
    embeddings: np.ndarray, labels: np.ndarray, *, seed: int
) -> float:
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or int(counts.min()) < 5:
        raise CorrectionV42Error("SCANNER_PROBE_CLASS_SUPPORT_INSUFFICIENT")
    splitter = StratifiedKFold(
        n_splits=min(3, int(counts.min())), shuffle=True, random_state=int(seed)
    )
    aucs: list[float] = []
    for train_idx, test_idx in splitter.split(embeddings, labels):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs", random_state=int(seed)),
        )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(embeddings[train_idx], labels[train_idx])
        if any(issubclass(item.category, ConvergenceWarning) for item in recorded):
            raise CorrectionV42Error("SCANNER_PROBE_NONCONVERGED")
        probabilities = model.predict_proba(embeddings[test_idx])
        if len(classes) == 2:
            auc = roc_auc_score(labels[test_idx], probabilities[:, 1])
        else:
            auc = roc_auc_score(
                labels[test_idx], probabilities, labels=classes,
                multi_class="ovr", average="macro",
            )
        if not math.isfinite(float(auc)):
            raise CorrectionV42Error("SCANNER_PROBE_AUC_NONFINITE")
        aucs.append(float(auc))
    return float(np.mean(aucs))


def _probe_one_fold(
    embeddings_by_subject: dict[str, list[float]],
    manufacturer_by_subject: dict[str, str],
    *, seed: int, permutations: int,
) -> dict[str, Any]:
    if int(permutations) < 19:
        raise CorrectionV42Error("SCANNER_PROBE_PERMUTATIONS_TOO_FEW")
    if set(embeddings_by_subject) != set(manufacturer_by_subject):
        raise CorrectionV42Error("SCANNER_PROBE_METADATA_MEMBERSHIP_MISMATCH")
    support = dict(sorted(Counter(manufacturer_by_subject.values()).items()))
    # UNKNOWN is missing metadata, not a scanner manufacturer.  Rare known
    # manufacturers are reported but lack enough samples for stable CV.
    eligible = sorted(k for k, n in support.items() if k != "UNKNOWN" and n >= 5)
    excluded = {k: n for k, n in support.items() if k not in eligible}
    if len(eligible) < 2:
        return {
            "status": "NOT_ESTIMABLE", "reason": "LESS_THAN_TWO_SUPPORTED_MANUFACTURERS",
            "manufacturer_support": support, "excluded_from_probe": excluded,
        }
    subjects = sorted(s for s, name in manufacturer_by_subject.items() if name in eligible)
    vectors = np.asarray([embeddings_by_subject[s] for s in subjects], dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != len(subjects) or not np.isfinite(vectors).all():
        raise CorrectionV42Error("SCANNER_PROBE_EMBEDDINGS_INVALID")
    labels = LabelEncoder().fit_transform([manufacturer_by_subject[s] for s in subjects])
    actual = _cross_validated_auc(vectors, labels, seed=seed)
    rng = np.random.default_rng(int(seed))
    null = [
        _cross_validated_auc(vectors, rng.permutation(labels), seed=seed + i + 1)
        for i in range(int(permutations))
    ]
    return {
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "manufacturer_support": support,
        "excluded_from_probe": excluded,
        "n_probe_subjects": len(subjects),
        "eligible_manufacturers": eligible,
        "within_model_cv_macro_auc": actual,
        "permutation_null_mean_auc": float(np.mean(null)),
        "permutation_p_descriptive": (1 + sum(v >= actual for v in null)) / (1 + len(null)),
        "permutation_count": len(null),
        "embedding_coordinate_scope": "ONE_OUTER_FOLD_MODEL_ONLY",
        "scaler_scope": "PROBE_TRAIN_SPLIT_ONLY",
    }


def build_scanner_correction(
    root: str | Path, correction_path: str | Path, *, permutations: int = 49,
    seed: int = 20260714,
) -> dict[str, Any]:
    root = Path(root).resolve()
    raw_receipt = Path(correction_path).read_bytes()
    correction = json.loads(raw_receipt)
    if correction.get("report_sha256") != sha256_mapping(
        {k: v for k, v in correction.items() if k != "report_sha256"}
    ):
        raise CorrectionV42Error("CORRECTION_RECEIPT_HASH_MISMATCH")
    if correction.get("source_root") != str(root):
        raise CorrectionV42Error("CORRECTION_SOURCE_ROOT_MISMATCH")
    contract_path = root / "run_contract.json"
    contract_raw = contract_path.read_bytes()
    if hashlib.sha256(contract_raw).hexdigest() != correction["input_sha256"]["run_contract.json"]:
        raise CorrectionV42Error("SCANNER_SOURCE_CONTRACT_CHANGED")
    contract = json.loads(contract_raw)
    inputs: dict[str, str] = {}
    results: dict[str, Any] = {}
    for branch in BRANCHES:
        membership_ref = contract["input_fingerprints"][f"{branch}_membership"]
        path = Path(membership_ref["path"]).resolve()
        if not path.is_relative_to(root.parents[1]):
            raise CorrectionV42Error("MEMBERSHIP_OUTSIDE_PROJECT")
        raw_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if raw_digest != membership_ref["sha256"]:
            raise CorrectionV42Error("MEMBERSHIP_HASH_MISMATCH")
        inputs[f"{branch}_membership"] = raw_digest
        frame = pd.read_parquet(path)
        development = frame[frame["role"] == "development"]
        if development["patient_hash"].duplicated().any():
            raise CorrectionV42Error("SCANNER_MEMBERSHIP_DUPLICATE_SUBJECT")
        manufacturers = dict(zip(
            development["patient_hash"].astype(str),
            development["manufacturer"].fillna("UNKNOWN").astype(str).str.upper(),
            strict=True,
        ))
        branch_results: list[dict[str, Any]] = []
        for fold_row in correction["branches"][branch]["folds"]:
            fold = int(fold_row["outer_fold"])
            for arm in sorted({"A0", str(fold_row["selected_arm"])}):
                outer = _verified_outer(root, inputs, branch=branch, arm=arm, fold=fold)
                if arm == fold_row["selected_arm"] and (
                    outer["private_sha256"] != fold_row["selected_outer_private_sha256"]
                ):
                    raise CorrectionV42Error("SELECTED_OUTER_ARTIFACT_CHANGED")
                metrics = outer["metrics"]
                embeddings = {
                    str(row["patient_hash"]): row["embedding"]
                    for row in metrics.get("oof_embeddings", [])
                }
                if set(embeddings) != set(outer["keyed_outcomes"]):
                    raise CorrectionV42Error("SCANNER_EMBEDDING_SUBJECT_COVERAGE_MISMATCH")
                if not set(embeddings).issubset(manufacturers):
                    raise CorrectionV42Error("SCANNER_METADATA_SUBJECT_MISSING")
                probe = _probe_one_fold(
                    embeddings,
                    {s: manufacturers[s] for s in embeddings},
                    seed=int(seed) + (0 if branch == "amyloid" else 1000) + fold * 10
                    + int(arm[1]),
                    permutations=int(permutations),
                )
                branch_results.append({
                    "fold": fold, "arm": arm,
                    "private_record_sha256": outer["private_sha256"],
                    "probe": probe,
                })
        results[branch] = {
            "fold_model_probes": branch_results,
            "completed_probes": sum(
                row["probe"]["status"] == "COMPLETED_DIAGNOSTIC_ONLY"
                for row in branch_results
            ),
            "total_probe_cells": len(branch_results),
        }
    for name, digest in inputs.items():
        if name.endswith("_membership"):
            continue
        if correction["input_sha256"].get(name) != digest:
            raise CorrectionV42Error("SCANNER_INPUT_CHANGED:" + name)
    return {
        "schema_version": "4.2",
        "status": "COMPLETED_WITH_ESTIMABILITY_GATES",
        "science_gate_status": "BLOCKED",
        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
        "pooled_oof_embedding_probe_status": "INVALID_CROSS_MODEL_COORDINATE_POOLING",
        "leave_manufacturer_out_performance_status": "NOT_EXECUTED",
        "correction_receipt_sha256": hashlib.sha256(raw_receipt).hexdigest(),
        "source_run_contract_sha256": correction["source_run_contract_sha256"],
        "permutations_per_estimable_probe": int(permutations),
        "results": results,
        "input_sha256": dict(sorted(inputs.items())),
    }


def write_scanner_correction(
    root: str | Path, correction_path: str | Path, output_path: str | Path,
    *, permutations: int = 49, seed: int = 20260714,
) -> dict[str, Any]:
    report = build_scanner_correction(
        root, correction_path, permutations=permutations, seed=seed
    )
    report["source_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["report_sha256"] = sha256_mapping(report)
    output = Path(output_path)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != report:
            raise CorrectionV42Error("SCANNER_OUTPUT_CONTRACT_MISMATCH")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    return report
