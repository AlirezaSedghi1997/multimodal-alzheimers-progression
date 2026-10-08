"""Outcome-blind, cell-balanced sampling and subject-clustered XAI summaries.

This is a post-hoc inference-only sensitivity study. A successful computation
does not repair PET quantitation, scanner transport, or external validation.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .posttraining_correction_v4_2 import CorrectionV42Error


def select_subjects(cell: dict[str, Any], *, seed: int, n: int = 2) -> list[str]:
    """Select without outcomes; identical subjects for all arms of one fold."""
    subjects = [str(value) for value in cell["firewall_subjects"]]
    if len(subjects) != len(set(subjects)) or len(subjects) < n or n < 2:
        raise CorrectionV42Error("XAI_BALANCED_SELECTION_NOT_POSSIBLE")
    prefix = f"{seed}|{cell['branch']}|{cell['outer']}|"
    return sorted(subjects, key=lambda subject: (
        hashlib.sha256((prefix + subject).encode("utf-8")).hexdigest(), subject
    ))[:n]


def map_agreement(first: np.ndarray, second: np.ndarray, *, stride: int = 4) -> dict[str, Any]:
    """One subject is one observation; the sampled voxels are NOT replicates."""
    a, b = np.asarray(first), np.asarray(second)
    if a.shape != b.shape or a.ndim != 3 or stride < 1:
        raise CorrectionV42Error("XAI_MAP_GEOMETRY_MISMATCH")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise CorrectionV42Error("XAI_MAP_NONFINITE")
    av = a[::stride, ::stride, ::stride].ravel()
    bv = b[::stride, ::stride, ::stride].ravel()
    informative = (av != 0) | (bv != 0)
    av, bv = av[informative], bv[informative]
    if len(av) < 100 or np.ptp(av) <= 1e-12 or np.ptp(bv) <= 1e-12:
        return {"status": "NOT_ESTIMABLE_CONSTANT_OR_EMPTY", "rho": None,
                "sampled_voxels": int(len(av)), "unit": "SUBJECT_MAP"}
    rho = float(spearmanr(av, bv).statistic)
    return {"status": "COMPUTED_DESCRIPTIVE", "rho": rho,
            "sampled_voxels": int(len(av)), "unit": "SUBJECT_MAP"}


def cluster_bootstrap(
    rows: list[dict[str, Any]], *, field: str, seed: int, draws: int = 1000,
) -> dict[str, Any]:
    """Resample folds then subjects within folds; no voxel-level uncertainty."""
    if draws < 1000:
        raise CorrectionV42Error("XAI_CLUSTER_BOOTSTRAP_TOO_SMALL")
    groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            groups[int(row["fold"])].append(float(value))
    if set(groups) != set(range(5)) or any(len(group) < 2 for group in groups.values()):
        return {"status": "NOT_ESTIMABLE_FOLD_OR_SUBJECT_COVERAGE", "mean_rho": None,
                "ci95": None, "fold_count": len(groups)}
    rng = np.random.default_rng(seed)
    fold_ids = sorted(groups)
    samples = []
    for _ in range(draws):
        chosen = rng.choice(fold_ids, size=len(fold_ids), replace=True)
        fold_means = []
        for fold in chosen:
            values = groups[int(fold)]
            fold_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        samples.append(float(np.mean(fold_means)))
    return {"status": "CONDITIONAL_POST_HOC_EXPLORATORY", "mean_rho": float(np.mean([
        np.mean(group) for group in groups.values()
    ])), "ci95": [float(v) for v in np.quantile(samples, [0.025, 0.975])],
        "fold_count": 5, "subject_maps": sum(map(len, groups.values())),
        "bootstrap_unit": "FOLD_THEN_SUBJECT", "draws": draws,
        "selection_and_training_uncertainty": "NOT_INCLUDED"}
