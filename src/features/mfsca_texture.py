"""MFSCA texture surrogate features (arXiv:2606.10222).

Full 3D Hilbert-curve MFDFA requires slice pipeline (R3 stretch).
This module provides header-stat proxies aligned with MFSCA literature:
  - Hurst proxy H ≈ f(q=2) from intensity dispersion
  - spectrum width Δα proxy from percentile spread
  - asymmetry A proxy from mean/std skew
"""
from __future__ import annotations

import math
from typing import Mapping


MFSCA_FEATURE_NAMES = [
    "mfsca_hurst_proxy",
    "mfsca_spectrum_width_proxy",
    "mfsca_asymmetry_proxy",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def hurst_proxy(std: float, mean: float) -> float:
    """Monofractal fallback when full MFDFA unavailable."""
    if std <= 0.0 or mean <= 0.0:
        return 0.5
    cv = std / mean
    return float(max(0.05, min(0.95, 0.5 + 0.1 * math.tanh(cv - 1.0))))


def spectrum_width_proxy(p90: float, p10: float) -> float:
    return float(max(0.0, p90 - p10))


def asymmetry_proxy(mean: float, std: float, p50: float) -> float:
    if std <= 0.0:
        return 0.0
    return float((mean - p50) / std)


def extract_mfsca_from_mri_row(row: Mapping[str, object]) -> dict[str, float]:
    mean = _safe_float(row.get("mri_mean"))
    std = _safe_float(row.get("mri_std"))
    p10 = _safe_float(row.get("mri_p10"))
    p50 = _safe_float(row.get("mri_p50"))
    p90 = _safe_float(row.get("mri_p90"))
    return {
        "mfsca_hurst_proxy": hurst_proxy(std, mean),
        "mfsca_spectrum_width_proxy": spectrum_width_proxy(p90, p10),
        "mfsca_asymmetry_proxy": asymmetry_proxy(mean, std, p50),
    }
