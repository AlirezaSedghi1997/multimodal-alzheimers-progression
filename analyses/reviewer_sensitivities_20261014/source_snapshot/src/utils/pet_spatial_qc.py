from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

RAW_ID_PATTERNS = [
    re.compile(r"\bB\d{4,}\b", re.IGNORECASE),
    re.compile(r"\bPREVBID\b", re.IGNORECASE),
    re.compile(r"\bLRN_?\d{4,}\b", re.IGNORECASE),
    re.compile(r"(/[A-Za-z0-9_.\- ()]+)+"),
    re.compile(r"[A-Za-z]:\\[^\s]+"),
]

REQUIRED_FORBIDDEN_TERMS = {
    "pet", "suvr", "centiloid", "amyloid", "tau", "fbp", "ftp", "gap", "days_t0"
}


def scrub_sensitive_text(text: Any) -> str:
    s = str(text)
    for pat in RAW_ID_PATTERNS:
        s = pat.sub("<REDACTED>", s)
    s = re.sub(r"A4[ _-]?Learn[^\s,;]*", "<REDACTED_DATA_ROOT>", s, flags=re.IGNORECASE)
    return s[:500]


def sha256_file(path: str | Path) -> str:
    h = __import__("hashlib").sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    Path(tmp).replace(path)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False))


def load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("PyYAML is required for Method Y PET policy loading.") from e
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_single_source_forbidden_terms(policy: Mapping[str, Any]) -> set[str]:
    terms = set()
    if "method_x_boundary" in policy:
        terms.update(policy["method_x_boundary"].get("forbidden_model_input_terms", []))
    terms.update(policy.get("forbidden_model_input_terms", []))
    missing = REQUIRED_FORBIDDEN_TERMS - {str(t).lower() for t in terms}
    if missing:
        raise RuntimeError(
            "Policy missing required forbidden_model_input_terms: " + ", ".join(sorted(missing))
        )
    return {str(t).lower() for t in terms}


def assert_method_y_output_root(out_root: str | Path, policy: Optional[Mapping[str, Any]] = None) -> Path:
    out_root = Path(out_root)
    normalized = out_root.as_posix().rstrip("/")
    forbidden = {"data/cached", "data/processed"}
    if policy:
        forbidden.update(policy.get("method_x_boundary", {}).get("method_x_cache_roots_forbidden_for_pet", []))
    for bad in forbidden:
        bad = bad.rstrip("/")
        if normalized == bad or normalized.startswith(bad + "/"):
            raise RuntimeError(f"PET output root cannot be inside Method X cache/root: {out_root}")
    if "preprocessed_pet" not in normalized and "cached_method_y_pet" not in normalized:
        raise RuntimeError("PET output root must include 'preprocessed_pet' or 'cached_method_y_pet'.")
    return out_root


def foreground_mask(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros(arr.shape, dtype=bool)
    vals = arr[finite]
    threshold = max(float(np.percentile(vals, 5)), eps)
    return finite & (arr > threshold)


def normalize_pet_array(arr: np.ndarray, mode: str = "percentile_clip_foreground_zscore", reference_mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict]:
    arr = np.asarray(arr, dtype=np.float32)
    mask = foreground_mask(arr)
    info: dict[str, Any] = {"mode": mode, "foreground_voxels": int(mask.sum())}
    if mask.sum() == 0:
        raise RuntimeError("No foreground voxels found for PET normalization.")
    if mode == "suvr_image":
        if reference_mask is None:
            raise RuntimeError("suvr_image normalization requires a reference mask or precomputed reference output.")
        return compute_suvr_image(arr, reference_mask)
    x = arr.copy()
    if mode == "percentile_clip_foreground_zscore":
        lo = float(np.percentile(x[mask], 0.5))
        hi = float(np.percentile(x[mask], 99.5))
        x = np.clip(x, lo, hi)
        info.update({"clip_low": lo, "clip_high": hi})
    elif mode == "foreground_zscore":
        pass
    else:
        raise RuntimeError(f"Unsupported normalization mode: {mode}")
    mean = float(np.mean(x[mask]))
    std = float(np.std(x[mask]))
    if std < 1e-8:
        raise RuntimeError("PET foreground std too small; invalid image or mask.")
    x[mask] = (x[mask] - mean) / std
    x[~mask] = 0.0
    info.update({"foreground_mean_before": mean, "foreground_std_before": std})
    return x.astype(np.float32), info


def compute_suvr_image(arr: np.ndarray, reference_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    arr = np.asarray(arr, dtype=np.float32)
    reference_mask = np.asarray(reference_mask).astype(bool)
    if reference_mask.sum() == 0:
        raise RuntimeError("Reference region is empty; check atlas/coregistration.")
    ref_mean = float(np.mean(arr[reference_mask]))
    if ref_mean < 1e-6 or not math.isfinite(ref_mean):
        raise RuntimeError("Reference region mean invalid; check image/reference mask.")
    return (arr / ref_mean).astype(np.float32), {"mode": "suvr_image", "reference_mean": ref_mean, "reference_voxels": int(reference_mask.sum())}


def centiloid_from_suvr(suvr: float, tracer: str, suvr_policy: Mapping[str, Any]) -> Optional[float]:
    tracer = tracer.lower()
    tracers = suvr_policy.get("amyloid_pet", {}).get("tracers", {})
    if tracer not in tracers:
        return None
    slope = tracers[tracer].get("centiloid_slope")
    intercept = tracers[tracer].get("centiloid_intercept")
    if slope is None or intercept is None:
        return None
    return float(slope) * float(suvr) + float(intercept)


def compute_volume_qc(arr: np.ndarray, affine: Optional[np.ndarray] = None, centiloid: Optional[float] = None) -> dict:
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    finite_values = arr[finite]
    qc: dict[str, Any] = {
        "shape": list(arr.shape),
        "ndim": int(arr.ndim),
        "finite_fraction": float(finite.mean()) if arr.size else 0.0,
        "foreground_fraction": float(foreground_mask(arr).mean()) if arr.size else 0.0,
        "std": float(np.std(finite_values)) if finite_values.size else 0.0,
        "mean": float(np.mean(finite_values)) if finite_values.size else 0.0,
        "p01": float(np.percentile(finite_values, 1)) if finite_values.size else None,
        "p99": float(np.percentile(finite_values, 99)) if finite_values.size else None,
        "affine_valid": True,
        "centiloid": centiloid,
    }
    if affine is not None:
        a = np.asarray(affine)
        qc["affine_valid"] = bool(a.shape == (4, 4) and np.isfinite(a).all())
    return qc


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return float("nan")
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)
    else:
        mask = mask.astype(bool) & np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    x = a[mask] - float(np.mean(a[mask]))
    y = b[mask] - float(np.mean(b[mask]))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def assess_qc_gates(qc: Mapping[str, Any], policy: Mapping[str, Any], spatial_status: Optional[Mapping[str, Any]] = None) -> dict:
    gates = policy.get("qc_gates", {})
    failures = []
    warnings = []
    if qc.get("ndim") != 3:
        failures.append("NOT_3D")
    if qc.get("finite_fraction", 0.0) < float(gates.get("finite_fraction_min", 0.95)):
        failures.append("LOW_FINITE_FRACTION")
    if qc.get("foreground_fraction", 0.0) < float(gates.get("foreground_fraction_min", 0.01)):
        failures.append("LOW_FOREGROUND_FRACTION")
    if qc.get("std", 0.0) < float(gates.get("std_min", 1e-6)):
        failures.append("LOW_STD")
    if not bool(qc.get("affine_valid", True)):
        failures.append("INVALID_AFFINE")
    p99 = qc.get("p99")
    if p99 is not None and abs(float(p99)) > float(gates.get("p99_abs_max_after_normalization", 20.0)):
        failures.append("P99_OUT_OF_RANGE")
    cl = qc.get("centiloid")
    if cl is not None:
        lo, hi = gates.get("centiloid_valid_range", [-20, 200])
        if float(cl) < float(lo) or float(cl) > float(hi):
            warnings.append("CENTILOID_OUTSIDE_EXPECTED_RANGE_REVIEW")
    if gates.get("fail_if_registration_missing", False) and spatial_status:
        if not spatial_status.get("registration_performed", False):
            failures.append("REGISTRATION_MISSING")
    ncc = qc.get("ncc_pet_t1")
    ncc_fail = float(gates.get("ncc_fail_below", 0.60))
    ncc_warn = float(gates.get("ncc_warn_below", 0.75))
    if ncc is not None and math.isfinite(float(ncc)):
        if float(ncc) < ncc_fail:
            failures.append("NCC_PET_T1_FAIL")
        elif float(ncc) < ncc_warn:
            warnings.append("LOW_NCC_REVIEW")
    official_cl = qc.get("official_centiloid")
    derived_cl = qc.get("derived_centiloid")
    if official_cl is not None and derived_cl is not None:
        try:
            delta = abs(float(derived_cl) - float(official_cl))
            qc_delta_max = float(gates.get("centiloid_abs_delta_max", 15.0))
            if delta > qc_delta_max:
                failures.append("CENTILOID_DELTA_FAIL")
            elif delta > qc_delta_max / 2:
                warnings.append("CENTILOID_DELTA_REVIEW")
        except (TypeError, ValueError):
            pass
    return {"status": "PASS" if not failures else "FAIL", "failures": failures, "warnings": warnings}


def write_overlay_png(path: str | Path, base_arr: np.ndarray, pet_arr: np.ndarray, title: str = "PET overlay") -> dict:
    """Private visual QC hook. Writes simple middle-slice overlay if matplotlib is available."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        base = np.asarray(base_arr)
        pet = np.asarray(pet_arr)
        if base.shape != pet.shape:
            return {"status": "SKIPPED_SHAPE_MISMATCH", "path_redacted": "results/qc/method_y_pet/overlays/<file>.png"}
        sl = base.shape[2] // 2
        plt.figure(figsize=(5, 5))
        plt.imshow(base[:, :, sl].T, cmap="gray", origin="lower")
        plt.imshow(pet[:, :, sl].T, cmap="hot", alpha=0.35, origin="lower")
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        return {"status": "WROTE", "path_redacted": "results/qc/method_y_pet/overlays/<file>.png"}
    except Exception as e:
        return {"status": "FAILED", "error_message": scrub_sensitive_text(str(e))}


def no_raw_paths_in_exportable_json(obj: Any) -> bool:
    text = json.dumps(obj, ensure_ascii=False)
    return text == scrub_sensitive_text(text)
