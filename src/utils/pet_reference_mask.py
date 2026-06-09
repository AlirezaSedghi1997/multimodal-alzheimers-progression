from __future__ import annotations

import numpy as np

from src.utils.pet_spatial_qc import centiloid_from_suvr, foreground_mask


def infer_cerebellum_reference_mask(arr: np.ndarray) -> np.ndarray:
    """Infer inferior-posterior cerebellum proxy mask in aligned MNI-like space.

    This is an interim whole-cerebellum surrogate until atlas-based masks are wired.
    Used for SUVR-image normalization QC only — not a training feature.
    """
    arr = np.asarray(arr, dtype=np.float32)
    brain = foreground_mask(arr)
    if int(brain.sum()) < 100:
        raise RuntimeError("Cannot infer cerebellum reference mask: foreground too small.")
    zz, yy, _ = np.indices(arr.shape)
    z_cut = float(np.quantile(zz[brain], 0.45))
    y_cut = float(np.quantile(yy[brain], 0.40))
    mask = brain & (zz <= z_cut) & (yy <= y_cut)
    if int(mask.sum()) < 25:
        raise RuntimeError("Cerebellum proxy mask too small; check alignment or provide --reference_mask.")
    return mask.astype(bool)


def composite_foreground_suvr(arr: np.ndarray, reference_mask: np.ndarray) -> float | None:
    arr = np.asarray(arr, dtype=np.float32)
    reference_mask = np.asarray(reference_mask).astype(bool)
    fg = foreground_mask(arr)
    cortex = fg & ~reference_mask
    if int(cortex.sum()) < 25 or int(reference_mask.sum()) < 25:
        return None
    ref_mean = float(arr[reference_mask].mean())
    if ref_mean <= 0 or not np.isfinite(ref_mean):
        return None
    comp_mean = float(arr[cortex].mean())
    return comp_mean / ref_mean


def image_derived_centiloid(
    arr: np.ndarray,
    reference_mask: np.ndarray,
    *,
    tracer: str,
    suvr_policy: dict,
) -> tuple[float | None, float | None]:
    suvr = composite_foreground_suvr(arr, reference_mask)
    if suvr is None:
        return None, None
    cl = centiloid_from_suvr(suvr, tracer, suvr_policy)
    return suvr, cl
