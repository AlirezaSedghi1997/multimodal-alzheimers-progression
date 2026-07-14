from __future__ import annotations

import numpy as np


class GeometryError(ValueError):
    """Raised when an image cannot be placed on the locked physical grid."""


def validate_affine(affine: np.ndarray) -> None:
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise GeometryError("INVALID_AFFINE")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-8:
        raise GeometryError("SINGULAR_AFFINE")


def resample_to_reference(source_image, reference_image, *, order: int = 1):
    """Resample in world coordinates; never infer physical geometry from array shape."""

    from nibabel.processing import resample_from_to

    if len(source_image.shape) != 3 or len(reference_image.shape) != 3:
        raise GeometryError("THREE_DIMENSIONAL_IMAGES_REQUIRED")
    validate_affine(source_image.affine)
    validate_affine(reference_image.affine)
    result = resample_from_to(
        source_image,
        (reference_image.shape, reference_image.affine),
        order=int(order),
        mode="constant",
        cval=0.0,
    )
    if result.shape != reference_image.shape:
        raise GeometryError("REFERENCE_SHAPE_MISMATCH")
    if not np.allclose(result.affine, reference_image.affine, atol=1e-6, rtol=0.0):
        raise GeometryError("REFERENCE_AFFINE_MISMATCH")
    if not np.isfinite(np.asarray(result.dataobj)).all():
        raise GeometryError("NONFINITE_RESAMPLED_IMAGE")
    return result
