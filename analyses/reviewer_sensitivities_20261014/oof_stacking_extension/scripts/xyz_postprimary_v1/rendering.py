"""Private actual-subject overlays; native remapping is visualization, not lesion truth."""
from __future__ import annotations

import io
import os
from pathlib import Path

import nibabel as nib
import numpy as np


def atomic_bytes(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def save_nifti(path, array, affine):
    atomic_bytes(path, nib.Nifti1Image(np.asarray(array, dtype="float32"), affine).to_bytes())


def overlay(background, maps, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg = nib.as_closest_canonical(background).get_fdata(dtype=np.float32)
    # Reserve titles and the shared colorbar in the layout instead of covering anatomy.
    # https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html
    fig, axes = plt.subplots(len(maps), 3, figsize=(12.5, 1.1+3.6*len(maps)),
                             squeeze=False, layout="constrained")
    finite = bg[np.isfinite(bg) & (bg != 0)]
    lo, hi = np.percentile(finite, [1, 99]) if len(finite) else (0, 1)
    for row, (name, image) in enumerate(maps.items()):
        value = nib.as_closest_canonical(image).get_fdata(dtype=np.float32)
        if value.shape != bg.shape: raise ValueError("OVERLAY_GEOMETRY_MISMATCH")
        nonzero = np.abs(value[value != 0])
        scale = max(float(np.percentile(nonzero, 99)) if len(nonzero) else 0, 1e-12)
        for axis, label in enumerate(("Sagittal", "Coronal", "Axial")):
            sl = bg.shape[axis] // 2
            axes[row, axis].imshow(np.rot90(np.take(bg, sl, axis)), cmap="gray", vmin=lo, vmax=hi)
            heat = np.rot90(np.take(value, sl, axis))
            im = axes[row, axis].imshow(np.ma.masked_where(abs(heat) < .1*scale, heat),
                cmap="coolwarm", vmin=-scale, vmax=scale, alpha=.65)
            readable = name.replace("_", " ")
            axes[row, axis].set_title(readable + "\n" + label, fontsize=9)
            axes[row, axis].axis("off")
        fig.colorbar(im, ax=axes[row].tolist(), shrink=.65, label="Signed attribution; display clipped at p99")
    fig.suptitle(title + "\nRED increases / BLUE decreases model target; NOT an Alzheimer lesion map",
                 fontsize=10)
    stream = io.BytesIO(); fig.savefig(stream, format="png", dpi=120); plt.close(fig)
    atomic_bytes(path, stream.getvalue())


def native_maps(raw_path, model_path, transform_folder, model_maps, mask, out):
    # Explicit inverse affine + inverse warp order, per ANTs API:
    # https://antspy.readthedocs.io/en/v0.6.1/_modules/ants/registration/apply_transforms.html
    import ants
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    raw = ants.image_read(str(raw_path)); fixed = ants.image_read(str(model_path))
    folder = Path(transform_folder)
    inverse = [str(folder / "registration_0GenericAffine.mat"), str(folder / "registration_1InverseWarp.nii.gz")]
    forward = [str(folder / "registration_1Warp.nii.gz"), str(folder / "registration_0GenericAffine.mat")]
    model_mask = fixed.new_image_like(np.asarray(mask, dtype="float32"))
    natmask = ants.apply_transforms(raw, model_mask, inverse, whichtoinvert=[True, False], interpolator="nearestNeighbor")
    back = ants.apply_transforms(fixed, natmask, forward, whichtoinvert=[False, False], interpolator="nearestNeighbor").numpy() > .5
    dice = float(2*np.count_nonzero(back & mask) / (np.count_nonzero(back) + np.count_nonzero(mask)))
    if not np.isfinite(dice) or dice < .9: raise ValueError("NATIVE_MASK_ROUNDTRIP_FAILED")
    reference = nib.load(raw_path); products = {}
    for name, array in model_maps.items():
        moved = ants.apply_transforms(raw, fixed.new_image_like(np.asarray(array, dtype="float32")), inverse,
                                     whichtoinvert=[True, False], interpolator="linear")
        values = moved.numpy()
        if tuple(values.shape) != reference.shape or not np.isfinite(values).all():
            raise ValueError("NATIVE_ATTRIBUTION_GEOMETRY_OR_SIGNAL")
        path = out / (name + "_native.nii")
        save_nifti(path, values, reference.affine); products[name] = nib.load(path)
    overlay(reference, products, out / "native_overlay.png", "Actual individual native MRI; resampled attribution display")
    return {"mask_roundtrip_dice": dice, "interpolation": "LINEAR_SIGNED_SCALAR_FOR_DISPLAY",
            "native_attribution_sum_is_not_a_completeness_measure": True,
            "disease_localization_allowed": False}
