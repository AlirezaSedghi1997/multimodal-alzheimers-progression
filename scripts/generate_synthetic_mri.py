#!/usr/bin/env python3
"""Generate synthetic MRI volumes using noise augmentation and copy-paste."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_cache(cache_path: Path) -> dict:
    import h5py
    data = {}
    with h5py.File(cache_path, "r") as h5:
        data["image"] = h5["image"][:]
        data["label"] = h5["label"][:]
        data["sample_hash"] = [v.decode() if isinstance(v, bytes) else v for v in h5["sample_hash"][:]]
        data["tabular"] = h5["tabular"][:]
        data["status"] = [v.decode() if isinstance(v, bytes) else v for v in h5["status"][:]]
    return data


def augment_noise(image: np.ndarray, std: float = 0.05) -> np.ndarray:
    noise = np.random.normal(0, std, image.shape).astype(image.dtype)
    return np.clip(image + noise, 0, 1)


def augment_flip(image: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.flip(image, axis=axis).copy()


def augment_rotate(image: np.ndarray, angle: float = 0.1) -> np.ndarray:
    from scipy.ndimage import rotate as ndrotate
    axes = [(1, 2), (0, 2), (0, 1)]
    ax1, ax2 = axes[np.random.randint(3)]
    img_float = image.astype(np.float32)
    result = ndrotate(img_float, angle=np.random.uniform(-angle, angle) * 180 / np.pi, axes=(ax1, ax2), reshape=False, mode='constant')
    return result.reshape(image.shape).astype(image.dtype)


def augment_scale(image: np.ndarray, factor: float = 0.1) -> np.ndarray:
    scale = 1.0 + np.random.uniform(-factor, factor)
    return np.clip(image * scale, 0, 1)


def augment_elastic(image: np.ndarray, alpha: float = 20.0, sigma: float = 3.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter, map_coordinates
    shape = image.shape[1:]
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    grid_z, grid_y, grid_x = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
    )
    indices = np.array([grid_z + dx, grid_y + dy, grid_x + dz])
    indices = np.clip(indices, 0, np.array(shape).reshape(3, 1, 1, 1) - 1)
    img_float = image[0].astype(np.float32)
    result = map_coordinates(img_float, indices, order=1)
    return result.reshape(image.shape).astype(image.dtype)


def generate_synthetic(data: dict, n_synthetic: int = 500) -> dict:
    images = data["image"]
    labels = data["label"]
    tabular = data["tabular"]
    sample_hashes = data["sample_hash"]

    synthetic_images = []
    synthetic_labels = []
    synthetic_tabular = []
    synthetic_hashes = []

    for i in range(n_synthetic):
        idx = np.random.randint(len(images))
        img = images[idx].copy()
        label = labels[idx]
        tab = tabular[idx].copy()

        aug_type = np.random.choice(["noise", "flip", "rotate", "scale", "elastic", "combined"])
        if aug_type == "noise":
            img = augment_noise(img, std=0.03)
        elif aug_type == "flip":
            img = augment_flip(img, axis=np.random.choice([0, 1, 2]))
        elif aug_type == "rotate":
            img = augment_rotate(img, angle=0.05)
        elif aug_type == "scale":
            img = augment_scale(img, factor=0.05)
        elif aug_type == "elastic":
            img = augment_elastic(img, alpha=10.0, sigma=2.0)
        else:
            img = augment_noise(img, std=0.02)
            img = augment_flip(img, axis=np.random.choice([0, 1, 2]))
            img = augment_scale(img, factor=0.03)

        tab = tab + np.random.normal(0, 0.01, tab.shape).astype(tab.dtype)

        synthetic_images.append(img)
        synthetic_labels.append(label)
        synthetic_tabular.append(tab)
        synthetic_hashes.append(f"syn_{sample_hashes[idx]}_{i}")

    return {
        "image": np.array(synthetic_images),
        "label": np.array(synthetic_labels),
        "tabular": np.array(synthetic_tabular),
        "sample_hash": synthetic_hashes,
    }


def save_synthetic_cache(synthetic: dict, output_path: Path) -> None:
    import h5py
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("image", data=synthetic["image"], dtype="float16", compression="gzip")
        h5.create_dataset("label", data=synthetic["label"], dtype="int8")
        h5.create_dataset("tabular", data=synthetic["tabular"], dtype="float32")
        h5.create_dataset("sample_hash", data=[s.encode() for s in synthetic["sample_hash"]])
        h5.create_dataset("status", data=["OK"] * len(synthetic["label"]))
        h5.attrs["synthetic"] = True
        h5.attrs["n_synthetic"] = len(synthetic["label"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_128.h5")
    ap.add_argument("--out", default="data/cached/t1_synthetic_128.h5")
    ap.add_argument("--n_synthetic", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    cache_path = PROJECT_ROOT / args.cache
    out_path = PROJECT_ROOT / args.out

    print(f"Loading cache: {cache_path}")
    data = load_cache(cache_path)
    print(f"  Real samples: {len(data['label'])}")

    print(f"Generating {args.n_synthetic} synthetic samples...")
    synthetic = generate_synthetic(data, args.n_synthetic)
    print(f"  Synthetic samples: {len(synthetic['label'])}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_synthetic_cache(synthetic, out_path)
    print(f"Saved to: {out_path}")

    summary = {
        "n_real": len(data["label"]),
        "n_synthetic": len(synthetic["label"]),
        "total": len(data["label"]) + len(synthetic["label"]),
        "synthetic_ratio": round(len(synthetic["label"]) / len(data["label"]), 2),
        "augmentation_types": ["noise", "flip", "rotate", "scale", "elastic", "combined"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
