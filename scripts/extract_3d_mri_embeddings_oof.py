from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.train_3d_mri import H5MRIDataset, cache_index, decode, split_indices
from src.models.backbones.lightweight_3d import MethodX3DClassifier
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.utils.state_manager import StateManager


def read_catalog(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["sample_hash"]: row for row in csv.DictReader(f)}


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def extract_fold_embeddings(
    checkpoint_path: Path,
    cache_path: Path,
    indices: list[int],
    device,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    import torch
    from torch.utils.data import DataLoader

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tab_mean = ckpt["tab_mean"]
    tab_std = ckpt["tab_std"]
    fusion = ckpt.get("fusion", "cross_attention")
    model = MethodX3DClassifier(fusion=fusion).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = H5MRIDataset(cache_path, indices, tab_mean, tab_std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    latents: list[np.ndarray] = []
    sample_hashes: list[str] = []
    import h5py

    with h5py.File(cache_path, "r") as h5:
        hash_lookup = {i: decode(h5["sample_hash"][i]) for i in indices}
    offset = 0
    with torch.no_grad():
        for image, tabular, _ in loader:
            image = image.to(device)
            tabular = tabular.to(device)
            mri_latent, _ = model.encode_latents(image, tabular)
            latents.append(mri_latent.detach().cpu().numpy())
            batch_n = image.shape[0]
            for j in range(batch_n):
                sample_hashes.append(hash_lookup[indices[offset + j]])
            offset += batch_n
    return np.concatenate(latents, axis=0), sample_hashes


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract out-of-fold 3D MRI latents from fold checkpoints.")
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_96.h5")
    ap.add_argument("--checkpoints_dir", default="results/final_results/three_d")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--out", default="data/processed/method_x_3d_mri_embeddings_oof.npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        obj = np.load(out_path, allow_pickle=True)
        print(json.dumps({"status": "RESUMED", "n": int(obj["embeddings"].shape[0]), "dim": int(obj["embeddings"].shape[1])}, indent=2))
        return 0

    cache_path = Path(args.cache)
    split_dir = Path(args.splits)
    ckpt_dir = Path(args.checkpoints_dir)
    catalog = read_catalog(Path(args.catalog))
    sample_to_idx = cache_index(cache_path)
    device = resolve_device(args.device)

    all_embeddings: list[np.ndarray] = []
    all_hashes: list[str] = []
    fold_manifest: list[dict[str, object]] = []

    for fold in range(args.folds):
        ckpt = ckpt_dir / f"method_x_3d_fold_{fold}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        test_idx = split_indices(split_dir / f"fold_{fold}_test.csv", sample_to_idx)
        if not test_idx:
            raise RuntimeError(f"Fold {fold} has no cache-overlapping test samples.")
        emb, hashes = extract_fold_embeddings(ckpt, cache_path, test_idx, device, args.batch_size)
        substudies = [catalog[h]["substudy"] for h in hashes if h in catalog]
        if len(substudies) != len(hashes):
            raise RuntimeError("Catalog join failed for OOF embedding extraction.")
        all_embeddings.append(emb)
        all_hashes.extend(hashes)
        fold_manifest.append({"fold": fold, "n": len(hashes), "checkpoint": str(ckpt)})

    embeddings = np.concatenate(all_embeddings, axis=0)
    substudy = np.array([catalog[h]["substudy"] for h in all_hashes], dtype=object)
    patient_hash = np.array([catalog[h]["patient_hash"] for h in all_hashes], dtype=object)

    import torch

    fusion = torch.load(ckpt_dir / "method_x_3d_fold_0.pt", map_location="cpu", weights_only=True).get("fusion", "cross_attention")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings.astype(np.float32),
        sample_hash=np.array(all_hashes, dtype=object),
        substudy=substudy,
        patient_hash=patient_hash,
        embedding_type=np.array("mri_latent_oof"),
        latent_dim=np.int32(embeddings.shape[1]),
        folds=np.int32(args.folds),
        fusion=np.array(fusion),
        fold_manifest=np.array(json.dumps(fold_manifest), dtype=object),
    )

    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "out": str(out_path),
        "n": int(embeddings.shape[0]),
        "latent_dim": int(embeddings.shape[1]),
        "folds": args.folds,
        "fusion": fusion,
        "oof_discipline": "Each sample embedded by its held-out fold checkpoint only.",
        "dev_set_only_excludes_locked_pseudo_external_holdout": True,
        "expected_n_dev": 1359,
        "raw_identifiers_written": False,
        "subject_level_predictions_exported": False,
        "public_export_allowed": False,
    }
    atomic_write_json("results/tables/cohort_probe/method_x_3d_embedding_extraction_summary.json", summary)
    atomic_write_text(
        "results/tables/cohort_probe/method_x_3d_embedding_extraction_summary.md",
        "\n".join(
            [
                "# Method X 3D MRI OOF Embedding Extraction\n",
                f"- n: `{summary['n']}`\n",
                f"- latent_dim: `{summary['latent_dim']}`\n",
                f"- fusion: `{fusion}`\n",
                f"- oof_discipline: `{summary['oof_discipline']}`\n",
                "- public_export_allowed: `False`\n",
            ]
        ),
    )
    write_run_manifest(
        "results/logs/method_x_3d_embedding_extraction_manifest.json",
        stage="extract_3d_mri_embeddings_oof",
        data_root=None,
        extra=summary,
    )
    StateManager().append_event("extract_3d_mri_embeddings_oof", "COMPLETED", f"OOF MRI latents extracted for {summary['n']} samples.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
