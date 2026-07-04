from __future__ import annotations

"""Real Method Y training loops for stages S1 (PET-only), S2 (joint MRI+PET) and S3
(Y->X distillation). These functions are imported by the stage scripts under
``scripts/method_y/`` and executed only in ``dry_run`` / ``train`` mode when the
required data caches and checkpoints are present. They never run in plan mode and never
auto-trigger heavy training (the mode/authorization gate in ``_stage_common`` enforces
this before these functions are reached).

Scientific defaults are fixed by ``final_delivery/docs/method_y_pet_pipeline_plan.md``
and the stage YAML configs; they must not be changed silently.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object  # type: ignore[misc,assignment]

from src.models.losses.distillation import extract_teacher_mri_latent, method_y_distillation_loss
from src.models.losses.evidential import (
    centiloid_huber_loss,
    evidential_cross_entropy_loss,
    kl_anneal_factor,
)
from src.models.method_y.joint_model import JointMRIPETModel, MethodYDistilledStudent
from src.models.method_y.petnet_lite import PETNetLite, pet_intensity_augment
from scripts.method_y._stage_common import (
    DEFAULT_METHOD_Y_PET_CACHE,
    DEFAULT_METHOD_Y_PET_INPUT_LIST,
    VRAM8_DEFAULTS,
    compute_config_diff,
)

TRACER_INDEX = {"FBP": 0, "FBB": 1, "PIB": 2}
ANNEAL_STEPS = 1000
EARLY_STOP_PATIENCE = 8
CL_PRIMARY = 24.0
CL_SECONDARY = 30.0
DEFAULT_FOLD_MANIFEST = Path("results/method_y_execution/fold_manifest.json")
DEFAULT_VALIDATION_ARTIFACT_DIR = Path("results/method_y_execution/validation_artifacts")
DEFAULT_PET_INPUT_LIST = Path(DEFAULT_METHOD_Y_PET_INPUT_LIST)


class MethodYDataContractError(RuntimeError):
    """Raised when Method Y cannot satisfy a required PET/MRI data contract."""


def _decode_h5_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path.cwd() / p


def _load_pet_input_list_metadata(path: str | Path = DEFAULT_PET_INPUT_LIST) -> dict[str, dict[str, str]]:
    target = _project_path(path)
    if not target.exists():
        return {}
    keep = {"tracer", "cohort", "scanner", "paired_mri_subject_id_hash"}
    out: dict[str, dict[str, str]] = {}
    with target.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            subject = str(row.get("subject_id_hash") or "").strip()
            if not subject:
                continue
            out[subject] = {key: str(row.get(key) or "").strip() for key in keep}
    return out


def _h5_subject_metadata(grp, *, csv_meta: dict[str, str] | None = None) -> dict[str, Any]:
    csv_meta = csv_meta or {}
    centiloid = float(grp["centiloid"][()]) if "centiloid" in grp else 0.0
    label = int(grp["amyloid_pos"][()]) if "amyloid_pos" in grp else int(centiloid >= CL_PRIMARY)
    cohort = csv_meta.get("cohort") or (_decode_h5_value(grp["cohort"][()]) if "cohort" in grp else "OTHER")
    scanner = csv_meta.get("scanner") or (_decode_h5_value(grp["scanner"][()]) if "scanner" in grp else "UNKNOWN")
    tracer = csv_meta.get("tracer") or (_decode_h5_value(grp["tracer"][()]) if "tracer" in grp else "")
    return {
        "label_cl24": int(label),
        "cohort": str(cohort or "OTHER").upper(),
        "scanner": str(scanner or "UNKNOWN"),
        "tracer": str(tracer or "").upper(),
        "equivocal": int(grp["equivocal"][()]) if "equivocal" in grp else int(18.0 <= centiloid < 35.0),
    }

# VRAM=8 GB shrink profile (operator-approved). Canonical 16 GB defaults and the diff
# vs them live in scripts.method_y._stage_common (single source of truth) and are logged
# in every stage report per prompt Section 4.

def _compute_profile(cfg: dict) -> dict:
    """Merge VRAM=8 shrink defaults with config overrides (compute + memory_controls)."""
    profile = dict(VRAM8_DEFAULTS)
    profile.update(cfg.get("compute", {}) or {})
    profile.update(cfg.get("memory_controls", {}) or {})
    profile.setdefault("num_workers", 0)
    if "checkpointing" in profile and "gradient_checkpointing" not in profile:
        profile["gradient_checkpointing"] = bool(profile["checkpointing"])
    if "gradient_checkpointing" in profile:
        profile.setdefault("checkpointing", bool(profile["gradient_checkpointing"]))
    return profile


def _check_vram_budget(device, budget_gb: float) -> str | None:
    """Return a blocker string if peak VRAM exceeded the budget, else None."""
    if torch is None or device.type != "cuda":
        return None
    try:
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    except Exception:
        return None
    if peak > float(budget_gb):
        return f"BLOCKED_VRAM_BUDGET_EXCEEDED:{peak:.2f}GB>{budget_gb:.2f}GB"
    return None


def _device(prefer_gpu: bool = True):
    if torch is None:
        raise RuntimeError("PyTorch is required for Method Y training")
    return torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")


def _spearman(a, b):
    try:
        import numpy as np
        from scipy.stats import spearmanr
    except Exception:
        return None
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3:
        return None
    r = spearmanr(a, b).correlation
    return None if r != r else float(r)


def _auc(y, s):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s))
    except Exception:
        return None


def _ece_and_brier(labels: list[int], probs: list[float], n_bins: int = 10) -> tuple[float | None, float | None]:
    try:
        import numpy as np
    except Exception:
        return None, None
    y = np.asarray(labels, dtype=float).reshape(-1)
    p = np.asarray(probs, dtype=float).reshape(-1)
    if y.size == 0 or y.size != p.size:
        return None, None
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if not mask.any():
            continue
        ece += float(mask.sum() / p.size) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece), brier


def _linear_probe_auc(
    train_features: list[list[float]],
    train_labels: list[int],
    val_features: list[list[float]],
    val_labels: list[int],
) -> float | None:
    """Train a simple linear probe and return binary or macro OVR AUC."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None

    x_train = np.asarray(train_features, dtype=float)
    x_val = np.asarray(val_features, dtype=float)
    y_train = np.asarray(train_labels, dtype=int)
    y_val = np.asarray(val_labels, dtype=int)
    keep_train = y_train >= 0
    keep_val = y_val >= 0
    x_train, y_train = x_train[keep_train], y_train[keep_train]
    x_val, y_val = x_val[keep_val], y_val[keep_val]
    if x_train.ndim != 2 or x_val.ndim != 2 or len(set(y_train.tolist())) < 2 or len(set(y_val.tolist())) < 2:
        return None
    try:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(x_train, y_train)
        if len(clf.classes_) == 2:
            scores = clf.predict_proba(x_val)[:, 1]
            positive = int(clf.classes_[1])
            return float(roc_auc_score((y_val == positive).astype(int), scores))
        probs = clf.predict_proba(x_val)
        return float(roc_auc_score(y_val, probs, multi_class="ovr", average="macro", labels=clf.classes_))
    except Exception:
        return None


def masked_evidential_cross_entropy_loss(logits, target, equivocal_mask=None, *, step: int = 0, anneal_steps: int = ANNEAL_STEPS):
    """Evidential binary loss excluding equivocal Centiloid-zone subjects."""
    if equivocal_mask is None:
        return evidential_cross_entropy_loss(logits, target, step=step, anneal_steps=anneal_steps)
    keep = ~equivocal_mask.to(device=logits.device).bool().reshape(-1)
    if not bool(keep.any()):
        return logits.sum() * 0.0
    return evidential_cross_entropy_loss(logits[keep], target.to(device=logits.device)[keep], step=step, anneal_steps=anneal_steps)


class PETCacheDataset(Dataset):
    """Reads the Method Y MNI PET HDF5 cache. Returns PET image + per-subject labels."""

    def __init__(
        self,
        h5_path: str,
        subject_ids: list[str] | None = None,
        *,
        pet_input_list_path: str | Path = DEFAULT_PET_INPUT_LIST,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for PETCacheDataset")
        import h5py

        self.h5_path = str(h5_path)
        self.csv_metadata = _load_pet_input_list_metadata(pet_input_list_path)
        with h5py.File(self.h5_path, "r") as h5:
            all_keys = list(h5.keys())
        self.keys = [k for k in all_keys if subject_ids is None or k in set(subject_ids)]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        import h5py
        import numpy as np

        with h5py.File(self.h5_path, "r") as h5:
            grp = h5[self.keys[idx]]
            image = np.asarray(grp["image"], dtype="float32")
            tracer = _decode_h5_value(grp["tracer"][()])
            centiloid = float(grp["centiloid"][()])
            amyloid_pos = int(grp["amyloid_pos"][()])
            equivocal = int(grp["equivocal"][()])
            cohort = _decode_h5_value(grp["cohort"][()])
            scanner = _decode_h5_value(grp["scanner"][()]) if "scanner" in grp else "UNKNOWN"
            paired_mri = _decode_h5_value(grp["paired_mri_subject_id_hash"][()]) if "paired_mri_subject_id_hash" in grp else ""
        csv_meta = self.csv_metadata.get(self.keys[idx], {})
        cohort = csv_meta.get("cohort") or cohort
        scanner = csv_meta.get("scanner") or scanner
        image = torch.from_numpy(image)
        tracer_idx = TRACER_INDEX.get(str(tracer).upper(), -1)
        return {
            "subject_id_hash": self.keys[idx],
            "paired_mri_subject_id_hash": paired_mri,
            "image": image,
            "tracer_idx": int(tracer_idx),
            "centiloid": float(centiloid),
            "amyloid_pos": int(amyloid_pos),
            "equivocal": int(equivocal),
            "cohort": str(cohort),
            "scanner": str(scanner),
            "is_synthetic": 0,
        }


class MRICacheReader:
    """Array-style Method X MRI cache reader keyed by sample_hash."""

    def __init__(self, h5_path: str) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for MRICacheReader")
        import h5py

        self.h5_path = str(h5_path)
        if not Path(self.h5_path).exists():
            raise MethodYDataContractError(f"BLOCKED_MRI_CACHE_MISSING:{self.h5_path}")
        with h5py.File(self.h5_path, "r") as h5:
            required = {"image", "sample_hash", "status"}
            missing = sorted(required - set(h5.keys()))
            if missing:
                raise MethodYDataContractError(f"BLOCKED_MRI_CACHE_SCHEMA_INVALID:{missing}")
            hashes = [_decode_h5_value(v) for v in h5["sample_hash"][:]]
            statuses = [_decode_h5_value(v) for v in h5["status"][:]]
        self.index: dict[str, int] = {}
        for idx, (sample_hash, status) in enumerate(zip(hashes, statuses, strict=True)):
            if str(status).upper() == "OK" and sample_hash not in self.index:
                self.index[sample_hash] = idx

    def has(self, sample_hash: str) -> bool:
        return str(sample_hash) in self.index

    def read(self, sample_hash: str):
        import h5py
        import numpy as np

        key = str(sample_hash)
        if key not in self.index:
            raise MethodYDataContractError(f"BLOCKED_PAIRED_MRI_NOT_FOUND:{key}")
        idx = self.index[key]
        with h5py.File(self.h5_path, "r") as h5:
            image = np.asarray(h5["image"][idx], dtype="float32")
            tabular = np.asarray(h5["tabular"][idx], dtype="float32") if "tabular" in h5 else np.zeros(5, dtype="float32")
        return image, tabular


class PairedMRIPETCacheDataset(PETCacheDataset):
    """PET cache joined to the Method X MRI cache by paired_mri_subject_id_hash."""

    def __init__(
        self,
        pet_h5_path: str,
        mri_h5_path: str,
        subject_ids: list[str] | None = None,
        *,
        pet_input_list_path: str | Path = DEFAULT_PET_INPUT_LIST,
    ) -> None:
        super().__init__(pet_h5_path, subject_ids=subject_ids, pet_input_list_path=pet_input_list_path)
        self.mri = MRICacheReader(mri_h5_path)
        missing: list[str] = []
        import h5py

        with h5py.File(self.h5_path, "r") as h5:
            for subject in self.keys:
                grp = h5[subject]
                paired = _decode_h5_value(grp["paired_mri_subject_id_hash"][()]) if "paired_mri_subject_id_hash" in grp else ""
                if not paired or not self.mri.has(paired):
                    missing.append(f"{subject}->{paired or '<empty>'}")
        if missing:
            preview = ", ".join(missing[:5])
            raise MethodYDataContractError(
                f"BLOCKED_PAIRED_MRI_LINK_MISSING:{len(missing)} subjects; examples: {preview}"
            )

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        mri_image, tabular = self.mri.read(item["paired_mri_subject_id_hash"])
        item["pet"] = item["image"]
        item["mri"] = torch.from_numpy(mri_image)
        item["tabular"] = torch.from_numpy(tabular) if tabular is not None else None
        return item


def make_subject_folds(
    subject_ids: list[str],
    n_folds: int = 4,
    seed: int = 42,
    *,
    strata_by_subject: dict[str, str] | None = None,
) -> list[dict[str, list[str]]]:
    """Subject-level grouped K-fold, optionally stratified by cached PET metadata."""
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    ids = sorted(set(subject_ids))
    if strata_by_subject:
        buckets: dict[str, list[str]] = defaultdict(list)
        for subject in ids:
            buckets[str(strata_by_subject.get(subject, "UNSTRATIFIED"))].append(subject)
        val_folds: list[list[str]] = [[] for _ in range(n_folds)]
        for key in sorted(buckets):
            members = list(buckets[key])
            rng.shuffle(members)
            for subject in members:
                target_fold = min(range(n_folds), key=lambda idx: (len(val_folds[idx]), idx))
                val_folds[target_fold].append(subject)
    else:
        rng.shuffle(ids)
        val_folds = [ids[k::n_folds] for k in range(n_folds)]
    folds: list[dict[str, list[str]]] = []
    for k in range(n_folds):
        val = sorted(val_folds[k])
        train = [s for s in ids if s not in set(val)]
        folds.append({"train": train, "val": val})
    return folds


def _subject_strata_from_h5(
    h5_path: str | Path,
    subject_ids: list[str],
    *,
    pet_input_list_path: str | Path = DEFAULT_PET_INPUT_LIST,
) -> dict[str, str]:
    import h5py

    csv_metadata = _load_pet_input_list_metadata(pet_input_list_path)
    out: dict[str, str] = {}
    with h5py.File(h5_path, "r") as h5:
        for subject in sorted(set(str(s) for s in subject_ids)):
            if subject not in h5:
                continue
            meta = _h5_subject_metadata(h5[subject], csv_meta=csv_metadata.get(subject, {}))
            out[subject] = (
                f"eq{int(meta['equivocal'])}|cl24{int(meta['label_cl24'])}|"
                f"cohort={meta['cohort']}|scanner={meta['scanner']}"
            )
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rel_or_abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path.cwd() / p


def load_or_create_subject_folds(
    subject_ids: list[str],
    *,
    manifest_path: str | Path = DEFAULT_FOLD_MANIFEST,
    n_folds: int = 4,
    seed: int = 42,
    mode: str = "train",
    strata_by_subject: dict[str, str] | None = None,
) -> tuple[list[dict[str, list[str]]], dict[str, Any]]:
    """Persist and reuse deterministic subject-level folds.

    The stage scripts used to create folds in memory, which made S1/S2/S3 easy to
    drift apart. This manifest is intentionally keyed by subject hashes only. If
    the subject set changes (for example a dry-run limit vs full training), a new
    manifest is written and the report records the regeneration reason.
    """
    target = _rel_or_abs(manifest_path)
    subjects = sorted(set(str(s) for s in subject_ids))
    fold_strategy = "stratified_label_cohort_scanner_equivocal" if strata_by_subject else "deterministic_shuffle"
    strata_signature = {subject: str(strata_by_subject.get(subject, "")) for subject in subjects} if strata_by_subject else {}
    existing: dict[str, Any] | None = None
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            existing = None
    if (
        existing
        and existing.get("schema_version") == 1
        and existing.get("subject_id_hashes") == subjects
        and int(existing.get("n_folds", -1)) == int(n_folds)
        and int(existing.get("seed", -1)) == int(seed)
        and existing.get("fold_strategy", "deterministic_shuffle") == fold_strategy
        and (existing.get("strata_by_subject", {}) or {}) == strata_signature
    ):
        folds = [
            {"train": list(fold["train_subject_id_hashes"]), "val": list(fold["val_subject_id_hashes"])}
            for fold in existing.get("folds", [])
        ]
        return folds, {
            "manifest_path": str(target),
            "manifest_status": "REUSED",
            "n_subjects": len(subjects),
            "n_folds": n_folds,
            "mode": mode,
        }

    folds = make_subject_folds(subjects, n_folds=n_folds, seed=seed, strata_by_subject=strata_by_subject)
    reason = "CREATED" if existing is None else "REGENERATED_SUBJECT_SET_CHANGED"
    payload = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "manifest_status": reason,
        "mode_when_written": mode,
        "n_subjects": len(subjects),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "fold_strategy": fold_strategy,
        "strata_by_subject": strata_signature,
        "subject_id_hashes": subjects,
        "folds": [
            {
                "fold": idx,
                "train_subject_id_hashes": fold["train"],
                "val_subject_id_hashes": fold["val"],
            }
            for idx, fold in enumerate(folds)
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return folds, {
        "manifest_path": str(target),
        "manifest_status": reason,
        "n_subjects": len(subjects),
        "n_folds": n_folds,
        "mode": mode,
    }


def _mode_suffix(mode: str) -> str:
    return "_dryrun" if mode == "dry_run" else ""


def _alpha_uncertainty(logits) -> Any:
    alpha = F.softplus(logits) + 1.0
    return (2.0 / alpha.sum(dim=-1)).detach().cpu()


def write_validation_artifact(path: str | Path, **arrays: Any) -> str:
    """Write a compressed validation artifact and enforce equal row counts."""
    import numpy as np

    target = _rel_or_abs(path)
    if "centiloid_true" not in arrays or "labels_cl24" not in arrays or "route_uncertainty" not in arrays:
        raise MethodYDataContractError("BLOCKED_VALIDATION_ARTIFACT_SCHEMA_MISSING")
    converted = {key: np.asarray(value) for key, value in arrays.items()}
    row_counts = {
        key: int(value.shape[0])
        for key, value in converted.items()
        if value.ndim >= 1 and key != "feature_order"
    }
    if not row_counts:
        raise MethodYDataContractError("BLOCKED_VALIDATION_ARTIFACT_EMPTY")
    if len(set(row_counts.values())) != 1:
        raise MethodYDataContractError(f"BLOCKED_VALIDATION_ARTIFACT_ROW_MISMATCH:{row_counts}")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **converted)
    return str(target)


def validate_validation_artifact(path: str | Path, required: tuple[str, ...] = ("centiloid_true", "labels_cl24", "route_uncertainty")) -> dict[str, Any]:
    """Return a small schema summary for a saved validation .npz artifact."""
    import numpy as np

    target = _rel_or_abs(path)
    with np.load(target, allow_pickle=False) as data:
        keys = set(data.files)
        missing = sorted(set(required) - keys)
        if missing:
            raise MethodYDataContractError(f"BLOCKED_VALIDATION_ARTIFACT_SCHEMA_MISSING:{missing}")
        n = int(data[required[0]].shape[0])
        for key in required:
            if int(data[key].shape[0]) != n:
                raise MethodYDataContractError(f"BLOCKED_VALIDATION_ARTIFACT_ROW_MISMATCH:{key}")
        return {"path": str(target), "n_rows": n, "keys": sorted(keys)}


def cohort_probe_gate_summary(fold_metrics: list[dict[str, Any]], *, ceiling: float = 0.65) -> dict[str, Any]:
    probes = [float(m["cohort_probe_auc"]) for m in fold_metrics if m.get("cohort_probe_auc") is not None]
    return {
        "measured": bool(probes),
        "max_auc": max(probes) if probes else None,
        "passed": bool(probes) and max(probes) <= float(ceiling),
    }


def select_balanced_dry_run_subjects(h5_path: str | Path, subject_ids: list[str], *, limit: int) -> list[str]:
    """Select a small dry-run subset with label/cohort/scanner/equivocal diversity."""
    if limit <= 0 or len(subject_ids) <= limit:
        return list(subject_ids)
    import h5py
    from collections import defaultdict

    buckets: dict[tuple[int, int, str, str], list[str]] = defaultdict(list)
    csv_metadata = _load_pet_input_list_metadata()
    subject_set = set(subject_ids)
    with h5py.File(h5_path, "r") as h5:
        for subject in sorted(subject_set):
            if subject not in h5:
                continue
            meta = _h5_subject_metadata(h5[subject], csv_meta=csv_metadata.get(subject, {}))
            label = int(meta["label_cl24"])
            cohort = str(meta["cohort"])
            cohort = cohort if cohort in {"A4", "LEARN"} else "OTHER"
            scanner = str(meta["scanner"] or "UNKNOWN").upper()
            equivocal = int(meta["equivocal"])
            buckets[(equivocal, label, cohort, scanner)].append(subject)
    selected: list[str] = []
    while len(selected) < limit and any(buckets.values()):
        for key in sorted(buckets.keys()):
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                if len(selected) >= limit:
                    break
    return selected


def _sample_pet_affine_theta(image):
    """Sample one 3D affine transform using normalized affine_grid translations."""
    B = image.shape[0]
    angle = (torch.rand(1).item() * 2 - 1) * 2.5 * 3.14159265 / 180.0
    scale = 1.0 + (torch.rand(1).item() * 2 - 1) * 0.02
    cos_a, sin_a = float(torch.cos(torch.tensor(angle))), float(torch.sin(torch.tensor(angle)))
    _, _, depth, height, width = image.shape

    def vox_to_norm(vox: float, size: int) -> float:
        return 2.0 * float(vox) / max(float(size - 1), 1.0)

    tx = vox_to_norm((torch.rand(1).item() * 2 - 1) * 1.5, width)
    ty = vox_to_norm((torch.rand(1).item() * 2 - 1) * 1.5, height)
    tz = vox_to_norm((torch.rand(1).item() * 2 - 1) * 1.5, depth)
    theta = torch.eye(3, 4, dtype=image.dtype, device=image.device)
    theta[0, 0] = cos_a * scale
    theta[0, 1] = -sin_a * scale
    theta[1, 0] = sin_a * scale
    theta[1, 1] = cos_a * scale
    theta[2, 2] = scale
    theta[0, 3] = tx
    theta[1, 3] = ty
    theta[2, 3] = tz
    return theta.unsqueeze(0).expand(B, 3, 4).contiguous()


def _apply_geometric_transform(image, theta, *, sagittal_flip: bool):
    out = torch.flip(image, dims=[-1]) if sagittal_flip else image
    grid = F.affine_grid(theta, out.shape, align_corners=False)
    return F.grid_sample(out, grid, align_corners=False, mode="bilinear", padding_mode="border")


def paired_geometric_augment(image_a, image_b):
    """Apply the same small rigid affine transform to an MRI/PET pair."""
    if torch is None:
        raise RuntimeError("PyTorch is required for PET augmentation")
    theta = _sample_pet_affine_theta(image_a)
    return (
        _apply_geometric_transform(image_a, theta, sagittal_flip=False),
        _apply_geometric_transform(image_b, theta, sagittal_flip=False),
    )


def pet_train_augment(image):
    """PET-safe train-time augmentation: small rigid affine plus very light noise.

    No flips, gamma, bias-field, elastic deformation, smoothing, scaling, MixUp, or CutMix.
    ``image`` is expected as [B, C, D, H, W].
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for PET augmentation")
    out, _ = paired_geometric_augment(image, image)
    return pet_intensity_augment(out)


def pet_tta_predict(model, image):
    """Test-time augmentation: average sigmoid over identity + sagittal + A-P flips."""
    if torch is None:
        raise RuntimeError("PyTorch is required for TTA")
    model.eval()
    versions = [image, torch.flip(image, dims=[-1]), torch.flip(image, dims=[-2])]
    probs = []
    with torch.no_grad():
        for v in versions:
            out = model(v)
            logits = out["logits"]
            probs.append(F.softmax(logits, dim=-1)[:, 0])
    return torch.stack(probs).mean(dim=0)


def _collect_pet_validation(model, loader, device, *, use_tta: bool = True) -> dict[str, Any]:
    model.eval()
    all_logits, all_prob, all_uncertainty = [], [], []
    all_cl_pred, all_cl_true, all_y24, all_y30, all_tracer = [], [], [], [], []
    all_cohort, all_scanner, all_equivocal, all_latent, all_subjects = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            out = model(image)
            logits = out["logits"]
            prob = pet_tta_predict(model, image) if use_tta else F.softmax(logits, dim=-1)[:, 0]
            all_logits.extend(logits.detach().cpu().tolist())
            all_prob.extend(prob.cpu().tolist())
            all_uncertainty.extend(_alpha_uncertainty(logits).tolist())
            all_latent.extend(out["z_pet"].cpu().tolist())
            all_cl_pred.extend(out["cl_pred"].cpu().tolist())
            all_cl_true.extend([float(c) for c in batch["centiloid"]])
            all_y24.extend([int(c >= CL_PRIMARY) for c in batch["centiloid"]])
            all_y30.extend([int(c >= CL_SECONDARY) for c in batch["centiloid"]])
            all_tracer.extend([int(t) for t in batch["tracer_idx"]])
            all_cohort.extend([str(c) for c in batch["cohort"]])
            all_scanner.extend([str(s) for s in batch["scanner"]])
            all_equivocal.extend([int(e) for e in batch["equivocal"]])
            all_subjects.extend([str(s) for s in batch["subject_id_hash"]])
    return {
        "pet_logits": all_logits,
        "prob_pet": all_prob,
        "route_uncertainty": all_uncertainty,
        "cl_pred": all_cl_pred,
        "centiloid_true": all_cl_true,
        "labels_cl24": all_y24,
        "labels_cl30": all_y30,
        "tracer_idx": all_tracer,
        "cohort": all_cohort,
        "scanner": all_scanner,
        "equivocal": all_equivocal,
        "z_pet": all_latent,
        "subject_id_hash": all_subjects,
    }


def _pet_metrics_for_scope(collected: dict[str, Any], keep_mask: list[bool]) -> dict[str, Any]:
    all_prob = [value for value, keep in zip(collected["prob_pet"], keep_mask, strict=True) if keep]
    all_cl_pred = [value for value, keep in zip(collected["cl_pred"], keep_mask, strict=True) if keep]
    all_cl_true = [value for value, keep in zip(collected["centiloid_true"], keep_mask, strict=True) if keep]
    all_y24 = [value for value, keep in zip(collected["labels_cl24"], keep_mask, strict=True) if keep]
    all_y30 = [value for value, keep in zip(collected["labels_cl30"], keep_mask, strict=True) if keep]
    ece, brier = _ece_and_brier(all_y24, all_prob)
    return {
        "auroc_cl24": _auc(all_y24, all_prob),
        "auroc_cl30": _auc(all_y30, all_prob),
        "spearman_cl": _spearman(all_cl_pred, all_cl_true),
        "ece_cl24": ece,
        "brier_cl24": brier,
        "n_val": len(all_y24),
    }


def _evaluate_pet_only(model, loader, device, *, gate_metric_subjects: str = "non_equivocal_only"):
    collected = _collect_pet_validation(model, loader, device, use_tta=True)
    n = len(collected["labels_cl24"])
    all_mask = [True] * n
    non_equivocal_mask = [not bool(value) for value in collected["equivocal"]]
    metrics_by_scope = {
        "all_subjects": _pet_metrics_for_scope(collected, all_mask),
        "non_equivocal_only": _pet_metrics_for_scope(collected, non_equivocal_mask),
    }
    scope = gate_metric_subjects if gate_metric_subjects in metrics_by_scope else "non_equivocal_only"
    metrics = dict(metrics_by_scope[scope])
    metrics["metrics_by_subject_scope"] = metrics_by_scope
    metrics["gate_metric_subjects"] = scope
    return metrics


def _collect_pet_latents(model, loader, device) -> tuple[list[list[float]], list[int], list[str]]:
    model.eval()
    features: list[list[float]] = []
    tracers: list[int] = []
    cohorts: list[str] = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            features.extend(out["z_pet"].cpu().tolist())
            tracers.extend([int(t) for t in batch["tracer_idx"]])
            cohorts.extend([str(c) for c in batch["cohort"]])
    return features, tracers, cohorts


def _binary_cohort_labels(cohorts: list[str]) -> list[int]:
    mapping = {"LEARN": 0, "A4": 1}
    return [mapping.get(str(value).upper(), -1) for value in cohorts]


def _probe_measurability_status(labels: list[int], *, single_class_status: str) -> str:
    observed = sorted({int(value) for value in labels if int(value) >= 0})
    if len(observed) >= 2:
        return "MEASURABLE"
    if len(observed) == 1:
        return single_class_status
    return "NOT_MEASURABLE_NO_VALID_LABELS"


def _observed_tracer_classes(h5_path: str | Path, subject_ids: list[str]) -> set[int]:
    import h5py

    out: set[int] = set()
    with h5py.File(h5_path, "r") as h5:
        for subject in subject_ids:
            if subject not in h5:
                continue
            tracer = _decode_h5_value(h5[subject]["tracer"][()]) if "tracer" in h5[subject] else ""
            idx = TRACER_INDEX.get(str(tracer).upper(), -1)
            if idx >= 0:
                out.add(int(idx))
    return out


def train_pet_only_fold(fold_idx: int, train_ids, val_ids, h5_path, *, epochs: int, lr: float = 1e-4,
                        grad_accum: int = 4, device=None, amp: bool = True, base_channels: int = 4,
                        batch_size: int = 1, gradient_checkpointing: bool = True,
                        vram_budget_gb: float = 7.0, mode: str = "train",
                        loss_weights: dict[str, float] | None = None,
                        model_cfg: dict[str, Any] | None = None,
                        gate_metric_subjects: str = "non_equivocal_only",
                        validation_artifact_dir: str | Path = DEFAULT_VALIDATION_ARTIFACT_DIR,
                        checkpoint_dir: str | Path = "models/method_y",
                        pet_input_list_path: str | Path = DEFAULT_PET_INPUT_LIST,
                        num_workers: int = 0) -> dict[str, Any]:
    """Stage S1 PET-only training for one fold. Returns metrics + checkpoint path."""
    device = device or _device()
    train_ds = PETCacheDataset(h5_path, subject_ids=train_ids, pet_input_list_path=pet_input_list_path)
    val_ds = PETCacheDataset(h5_path, subject_ids=val_ids, pet_input_list_path=pet_input_list_path)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers))
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))
    model_cfg = model_cfg or {}
    model = PETNetLite(in_channels=1, base_channels=int(base_channels),
                       output_dim=int(model_cfg.get("output_dim", 256) or 256), n_tracers=3,
                       lambda_adv=0.1, gradient_checkpointing=bool(gradient_checkpointing),
                       normalization=str(model_cfg.get("normalization", "batch"))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    weights = {
        "evidential_weight": 1.0,
        "centiloid_weight": 0.5,
        "tracer_adversarial_weight": 0.1,
        "cohort_adversarial_weight": 0.0,
    }
    if loss_weights:
        weights.update({k: float(v) for k, v in loss_weights.items() if k in weights})
    tracer_classes = _observed_tracer_classes(h5_path, list(train_ids) + list(val_ids))
    tracer_adversarial_enabled = len(tracer_classes) >= 2 and weights["tracer_adversarial_weight"] > 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_val_loss = float("inf")
    patience = 0
    step = 0
    for epoch in range(int(epochs)):
        model.train()
        for batch in train_loader:
            image = batch["image"].to(device)
            image = pet_train_augment(image)
            target = batch["amyloid_pos"].to(device).long()
            cl_true = batch["centiloid"].to(device).float()
            tracer_idx = batch["tracer_idx"].to(device).long()
            cohort_idx = torch.as_tensor(
                _binary_cohort_labels([str(value) for value in batch["cohort"]]),
                device=device,
                dtype=torch.long,
            )
            synthetic_mask = batch["is_synthetic"].to(device).bool()
            equivocal_mask = batch["equivocal"].to(device).bool()
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                out = model(image)
                loss_ev = masked_evidential_cross_entropy_loss(
                    out["logits"], target, equivocal_mask, step=step, anneal_steps=ANNEAL_STEPS
                )
                loss_cl = centiloid_huber_loss(out["cl_pred"], cl_true, synthetic_mask=synthetic_mask)
                tracer_mask = tracer_idx >= 0
                loss_adv = (
                    F.cross_entropy(out["tracer_logits"][tracer_mask], tracer_idx[tracer_mask])
                    if tracer_adversarial_enabled and tracer_mask.any()
                    else out["tracer_logits"].sum() * 0.0
                )
                cohort_mask = cohort_idx >= 0
                loss_cohort_adv = (
                    F.cross_entropy(out["cohort_logits"][cohort_mask], cohort_idx[cohort_mask])
                    if weights["cohort_adversarial_weight"] > 0.0 and cohort_mask.any()
                    else out["cohort_logits"].sum() * 0.0
                )
                loss = (
                    weights["evidential_weight"] * loss_ev
                    + weights["centiloid_weight"] * loss_cl
                    + weights["tracer_adversarial_weight"] * loss_adv
                    + weights["cohort_adversarial_weight"] * loss_cohort_adv
                )
            scaler.scale(loss / grad_accum).backward()
            step += 1
            if step % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if step % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        # validation evidential loss for early stopping
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                target = batch["amyloid_pos"].to(device).long()
                equivocal_mask = batch["equivocal"].to(device).bool()
                out = model(image)
                val_losses.append(float(masked_evidential_cross_entropy_loss(out["logits"], target, equivocal_mask).cpu()))
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break
    metrics = _evaluate_pet_only(model, val_loader, device, gate_metric_subjects=gate_metric_subjects)
    train_latent, train_tracer, train_cohort = _collect_pet_latents(model, train_loader, device)
    val_latent, val_tracer, val_cohort = _collect_pet_latents(model, val_loader, device)
    tracer_probe_status = _probe_measurability_status(
        train_tracer + val_tracer,
        single_class_status="NOT_MEASURABLE_SINGLE_TRACER",
    )
    metrics["tracer_probe_status"] = tracer_probe_status
    metrics["tracer_adversarial_status"] = (
        "ENABLED_MULTITRACER" if tracer_adversarial_enabled else "DISABLED_SINGLE_TRACER"
    )
    metrics["cohort_adversarial_status"] = (
        "ENABLED" if weights["cohort_adversarial_weight"] > 0.0 else "DISABLED"
    )
    metrics["tracer_probe_auc"] = (
        _linear_probe_auc(train_latent, train_tracer, val_latent, val_tracer)
        if tracer_probe_status == "MEASURABLE"
        else None
    )
    metrics["cohort_probe_auc"] = _linear_probe_auc(
        train_latent, _binary_cohort_labels(train_cohort), val_latent, _binary_cohort_labels(val_cohort)
    )
    artifact = _collect_pet_validation(model, val_loader, device, use_tta=True)
    artifact_path = Path(validation_artifact_dir) / f"stage_s1_fold{fold_idx}{_mode_suffix(mode)}.npz"
    metrics["validation_artifact"] = write_validation_artifact(
        artifact_path,
        pet_logits=artifact["pet_logits"],
        prob_pet=artifact["prob_pet"],
        route_uncertainty=artifact["route_uncertainty"],
        centiloid_true=artifact["centiloid_true"],
        labels_cl24=artifact["labels_cl24"],
        labels_cl30=artifact["labels_cl30"],
        tracer_idx=artifact["tracer_idx"],
        cohort=artifact["cohort"],
        scanner=artifact["scanner"],
        equivocal=artifact["equivocal"],
        z_pet=artifact["z_pet"],
        subject_id_hash=artifact["subject_id_hash"],
    )
    vram_blocker = _check_vram_budget(device, vram_budget_gb)
    if vram_blocker:
        metrics["vram_blocker"] = vram_blocker
    suffix = "_dryrun" if mode == "dry_run" else ""
    ckpt_path = str(Path(checkpoint_dir) / f"pet_only_fold{fold_idx}{suffix}.pt")
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "fold": fold_idx, "metrics": metrics}, ckpt_path)
    metrics["checkpoint"] = ckpt_path
    metrics["best_val_evidential_loss"] = best_val_loss
    return metrics


def run_stage_s1(cfg: dict, *, mode: str, limit: int, epochs: int) -> dict[str, Any]:
    """Run Stage S1 across 4 folds and evaluate the acceptance gate."""
    h5_path = cfg.get("inputs", {}).get("pet_h5", DEFAULT_METHOD_Y_PET_CACHE)
    pet_input_list_path = cfg.get("inputs", {}).get("pet_input_list", DEFAULT_PET_INPUT_LIST)
    profile = _compute_profile(cfg)
    gate_cfg = cfg.get("acceptance_gate", {}) or {}
    gate_metric_subjects = str(gate_cfg.get("metric_subjects", "non_equivocal_only"))
    allow_single_tracer_promotion = bool(gate_cfg.get("allow_single_tracer_promotion", False))
    cohort_probe_hard_gate = bool(gate_cfg.get("cohort_probe_hard_gate_for_s2_s5", True))
    cohort_probe_deferred_policy = gate_cfg.get("cohort_probe_deferred_policy")
    import h5py

    with h5py.File(h5_path, "r") as h5:
        subjects = list(h5.keys())
    if mode == "dry_run":
        subjects = select_balanced_dry_run_subjects(h5_path, subjects, limit=limit)
    strata_by_subject = _subject_strata_from_h5(h5_path, subjects, pet_input_list_path=pet_input_list_path)
    fold_cfg = cfg.get("folds", {}) or {}
    folds, fold_manifest = load_or_create_subject_folds(
        subjects,
        manifest_path=fold_cfg.get("manifest", DEFAULT_FOLD_MANIFEST),
        n_folds=int(fold_cfg.get("n_folds", 4)),
        seed=int(fold_cfg.get("seed", 42)),
        mode=mode,
        strata_by_subject=strata_by_subject,
    )
    validation_dir = cfg.get("outputs", {}).get("validation_dir", DEFAULT_VALIDATION_ARTIFACT_DIR)
    checkpoint_dir = cfg.get("outputs", {}).get("checkpoint_dir", "models/method_y")
    num_workers = int(profile.get("num_workers", 0) or 0)
    fold_metrics = []
    vram_blocked = False
    for k, fold in enumerate(folds):
        m = train_pet_only_fold(k, fold["train"], fold["val"], h5_path, epochs=epochs,
                                grad_accum=profile["grad_accum"], amp=profile["amp"],
                                base_channels=profile["base_channels"], batch_size=profile["batch_size"],
                                 gradient_checkpointing=profile["gradient_checkpointing"],
                                 vram_budget_gb=profile["vram_budget_gb"], mode=mode,
                                 loss_weights=cfg.get("losses", {}),
                                 gate_metric_subjects=gate_metric_subjects,
                                 validation_artifact_dir=validation_dir,
                                 checkpoint_dir=checkpoint_dir,
                                 model_cfg=cfg.get("model", {}),
                                 pet_input_list_path=pet_input_list_path,
                                 num_workers=num_workers)
        fold_metrics.append(m)
        if m.get("vram_blocker"):
            vram_blocked = True
            break
    aurocs = [m["auroc_cl24"] for m in fold_metrics if m["auroc_cl24"] is not None]
    spearmans = [m["spearman_cl"] for m in fold_metrics if m["spearman_cl"] is not None]
    tracer_probes = [m["tracer_probe_auc"] for m in fold_metrics if m.get("tracer_probe_auc") is not None]
    csv_metadata = _load_pet_input_list_metadata(pet_input_list_path)
    with h5py.File(h5_path, "r") as h5:
        observed_tracers = sorted(
            {
                _h5_subject_metadata(h5[subject], csv_meta=csv_metadata.get(subject, {}))["tracer"]
                for subject in subjects
                if subject in h5
            }
        )
    if len([value for value in observed_tracers if value]) <= 1:
        tracer_probe_status = "NOT_MEASURABLE_SINGLE_TRACER"
    elif tracer_probes:
        tracer_probe_status = "MEASURED"
    else:
        tracer_probe_status = "NOT_MEASURABLE_INSUFFICIENT_FOLD_CLASSES"
    tracer_gate_passed = (
        (bool(tracer_probes) and max(tracer_probes) <= 0.65)
        or (tracer_probe_status == "NOT_MEASURABLE_SINGLE_TRACER" and allow_single_tracer_promotion)
    )
    cohort_gate = cohort_probe_gate_summary(fold_metrics)
    ge085 = sum(1 for a in aurocs if a >= 0.85)
    gate = (
        (not vram_blocked)
        and ge085 >= 3
        and bool(spearmans)
        and min(spearmans) >= 0.65
        and tracer_gate_passed
        and (cohort_gate["passed"] or not cohort_probe_hard_gate)
    )
    blocker = None
    blocker_reasons: list[str] = []
    deferred_risks: list[str] = []
    if vram_blocked:
        blocker = next((m.get("vram_blocker") for m in fold_metrics if m.get("vram_blocker")), None)
        blocker_reasons.append(str(blocker))
    if ge085 < 3:
        blocker_reasons.append("S1_AUROC_GATE_FAILED")
    if not spearmans or min(spearmans) < 0.65:
        blocker_reasons.append("S1_SPEARMAN_GATE_FAILED")
    if tracer_probe_status == "NOT_MEASURABLE_SINGLE_TRACER" and not allow_single_tracer_promotion:
        blocker_reasons.append("BLOCKED_MULTITRACER_REQUIRED_FOR_TRACER_PROBE")
    elif tracer_probe_status != "MEASURED" and not tracer_gate_passed:
        blocker_reasons.append("BLOCKED_TRACER_PROBE_NOT_MEASURABLE")
    if not cohort_gate["passed"] and cohort_probe_hard_gate:
        blocker_reasons.append("S1_COHORT_PROBE_GATE_FAILED")
    elif not cohort_gate["passed"]:
        deferred_risks.append("S1_COHORT_PROBE_GATE_DEFERRED_TO_FINAL_REVIEW")
    if not gate and blocker is None and blocker_reasons:
        blocker = "+".join(dict.fromkeys(blocker_reasons))
    return {"fold_metrics": fold_metrics, "acceptance_gate_passed": bool(gate),
            "folds_auroc_ge_0.85": ge085, "min_spearman": min(spearmans) if spearmans else None,
            "max_tracer_probe_auc": max(tracer_probes) if tracer_probes else None,
            "tracer_probe_status": tracer_probe_status,
            "tracer_probe_gate": {
                "status": tracer_probe_status,
                "passed": bool(tracer_gate_passed),
                "allow_single_tracer_promotion": allow_single_tracer_promotion,
                "tracers_observed": observed_tracers,
            },
            "gate_metric_subjects": gate_metric_subjects,
            "max_cohort_probe_auc": cohort_gate["max_auc"],
            "cohort_probe_gate": {
                **cohort_gate,
                "hard_gate_for_s2_s5": cohort_probe_hard_gate,
                "deferred_to_final_review": bool(deferred_risks),
                "deferred_policy": cohort_probe_deferred_policy,
            },
            "fold_manifest": fold_manifest,
            "vram_blocker": next((m.get("vram_blocker") for m in fold_metrics if m.get("vram_blocker")), None),
            "blocker": blocker,
            "blocker_reasons": list(dict.fromkeys(blocker_reasons)),
            "deferred_risks": list(dict.fromkeys(deferred_risks)),
            "config_diff_vs_canonical": compute_config_diff(cfg),
            "compute_profile": profile}


def build_method_x_encoder(ckpt_path: str, backbone: str = "resnet10_3d"):
    """Load a frozen Method X MRI encoder from a checkpoint (schema must be compatible)."""
    from src.models.backbones.lightweight_3d import MethodX3DClassifier

    raw = torch.load(ckpt_path, map_location="cpu")
    meta = raw if isinstance(raw, dict) else {}
    ckpt_backbone = str(meta.get("backbone", "")).lower().strip()
    requested_backbone = str(backbone).lower().strip()
    if ckpt_backbone and ckpt_backbone != requested_backbone:
        raise MethodYDataContractError(
            f"BLOCKED_METHOD_X_CHECKPOINT_BACKBONE_MISMATCH:{ckpt_backbone}!={requested_backbone}"
        )
    latent_dim = int(meta.get("latent_dim", 128) or 128)
    model = MethodX3DClassifier(
        backbone=requested_backbone,
        fusion=str(meta.get("fusion", "cross_attention")),
        latent_dim=latent_dim,
        dropout=float(meta.get("dropout", 0.1) or 0.1),
        tab_hidden_dim=int(meta.get("tab_hidden_dim", 32) or 32),
        bottleneck_dim=int(meta.get("bottleneck_dim", 32) or 32),
        fusion_hidden_dim=meta.get("fusion_hidden_dim", None),
    )
    if isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw
    state = state if isinstance(state, dict) else {}
    if state and all(str(key).startswith("module.") for key in state.keys()):
        state = {str(key)[7:]: value for key, value in state.items()}
    model_keys = set(model.state_dict().keys())
    matched = sorted(model_keys.intersection(state.keys()))
    if len(matched) < max(1, int(0.25 * len(model_keys))):
        raise MethodYDataContractError(
            f"BLOCKED_METHOD_X_CHECKPOINT_SCHEMA_INCOMPATIBLE:matched_keys={len(matched)}/{len(model_keys)}"
        )
    load_report = model.load_state_dict(state, strict=False)
    model._method_y_checkpoint_load_report = {
        "checkpoint": ckpt_path,
        "requested_backbone": requested_backbone,
        "checkpoint_backbone": ckpt_backbone or None,
        "matched_keys": len(matched),
        "total_model_keys": len(model_keys),
        "missing_keys": list(load_report.missing_keys),
        "unexpected_keys": list(load_report.unexpected_keys),
    }
    return model


def _stage_report_path(stage: str, execution_root: str | Path | None = None) -> Path:
    root = Path(execution_root) if execution_root else Path("results") / "method_y_execution"
    return root / f"stage_{stage.lower()}_report.json"


def _load_stage_report(stage: str, execution_root: str | Path | None = None) -> dict[str, Any]:
    import json

    path = _stage_report_path(stage, execution_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_stage_fold_metrics(stage: str, execution_root: str | Path | None = None) -> list[dict[str, Any]]:
    payload = _load_stage_report(stage, execution_root)
    metrics = payload.get("fold_metrics")
    return metrics if isinstance(metrics, list) else []


def _dependency_report_root(cfg: dict[str, Any]) -> str | Path | None:
    inputs = cfg.get("inputs", {}) or {}
    outputs = cfg.get("outputs", {}) or {}
    reports = cfg.get("reports", {}) or {}
    return inputs.get("stage_report_root") or reports.get("stage_report_root") or outputs.get("stage_report_root")


def _fold_checkpoint_path(pattern: str, fold_idx: int, *, mode: str, dryrun_suffix: bool = True) -> str:
    path = pattern.format(fold=fold_idx)
    if mode == "dry_run" and dryrun_suffix and path.endswith(".pt") and "_dryrun" not in path:
        return path[:-3] + "_dryrun.pt"
    return path


def _evaluate_joint(model, loader, device):
    collected = _collect_joint_validation(model, loader, device)
    all_prob = collected["prob_y_fused"]
    all_y24 = collected["labels_cl24"]
    ece, brier = _ece_and_brier(all_y24, all_prob)
    return {"auroc_cl24": _auc(all_y24, all_prob), "ece_cl24": ece, "brier_cl24": brier,
            "cohort_probe_auc": None, "n_val": len(all_y24)}


def _collect_joint_validation(model, loader, device) -> dict[str, Any]:
    model.eval()
    all_logits, all_prob, all_uncertainty = [], [], []
    all_y24, all_y30, all_cl, all_latent, all_cohort, all_subjects = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            mri = batch["mri"].to(device)
            pet = batch["pet"].to(device)
            tabular = batch.get("tabular")
            tabular = tabular.to(device) if tabular is not None else None
            out = model(mri, pet, tabular=tabular)
            logits = out["logits"]
            all_logits.extend(logits.detach().cpu().tolist())
            all_prob.extend(F.softmax(logits, dim=-1)[:, 0].cpu().tolist())
            all_uncertainty.extend(_alpha_uncertainty(logits).tolist())
            all_y24.extend([int(c >= CL_PRIMARY) for c in batch["centiloid"]])
            all_y30.extend([int(c >= CL_SECONDARY) for c in batch["centiloid"]])
            all_cl.extend([float(c) for c in batch["centiloid"]])
            all_latent.extend(out["z_fused"].cpu().tolist())
            all_cohort.extend([str(c) for c in batch["cohort"]])
            all_subjects.extend([str(s) for s in batch["subject_id_hash"]])
    return {
        "joint_logits": all_logits,
        "prob_y_fused": all_prob,
        "route_uncertainty": all_uncertainty,
        "centiloid_true": all_cl,
        "labels_cl24": all_y24,
        "labels_cl30": all_y30,
        "z_fused": all_latent,
        "cohort": all_cohort,
        "subject_id_hash": all_subjects,
    }


def train_joint_fold(fold_idx, train_ids, val_ids, h5_path, mri_ckpt, *, epochs, lr=1e-4, grad_accum=4,
                     device=None, amp=True, base_channels=4, hidden_dim=128, bottleneck_dim=16,
                     num_heads=2, batch_size=1, mri_backbone="resnet10_3d",
                     gradient_checkpointing=True, vram_budget_gb=7.0, mode: str = "train",
                     mri_h5_path: str = "data/cached/t1_combined_foldlocal_4000_128.h5",
                     pet_ckpt_path: str | None = None,
                     loss_weights: dict[str, float] | None = None,
                     validation_artifact_dir: str | Path = DEFAULT_VALIDATION_ARTIFACT_DIR,
                     checkpoint_dir: str | Path = "models/method_y",
                     pet_input_list_path: str | Path = DEFAULT_PET_INPUT_LIST,
                     num_workers: int = 0):
    """Stage S2 joint MRI+PET fine-tune with frozen MRI encoder."""
    device = device or _device()
    mri_encoder = build_method_x_encoder(mri_ckpt, backbone=mri_backbone).to(device)
    pet_branch = PETNetLite(in_channels=1, base_channels=int(base_channels), output_dim=256, n_tracers=3,
                            lambda_adv=0.1, gradient_checkpointing=bool(gradient_checkpointing)).to(device)
    if pet_ckpt_path:
        state = torch.load(pet_ckpt_path, map_location="cpu")
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        pet_branch.load_state_dict(state_dict, strict=False)
    mri_dim = int(getattr(mri_encoder, "out_dim", 128) or 128)
    model = JointMRIPETModel(mri_encoder, pet_branch, mri_dim=mri_dim, pet_dim=256,
                             hidden_dim=int(hidden_dim), bottleneck_dim=int(bottleneck_dim),
                             num_heads=int(num_heads)).to(device)
    train_ds = PairedMRIPETCacheDataset(
        h5_path, mri_h5_path, subject_ids=train_ids, pet_input_list_path=pet_input_list_path
    )
    val_ds = PairedMRIPETCacheDataset(
        h5_path, mri_h5_path, subject_ids=val_ids, pet_input_list_path=pet_input_list_path
    )
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers))
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    weights = {
        "pet_evidential_weight": 1.0,
        "centiloid_weight": 0.5,
        "tracer_adversarial_weight": 0.1,
        "fused_evidential_weight": 1.0,
    }
    if loss_weights:
        weights.update({k: float(v) for k, v in loss_weights.items() if k in weights})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step = 0
    for epoch in range(int(epochs)):
        model.train()
        for batch in train_loader:
            pet = batch["image"].to(device)
            mri = batch["mri"].to(device)
            mri, pet = paired_geometric_augment(mri, pet)
            pet = pet_intensity_augment(pet)
            tabular = batch.get("tabular")
            tabular = tabular.to(device) if tabular is not None else None
            target = batch["amyloid_pos"].to(device).long()
            cl_true = batch["centiloid"].to(device).float()
            tracer_idx = batch["tracer_idx"].to(device).long()
            equivocal_mask = batch["equivocal"].to(device).bool()
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                out = model(mri, pet, tabular=tabular)
                pet_out = out["pet"]
                loss_ev = masked_evidential_cross_entropy_loss(
                    pet_out["logits"], target, equivocal_mask, step=step, anneal_steps=ANNEAL_STEPS
                )
                loss_cl = centiloid_huber_loss(pet_out["cl_pred"], cl_true, synthetic_mask=batch["is_synthetic"].to(device).bool())
                tracer_mask = tracer_idx >= 0
                loss_adv = F.cross_entropy(pet_out["tracer_logits"][tracer_mask], tracer_idx[tracer_mask]) if tracer_mask.any() else pet_out["tracer_logits"].sum() * 0.0
                loss_fused = masked_evidential_cross_entropy_loss(
                    out["logits"], target, equivocal_mask, step=step, anneal_steps=ANNEAL_STEPS
                )
                loss = (
                    weights["pet_evidential_weight"] * loss_ev
                    + weights["centiloid_weight"] * loss_cl
                    + weights["tracer_adversarial_weight"] * loss_adv
                    + weights["fused_evidential_weight"] * loss_fused
                )
            scaler.scale(loss / grad_accum).backward()
            step += 1
            if step % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if step % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    metrics = _evaluate_joint(model, val_loader, device)
    train_artifact = _collect_joint_validation(model, train_loader, device)
    val_artifact = _collect_joint_validation(model, val_loader, device)
    metrics["cohort_probe_auc"] = _linear_probe_auc(
        train_artifact["z_fused"],
        _binary_cohort_labels(train_artifact["cohort"]),
        val_artifact["z_fused"],
        _binary_cohort_labels(val_artifact["cohort"]),
    )
    artifact_path = Path(validation_artifact_dir) / f"stage_s2_fold{fold_idx}{_mode_suffix(mode)}.npz"
    metrics["validation_artifact"] = write_validation_artifact(
        artifact_path,
        joint_logits=val_artifact["joint_logits"],
        prob_y_fused=val_artifact["prob_y_fused"],
        route_uncertainty=val_artifact["route_uncertainty"],
        centiloid_true=val_artifact["centiloid_true"],
        labels_cl24=val_artifact["labels_cl24"],
        labels_cl30=val_artifact["labels_cl30"],
        z_fused=val_artifact["z_fused"],
        cohort=val_artifact["cohort"],
        subject_id_hash=val_artifact["subject_id_hash"],
    )
    vram_blocker = _check_vram_budget(device, vram_budget_gb)
    if vram_blocker:
        metrics["vram_blocker"] = vram_blocker
    suffix = "_dryrun" if mode == "dry_run" else ""
    ckpt_path = str(Path(checkpoint_dir) / f"joint_fold{fold_idx}{suffix}.pt")
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "fold": fold_idx, "metrics": metrics}, ckpt_path)
    metrics["checkpoint"] = ckpt_path
    return metrics


def run_stage_s2(cfg: dict, *, mode: str, limit: int, epochs: int, s1_auroc: float | None = None) -> dict[str, Any]:
    h5_path = cfg.get("inputs", {}).get("pet_h5", DEFAULT_METHOD_Y_PET_CACHE)
    pet_input_list_path = cfg.get("inputs", {}).get("pet_input_list", DEFAULT_PET_INPUT_LIST)
    mri_ckpt = cfg.get("inputs", {}).get("mri_ckpt", "models/method_x/best.pt")
    mri_h5 = cfg.get("inputs", {}).get("mri_h5", "data/cached/t1_combined_foldlocal_4000_128.h5")
    input_cfg = cfg.get("inputs", {}) or {}
    pet_ckpt_pattern = input_cfg.get("s1_pet_ckpt_pattern", "models/method_y/pet_only_fold{fold}.pt")
    pet_ckpt_dryrun_suffix = bool(input_cfg.get("s1_pet_ckpt_dryrun_suffix", True))
    profile = _compute_profile(cfg)
    dependency_root = _dependency_report_root(cfg)
    s1_report = _load_stage_report("S1", dependency_root)
    if s1_report.get("acceptance_gate_passed") is not True:
        return {"fold_metrics": [], "acceptance_gate_passed": False,
                "folds_auroc_ge_s1_plus_0.01": 0, "max_cohort_probe_auc": None,
                "vram_blocker": None, "blocker": "BLOCKED_S1_GATE_FAILED",
                "s1_report_status": s1_report.get("status"),
                "s1_blocker": s1_report.get("blocker") or "S1_NOT_PASSED",
                "config_diff_vs_canonical": compute_config_diff(cfg), "compute_profile": profile}
    if not Path(mri_ckpt).exists():
        return {"fold_metrics": [], "acceptance_gate_passed": False,
                "folds_auroc_ge_s1_plus_0.01": 0, "max_cohort_probe_auc": None,
                "vram_blocker": None, "blocker": "BLOCKED_METHOD_X_CHECKPOINT_MISSING",
                "config_diff_vs_canonical": compute_config_diff(cfg), "compute_profile": profile}
    s1_fold_metrics = _load_stage_fold_metrics("S1", dependency_root)
    if not s1_fold_metrics:
        return {"fold_metrics": [], "acceptance_gate_passed": False,
                "folds_auroc_ge_s1_plus_0.01": 0, "max_cohort_probe_auc": None,
                "vram_blocker": None, "blocker": "BLOCKED_S1_METRICS_MISSING",
                "config_diff_vs_canonical": compute_config_diff(cfg), "compute_profile": profile}
    import h5py

    with h5py.File(h5_path, "r") as h5:
        subjects = list(h5.keys())
    if mode == "dry_run":
        subjects = select_balanced_dry_run_subjects(h5_path, subjects, limit=limit)
    strata_by_subject = _subject_strata_from_h5(h5_path, subjects, pet_input_list_path=pet_input_list_path)
    fold_cfg = cfg.get("folds", {}) or {}
    folds, fold_manifest = load_or_create_subject_folds(
        subjects,
        manifest_path=fold_cfg.get("manifest", DEFAULT_FOLD_MANIFEST),
        n_folds=int(fold_cfg.get("n_folds", 4)),
        seed=int(fold_cfg.get("seed", 42)),
        mode=mode,
        strata_by_subject=strata_by_subject,
    )
    validation_dir = cfg.get("outputs", {}).get("validation_dir", DEFAULT_VALIDATION_ARTIFACT_DIR)
    checkpoint_dir = cfg.get("outputs", {}).get("checkpoint_dir", "models/method_y")
    num_workers = int(profile.get("num_workers", 0) or 0)
    fold_metrics = []
    vram_blocked = False
    for k, fold in enumerate(folds):
        pet_ckpt = _fold_checkpoint_path(pet_ckpt_pattern, k, mode=mode, dryrun_suffix=pet_ckpt_dryrun_suffix)
        if not Path(pet_ckpt).exists():
            return {"fold_metrics": fold_metrics, "acceptance_gate_passed": False,
                    "folds_auroc_ge_s1_plus_0.01": 0,
                    "max_cohort_probe_auc": None, "vram_blocker": None,
                    "blocker": f"BLOCKED_S1_PET_CHECKPOINT_MISSING:{pet_ckpt}",
                    "config_diff_vs_canonical": compute_config_diff(cfg), "compute_profile": profile}
        try:
            m = train_joint_fold(k, fold["train"], fold["val"], h5_path, mri_ckpt, epochs=epochs,
                                 grad_accum=profile["grad_accum"], amp=profile["amp"],
                                 base_channels=profile["base_channels"], hidden_dim=profile["hidden_dim"],
                                 bottleneck_dim=profile["bottleneck_dim"], num_heads=profile["num_heads"],
                                 batch_size=profile["batch_size"], mri_backbone=profile["mri_backbone"],
                                 gradient_checkpointing=profile["gradient_checkpointing"],
                                 vram_budget_gb=profile["vram_budget_gb"], mode=mode,
                                 mri_h5_path=mri_h5, pet_ckpt_path=pet_ckpt,
                                 loss_weights=cfg.get("losses", {}),
                                 validation_artifact_dir=validation_dir,
                                 checkpoint_dir=checkpoint_dir,
                                 pet_input_list_path=pet_input_list_path,
                                 num_workers=num_workers)
        except MethodYDataContractError as exc:
            return {"fold_metrics": fold_metrics, "acceptance_gate_passed": False,
                    "folds_auroc_ge_s1_plus_0.01": 0, "max_cohort_probe_auc": None,
                    "vram_blocker": None, "blocker": str(exc),
                    "config_diff_vs_canonical": compute_config_diff(cfg), "compute_profile": profile}
        fold_metrics.append(m)
        if m.get("vram_blocker"):
            vram_blocked = True
            break
    aurocs = [m["auroc_cl24"] for m in fold_metrics if m["auroc_cl24"] is not None]
    cohort_gate = cohort_probe_gate_summary(fold_metrics)
    s1_aurocs = [float(m["auroc_cl24"]) for m in s1_fold_metrics if m.get("auroc_cl24") is not None]
    s1_eces = [float(m["ece_cl24"]) for m in s1_fold_metrics if m.get("ece_cl24") is not None]
    s2_eces = [float(m["ece_cl24"]) for m in fold_metrics if m.get("ece_cl24") is not None]
    if s1_auroc is not None:
        s1_aurocs = [float(s1_auroc)] * len(aurocs)
    ge = sum(
        1
        for idx, auc in enumerate(aurocs)
        if idx < len(s1_aurocs) and auc >= (s1_aurocs[idx] + 0.01)
    )
    ece_gate = bool(s1_eces) and bool(s2_eces) and max(s2_eces) <= max(s1_eces)
    gate = (not vram_blocked) and ge >= 3 and cohort_gate["passed"] and ece_gate
    return {"fold_metrics": fold_metrics, "acceptance_gate_passed": bool(gate),
            "folds_auroc_ge_s1_plus_0.01": ge, "max_cohort_probe_auc": cohort_gate["max_auc"],
            "cohort_probe_gate": cohort_gate,
            "s1_reference_aurocs": s1_aurocs, "s1_reference_eces": s1_eces,
            "ece_gate_passed": ece_gate,
            "fold_manifest": fold_manifest,
            "vram_blocker": next((m.get("vram_blocker") for m in fold_metrics if m.get("vram_blocker")), None),
            "config_diff_vs_canonical": compute_config_diff(cfg),
            "compute_profile": profile}


def _collect_student_validation(model, loader, device) -> dict[str, Any]:
    model.eval()
    all_logits, all_prob, all_uncertainty = [], [], []
    all_y24, all_y30, all_cl, all_latent, all_cohort, all_subjects = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            mri = batch["mri"].to(device)
            tabular = batch.get("tabular")
            tabular = tabular.to(device) if tabular is not None else None
            out = model(mri, tabular=tabular)
            logits = out["logits"]
            all_logits.extend(logits.detach().cpu().tolist())
            all_prob.extend(F.softmax(logits, dim=-1)[:, 0].cpu().tolist())
            all_uncertainty.extend(_alpha_uncertainty(logits).tolist())
            all_y24.extend([int(c >= CL_PRIMARY) for c in batch["centiloid"]])
            all_y30.extend([int(c >= CL_SECONDARY) for c in batch["centiloid"]])
            all_cl.extend([float(c) for c in batch["centiloid"]])
            all_latent.extend(out["z_mri"].detach().cpu().tolist())
            all_cohort.extend([str(c) for c in batch["cohort"]])
            all_subjects.extend([str(s) for s in batch["subject_id_hash"]])
    return {
        "distill_logits": all_logits,
        "prob_yd": all_prob,
        "route_uncertainty": all_uncertainty,
        "centiloid_true": all_cl,
        "labels_cl24": all_y24,
        "labels_cl30": all_y30,
        "z_mri": all_latent,
        "cohort": all_cohort,
        "subject_id_hash": all_subjects,
    }


def _try_write_calibration_input_artifact(
    fold_idx: int,
    *,
    mode: str,
    validation_artifact_dir: str | Path,
    distill_artifact_path: str | Path,
) -> str | None:
    import numpy as np

    val_dir = _rel_or_abs(validation_artifact_dir)
    joint_path = val_dir / f"stage_s2_fold{fold_idx}{_mode_suffix(mode)}.npz"
    if not joint_path.exists():
        return None
    with np.load(joint_path, allow_pickle=False) as joint, np.load(_rel_or_abs(distill_artifact_path), allow_pickle=False) as distill:
        joint_subjects = joint["subject_id_hash"].astype(str)
        distill_subjects = distill["subject_id_hash"].astype(str)
        if joint_subjects.shape != distill_subjects.shape or not bool((joint_subjects == distill_subjects).all()):
            return None
        route_uncertainty = np.maximum(joint["route_uncertainty"].astype(float), distill["route_uncertainty"].astype(float))
        out_path = val_dir / f"calibration_fold{fold_idx}{_mode_suffix(mode)}_inputs.npz"
        return write_validation_artifact(
            out_path,
            joint_logits=joint["joint_logits"],
            distill_logits=distill["distill_logits"],
            centiloid_true=joint["centiloid_true"],
            labels_cl24=joint["labels_cl24"],
            labels_cl30=joint["labels_cl30"],
            route_uncertainty=route_uncertainty,
            subject_id_hash=joint_subjects,
        )


def train_distill_fold(fold_idx, train_ids, val_ids, h5_path, teacher_ckpt, *, epochs, lr=1e-4, T=2.0,
                       device=None, amp=True, batch_size=1, grad_accum: int = 4, mri_backbone="resnet10_3d",
                       gradient_checkpointing=True, vram_budget_gb=7.0, mri_ckpt="models/method_x/best.pt",
                       mode: str = "train", mri_h5_path: str = "data/cached/t1_combined_foldlocal_4000_128.h5",
                       loss_weights: dict[str, float] | None = None,
                       validation_artifact_dir: str | Path = DEFAULT_VALIDATION_ARTIFACT_DIR,
                       checkpoint_dir: str | Path = "models/method_y",
                       pet_input_list_path: str | Path = DEFAULT_PET_INPUT_LIST,
                       num_workers: int = 0):
    """Stage S3 Y->X distillation: PET-free MRI student from PET+MRI teacher."""
    device = device or _device()
    mri_encoder = build_method_x_encoder(mri_ckpt, backbone=mri_backbone).to(device)
    latent_dim = int(getattr(mri_encoder, "out_dim", 128) or 128)
    student = MethodYDistilledStudent(mri_encoder, latent_dim=latent_dim).to(device)
    teacher = JointMRIPETModel(build_method_x_encoder(mri_ckpt, backbone=mri_backbone),
                               PETNetLite(base_channels=4, gradient_checkpointing=bool(gradient_checkpointing)),
                               mri_dim=latent_dim, pet_dim=256, hidden_dim=128, bottleneck_dim=16).to(device)
    tstate = torch.load(teacher_ckpt, map_location="cpu")
    teacher.load_state_dict(tstate["model"] if "model" in tstate else tstate, strict=False)
    teacher.eval()
    train_ds = PairedMRIPETCacheDataset(
        h5_path, mri_h5_path, subject_ids=train_ids, pet_input_list_path=pet_input_list_path
    )
    val_ds = PairedMRIPETCacheDataset(
        h5_path, mri_h5_path, subject_ids=val_ids, pet_input_list_path=pet_input_list_path
    )
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers))
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    weights = {
        "kd_logits_weight": 1.0,
        "kd_feature_weight": 0.5,
        "supervised_weight": 1.0,
    }
    if loss_weights:
        weights.update({k: float(v) for k, v in loss_weights.items() if k in weights})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(int(epochs)):
        student.train()
        step = 0
        for batch in train_loader:
            pet = batch["image"].to(device)
            mri = batch["mri"].to(device)
            tabular = batch.get("tabular")
            tabular = tabular.to(device) if tabular is not None else None
            target = batch["amyloid_pos"].to(device).float()
            with torch.no_grad():
                t_out = teacher(mri, pet, tabular=tabular)
                teacher_logits = t_out["logits"]
                teacher_feat = t_out["z_mri"]
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                s_out = student(mri, tabular=tabular)
                loss, _ = method_y_distillation_loss(
                    s_out["logits"], teacher_logits, s_out["z_mri"], teacher_feat, target,
                    temperature=T, lambda_kd_logits=weights["kd_logits_weight"],
                    lambda_kd_feat=weights["kd_feature_weight"], lambda_ce=weights["supervised_weight"],
                )
            scaler.scale(loss / grad_accum).backward()
            step += 1
            if step % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if step % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    train_artifact = _collect_student_validation(student, train_loader, device)
    val_artifact = _collect_student_validation(student, val_loader, device)
    all_prob = val_artifact["prob_yd"]
    all_y24 = val_artifact["labels_cl24"]
    metrics = {"auroc_cl24": _auc(all_y24, all_prob), "uses_pet_as_input": False,
               "uses_synthetic_pet_as_input": False,
               "cohort_probe_auc": _linear_probe_auc(
                   train_artifact["z_mri"],
                   _binary_cohort_labels(train_artifact["cohort"]),
                   val_artifact["z_mri"],
                   _binary_cohort_labels(val_artifact["cohort"]),
               )}
    artifact_path = Path(validation_artifact_dir) / f"stage_s3_fold{fold_idx}{_mode_suffix(mode)}.npz"
    distill_artifact = write_validation_artifact(
        artifact_path,
        distill_logits=val_artifact["distill_logits"],
        prob_yd=val_artifact["prob_yd"],
        route_uncertainty=val_artifact["route_uncertainty"],
        centiloid_true=val_artifact["centiloid_true"],
        labels_cl24=val_artifact["labels_cl24"],
        labels_cl30=val_artifact["labels_cl30"],
        z_mri=val_artifact["z_mri"],
        cohort=val_artifact["cohort"],
        subject_id_hash=val_artifact["subject_id_hash"],
    )
    metrics["validation_artifact"] = distill_artifact
    combined = _try_write_calibration_input_artifact(
        fold_idx,
        mode=mode,
        validation_artifact_dir=validation_artifact_dir,
        distill_artifact_path=distill_artifact,
    )
    if combined:
        metrics["calibration_input_artifact"] = combined
    vram_blocker = _check_vram_budget(device, vram_budget_gb)
    if vram_blocker:
        metrics["vram_blocker"] = vram_blocker
    suffix = "_dryrun" if mode == "dry_run" else ""
    ckpt_path = str(Path(checkpoint_dir) / f"method_x_distilled_fold{fold_idx}{suffix}.pt")
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": student.state_dict(), "fold": fold_idx, "metrics": metrics,
                "model_card": student.model_card()}, ckpt_path)
    metrics["checkpoint"] = ckpt_path
    return metrics


def run_stage_s3(cfg: dict, *, mode: str, limit: int, epochs: int, teacher_auroc: float | None = None) -> dict[str, Any]:
    h5_path = cfg.get("inputs", {}).get("pet_h5", DEFAULT_METHOD_Y_PET_CACHE)
    pet_input_list_path = cfg.get("inputs", {}).get("pet_input_list", DEFAULT_PET_INPUT_LIST)
    teacher_ckpt = cfg.get("teacher", {}).get("joint_ckpt", "models/method_y/joint_fold0.pt")
    teacher_pattern = cfg.get("teacher", {}).get("joint_ckpt_pattern", "models/method_y/joint_fold{fold}.pt")
    profile = _compute_profile(cfg)
    mri_ckpt = cfg.get("inputs", {}).get("mri_ckpt", "models/method_x/best.pt")
    mri_h5 = cfg.get("inputs", {}).get("mri_h5", "data/cached/t1_combined_foldlocal_4000_128.h5")
    dependency_root = _dependency_report_root(cfg)
    s2_report = _load_stage_report("S2", dependency_root)
    if s2_report.get("acceptance_gate_passed") is not True:
        return {"fold_metrics": [], "acceptance_gate_passed": False,
                "folds_auroc_ge_0.95x_teacher": 0, "teacher_auroc": None,
                "vram_blocker": None, "blocker": "BLOCKED_S2_GATE_FAILED",
                "s2_report_status": s2_report.get("status"),
                "s2_blocker": s2_report.get("blocker") or "S2_NOT_PASSED",
                "config_diff_vs_canonical": compute_config_diff(cfg), "compute_profile": profile}
    s2_fold_metrics = _load_stage_fold_metrics("S2", dependency_root)
    if not s2_fold_metrics and teacher_auroc is None:
        return {"fold_metrics": [], "acceptance_gate_passed": False,
                "folds_auroc_ge_0.95x_teacher": 0, "teacher_auroc": None,
                "vram_blocker": None, "blocker": "BLOCKED_S2_METRICS_MISSING",
                "config_diff_vs_canonical": compute_config_diff(cfg), "compute_profile": profile}
    import h5py

    with h5py.File(h5_path, "r") as h5:
        subjects = list(h5.keys())
    if mode == "dry_run":
        subjects = select_balanced_dry_run_subjects(h5_path, subjects, limit=limit)
    strata_by_subject = _subject_strata_from_h5(h5_path, subjects, pet_input_list_path=pet_input_list_path)
    fold_cfg = cfg.get("folds", {}) or {}
    folds, fold_manifest = load_or_create_subject_folds(
        subjects,
        manifest_path=fold_cfg.get("manifest", DEFAULT_FOLD_MANIFEST),
        n_folds=int(fold_cfg.get("n_folds", 4)),
        seed=int(fold_cfg.get("seed", 42)),
        mode=mode,
        strata_by_subject=strata_by_subject,
    )
    validation_dir = cfg.get("outputs", {}).get("validation_dir", DEFAULT_VALIDATION_ARTIFACT_DIR)
    checkpoint_dir = cfg.get("outputs", {}).get("checkpoint_dir", "models/method_y")
    num_workers = int(profile.get("num_workers", 0) or 0)
    fold_metrics = []
    vram_blocked = False
    for k, fold in enumerate(folds):
        fold_teacher = _fold_checkpoint_path(teacher_pattern, k, mode=mode)
        if not Path(fold_teacher).exists():
            fold_teacher = teacher_ckpt
        if not Path(fold_teacher).exists():
            return {"fold_metrics": fold_metrics, "acceptance_gate_passed": False,
                    "folds_auroc_ge_0.95x_teacher": 0, "teacher_auroc": teacher_auroc,
                    "vram_blocker": None, "blocker": f"BLOCKED_S2_TEACHER_CHECKPOINT_MISSING:{fold_teacher}",
                    "config_diff_vs_canonical": compute_config_diff(cfg), "compute_profile": profile}
        try:
            m = train_distill_fold(k, fold["train"], fold["val"], h5_path, fold_teacher, epochs=epochs,
                                   batch_size=profile["batch_size"], grad_accum=profile["grad_accum"],
                                   mri_backbone=profile["mri_backbone"],
                                   gradient_checkpointing=profile["gradient_checkpointing"],
                                   vram_budget_gb=profile["vram_budget_gb"], mri_ckpt=mri_ckpt, mode=mode,
                                   mri_h5_path=mri_h5, loss_weights=cfg.get("losses", {}),
                                   validation_artifact_dir=validation_dir,
                                   checkpoint_dir=checkpoint_dir,
                                   pet_input_list_path=pet_input_list_path,
                                   num_workers=num_workers)
        except MethodYDataContractError as exc:
            return {"fold_metrics": fold_metrics, "acceptance_gate_passed": False,
                    "folds_auroc_ge_0.95x_teacher": 0, "teacher_auroc": teacher_auroc,
                    "vram_blocker": None, "blocker": str(exc),
                    "config_diff_vs_canonical": compute_config_diff(cfg), "compute_profile": profile}
        fold_metrics.append(m)
        if m.get("vram_blocker"):
            vram_blocked = True
            break
    aurocs = [m["auroc_cl24"] for m in fold_metrics if m["auroc_cl24"] is not None]
    teacher_aurocs = [float(m["auroc_cl24"]) for m in s2_fold_metrics if m.get("auroc_cl24") is not None]
    if teacher_auroc is not None:
        teacher_aurocs = [float(teacher_auroc)] * len(aurocs)
    ge = sum(
        1
        for idx, auc in enumerate(aurocs)
        if idx < len(teacher_aurocs) and auc >= (0.95 * teacher_aurocs[idx])
    )
    cohort_gate = cohort_probe_gate_summary(fold_metrics)
    gate = (not vram_blocked) and ge >= 3 and cohort_gate["passed"]
    return {"fold_metrics": fold_metrics, "acceptance_gate_passed": bool(gate),
            "folds_auroc_ge_0.95x_teacher": ge, "teacher_aurocs": teacher_aurocs,
            "max_cohort_probe_auc": cohort_gate["max_auc"],
            "cohort_probe_gate": cohort_gate,
            "fold_manifest": fold_manifest,
            "vram_blocker": next((m.get("vram_blocker") for m in fold_metrics if m.get("vram_blocker")), None),
            "config_diff_vs_canonical": compute_config_diff(cfg),
            "compute_profile": profile}
