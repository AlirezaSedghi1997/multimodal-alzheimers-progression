#!/usr/bin/env python3
"""Train Method X 3D MRI + tabular fusion with optional SynIB and latent alignment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import signal
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_baseline import average_precision, brier, roc_auc
from scripts.train_pet_method_y import resolve_training_device
from src.utils.dataloader_config import resolve_dataloader_kwargs
from src.training.method_x_resume import (
    assert_resume_contract,
    build_runtime_resume_contract,
    build_run_contract,
    build_locked_outer_result,
    outer_evaluation_resume_action,
    protocol_epochs_per_fold,
    read_fold_state,
    reuse_committed_outer_result,
    sha256_path,
    legacy_source_metadata_normalization,
    scientific_contract_digest,
    source_contract_migration_allowed,
    source_history_binds_digest,
    storage_policy_digest,
    transition_fold_state,
    validate_locked_outer_result,
)
from src.training.checkpoint_staging import (
    RESUMABLE_EXIT_CODE,
    CheckpointStager,
    add_staging_arguments,
    build_stager_from_args,
    rng_state_digest,
)
from src.training.method_x_finetune import (
    apply_prespecified_unfreeze_schedule,
    encoder_stage_map,
    load_fold_local_encoder_weights,
    validate_pretrained_fold_manifest,
)
from src.data.volume_augmentation import build_augmentation_policy, maybe_wrap_train_dataset
from src.evaluation.calibration import (
    apply_temperature_to_logits,
    calibration_analysis,
    calibration_metric_summary,
    fit_temperature_scaling,
    reliability_diagram_data,
    sigmoid_logits,
)
from src.evaluation.threshold_metrics import evaluate_with_train_selected_threshold
from src.evaluation.tta import TTAPredictor
from src.evaluation.cohort_probe import train_cohort_probe
from src.models.backbones.attention import AVAILABLE_ATTENTION
from src.models.backbones.backbone_factory import available_3d_backbones
from src.models.backbones.lightweight_3d import MethodX3DClassifier
from src.models.losses.latent_alignment import latent_alignment_loss
from src.models.losses.contrastive_alignment import ContrastiveAlignmentLoss, SupervisedContrastiveAlignmentLoss
from src.models.losses.phenotypic_contrastive import phenotypic_contrastive_loss
from src.models.losses.phase1_losses import SUPPORTED_TASK_LOSSES, compute_phase1_task_loss
from src.models.losses.synib_synergy_loss import SynIBConfig, synib_multimodal_loss
from src.models.fusion.registry import available_fusion_modes
from src.utils.final_claim_policy import (
    claim_from_three_d_aggregate,
    evaluate_three_d_final_claim,
    spm_progress,
    validate_claim,
)
from src.utils.reproducibility import (
    canonical_digest,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    seed_worker,
    seeded_torch_generator,
    split_membership_digest,
)
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest
from src.data.combined_mri_cache import (
    cache_has_supplementary,
    merge_train_with_supplementary,
    read_cache_sources,
    read_source_sample_hashes,
    read_source_folds,
    read_source_train_only,
    phenotype_features_from_tabular,
    METHOD_X_PHENOTYPE_FEATURES,
    CACHE_TABULAR_FEATURES,
    supplementary_train_indices,
    validate_phenotype_feature_names,
)
from src.utils.reproducibility import source_tree_digest, runtime_version_contract


# Runtime-only knobs are deliberately kept out of the scientific training
# contract.  The distinction is important for the live battery: moving a
# checkpoint between the H authority and the C staging volume must not make an
# otherwise identical experiment look scientifically different.  Keep this as
# a public, immutable tuple so audit tools can inspect it without importing the
# argument parser.
_NON_TRAINING_CONTRACT_ARGS = frozenset(
    {
        "checkpoint_staging_root",
        "checkpoint_full_cadence",
        "checkpoint_h_mirror",
        "resume_replay_policy",
        "allow_tf32",
        "torch_compile",
        "storage_policy",
        "throughput_profile_only",
    }
)

# The first Method X battery was issued by a trainer revision that bound the
# complete argparse scientific surface (not only the compact model signature)
# into ``parameters``.  Keeping this allow-list explicit is important: using
# ``vars(args)`` would accidentally bind output paths, resume controls, or the
# C staging location, while using ``requested_model_signature`` would strand
# the already-issued folds.  These names are the stable scientific/runtime
# provenance fields present in the v3 battery contract.
_METHOD_X_CONTRACT_PARAMETER_KEYS = (
    "alignment_head",
    "alignment_head_mode",
    "amp",
    "attention",
    "auc_loss_weight",
    "auc_margin",
    "aug_prob_scale",
    "augmentation_exposures_per_real",
    "backbone",
    "bottleneck_dim",
    "causal_adjustment",
    "causal_intervention_enabled",
    "checkpoint",
    "contrastive_alignment",
    "contrastive_temperature",
    "device",
    "discriminative_lr",
    "drop_tabular_feature",
    "dropout",
    "early_stopping_patience",
    "ema_decay",
    "epochs",
    "finetune_lr",
    "focal_alpha",
    "focal_gamma",
    "folds",
    "freeze_encoder",
    "freeze_until",
    "fusion",
    "fusion_hidden_dim",
    "fusion_num_heads",
    "grad_accum",
    "gradual_unfreeze",
    "hpo_selection_only",
    "hybrid_attention_heads",
    "hybrid_attention_window_size",
    "hybrid_variant",
    "include_global_supplementary_train",
    "input_mode",
    "label_smoothing",
    "latent_alignment_lambda",
    "latent_dim",
    "layerwise_lr_decay",
    "limit",
    "lr",
    "modality_dropout",
    "moe_load_balance_lambda",
    "monai_augment",
    "n_experts",
    "no_supplementary_train",
    "phase",
    "phenotypic_contrastive_alignment",
    "phenotypic_contrastive_lambda",
    "phenotypic_temperature",
    "pos_weight",
    "pretrained_checkpoint",
    "pretrained_fold_manifest",
    "pretrained_manifest",
    "pretrained_min_coverage",
    "pretrained_sha256",
    "q2_channel_gate",
    "require_gpu",
    "require_pretrained_provenance",
    "resolution",
    "restore_best_at_end",
    "sanity_random_labels",
    "sanity_random_noise_images",
    "scheduler",
    "seed",
    "step_gamma",
    "step_size",
    "supervised_contrastive_alignment",
    "swa",
    "swa_start_fraction",
    "synib_lambda",
    "tab_encoder",
    "tab_hidden_dim",
    "task_loss",
    "tta",
    "tta_augmentations",
    "tta_mode",
    "unfreeze_epoch",
    "unfreeze_schedule",
    "val_fraction",
    "warmup_epochs",
    "wd_exclude_bias_norm",
)


def build_method_x_contract_parameters(
    args: Any,
    *,
    source_contract_digest: str,
    runtime_versions: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete, path-free v3 training contract parameter map.

    Missing fields are filled with the values used by the issued v3 battery.
    This makes the builder compatible with older parser revisions while still
    rejecting scientific changes through the normal contract digest gate.
    ``source_contract_digest`` and ``runtime_versions`` are provenance fields;
    they are intentionally retained in the body so an audited source hop can
    be represented without mutating the scientific projection.
    """

    values = vars(args) if hasattr(args, "__dict__") else {}
    defaults: dict[str, Any] = {
        "alignment_head": "linear",
        # The issued v3 contract used null for the default linear mode.  The
        # model still receives the normalized ``linear`` parser value; this is
        # only a compatibility representation in the contract.
        "alignment_head_mode": None,
        "amp": False,
        "attention": "none",
        "auc_loss_weight": 1.0,
        "auc_margin": 1.0,
        "aug_prob_scale": 0.2,
        "augmentation_exposures_per_real": 4,
        "backbone": "resnet18_3d",
        "bottleneck_dim": 32,
        "causal_adjustment": "none",
        "causal_intervention_enabled": False,
        "checkpoint": False,
        "contrastive_alignment": False,
        "contrastive_temperature": 0.07,
        "device": "auto",
        "discriminative_lr": False,
        "drop_tabular_feature": "none",
        "dropout": 0.1,
        "early_stopping_patience": 5,
        "ema_decay": 0.0,
        "epochs": 100,
        "finetune_lr": 5e-5,
        "focal_alpha": -1.0,
        "focal_gamma": 2.0,
        "folds": 4,
        "freeze_encoder": False,
        "freeze_until": "",
        "fusion": "cross_attention",
        "fusion_hidden_dim": None,
        "fusion_num_heads": 2,
        "grad_accum": 1,
        "gradual_unfreeze": False,
        "hpo_selection_only": False,
        "hybrid_attention_heads": 2,
        "hybrid_attention_window_size": 4,
        "hybrid_variant": "baseline",
        "include_global_supplementary_train": False,
        "input_mode": "mri_plus_tabular",
        "label_smoothing": 0.0,
        "latent_alignment_lambda": 0.0,
        "latent_dim": 128,
        "layerwise_lr_decay": 1.0,
        "limit": 0,
        "lr": 1e-4,
        "modality_dropout": 0.5,
        "moe_load_balance_lambda": 0.0,
        "monai_augment": False,
        "n_experts": 4,
        "no_supplementary_train": False,
        "phase": "dev",
        "phenotypic_contrastive_alignment": False,
        "phenotypic_contrastive_lambda": 0.0,
        "phenotypic_temperature": 0.1,
        "pos_weight": "auto",
        "pretrained_checkpoint": "",
        "pretrained_fold_manifest": "",
        "pretrained_manifest": "",
        "pretrained_min_coverage": 0.95,
        "pretrained_sha256": "",
        "q2_channel_gate": "none",
        "require_gpu": False,
        "require_pretrained_provenance": False,
        "resolution": 128,
        "restore_best_at_end": False,
        "sanity_random_labels": False,
        "sanity_random_noise_images": False,
        "scheduler": "cosine",
        "seed": 13,
        "step_gamma": 0.5,
        "step_size": 10,
        "supervised_contrastive_alignment": False,
        "swa": False,
        "swa_start_fraction": 0.8,
        "synib_lambda": 0.0,
        "tab_encoder": "linear",
        "tab_hidden_dim": 32,
        "task_loss": "bce",
        "tta": False,
        "tta_augmentations": 5,
        "tta_mode": "basic",
        "unfreeze_epoch": 8,
        "unfreeze_schedule": "none",
        "val_fraction": 0.15,
        "warmup_epochs": 5,
        "wd_exclude_bias_norm": True,
    }
    result: dict[str, Any] = {}
    for key in _METHOD_X_CONTRACT_PARAMETER_KEYS:
        value = values.get(key, defaults[key])
        # ``alignment_head_mode=linear`` is the normalized parser default; the
        # issued contract encoded that same behavior as null.  Preserve
        # explicit nonlinear mode, which is scientifically distinct.
        if key == "alignment_head_mode" and value == "linear":
            value = None
        result[key] = value
    result["source_contract_digest"] = str(source_contract_digest)
    result["runtime_versions"] = dict(runtime_versions)
    return result


class SafeOptimizerBoundaryStop(RuntimeError):
    """Controlled stop raised only after a committed optimizer boundary."""


def _handle_training_signal(
    signum: int,
    frame: Any,
    training_stop_request: dict[str, Any],
) -> None:
    """Record a stop request without doing I/O from an async signal handler."""

    del frame
    training_stop_request["requested"] = True
    training_stop_request["signal"] = int(signum)


def _install_training_signal_handlers(
    training_stop_request: dict[str, Any],
) -> tuple[Any, Any]:
    """Install deferred SIGINT/SIGTERM handlers and return prior handlers."""

    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, lambda signum, frame: _handle_training_signal(signum, frame, training_stop_request))
    signal.signal(signal.SIGTERM, lambda signum, frame: _handle_training_signal(signum, frame, training_stop_request))
    return previous_sigint_handler, previous_sigterm_handler


def reject_method_x_nondeterministic_acceleration(args: Any) -> None:
    """Fail closed on opt-in accelerators that can change the bitwise path.

    TF32 and ``torch.compile`` can be useful for exploratory profiling, but
    they are not part of the locked Method X reproducibility protocol.  They
    therefore require a separate, versioned experiment rather than silently
    changing an existing run.
    """

    if bool(getattr(args, "allow_tf32", False)):
        raise SystemExit(79)
    if bool(getattr(args, "torch_compile", False)):
        raise SystemExit(79)


def enforce_method_x_strict_determinism() -> dict[str, Any]:
    """Lock CUDA attention to deterministic kernels for Method X.

    ``nn.MultiheadAttention`` may otherwise select the memory-efficient SDP
    backward kernel, which PyTorch explicitly reports as non-deterministic.
    A warning-only policy is insufficient for exact resume or same-seed model
    selection, so MX-04 and later runs use strict algorithms and the math SDP
    backend.  Flash/memory-efficient kernels remain available to other entry
    points because this function is called only by this trainer.
    """

    import torch

    torch.use_deterministic_algorithms(True, warn_only=False)
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is not None:
        if hasattr(cuda_backend, "enable_flash_sdp"):
            cuda_backend.enable_flash_sdp(False)
        if hasattr(cuda_backend, "enable_mem_efficient_sdp"):
            cuda_backend.enable_mem_efficient_sdp(False)
        if hasattr(cuda_backend, "enable_math_sdp"):
            cuda_backend.enable_math_sdp(True)
    return {
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "flash_sdp_enabled": bool(cuda_backend.flash_sdp_enabled())
        if cuda_backend is not None and hasattr(cuda_backend, "flash_sdp_enabled")
        else None,
        "mem_efficient_sdp_enabled": bool(cuda_backend.mem_efficient_sdp_enabled())
        if cuda_backend is not None and hasattr(cuda_backend, "mem_efficient_sdp_enabled")
        else None,
        "math_sdp_enabled": bool(cuda_backend.math_sdp_enabled())
        if cuda_backend is not None and hasattr(cuda_backend, "math_sdp_enabled")
        else None,
    }


def training_contract_parameters(args: Any) -> dict[str, Any]:
    """Return deterministic scientific parameters while dropping runtime flags."""

    values = vars(args) if hasattr(args, "__dict__") else {}
    return {
        str(key): value
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
        if str(key) not in _NON_TRAINING_CONTRACT_ARGS
        and not str(key).startswith("_")
    }


def verify_loaded_staged_recovery(
    staged_recovery_plan: Any,
    progress: Mapping[str, Any] | None,
) -> bool:
    """Verify a staged payload after its model/runtime state is restored.

    A cold start has no replay evidence and returns ``False``.  For a staged
    payload the journal's accumulator and RNG digest are checked through the
    backend's fail-closed verifier.  The helper is intentionally small so it
    can be exercised independently by acceptance tests.
    """

    if staged_recovery_plan is None or getattr(staged_recovery_plan, "payload_path", None) is None:
        return False
    payload = progress or {}
    staged_recovery_plan.verify_replay(
        rng_digest=rng_state_digest(),
        loss_sum=float(payload.get("epoch_loss_sum", payload.get("loss_sum", 0.0))),
        loss_count=int(payload.get("epoch_loss_count", payload.get("loss_count", 0))),
    )
    return True


def seed_offloaded_best_state_sidecar(
    checkpoint_stager: CheckpointStager | None,
    *,
    best_model_state: Mapping[str, Any] | None,
    best_averaged_state: Mapping[str, Any] | None,
    best_averaged_source: str | None,
    best_epoch: int | None,
    best_val: float,
) -> dict[str, Any] | None:
    """Persist the validation-best weights on H exactly once per improvement."""

    if checkpoint_stager is None or best_model_state is None:
        return None
    import math

    if best_epoch is None or not math.isfinite(float(best_val)):
        return None
    state: dict[str, Any] = {
        "best_model_state_dict": dict(best_model_state),
        "best_averaged_state_dict": (
            dict(best_averaged_state) if best_averaged_state is not None else None
        ),
        "best_averaged_source": best_averaged_source,
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
    }
    return checkpoint_stager.record_best_state(state, metric=float(best_val))


def persist_resume_boundary(
    *,
    checkpoint_stager: CheckpointStager | None,
    payload_factory: Any,
    progress_path: Path,
    optimizer_step: int,
    epoch: int,
    epoch_batch_cursor: int,
    sampler_cursor: int,
    loss_sum: float,
    loss_count: int,
    reason: str,
) -> dict[str, Any]:
    """Commit a complete boundary through staging, or atomically on H legacy."""

    if checkpoint_stager is not None:
        result = checkpoint_stager.commit_boundary(
            payload_factory=payload_factory,
            optimizer_step=int(optimizer_step),
            epoch=int(epoch),
            epoch_batch_cursor=int(epoch_batch_cursor),
            sampler_cursor=int(sampler_cursor),
            loss_sum=float(loss_sum),
            loss_count=int(loss_count),
            rng_digest=rng_state_digest(),
            reason=str(reason),
        )
        return result
    atomic_torch_save(payload_factory(), progress_path)
    return {"status": "LEGACY_H_PROGRESS_COMMITTED", "path": str(progress_path)}


def method_x_training_source_paths() -> list[Path]:
    scoped_roots = [
        PROJECT_ROOT / "scripts/train_3d_mri.py",
        PROJECT_ROOT / "scripts/train_baseline.py",
        PROJECT_ROOT / "src/data/combined_mri_cache.py",
        PROJECT_ROOT / "src/data/volume_augmentation.py",
        PROJECT_ROOT / "src/evaluation/calibration.py",
        PROJECT_ROOT / "src/evaluation/cohort_probe.py",
        PROJECT_ROOT / "src/evaluation/threshold_metrics.py",
        PROJECT_ROOT / "src/evaluation/tta.py",
        PROJECT_ROOT / "src/models/__init__.py",
        PROJECT_ROOT / "src/models/backbones/__init__.py",
        PROJECT_ROOT / "src/models/backbones/amuse_x_hybrid3d.py",
        PROJECT_ROOT / "src/models/backbones/attention.py",
        PROJECT_ROOT / "src/models/backbones/backbone_factory.py",
        PROJECT_ROOT / "src/models/backbones/deterministic_pool.py",
        PROJECT_ROOT / "src/models/backbones/lightweight_3d.py",
        PROJECT_ROOT / "src/models/backbones/modern_3d.py",
        PROJECT_ROOT / "src/models/backbones/pretrained_loader.py",
        PROJECT_ROOT / "src/models/causal_intervention_block.py",
        PROJECT_ROOT / "src/models/fusion/__init__.py",
        PROJECT_ROOT / "src/models/fusion/amplitude_phase_fusion.py",
        PROJECT_ROOT / "src/models/fusion/bottleneck_cross_attention.py",
        PROJECT_ROOT / "src/models/fusion/channel_gate.py",
        PROJECT_ROOT / "src/models/fusion/conv_feature_fusion.py",
        PROJECT_ROOT / "src/models/fusion/healnet_fusion.py",
        PROJECT_ROOT / "src/models/fusion/moe_fusion.py",
        PROJECT_ROOT / "src/models/fusion/registry.py",
        PROJECT_ROOT / "src/models/losses/__init__.py",
        PROJECT_ROOT / "src/models/losses/contrastive_alignment.py",
        PROJECT_ROOT / "src/models/losses/distillation.py",
        PROJECT_ROOT / "src/models/losses/evidential.py",
        PROJECT_ROOT / "src/models/losses/latent_alignment.py",
        PROJECT_ROOT / "src/models/losses/phase1_losses.py",
        PROJECT_ROOT / "src/models/losses/phenotypic_contrastive.py",
        PROJECT_ROOT / "src/models/losses/synib_synergy_loss.py",
        PROJECT_ROOT / "src/training/__init__.py",
        PROJECT_ROOT / "src/training/method_x_autosupervisor.py",
        PROJECT_ROOT / "src/training/method_x_loop_guard.py",
        PROJECT_ROOT / "src/training/method_x_outer_compatibility.py",
        PROJECT_ROOT / "src/training/method_x_registry.py",
        PROJECT_ROOT / "src/training/method_x_registry_lifecycle.py",
        PROJECT_ROOT / "src/training/method_x_repair_gate.py",
        PROJECT_ROOT / "src/training/method_x_resume.py",
        PROJECT_ROOT / "src/training/method_x_source_compatibility.py",
        PROJECT_ROOT / "src/training/method_x_spm_snapshot.py",
        PROJECT_ROOT / "src/training/method_x_stage_maps.py",
        PROJECT_ROOT / "src/training/method_x_stage_registry.py",
        PROJECT_ROOT / "src/utils/dataloader_config.py",
        PROJECT_ROOT / "src/utils/final_claim_policy.py",
        PROJECT_ROOT / "src/utils/io_helpers.py",
        PROJECT_ROOT / "src/utils/privacy.py",
        PROJECT_ROOT / "src/utils/reproducibility.py",
        PROJECT_ROOT / "src/utils/run_manifest.py",
    ]
    scoped_files: list[Path] = []
    for root in scoped_roots:
        if root.is_file():
            scoped_files.append(root)
        elif root.is_dir():
            scoped_files.extend(sorted(root.rglob("*.py")))
    scoped_files.extend(sorted((PROJECT_ROOT / "src/training").glob("method_x_*.py")))
    return sorted({p for p in scoped_files if p.is_file()})


def method_x_training_source_contract() -> tuple[str, dict[str, str | None]]:
    training_sources = method_x_training_source_paths()
    return (
        source_tree_digest(training_sources, base=PROJECT_ROOT),
        runtime_version_contract(),
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_sample_hashes(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [row["sample_hash"] for row in read_rows(path) if row.get("sample_hash")]


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


ALLOWED_METHOD_X_INPUT_MODES: tuple[str, ...] = ("mri_plus_tabular", "mri_only", "tabular_only")
PET_LIKE_TOKENS: tuple[str, ...] = (
    "pet",
    "synthetic_pet",
    "suvr",
    "centiloid",
    "amyloid_pet",
    "pet_embedding",
    "florbetapir",
    "av45",
)


def _normalized_token_text(value: object) -> str:
    return decode(value).lower().replace("-", "_").replace(" ", "_")


def _contains_forbidden_pet_token(value: object) -> bool:
    text = _normalized_token_text(value)
    return any(token in text for token in PET_LIKE_TOKENS)


def validate_method_x_input_policy(args, cache_path: Path) -> dict[str, object]:
    """Fail fast if Method X is configured with PET-like inputs or features."""

    input_mode = str(getattr(args, "input_mode", "mri_plus_tabular")).lower().strip()
    if input_mode not in ALLOWED_METHOD_X_INPUT_MODES:
        reason = "PET-like input modes are forbidden for Method X" if _contains_forbidden_pet_token(input_mode) else "Unsupported Method X input mode"
        raise SystemExit(f"{reason}: {input_mode}. Allowed modes: {', '.join(ALLOWED_METHOD_X_INPUT_MODES)}")

    import h5py

    forbidden_fields: list[str] = []
    feature_name_keys = {"feature_names", "tabular_feature_names", "tabular_columns", "columns"}
    with h5py.File(cache_path, "r") as h5:
        for key in h5.keys():
            if _contains_forbidden_pet_token(key):
                forbidden_fields.append(f"dataset:{key}")
        for key, value in h5.attrs.items():
            key_text = _normalized_token_text(key)
            if key_text in {"contains_pet_input", "contains_synthetic_pet_input", "uses_pet_input"} and bool(value):
                forbidden_fields.append(f"attr:{key}=true")
            elif _contains_forbidden_pet_token(value) and key_text not in {"method"}:
                forbidden_fields.append(f"attr:{key}")
        for key in feature_name_keys.intersection(set(h5.keys())):
            values = h5[key][:]
            for value in values:
                if _contains_forbidden_pet_token(value):
                    forbidden_fields.append(f"{key}:{decode(value)}")
    if forbidden_fields:
        raise SystemExit("Method X input policy rejected PET-like cache/tabular fields: " + ", ".join(forbidden_fields))
    return {
        "status": "PASSED",
        "input_mode": input_mode,
        "allowed_modes": list(ALLOWED_METHOD_X_INPUT_MODES),
        "method_x_pet_free": True,
        "pet_like_cache_fields_rejected": True,
    }


def method_x_split_contract_digest(split_dir: Path, folds: int | None = None, selection_only: bool = False) -> str:
    """Hash-bound digest of the locked outer split CSVs (order-stable).

    Legacy form ``(split_dir)`` digests every top-level ``*.csv`` and is
    byte-identical to all historical run contracts — it must never change.
    The scoped form selects ``fold_{i}_{train,test}.csv`` for
    ``i in range(folds)`` (``selection_only=True`` keeps train files only)
    for HPO-selection and comparison contexts.
    """

    import hashlib

    digest = hashlib.sha256()
    base = Path(split_dir)
    if folds is None and not selection_only:
        paths = sorted(base.glob("*.csv"))
    else:
        if folds is None or int(folds) < 1:
            raise ValueError("method_x_split_contract_digest requires folds>=1 for scoped digests")
        names = [
            f"fold_{fold}_{part}.csv"
            for fold in range(int(folds))
            for part in (("train",) if selection_only else ("train", "test"))
        ]
        paths = [base / name for name in sorted(names)]
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"SPLIT_CONTRACT_SCOPED_FILES_MISSING:{','.join(missing)}")
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_method_x_training_device(requested: str, *, require_gpu: bool):
    """Resolve the shared device policy and enforce Method X's final-GPU gate."""

    device = resolve_training_device(requested)
    if require_gpu and str(getattr(device, "type", "")) != "cuda":
        raise RuntimeError("METHOD_X_FINAL_TRAINING_REQUIRES_CUDA")
    return device


def migrate_source_only_resume_artifacts(*args, **kwargs):
    """Delegate to the shared source-only migration authority (monkeypatchable)."""

    from src.training.method_x_resume import (
        migrate_source_only_resume_artifacts as _impl,
    )

    return _impl(*args, **kwargs)


def cache_index(cache_path: Path) -> dict[str, int]:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        return {decode(v): i for i, v in enumerate(h5["sample_hash"][:])}


def ok_indices(cache_path: Path) -> set[int]:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        return {i for i, v in enumerate(h5["status"][:]) if decode(v) == "OK"}


def fit_source_counts(cache_path: Path, indices: tuple[int, ...] | list[int]) -> dict[str, int]:
    """Return source-aware counts for one fold's actual fit indices.

    ``train_n`` alone cannot distinguish real parents from materialized
    derivatives.  Keeping this decomposition at fold level prevents the
    aggregate report from labelling all fit exposures as real or augmented.
    """

    import h5py

    with h5py.File(cache_path, "r") as h5:
        hashes = [decode(v) for v in h5["sample_hash"][:]]
        statuses = [decode(v) for v in h5["status"][:]]
        sources = read_cache_sources(h5)
    supplementary = set(supplementary_train_indices(hashes, statuses, sources))
    augmented = sum(1 for index in indices if int(index) in supplementary)
    real = len(indices) - augmented
    return {
        "real_fit_n": int(real),
        "augmented_fit_n": int(augmented),
        "fit_exposure_n": int(len(indices)),
    }


class H5MethodXDataset:
    def __init__(
        self,
        cache_path: Path,
        indices: list[int],
        labels: list[int],
        *,
        sanity_random_labels: bool = False,
        sanity_random_noise_images: bool = False,
        return_phenotype_features: bool = False,
        input_mode: str = "mri_plus_tabular",
        drop_tabular_feature: str = "none",
        seed: int = 13,
    ) -> None:
        self.cache_path = cache_path
        self.indices = indices
        self.labels = list(labels)
        self.sanity_random_noise_images = bool(sanity_random_noise_images)
        self.return_phenotype_features = bool(return_phenotype_features)
        self.input_mode = str(input_mode).lower().strip()
        if self.input_mode not in ALLOWED_METHOD_X_INPUT_MODES:
            raise ValueError(f"Unsupported Method X input_mode: {input_mode}")
        drop = str(drop_tabular_feature or "none").lower().strip()
        if drop != "none" and drop not in METHOD_X_PHENOTYPE_FEATURES:
            raise ValueError(f"Unsupported Method X drop_tabular_feature: {drop_tabular_feature}")
        self.drop_tabular_index = -1 if drop == "none" else CACHE_TABULAR_FEATURES.index(drop)
        self.seed = int(seed)
        if sanity_random_labels:
            rng = random.Random(seed + len(indices))
            self.labels = [rng.randint(0, 1) for _ in self.labels]
        self._h5 = None

    def __len__(self) -> int:
        return len(self.indices)

    def _open(self):
        if self._h5 is None:
            import h5py

            self._h5 = h5py.File(self.cache_path, "r")
        return self._h5

    def __getitem__(self, item: int):
        import torch

        h5 = self._open()
        idx = self.indices[item]
        image = torch.from_numpy(h5["image"][idx].astype("float32"))
        if self.sanity_random_noise_images:
            generator = torch.Generator()
            generator.manual_seed(self.seed + int(idx))
            image = torch.randn(image.shape, dtype=image.dtype, generator=generator)
        tabular = torch.from_numpy(h5["tabular"][idx].astype("float32"))
        tab_presence = torch.tensor(1.0, dtype=torch.float32)
        if self.input_mode == "mri_only":
            tabular = torch.zeros_like(tabular)
            tab_presence = torch.tensor(0.0, dtype=torch.float32)
        if self.input_mode == "tabular_only":
            # Mirror of mri_only: the image is a constant zero volume, so all
            # learnable signal must come from the tabular branch.
            image = torch.zeros_like(image)
        if self.drop_tabular_index >= 0:
            # Feature-drop ablation: remove one canonical tabular column fully
            # (before phenotype derivation), leaving a constant that carries
            # no information. Normalization stats stay fold-local and full.
            tabular = tabular.clone()
            tabular[..., self.drop_tabular_index] = 0.0
        label = torch.tensor(float(self.labels[item]), dtype=torch.float32)
        if self.return_phenotype_features:
            phenotype = phenotype_features_from_tabular(tabular).to(torch.float32)
            return image, tabular, label, tab_presence, phenotype
        return image, tabular, label, tab_presence


def split_indices(
    split_path: Path,
    sample_to_idx: dict[str, int],
    ok: set[int],
    limit: int = 0,
) -> tuple[list[int], list[int]]:
    indices: list[int] = []
    labels: list[int] = []
    for row in read_rows(split_path):
        sample_hash = row.get("sample_hash", "")
        if sample_hash not in sample_to_idx:
            continue
        idx = sample_to_idx[sample_hash]
        if idx not in ok:
            continue
        indices.append(idx)
        labels.append(int(float(row["ab_label"])))
        if limit > 0 and len(indices) >= limit:
            break
    return indices, labels


def cohort_labels_for_indices(split_path: Path, sample_to_idx: dict[str, int], indices: list[int] | tuple[int, ...]) -> list[int]:
    lookup: dict[int, int] = {}
    for row in read_rows(split_path):
        sample_hash = row.get("sample_hash", "")
        if sample_hash not in sample_to_idx:
            continue
        cohort = str(row.get("substudy", "")).upper()
        if cohort == "A4":
            lookup[sample_to_idx[sample_hash]] = 1
        elif cohort == "LEARN":
            lookup[sample_to_idx[sample_hash]] = 0
    labels: list[int] = []
    for idx in indices:
        if int(idx) in lookup:
            labels.append(int(lookup[int(idx)]))
    return labels


def fold_from_split_path(split_path: Path) -> int | None:
    parts = split_path.stem.split("_")
    if len(parts) >= 3 and parts[0] == "fold":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def supplementary_context(cache_path: Path, ok: set[int], fold: int | None = None) -> tuple[list[int], list[int]]:
    return supplementary_context_with_policy(cache_path, ok, fold=fold, include_global=False)


def supplementary_context_with_policy(
    cache_path: Path,
    ok: set[int],
    fold: int | None = None,
    *,
    include_global: bool = False,
    allowed_source_hashes: set[str] | None = None,
) -> tuple[list[int], list[int]]:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        hashes = [decode(v) for v in h5["sample_hash"][:]]
        statuses = [decode(v) for v in h5["status"][:]]
        labels = [int(v) for v in h5["label"][:]]
        sources = read_cache_sources(h5)
        source_hashes = read_source_sample_hashes(h5)
        source_folds = read_source_folds(h5)
        source_train_only = read_source_train_only(h5)
    supp_idx = supplementary_train_indices(
        hashes,
        statuses,
        sources,
        source_folds=source_folds,
        source_train_only=source_train_only,
        fold=fold,
    )
    if fold is not None and include_global:
        global_idx = supplementary_train_indices(
            hashes,
            statuses,
            sources,
        )
        supp_idx = sorted(set(supp_idx).union(global_idx))
    if allowed_source_hashes is not None:
        # A materialized row is usable only when its parent is a current real
        # outer-train sample.  This prevents stale caches from importing
        # augmentations derived from a current test subject.
        if source_hashes is None:
            supp_idx = []
        else:
            supp_idx = [
                idx
                for idx in supp_idx
                if idx < len(source_hashes) and source_hashes[idx] in allowed_source_hashes
            ]
    supp_idx = [idx for idx in supp_idx if idx in ok]
    supp_labels = [labels[idx] for idx in supp_idx]
    return supp_idx, supp_labels


def merge_train_indices(
    real_indices: list[int],
    real_labels: list[int],
    supplementary_indices: list[int],
    supplementary_labels: list[int],
) -> tuple[list[int], list[int]]:
    merged_idx = merge_train_with_supplementary(real_indices, supplementary_indices)
    label_by_idx = dict(zip(real_indices, real_labels, strict=False))
    label_by_idx.update(dict(zip(supplementary_indices, supplementary_labels, strict=False)))
    merged_labels = [int(label_by_idx[idx]) for idx in merged_idx]
    return merged_idx, merged_labels


def supplementary_index_set(cache_path: Path, ok: set[int]) -> set[int]:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        hashes = [decode(v) for v in h5["sample_hash"][:]]
        statuses = [decode(v) for v in h5["status"][:]]
        sources = read_cache_sources(h5)
    supp_idx = supplementary_train_indices(hashes, statuses, sources)
    return {idx for idx in supp_idx if idx in ok}


def cache_sample_hashes(cache_path: Path) -> list[str]:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        return [decode(value) for value in h5["sample_hash"][:]]


def source_sample_hashes(cache_path: Path) -> list[str] | None:
    import h5py

    with h5py.File(cache_path, "r") as h5:
        return read_source_sample_hashes(h5)


def split_fit_val_indices(
    train_idx: list[int],
    train_y: list[int],
    *,
    cache_path: Path,
    ok: set[int],
    val_fraction: float,
    seed: int,
    fold: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    supplementary = supplementary_index_set(cache_path, ok)
    real_pairs = [(idx, y) for idx, y in zip(train_idx, train_y, strict=True) if idx not in supplementary]
    supplementary_pairs = [(idx, y) for idx, y in zip(train_idx, train_y, strict=True) if idx in supplementary]
    if len(real_pairs) < 2:
        raise RuntimeError("At least two real rows are required for a real-only inner validation split.")
    random.Random(seed + fold).shuffle(real_pairs)
    val_n = max(int(len(real_pairs) * val_fraction), 1)
    val_n = min(val_n, len(real_pairs) - 1)
    val_pairs = real_pairs[:val_n]
    all_hashes = cache_sample_hashes(cache_path)
    validation_source_hashes = {all_hashes[idx] for idx, _ in val_pairs}
    parent_hashes = source_sample_hashes(cache_path)
    # No derivative of an inner-validation image may stay in fit: otherwise
    # early stopping and hyperparameter selection see the held-out anatomy.
    if parent_hashes is None:
        supplementary_pairs = []
    else:
        supplementary_pairs = [
            (idx, label)
            for idx, label in supplementary_pairs
            if idx < len(parent_hashes)
            and parent_hashes[idx]
            and parent_hashes[idx] not in validation_source_hashes
        ]
    fit_pairs = real_pairs[val_n:] + supplementary_pairs
    fit_idx, fit_y = zip(*fit_pairs)
    val_idx, val_y = zip(*val_pairs)
    return (
        tuple(int(v) for v in fit_idx),
        tuple(int(v) for v in fit_y),
        tuple(int(v) for v in val_idx),
        tuple(int(v) for v in val_y),
    )


def inspect_exact_run_contract(
    out_dir: Path,
    *,
    run_contract: dict[str, Any],
    require_exact_resume_contract: bool,
) -> dict[str, Any]:
    """Read-only contract decision used by preflight and the write path.

    Scientific drift raises the original mismatch.  A source-only difference
    is reported without updating the contract, receipt, state, or checkpoint.
    """

    run_contract_path = out_dir / "run_contract.json"
    if not run_contract_path.exists():
        return {
            "status": "CREATE_REQUIRED",
            "run_contract_digest": run_contract["digest"],
        }
    if run_contract_path.exists() and require_exact_resume_contract:
        existing_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
        try:
            assert_resume_contract(existing_contract, run_contract)
        except RuntimeError:
            legacy = legacy_source_metadata_normalization(existing_contract, run_contract)
            if legacy is not None:
                return {
                    "status": "LEGACY_SOURCE_METADATA_NORMALIZATION_REQUIRED",
                    "previous_digest": existing_contract.get("run_contract_digest"),
                    "new_digest": run_contract["digest"],
                    "source_digest": legacy.get("source_digest_metadata"),
                }
            if not source_contract_migration_allowed(existing_contract, run_contract):
                raise
            return {
                "status": "SOURCE_ONLY_MIGRATION_REQUIRED",
                "previous_digest": existing_contract.get("run_contract_digest"),
                "new_digest": run_contract["digest"],
            }
        return {
            "status": "EXACT_MATCH",
            "run_contract_digest": run_contract["digest"],
        }
    if run_contract_path.exists():
        return {"status": "PRESENT_NOT_ENFORCED"}

    raise AssertionError("unreachable run-contract inspection state")


def ensure_exact_run_contract(
    out_dir: Path,
    *,
    run_contract: dict[str, Any],
    require_exact_resume_contract: bool,
) -> dict[str, Any]:
    """Persist or migrate the run contract before any training work starts."""

    decision = inspect_exact_run_contract(
        out_dir,
        run_contract=run_contract,
        require_exact_resume_contract=require_exact_resume_contract,
    )
    if decision["status"] == "SOURCE_ONLY_MIGRATION_REQUIRED":
        existing_contract = json.loads(
            (out_dir / "run_contract.json").read_text(encoding="utf-8")
        )
        migrate_source_only_resume_artifacts(
            out_dir,
            existing_contract=existing_contract,
            expected_contract=run_contract,
        )
        return {
            "status": "SOURCE_ONLY_MIGRATED",
            "previous_digest": decision["previous_digest"],
            "new_digest": decision["new_digest"],
        }
    if decision["status"] == "LEGACY_SOURCE_METADATA_NORMALIZATION_REQUIRED":
        existing_path = out_dir / "run_contract.json"
        existing_contract = json.loads(existing_path.read_text(encoding="utf-8"))
        # This repair is safe only before any training or evaluation bytes are
        # present.  A partially trained run must remain blocked and receive a
        # separately audited migration, never an in-place normalization.
        forbidden = []
        for candidate in out_dir.rglob("*"):
            if not candidate.is_file() or candidate == existing_path:
                continue
            name = candidate.name.lower()
            if (
                candidate.suffix.lower() in {".pt", ".pth", ".ckpt"}
                or name in {"three_d_metrics.json", "progress.json", "training.lock"}
                or "outer_eval" in name
                or "checkpoint" in name
                or "progress" in name
                or "metrics" in name
                or "lock" in name
            ):
                forbidden.append(candidate)
        if forbidden:
            raise RuntimeError("LEGACY_CONTRACT_NORMALIZATION_AFTER_TRAINING_FORBIDDEN")
        normalized = legacy_source_metadata_normalization(existing_contract, run_contract)
        if normalized is None:
            raise RuntimeError("LEGACY_SOURCE_METADATA_NORMALIZATION_INVALID")
        migration_dir = out_dir / "source_contract_migrations"
        migration_dir.mkdir(parents=True, exist_ok=True)
        old_file_sha = sha256_path(existing_path)
        archive = migration_dir / f"legacy_source_metadata_{old_file_sha}.json"
        if not archive.exists():
            shutil.copy2(existing_path, archive)
        receipt = {
            "schema_version": 1,
            "artifact": "method_x_legacy_source_metadata_normalization_receipt",
            "append_only": True,
            "old_file_sha256": old_file_sha,
            "new_file_sha256": None,
            "old_run_contract_digest": existing_contract.get("run_contract_digest"),
            "new_run_contract_digest": run_contract.get("digest"),
            "source_digest": normalized.get("source_digest_metadata"),
            "archived_path": archive.name,
            "training_artifact_present": False,
            "outer_test_touched": False,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        receipt_path = migration_dir / f"legacy_source_metadata_{old_file_sha}.receipt.json"
        # Preserve the source metadata outside the nested, digested body.
        atomic_write_json(
            existing_path,
            {
                "run_contract": normalized["run_contract"],
                "run_contract_digest": normalized["run_contract_digest"],
                "source_digest_metadata": normalized.get("source_digest_metadata"),
            },
        )
        receipt["new_file_sha256"] = sha256_path(existing_path)
        receipt["integrity_sha256"] = canonical_digest(
            {key: value for key, value in receipt.items() if key != "integrity_sha256"}
        )
        if not receipt_path.exists():
            atomic_write_json(receipt_path, receipt)
        return {
            "status": "LEGACY_SOURCE_METADATA_NORMALIZED",
            "previous_digest": decision["previous_digest"],
            "new_digest": decision["new_digest"],
            "receipt": str(receipt_path),
        }
    if decision["status"] == "CREATE_REQUIRED":
        atomic_write_json(
            out_dir / "run_contract.json",
            {"run_contract": run_contract, "run_contract_digest": run_contract["digest"]},
        )
        return {
            "status": "CREATED",
            "run_contract_digest": run_contract["digest"],
        }
    return decision


def resolve_train_indices(
    split_path: Path,
    sample_to_idx: dict[str, int],
    ok: set[int],
    *,
    cache_path: Path,
    include_supplementary: bool,
    include_global_supplementary: bool = False,
    limit: int = 0,
) -> tuple[list[int], list[int]]:
    real_idx, real_labels = split_indices(split_path, sample_to_idx, ok, limit=limit)
    if not include_supplementary:
        return real_idx, real_labels
    all_hashes = cache_sample_hashes(cache_path)
    supp_idx, supp_labels = supplementary_context_with_policy(
        cache_path,
        ok,
        fold=fold_from_split_path(split_path),
        include_global=include_global_supplementary,
        allowed_source_hashes={all_hashes[idx] for idx in real_idx},
    )
    return merge_train_indices(real_idx, real_labels, supp_idx, supp_labels)


def tabular_stats(cache_path: Path, indices: list[int]):
    import torch

    if not indices:
        return None, None
    dataset = H5MethodXDataset(cache_path, indices, [0] * len(indices))
    rows = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0)
    return rows.mean(dim=0), rows.std(dim=0).clamp_min(1.0)


def normalize_tabular(tabular, tab_mean, tab_std):
    if tab_mean is None or tab_std is None:
        return tabular
    tab_mean = tab_mean.to(tabular.device)
    tab_std = tab_std.to(tabular.device)
    return (tabular - tab_mean) / tab_std


def normalize_phenotype_features(phenotype_features, tab_mean, tab_std):
    if tab_mean is None or tab_std is None:
        return phenotype_features
    width = phenotype_features.shape[-1]
    if width != len(METHOD_X_PHENOTYPE_FEATURES):
        raise ValueError("Phenotype width differs from the canonical five-feature contract")
    indices = [CACHE_TABULAR_FEATURES.index(name) for name in METHOD_X_PHENOTYPE_FEATURES]
    tab_mean = tab_mean[indices].to(phenotype_features.device)
    tab_std = tab_std[indices].to(phenotype_features.device)
    return (phenotype_features - tab_mean) / tab_std


def unpack_batch(batch):
    if len(batch) == 3:
        image, tabular, label = batch
        return image, tabular, label, None, None
    if len(batch) == 4:
        image, tabular, label, tab_presence = batch
        return image, tabular, label, tab_presence, None
    if len(batch) == 5:
        image, tabular, label, tab_presence, phenotype_features = batch
        return image, tabular, label, tab_presence, phenotype_features
    raise ValueError(f"Unexpected Method X batch arity: {len(batch)}")


def compute_loss(
    model,
    image,
    tabular,
    label,
    *,
    pos_weight,
    latent_alignment_lambda: float,
    synib_cfg: SynIBConfig | None,
    use_synib: bool,
    contrastive_alignment: bool = False,
    contrastive_temperature: float = 0.07,
    contrastive_loss_fn=None,
    supervised_contrastive_alignment: bool = False,
    phenotypic_contrastive_lambda: float = 0.0,
    phenotypic_temperature: float = 0.1,
    phenotype_feature_names: tuple[str, ...] = (),
    tab_presence=None,
    phenotype_features=None,
    task_loss_name: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    auc_margin: float = 1.0,
    auc_loss_weight: float = 1.0,
):
    import torch

    mri_latent, tab_raw = model.encode(image, tabular)
    logits = model.logits_from_latents(mri_latent, tab_raw, tab_presence=tab_presence)
    moe_load_balance_term = getattr(getattr(model, "fuser", None), "last_load_balance_loss", None)
    aligned_tab = model.alignment_projection(tab_raw)
    task, task_parts = compute_phase1_task_loss(
        logits,
        label,
        loss_name=task_loss_name,
        pos_weight=pos_weight,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        auc_margin=auc_margin,
        auc_loss_weight=auc_loss_weight,
    )
    loss = task
    parts: dict[str, object] = {
        "task_loss": float(task.detach()),
        "latent_alignment_loss": 0.0,
        "contrastive_alignment_loss": 0.0,
        "phenotypic_contrastive_loss": 0.0,
        "phenotypic_contrastive_weighted_loss": 0.0,
        "moe_load_balance_loss": 0.0,
        "synib_loss": 0.0,
    }
    parts.update(task_parts)

    align_term = None
    if contrastive_alignment and contrastive_loss_fn is not None:
        if supervised_contrastive_alignment:
            align = contrastive_loss_fn(mri_latent, aligned_tab, label)
        elif phenotype_features is not None:
            align = contrastive_loss_fn(
                mri_latent,
                aligned_tab,
                phenotype_features=phenotype_features,
            )
        else:
            align = contrastive_loss_fn(mri_latent, aligned_tab)
        align_term = latent_alignment_lambda * align if latent_alignment_lambda > 0.0 else align
        loss = loss + align_term
        parts["contrastive_alignment"] = float(align.detach())
        parts["contrastive_alignment_loss"] = float(align.detach())
    elif latent_alignment_lambda > 0.0:
        align = latent_alignment_loss(mri_latent, aligned_tab)
        align_term = latent_alignment_lambda * align
        loss = loss + align_term
        parts["latent_alignment"] = float(align.detach())
        parts["latent_alignment_loss"] = float(align.detach())

    if float(phenotypic_contrastive_lambda) > 0.0:
        if phenotype_features is None:
            raise ValueError("phenotype_features are required when phenotypic_contrastive_lambda > 0")
        phen_loss = phenotypic_contrastive_loss(
            mri_latent,
            phenotype_features,
            list(phenotype_feature_names or METHOD_X_PHENOTYPE_FEATURES),
            temperature=float(phenotypic_temperature),
        )
        weighted_phen_loss = float(phenotypic_contrastive_lambda) * phen_loss
        loss = loss + weighted_phen_loss
        parts["phenotypic_contrastive_loss"] = float(phen_loss.detach())
        parts["phenotypic_contrastive_weighted_loss"] = float(weighted_phen_loss.detach())

    if use_synib and synib_cfg is not None and synib_cfg.lambda_syn > 0.0:
        zeros_mri = torch.zeros_like(mri_latent)
        zeros_tab = torch.zeros_like(tab_raw)
        logits_tab = model.logits_from_latents(zeros_mri, tab_raw, tab_presence=tab_presence)
        logits_mri = model.logits_from_latents(mri_latent, zeros_tab, tab_presence=torch.zeros_like(label))
        syn_total, syn_parts = synib_multimodal_loss(logits, logits_tab, logits_mri, label, config=synib_cfg, pos_weight=pos_weight)
        loss = syn_total
        if align_term is not None:
            loss = loss + align_term
        parts.update({f"synib_{k}": v for k, v in syn_parts.items()})
        parts["synib_loss"] = float(syn_parts.get("syn_penalty", 0.0))

    if moe_load_balance_term is not None:
        loss = loss + moe_load_balance_term
        parts["moe_load_balance_loss"] = float(moe_load_balance_term.detach())

    parts["total"] = float(loss.detach())
    parts["total_loss"] = float(loss.detach())
    return loss, parts


def eval_loss(
    model,
    loader,
    device,
    pos_weight,
    *,
    tab_mean,
    tab_std,
    task_loss_name: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    auc_margin: float = 1.0,
    auc_loss_weight: float = 1.0,
) -> float:
    import torch

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            image, tabular, label, tab_presence, _phenotype_features = unpack_batch(batch)
            image = image.to(device)
            tabular = normalize_tabular(tabular.to(device), tab_mean, tab_std)
            label = label.to(device)
            if tab_presence is not None:
                tab_presence = tab_presence.to(device)
            logits = model(image, tabular, tab_presence=tab_presence)
            loss, _ = compute_phase1_task_loss(
                logits,
                label,
                loss_name=task_loss_name,
                pos_weight=pos_weight,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                auc_margin=auc_margin,
                auc_loss_weight=auc_loss_weight,
            )
            losses.append(float(loss.detach()))
    return float(sum(losses) / max(len(losses), 1))


def clone_state_dict(state: dict) -> dict:
    return {
        key: value.detach().clone() if hasattr(value, "detach") else value
        for key, value in state.items()
    }


def initialize_weight_average_state(model) -> dict:
    return clone_state_dict(model.state_dict())


def update_ema_state(ema_state: dict, model, decay: float) -> dict:
    import torch

    if ema_state is None:
        ema_state = initialize_weight_average_state(model)
    decay = float(decay)
    for key, value in model.state_dict().items():
        current = value.detach()
        if key not in ema_state:
            ema_state[key] = current.clone()
        elif torch.is_floating_point(current):
            ema_state[key].mul_(decay).add_(current, alpha=1.0 - decay)
        else:
            ema_state[key].copy_(current)
    return ema_state


def update_swa_state(swa_state: dict | None, model, count: int) -> tuple[dict, int]:
    import torch

    if swa_state is None:
        return initialize_weight_average_state(model), 1
    next_count = int(count) + 1
    for key, value in model.state_dict().items():
        current = value.detach()
        if key not in swa_state:
            swa_state[key] = current.clone()
        elif torch.is_floating_point(current):
            swa_state[key].add_(current - swa_state[key], alpha=1.0 / next_count)
        else:
            swa_state[key].copy_(current)
    return swa_state, next_count


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    pos_weight,
    *,
    latent_alignment_lambda: float,
    synib_cfg: SynIBConfig | None,
    use_synib: bool,
    tab_mean,
    tab_std,
    use_amp: bool = False,
    scaler=None,
    max_grad_norm: float = 1.0,
    grad_accum_steps: int = 1,
    label_smoothing: float = 0.0,
    contrastive_alignment: bool = False,
    contrastive_temperature: float = 0.07,
    contrastive_loss_fn=None,
    supervised_contrastive_alignment: bool = False,
    phenotypic_contrastive_lambda: float = 0.0,
    phenotypic_temperature: float = 0.1,
    phenotype_feature_names: tuple[str, ...] = (),
    use_checkpoint: bool = False,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
    task_loss_name: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    auc_margin: float = 1.0,
    auc_loss_weight: float = 1.0,
    ema_state: dict | None = None,
    ema_decay: float = 0.0,
    optimizer_step_commit_callback=None,
    start_batch_cursor: int = 0,
    initial_loss_sum: float = 0.0,
    initial_loss_count: int = 0,
    optimizer_step_offset: int = 0,
    epoch_index: int = 0,
) -> float:
    import torch

    if int(start_batch_cursor) < 0:
        raise ValueError("start_batch_cursor must be non-negative")
    if int(initial_loss_count) < 0:
        raise ValueError("initial_loss_count must be non-negative")
    if int(optimizer_step_offset) < 0:
        raise ValueError("optimizer_step_offset must be non-negative")
    model.train()
    losses: list[float] = []
    loss_part_sums: dict[str, float] = {}
    loss_part_count = int(initial_loss_count)
    loss_sum = float(initial_loss_sum)
    local_optimizer_steps = 0
    amp_enabled = use_amp and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        # A staged checkpoint can represent a safe optimizer boundary in the
        # middle of an epoch.  The already-committed batches are deterministic
        # replay history and must not be optimized a second time.
        if step < int(start_batch_cursor):
            continue
        image, tabular, label, tab_presence, phenotype_features = unpack_batch(batch)
        image = image.to(device)
        tabular = normalize_tabular(tabular.to(device), tab_mean, tab_std)
        label = label.to(device)
        if phenotype_features is not None:
            phenotype_features = normalize_phenotype_features(phenotype_features.to(device), tab_mean, tab_std)
        if tab_presence is not None:
            tab_presence = tab_presence.to(device)
        if label_smoothing > 0.0:
            label = label * (1.0 - label_smoothing) + 0.5 * label_smoothing
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss, parts = compute_loss(
                model,
                image,
                tabular,
                label,
                pos_weight=pos_weight,
                latent_alignment_lambda=latent_alignment_lambda,
                synib_cfg=synib_cfg,
                use_synib=use_synib,
                contrastive_alignment=contrastive_alignment,
                contrastive_temperature=contrastive_temperature,
                contrastive_loss_fn=contrastive_loss_fn,
                supervised_contrastive_alignment=supervised_contrastive_alignment,
                phenotypic_contrastive_lambda=phenotypic_contrastive_lambda,
                phenotypic_temperature=phenotypic_temperature,
                phenotype_feature_names=phenotype_feature_names,
                tab_presence=tab_presence,
                phenotype_features=phenotype_features,
                task_loss_name=task_loss_name,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                auc_margin=auc_margin,
                auc_loss_weight=auc_loss_weight,
            )
            loss = loss / grad_accum_steps
        for key, value in parts.items():
            if isinstance(value, (int, float)):
                loss_part_sums[key] = loss_part_sums.get(key, 0.0) + float(value)
        loss_part_count += 1
        raw_loss_value = float(loss.detach().cpu()) * float(grad_accum_steps)
        loss_sum += raw_loss_value
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        is_update_step = (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader)
        if is_update_step:
            if scaler is not None and amp_enabled:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            if ema_state is not None and float(ema_decay) > 0.0:
                update_ema_state(ema_state, model, ema_decay)
            if scheduler is not None and scheduler_step_per_batch:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            local_optimizer_steps += 1
            if optimizer_step_commit_callback is not None:
                # The callback is invoked only after optimizer/scheduler state
                # is coherent.  It may persist a checkpoint and then request a
                # safe boundary stop; it must never run from the signal handler.
                optimizer_step_commit_callback(
                    local_optimizer_steps=int(local_optimizer_steps),
                    optimizer_step=int(optimizer_step_offset) + int(local_optimizer_steps),
                    global_optimizer_step=int(optimizer_step_offset) + int(local_optimizer_steps),
                    batch_cursor=int(step + 1),
                    sampler_cursor=int(step + 1),
                    epoch=int(epoch_index),
                    loss_sum=float(loss_sum),
                    loss_count=int(loss_part_count),
                    epoch_loss_sum=float(loss_sum),
                    epoch_loss_count=int(loss_part_count),
                )
        losses.append(raw_loss_value)
    model._last_epoch_loss_parts = {
        key: value / max(loss_part_count, 1)
        for key, value in loss_part_sums.items()
    }
    return float(loss_sum / max(loss_part_count, 1))


def predict(
    model,
    loader,
    device,
    tab_mean,
    tab_std,
    use_tta: bool = False,
    tta_augmentations: int = 5,
    tta_mode: str = "basic",
    *,
    return_logits: bool = False,
) -> tuple[list[int], list[float]] | tuple[list[int], list[float], list[float]]:
    import torch

    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    y_logits: list[float] = []
    tta = TTAPredictor(model, n_augmentations=tta_augmentations, mode=tta_mode) if use_tta else None
    with torch.no_grad():
        for batch in loader:
            image, tabular, label, tab_presence, _phenotype_features = unpack_batch(batch)
            image = image.to(device)
            tabular = normalize_tabular(tabular.to(device), tab_mean, tab_std)
            if tab_presence is not None:
                tab_presence = tab_presence.to(device)
            if tta is not None:
                probs = tta.predict(image, tabular, tab_presence=tab_presence)
                if return_logits:
                    logits = torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))
            else:
                logits = model(image, tabular, tab_presence=tab_presence)
                probs = torch.sigmoid(logits)
            y_true.extend(int(v) for v in label.tolist())
            y_prob.extend(float(v) for v in probs.detach().cpu().tolist())
            if return_logits:
                y_logits.extend(float(v) for v in logits.detach().cpu().tolist())
    if return_logits:
        return y_true, y_prob, y_logits
    return y_true, y_prob


def collect_mri_latents(model, loader, device, tab_mean, tab_std):
    import numpy as np
    import torch

    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            image, tabular, _label, _tab_presence, _phenotype_features = unpack_batch(batch)
            image = image.to(device)
            tabular = normalize_tabular(tabular.to(device), tab_mean, tab_std)
            mri_latent, _tab_latent = model.encode(image, tabular)
            rows.append(mri_latent.detach().cpu().numpy())
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.concatenate(rows, axis=0)


def temperature_scaling_report(
    *,
    validation_y: list[int],
    validation_logits: list[float],
    test_y: list[int],
    test_logits: list[float],
) -> tuple[dict, list[float]]:
    if not validation_y or not test_y:
        return (
            {
                "status": "NOT_RUN",
                "reason": "validation_or_test_predictions_missing",
                "fit_scope_required": "inner_validation_only",
                "test_labels_used_for_calibration": False,
            },
            [],
        )
    uncalibrated_test_prob = sigmoid_logits(test_logits)
    try:
        fit = fit_temperature_scaling(validation_logits, validation_y)
        scaled_test_logits = apply_temperature_to_logits(test_logits, float(fit["temperature"]))
        calibrated_test_prob = sigmoid_logits(scaled_test_logits)
    except Exception as exc:
        return (
            {
                "status": "NOT_RUN",
                "reason": f"temperature_scaling_failed:{type(exc).__name__}:{exc}",
                "fit_scope_required": "inner_validation_only",
                "test_labels_used_for_calibration": False,
                "test_uncalibrated": calibration_metric_summary(test_y, uncalibrated_test_prob),
            },
            [],
        )
    return (
        {
            "status": "FITTED_ON_INNER_VALIDATION_ONLY",
            "temperature": fit["temperature"],
            "n_validation": fit["n_validation"],
            "validation_before_nll": fit["before_nll"],
            "validation_after_nll": fit["after_nll"],
            "fit_scope": "inner_validation_only",
            "test_labels_used_for_calibration": False,
            "test_uncalibrated": calibration_metric_summary(test_y, uncalibrated_test_prob),
            "test_calibrated": calibration_metric_summary(test_y, calibrated_test_prob),
        },
        calibrated_test_prob,
    )


def resolve_mode(device, requested_phase: str, limit: int, *, require_gpu: bool = False) -> tuple[str, int, int, int]:
    if require_gpu and device.type != "cuda":
        raise RuntimeError("GPU is required (--require_gpu) but CUDA device is not available")
    if device.type != "cuda":
        return "CPU_DEBUG_ONLY", 1, 1, max(limit, 32)
    if requested_phase != "final":
        return "FULL_3D_TRAINING", 0, 0, 0
    return "FULL_3D_TRAINING", 0, 0, 0


def best_epoch_from_curve(val_curve: list[float]) -> int | None:
    if not val_curve:
        return None
    return min(range(len(val_curve)), key=lambda i: val_curve[i]) + 1


def reported_epochs_run(last_completed_epoch: int, train_curve: list[float]) -> int:
    """Report true completed epochs even when legacy checkpoints lack loss curves."""
    return max(int(last_completed_epoch or 0), len(train_curve))


def is_selection_only_run(args: Any) -> bool:
    """True when the run is forbidden from reading or evaluating outer-test rows."""

    return bool(
        getattr(args, "validity_inner_only", False)
        or getattr(args, "hpo_selection_only", False)
    )


def outer_test_evaluation_policy(selection_only: bool = False) -> dict[str, Any]:
    """Machine-readable exactly-once policy used by trainers and auditors."""

    if selection_only:
        return {
            "during_training_epochs": False,
            "after_checkpoint_lock_only": False,
            "max_evaluations_per_fold": 0,
            "selection_only_no_outer_test": True,
        }
    return {
        "during_training_epochs": False,
        "after_checkpoint_lock_only": True,
        "max_evaluations_per_fold": 1,
    }


def numeric_values(values: list[object]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def last_numeric(values: list[object]) -> float | None:
    nums = numeric_values(values)
    return nums[-1] if nums else None


def min_numeric(values: list[object]) -> float | None:
    nums = numeric_values(values)
    return min(nums) if nums else None


def parse_pos_weight(value: str, labels: tuple[int, ...] | list[int], device):
    import torch

    text = str(value).strip().lower()
    if text == "none":
        return None
    if text == "auto":
        return torch.tensor(
            [min(max(sum(1 for v in labels if v == 0), 1) / max(sum(labels), 1), 20.0)],
            device=device,
        )
    return torch.tensor([float(text)], device=device)


def parameter_counts(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameter_count": int(total), "trainable_parameter_count": int(trainable)}


def freeze_all(module) -> list[str]:
    names = []
    for name, param in module.named_parameters():
        param.requires_grad = False
        names.append(name)
    return names


def freeze_encoder_until(model, marker: str) -> list[str]:
    marker = str(marker).strip()
    if not marker:
        return []
    frozen: list[str] = []
    children = list(model.encoder.named_children())
    if marker.isdigit():
        limit = int(marker)
        for idx, (child_name, child) in enumerate(children):
            if idx >= limit:
                break
            for name, param in child.named_parameters(prefix=child_name):
                param.requires_grad = False
                frozen.append(name)
        return frozen
    freeze = True
    for name, param in model.encoder.named_parameters():
        if marker in name:
            freeze = False
        if freeze:
            param.requires_grad = False
            frozen.append(name)
    return frozen


def apply_finetuning_controls(model, args) -> list[str]:
    frozen: list[str] = []
    if args.freeze_encoder or args.gradual_unfreeze:
        frozen.extend(f"encoder.{name}" for name in freeze_all(model.encoder))
    if args.freeze_until:
        frozen.extend(f"encoder.{name}" for name in freeze_encoder_until(model, args.freeze_until))
    return sorted(set(frozen))


def gradual_unfreeze_step(model, *, epoch: int, epochs: int, enabled: bool) -> list[str]:
    if not enabled:
        return []
    children = list(model.encoder.named_children())
    if not children:
        return []
    step = min(int(epoch), len(children) - 1)
    child_name, child = children[-(step + 1)]
    unfrozen: list[str] = []
    for name, param in child.named_parameters(prefix=child_name):
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen.append(f"encoder.{name}")
    return unfrozen


def build_optimizer(model, *, base_lr: float, args):
    import torch

    weight_decay = 1e-4
    exclude_bias_norm = bool(getattr(args, "wd_exclude_bias_norm", True))

    def no_decay_param(name: str, param) -> bool:
        lowered = name.lower()
        return param.ndim < 2 or lowered.endswith(".bias") or "norm" in lowered or ".bn" in lowered

    def add_param_group(
        groups: list[dict],
        manifest_groups: list[dict],
        named_params: list[tuple[str, object]],
        *,
        scope: str,
        lr: float,
        include_frozen: bool,
    ) -> None:
        decay_params = []
        nodecay_params = []
        for name, param in named_params:
            if exclude_bias_norm and no_decay_param(name, param):
                nodecay_params.append(param)
            else:
                decay_params.append(param)
        for decay_scope, params, group_weight_decay in (
            ("decay", decay_params, weight_decay),
            ("no_decay", nodecay_params, 0.0),
        ):
            if not params:
                continue
            groups.append({"params": params, "lr": lr, "weight_decay": group_weight_decay})
            manifest_groups.append(
                {
                    "scope": f"{scope}.{decay_scope}",
                    "lr": float(lr),
                    "weight_decay": float(group_weight_decay),
                    "parameters": sum(p.numel() for p in params),
                    "trainable_parameters_at_optimizer_build": sum(p.numel() for p in params if p.requires_grad),
                    "includes_frozen_parameters_for_gradual_unfreeze": include_frozen,
                }
            )

    if not args.discriminative_lr and float(args.layerwise_lr_decay) == 1.0:
        groups: list[dict] = []
        manifest_groups: list[dict] = []
        add_param_group(
            groups,
            manifest_groups,
            list(model.named_parameters()),
            scope="all_parameters",
            lr=base_lr,
            include_frozen=False,
        )
        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay), {
            "mode": "single_lr",
            "base_lr": base_lr,
            "wd_exclude_bias_norm": exclude_bias_norm,
            "groups": manifest_groups,
        }
    decay = float(args.layerwise_lr_decay)
    groups = []
    manifest_groups = []
    encoder_children = list(model.encoder.named_children())
    seen_param_ids: set[int] = set()
    for depth, (name, child) in enumerate(encoder_children):
        named_params = []
        for param_name, param in child.named_parameters(prefix=f"encoder.{name}"):
            if id(param) in seen_param_ids:
                continue
            seen_param_ids.add(id(param))
            named_params.append((param_name, param))
        if not named_params:
            continue
        lr = base_lr * (decay ** max(len(encoder_children) - depth - 1, 0))
        add_param_group(
            groups,
            manifest_groups,
            named_params,
            scope=f"encoder.{name}",
            lr=lr,
            include_frozen=True,
        )
    head_params = [
        (name, p)
        for name, p in model.named_parameters()
        if not name.startswith("encoder.") and id(p) not in seen_param_ids
    ]
    if head_params:
        add_param_group(
            groups,
            manifest_groups,
            head_params,
            scope="tabular_fusion_head",
            lr=base_lr,
            include_frozen=True,
        )
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay), {
        "mode": "discriminative_lr",
        "base_lr": float(base_lr),
        "layerwise_lr_decay": decay,
        "wd_exclude_bias_norm": exclude_bias_norm,
        "coverage": "all_parameters_including_currently_frozen",
        "groups": manifest_groups,
    }


def build_scheduler(optimizer, *, args, epochs: int, steps_per_epoch: int, base_lr: float):
    import torch

    scheduler_name = args.scheduler.lower().strip()
    if scheduler_name in {"none", "onecycle"} and args.warmup_epochs > 0:
        raise SystemExit("--warmup_epochs is only supported with --scheduler cosine or step; pass --warmup_epochs 0 for onecycle/none.")
    if scheduler_name == "none":
        return None
    if scheduler_name == "onecycle":
        total_steps = max(int(epochs) * max(int(steps_per_epoch), 1), 1)
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=base_lr, total_steps=total_steps)
    if scheduler_name == "step":
        base = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.step_size, 1), gamma=args.step_gamma)
    elif scheduler_name == "cosine":
        base = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - args.warmup_epochs, 1),
            eta_min=1e-6,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    if args.warmup_epochs <= 0:
        return base
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, base], milestones=[args.warmup_epochs])


def memory_profile(device) -> dict[str, float | None]:
    profile: dict[str, float | None] = {
        "cpu_rss_mb": None,
        "cuda_max_memory_allocated_gb": None,
        "cuda_max_memory_reserved_gb": None,
    }
    try:
        import psutil

        profile["cpu_rss_mb"] = float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except Exception:
        pass
    if getattr(device, "type", None) == "cuda":
        try:
            import torch

            profile["cuda_max_memory_allocated_gb"] = float(torch.cuda.max_memory_allocated(device) / (1024**3))
            profile["cuda_max_memory_reserved_gb"] = float(torch.cuda.max_memory_reserved(device) / (1024**3))
        except Exception:
            pass
    return profile


def hparam_selection_payload(args) -> dict:
    payload = getattr(args, "_hparam_selection", None)
    if isinstance(payload, dict):
        return payload
    return {
        "status": "NOT_REQUESTED",
        "selection_source": "none",
        "external_or_pseudo_external_used_for_selection": False,
        "test_used_for_selection": False,
        "selection_applied": False,
    }


def apply_hparam_manifest(args) -> None:
    if not getattr(args, "use_hparam_manifest", False):
        args._hparam_selection = hparam_selection_payload(args)
        return
    manifest_path = PROJECT_ROOT / args.hparam_manifest
    if not manifest_path.exists():
        args._hparam_selection = {
            "status": "NO_COMPLETED_SEARCH_AVAILABLE",
            "manifest": args.hparam_manifest,
            "selection_source": "outer_train_inner_validation_only",
            "external_or_pseudo_external_used_for_selection": False,
            "test_used_for_selection": False,
            "selection_applied": False,
        }
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        args._hparam_selection = {
            "status": "INVALID_MANIFEST",
            "manifest": args.hparam_manifest,
            "selection_applied": False,
            "external_or_pseudo_external_used_for_selection": False,
            "test_used_for_selection": False,
        }
        return
    best_by_backbone = manifest.get("best_by_backbone") or {}
    backbone_specific = best_by_backbone.get(args.backbone) if isinstance(best_by_backbone, dict) else None
    best = backbone_specific or manifest.get("best") or {}
    params = best.get("params") or {}
    if not params:
        args._hparam_selection = {
            "status": manifest.get("status", "NO_BEST_CONFIG"),
            "manifest": args.hparam_manifest,
            "selection_source": manifest.get("selection_source", "outer_train_inner_validation_only"),
            "external_or_pseudo_external_used_for_selection": False,
            "test_used_for_selection": False,
            "selection_applied": False,
            "requested_backbone": args.backbone,
            "backbone_specific_candidate_available": False,
        }
        return
    selected_backbone = params.get("backbone")
    if selected_backbone and selected_backbone != args.backbone and not args.allow_hparam_backbone_override:
        args._hparam_selection = {
            "status": "BEST_BACKBONE_MISMATCH_NOT_APPLIED",
            "manifest": args.hparam_manifest,
            "selected_backbone": selected_backbone,
            "requested_backbone": args.backbone,
            "selection_source": manifest.get("selection_source", "outer_train_inner_validation_only"),
            "external_or_pseudo_external_used_for_selection": False,
            "test_used_for_selection": False,
            "selection_applied": False,
            "backbone_specific_candidate_available": False,
        }
        return
    applied: dict[str, object] = {}
    for key in [
        "backbone",
        "fusion",
        "lr",
        "dropout",
        "latent_dim",
        "tab_hidden_dim",
        "bottleneck_dim",
        "latent_alignment_lambda",
        "label_smoothing",
        "attention",
        "q2_channel_gate",
        "scheduler",
        "pos_weight",
        "task_loss",
        "focal_gamma",
        "focal_alpha",
        "auc_margin",
        "auc_loss_weight",
        "n_experts",
        "fusion_num_heads",
        "moe_load_balance_lambda",
        "restore_best_at_end",
        "supervised_contrastive_alignment",
        "phenotypic_contrastive_alignment",
        "phenotypic_contrastive_lambda",
        "phenotypic_temperature",
        "synib_lambda",
        "ema_decay",
        "swa",
        "swa_start_fraction",
        "layerwise_lr_decay",
        "discriminative_lr",
    ]:
        if key in params:
            setattr(args, key, params[key])
            applied[key] = params[key]
    args._hparam_selection = {
        "status": "APPLIED_BACKBONE_SPECIFIC" if backbone_specific else "APPLIED",
        "manifest": args.hparam_manifest,
        "selected_trial": best.get("name"),
        "selected_backbone": selected_backbone,
        "requested_backbone": args.backbone,
        "backbone_specific_candidate_available": bool(backbone_specific),
        "selection_source": manifest.get("selection_source", "outer_train_inner_validation_only"),
        "external_or_pseudo_external_used_for_selection": False,
        "test_used_for_selection": False,
        "selection_metric": manifest.get("selection_metric", "mean_best_val_loss"),
        "selection_applied": True,
        "applied_params": applied,
    }


def build_aggregate(
    *,
    mode: str,
    args,
    device,
    epochs: int,
    folds: int,
    use_synib: bool,
    use_amp: bool,
    cache_path: Path,
    cache_resolution: int | None,
    fold_metrics: list[dict],
    all_y: list[int],
    all_p: list[float],
    checkpoints: list[str],
    augmentation_policy: dict | None = None,
    threshold_rows: list[dict] | None = None,
    include_supplementary_train: bool = False,
    supplementary_ok_n: int = 0,
) -> dict:
    selection_only = is_selection_only_run(args)
    selection_reason = (
        "validity_inner_only"
        if bool(getattr(args, "validity_inner_only", False))
        else "hpo_selection_only"
    )
    claim_policy = evaluate_three_d_final_claim(
        mode=mode,
        phase=args.phase,
        resolution=args.resolution,
        folds=folds,
        cache_resolution=int(cache_resolution) if cache_resolution is not None else None,
    )
    if selection_only:
        claim_policy["final_claim_allowed"] = False
        claim_policy.setdefault("claim_blockers", []).append(
            "INNER_SELECTION_ONLY_OUTER_EVALUATION_DEFERRED"
        )
        claim_policy["reporting_mode"] = "INNER_SELECTION_ONLY_NOT_FINAL_REPORTING"
    if bool(getattr(args, "include_global_supplementary_train", False)):
        claim_policy["final_claim_allowed"] = False
        claim_policy.setdefault("claim_blockers", []).append("GLOBAL_SUPPLEMENTARY_TRAIN_AUGMENTATION_REQUIRES_CAVEAT")
        claim_policy["reporting_mode"] = "AUGMENTED_TRAINING_RESEARCH_REPORTING_ONLY"
    best_losses = [float(m["best_val_loss"]) for m in fold_metrics if m.get("best_val_loss") is not None]
    final_val_losses = [float(m["val_loss_last"]) for m in fold_metrics if m.get("val_loss_last") is not None]
    final_evaluation_losses = [
        float(m["evaluation_loss_last"]) for m in fold_metrics if m.get("evaluation_loss_last") is not None
    ]
    epochs_per_fold = (
        protocol_epochs_per_fold(epochs, folds)
        if bool(getattr(args, "require_exact_resume_contract", False))
        else int(epochs)
    )
    return {
        "model": "MethodX3DClassifier",
        "mode": mode,
        "phase": args.phase,
        "evaluation_scope": (
            "inner_validation_only"
            if selection_only
            else "locked_outer_test_reporting_only"
        ),
        "validity_inner_only": bool(getattr(args, "validity_inner_only", False)),
        "hpo_selection_only": bool(getattr(args, "hpo_selection_only", False)),
        "selection_only_reason": selection_reason if selection_only else None,
        "outer_test_access_count": 0 if selection_only else len(fold_metrics),
        "outer_test_evaluation_policy": outer_test_evaluation_policy(selection_only),
        "outer_test_used_for_selection": False,
        "throughput_profile_only": bool(getattr(args, "throughput_profile_only", False)),
        "determinism_runtime": getattr(args, "_determinism_runtime", None),
        "fusion": args.fusion,
        "backbone": args.backbone,
        "attention": getattr(args, "attention", "none"),
        "alignment_head": getattr(args, "alignment_head", "linear"),
        "tab_encoder": getattr(args, "tab_encoder", "linear"),
        "causal_adjustment": getattr(args, "causal_adjustment", "none"),
        "causal_intervention_enabled": bool(getattr(args, "causal_intervention_enabled", False)),
        "q2_channel_gate": getattr(args, "q2_channel_gate", "none"),
        "pretrained_checkpoint": getattr(args, "pretrained_checkpoint", "") or None,
        "pretrained_fold_manifest": getattr(args, "pretrained_fold_manifest", "") or None,
        "pretrained_load_reports": getattr(args, "_pretrained_load_reports", []),
        "frozen_layers": getattr(args, "_frozen_layers", []),
        "layerwise_lr_decay": getattr(args, "layerwise_lr_decay", 1.0),
        "discriminative_lr": bool(getattr(args, "discriminative_lr", False)),
        "gradual_unfreeze": bool(getattr(args, "gradual_unfreeze", False)),
        "unfreeze_schedule": getattr(args, "unfreeze_schedule", "none"),
        "unfreeze_epoch": int(getattr(args, "unfreeze_epoch", 8)),
        "unfreeze_transitions": getattr(args, "_unfreeze_transitions", []),
        "encoder_stage_map": getattr(args, "_encoder_stage_map", None),
        "trainable_parameter_count": getattr(args, "_trainable_parameter_count", None),
        "total_parameter_count": getattr(args, "_total_parameter_count", None),
        "input_mode": args.input_mode,
        "drop_tabular_feature": getattr(args, "drop_tabular_feature", "none"),
        "latent_alignment_lambda": args.latent_alignment_lambda,
        "phenotypic_contrastive_alignment": bool(getattr(args, "phenotypic_contrastive_alignment", False)),
        "phenotypic_contrastive_lambda": float(getattr(args, "phenotypic_contrastive_lambda", 0.0)),
        "phenotypic_temperature": float(getattr(args, "phenotypic_temperature", 0.1)),
        "supervised_contrastive_alignment": bool(getattr(args, "supervised_contrastive_alignment", False)),
        "aug_prob_scale": float(getattr(args, "aug_prob_scale", 0.2)),
        "synib_lambda": args.synib_lambda,
        "modality_dropout": args.modality_dropout,
        "lr": getattr(args, "lr", 1e-4),
        "dropout": getattr(args, "dropout", 0.1),
        "label_smoothing": getattr(args, "label_smoothing", 0.0),
        "task_loss": getattr(args, "task_loss", "bce"),
        "focal_gamma": float(getattr(args, "focal_gamma", 2.0)),
        "focal_alpha": (
            None
            if float(getattr(args, "focal_alpha", -1.0)) < 0.0
            else float(getattr(args, "focal_alpha", -1.0))
        ),
        "auc_surrogate": {
            "enabled": getattr(args, "task_loss", "bce") in {"auc_pairwise", "bce_auc"},
            "implementation": "dependency_free_pairwise_logistic_surrogate",
            "margin": float(getattr(args, "auc_margin", 1.0)),
            "weight": float(getattr(args, "auc_loss_weight", 1.0)),
            "fallback_to_bce_when_batch_has_single_class": True,
            "heavy_dependency_used": False,
        },
        "weight_averaging": {
            "ema_enabled": float(getattr(args, "ema_decay", 0.0)) > 0.0,
            "ema_decay": float(getattr(args, "ema_decay", 0.0)),
            "swa_enabled": bool(getattr(args, "swa", False)),
            "swa_start_fraction": float(getattr(args, "swa_start_fraction", 0.8)),
            "default_behavior_changed": False,
        },
        "pos_weight": getattr(args, "pos_weight", "auto"),
        "scheduler": getattr(args, "scheduler", "cosine"),
        "tta": bool(getattr(args, "tta", False)),
        "tta_mode": getattr(args, "tta_mode", "basic"),
        "tta_augmentations": int(getattr(args, "tta_augmentations", 5)),
        "optimizer_param_groups": getattr(args, "_optimizer_manifest", None),
        "latent_dim": getattr(args, "latent_dim", 128),
        "tab_hidden_dim": getattr(args, "tab_hidden_dim", 32),
        "tab_encoder": getattr(args, "tab_encoder", "linear"),
        "bottleneck_dim": getattr(args, "bottleneck_dim", 32),
        "fusion_hidden_dim": getattr(args, "fusion_hidden_dim", None),
        "n_experts": getattr(args, "n_experts", 4),
        "fusion_num_heads": getattr(args, "fusion_num_heads", 2),
        "moe_load_balance_lambda": float(getattr(args, "moe_load_balance_lambda", 0.0)),
        "restore_best_at_end": bool(getattr(args, "restore_best_at_end", False)),
        "q2": {
            "channel_gate": getattr(args, "q2_channel_gate", "none"),
            "implemented": getattr(args, "q2_channel_gate", "none") != "none",
            "default_behavior_changed": False,
            "method_x_pet_free": True,
            "allowed_conditioning_features": [
                "age",
                "sex",
                "education_years",
                "apoe4_carrier",
                "mmse_base",
            ]
            if getattr(args, "q2_channel_gate", "none") == "phenotype"
            else None,
            "forbidden_conditioning_features": [
                "cohort",
                "site",
                "scanner",
                "pet",
                "suvr",
                "centiloid",
            ],
            "ddpm_scope": "not_used_by_method_x_channel_gate",
        },
        "literature_anchor_synib": "LIT019 arXiv:2606.09853" if use_synib else None,
        "epochs": epochs,
        "epochs_per_fold": epochs_per_fold,
        "epoch_target": epochs_per_fold,
        "epoch_target_met_all_completed_folds": all(
            int(m.get("epochs_run") or 0) >= epochs_per_fold for m in fold_metrics
        )
        if fold_metrics
        else False,
        "folds": folds,
        "completed_folds": len(fold_metrics),
        "batch_size": args.batch_size,
        "val_fraction": args.val_fraction,
        "early_stopping_patience": args.early_stopping_patience,
        "requested_resolution": args.resolution,
        "cache_resolution": cache_resolution,
        "cache": str(Path(args.cache)),
        "device_requested": args.device,
        "device": device.type,
        "amp_enabled": use_amp,
        "num_workers": args.num_workers,
        "augmentation": augmentation_policy
        or build_augmentation_policy(
            enabled=args.monai_augment,
            real_train_n=sum(int(m.get("train_n") or 0) for m in fold_metrics),
            exposures_per_real=args.augmentation_exposures_per_real,
        ),
        "variable_modality_adjustment": {
            "tabular_covariates": "cache_tabular_covariates_only",
            "input_policy": "MRI_ONLY_TABULAR_ZEROED" if args.input_mode == "mri_only" else "MRI_PLUS_ALLOWED_TABULAR_COVARIATES",
            "method_x_input_policy": getattr(args, "_input_policy", None),
            "method_x_mri_only_pet_free": True,
            "pet_input_allowed": False,
            "synthetic_pet_input_allowed": False,
            "raw_path_allowed": False,
            "raw_id_allowed": False,
            "subject_level_public_vector_allowed": False,
            "augmentation_policy": "MONAI_TRAIN_FOLD_ONLY" if args.monai_augment else "DISABLED",
            "monai_aug_prob_scale": float(getattr(args, "aug_prob_scale", 0.2)),
            "fold_local_train_only_augmentation": True,
            "holdout_augmentation_enabled": False,
            "threshold_selection_source": "fold_internal_validation_only",
            "calibration_threshold_adjustment": "thresholds_selected_on_validation_fold_only_for_reporting",
            "evaluation_loss_reporting_only": True,
            "sanity_random_labels": bool(getattr(args, "sanity_random_labels", False)),
            "sanity_random_noise_images": bool(getattr(args, "sanity_random_noise_images", False)),
            "phenotype_similarity_features": [
                "age",
                "sex",
                "education_years",
                "apoe4_carrier",
                "mmse_base",
            ]
            if (
                bool(getattr(args, "phenotypic_contrastive_alignment", False))
                or float(getattr(args, "phenotypic_contrastive_lambda", 0.0)) > 0.0
            )
            else None,
            "cohort_features_used_for_similarity": False,
            "causal_adjustment": getattr(args, "causal_adjustment", "none"),
            "q2_channel_gate": getattr(args, "q2_channel_gate", "none"),
            "q2_conditioning_policy": (
                "approved_phenotype_fields_only"
                if getattr(args, "q2_channel_gate", "none") == "phenotype"
                else "mri_latent_only"
                if getattr(args, "q2_channel_gate", "none") == "latent_logistic"
                else "disabled"
            ),
        },
        "sanity_random_labels": bool(getattr(args, "sanity_random_labels", False)),
        "sanity_random_noise_images": bool(getattr(args, "sanity_random_noise_images", False)),
        "evaluation_loss_policy": {
            "source": (
                f"not_accessed_{selection_reason}"
                if selection_only
                else "locked_fold_test_split"
            ),
            "reporting_only": True,
            "used_for_early_stopping": False,
            "used_for_hyperparameter_selection": False,
            "used_for_threshold_selection": False,
            "used_for_model_selection": False,
        },
        "threshold_selection": {
            "source": "fold_internal_validation_only",
            "test_labels_used_for_threshold_selection": False,
            "evaluation_loss_used_for_threshold_selection": False,
            "folds": threshold_rows or [],
        },
        "loss_logging_fields": [
            "task_loss",
            "task_loss_name",
            "bce_loss",
            "focal_loss",
            "auc_pairwise_loss",
            "auc_pairwise_pairs",
            "latent_alignment_loss",
            "contrastive_alignment_loss",
            "phenotypic_contrastive_loss",
            "phenotypic_contrastive_weighted_loss",
            "moe_load_balance_loss",
            "synib_loss",
            "total_loss",
        ],
        "amuse_x_min": {
            "a2_phenotypic_contrastive_enabled": (
                bool(getattr(args, "phenotypic_contrastive_alignment", False))
                or float(getattr(args, "phenotypic_contrastive_lambda", 0.0)) > 0.0
            ),
            "a2_phenotypic_contrastive_lambda": float(getattr(args, "phenotypic_contrastive_lambda", 0.0)),
            "a2_phenotypic_temperature": float(getattr(args, "phenotypic_temperature", 0.1)),
            "a3_alignment_head": getattr(args, "alignment_head", "linear"),
            "b3_tab_encoder": getattr(args, "tab_encoder", "linear"),
            "a4_causal_adjustment": getattr(args, "causal_adjustment", "none"),
            "a4_causal_intervention_enabled": bool(getattr(args, "causal_intervention_enabled", False)),
            "a6_validate_claim_available": True,
        },
        "sanity_checks": {
            "random_labels_enabled": bool(getattr(args, "sanity_random_labels", False)),
            "random_noise_images_enabled": bool(getattr(args, "sanity_random_noise_images", False)),
            "allowed_in_final_reporting": False,
        },
        "claim_target": "amyloid_biomarker_status_preclinical_risk_only",
        "claim_scope": getattr(args, "claim_scope", "amyloid biomarker / preclinical risk only"),
        "framing_language": getattr(args, "framing_language", "MRI-only amyloid biomarker / preclinical risk estimate"),
        "framing_language_sample": getattr(args, "framing_language", "MRI-only amyloid biomarker / preclinical risk estimate"),
        "clinical_alzheimer_diagnosis_output_allowed": False,
        "uses_pet_as_input": False,
        "uses_synthetic_pet_as_input": False,
        "uses_raw_paths": False,
        "uses_raw_identifiers": False,
        "uses_subject_level_public_vectors": False,
        "n": len(all_y),
        "roc_auc": roc_auc(all_y, all_p) if all_y else None,
        "pr_auc": average_precision(all_y, all_p) if all_y else None,
        "brier": brier(all_y, all_p) if all_y else None,
        "checkpoints": checkpoints,
        "mean_best_val_loss": (sum(best_losses) / len(best_losses)) if best_losses else None,
        "mean_final_val_loss": (sum(final_val_losses) / len(final_val_losses)) if final_val_losses else None,
        "mean_final_evaluation_loss": (sum(final_evaluation_losses) / len(final_evaluation_losses))
        if final_evaluation_losses
        else None,
        "cohort_probe_auc": (
            sum(float(m["cohort_probe_auc"]) for m in fold_metrics if m.get("cohort_probe_auc") is not None)
            / len([m for m in fold_metrics if m.get("cohort_probe_auc") is not None])
        )
        if any(m.get("cohort_probe_auc") is not None for m in fold_metrics)
        else None,
        "cohort_probe": {
            "status": "COMPLETED"
            if any(m.get("cohort_probe_auc") is not None for m in fold_metrics)
            else "NOT_RUN",
            "folds": [
                {"fold": m.get("fold"), **m.get("cohort_probe", {})}
                for m in fold_metrics
                if isinstance(m.get("cohort_probe"), dict)
            ],
        },
        "best_epoch_by_fold": [
            {"fold": m.get("fold"), "best_epoch": m.get("best_epoch"), "best_val_loss": m.get("best_val_loss")}
            for m in fold_metrics
            if m.get("best_epoch") is not None or m.get("best_val_loss") is not None
        ],
        "evaluation_loss_by_fold": [
            {
                "fold": m.get("fold"),
                "evaluation_loss_last": m.get("evaluation_loss_last"),
                "best_evaluation_loss": m.get("best_evaluation_loss"),
            }
            for m in fold_metrics
            if m.get("evaluation_loss_last") is not None or m.get("best_evaluation_loss") is not None
        ],
        "training_continuation_by_fold": [
            {"fold": m.get("fold"), **m.get("training_continuation")}
            for m in fold_metrics
            if isinstance(m.get("training_continuation"), dict)
        ],
        "hparam_selection": hparam_selection_payload(args),
        "include_supplementary_train": include_supplementary_train,
        "supplementary_ok_n": supplementary_ok_n,
        "include_global_supplementary_train": bool(getattr(args, "include_global_supplementary_train", False)),
        "supplementary_policy": (
            "explicit_global_train_augmentation"
            if bool(getattr(args, "include_global_supplementary_train", False))
            else "fold_local_train_only"
        ),
        "claim_policy_report": validate_claim(
            claim_from_three_d_aggregate(
                {
                    "model": "MethodX3DClassifier",
                    "mode": mode,
                    "phase": args.phase,
                    "resolution": args.resolution,
                    "folds": folds,
                    "uses_pet_as_input": False,
                    "uses_synthetic_pet_as_input": False,
                    "uses_raw_identifiers": False,
                    "uses_raw_paths": False,
                    "clinical_alzheimer_diagnosis_output_allowed": False,
                    "roc_auc": roc_auc(all_y, all_p) if all_y else None,
                    "pr_auc": average_precision(all_y, all_p) if all_y else None,
                    "ece": None,
                    "claim_scope": getattr(args, "claim_scope", "amyloid biomarker / preclinical risk only"),
                    "framing_language": getattr(
                        args,
                        "framing_language",
                        "MRI-only amyloid biomarker / preclinical risk estimate",
                    ),
                    "spm_complete": claim_policy.get("spm_complete", False),
                    "validity_recommendation": claim_policy.get("validity_recommendation"),
                },
                strict=getattr(args, "validate_claim_strict", False),
            ),
            strict=getattr(args, "validate_claim_strict", False),
        ),
        "final_claim_allowed": claim_policy["final_claim_allowed"],
        "claim_policy": claim_policy,
        # Flat mirrors of the authoritative eligibility verdict.  The claim
        # projection reads these, so they must not live only inside the nested
        # ``claim_policy``: a silently dropped field makes the gate judge a
        # missing value instead of the real rule.
        "spm_complete": claim_policy.get("spm_complete"),
        "spm_snapshot_verified": claim_policy.get("spm_snapshot_verified"),
        "validity_recommendation": claim_policy.get("validity_recommendation"),
        "final_contract": {
            "final_requires_128": True,
            "final_requires_4fold": True,
            "pet_input_allowed": False,
            "synthetic_pet_input_allowed": False,
            "gradient_checkpointing": bool(getattr(args, "checkpoint", False)),
            "attention": getattr(args, "attention", "none"),
            "scheduler": getattr(args, "scheduler", "cosine"),
            "alignment_head": getattr(args, "alignment_head", "linear"),
            "alignment_head_mode": getattr(args, "alignment_head_mode", "linear"),
            "tab_encoder": getattr(args, "tab_encoder", "linear"),
            "causal_adjustment": getattr(args, "causal_adjustment", "none"),
            "q2_channel_gate": getattr(args, "q2_channel_gate", "none"),
            "restore_best_at_end": bool(getattr(args, "restore_best_at_end", False)),
            "task_loss": getattr(args, "task_loss", "bce"),
            "ema_enabled": float(getattr(args, "ema_decay", 0.0)) > 0.0,
            "swa_enabled": bool(getattr(args, "swa", False)),
        },
    }


def write_metrics_payload(
    out_dir: Path,
    aggregate: dict,
    fold_metrics: list[dict],
    *,
    status: str = "IN_PROGRESS",
) -> None:
    payload = {"aggregate": {**aggregate, "status": status}, "folds": fold_metrics}
    atomic_write_json(out_dir / "three_d_metrics.json", payload)


def atomic_torch_save(payload: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_torch_save_immutable(payload: dict, path: Path) -> None:
    """Write a terminal/locked checkpoint without silently replacing evidence.

    A fold can be resumed after a process dies between the lock and the outer
    evaluation.  In that situation the lock file is already authoritative; a
    second invocation must either prove that its contract and checkpoint
    identity are the same or fail closed.  It must never overwrite the first
    locked bytes with a newly initialised model.
    """

    import torch

    path = Path(path)
    if path.is_file():
        try:
            existing = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError("LOCKED_CHECKPOINT_UNREADABLE") from exc
        digest_mismatch = False
        for key in (
            "run_contract_digest",
            "fold",
            "backbone",
            "checkpoint_kind",
            "split_membership_digest",
        ):
            if existing.get(key) != payload.get(key):
                # A source-only hop rebinds progress/state linkage while the
                # locked weights stay byte-identical: the rebuilt lock payload
                # then carries the new digest next to the old locked bytes.
                # That single-key drift is admissible only when the selected
                # weights below prove byte-identical AND the two embedded
                # contracts differ by source digest alone; any scientific
                # drift still fails closed here.
                if key != "run_contract_digest":
                    raise RuntimeError(f"LOCKED_CHECKPOINT_IDENTITY_MISMATCH:{key}")
                digest_mismatch = True
        # The model state is the scientific payload.  Compare tensor shapes and
        # bytes rather than trusting a mutable timestamp or a self-reported
        # marker.  This check is only reached on recovery, not every epoch.
        expected_state = payload.get("model_state_dict")
        found_state = existing.get("model_state_dict")
        if not isinstance(expected_state, Mapping) or not isinstance(found_state, Mapping):
            raise RuntimeError("LOCKED_CHECKPOINT_MODEL_STATE_MISSING")
        if set(expected_state) != set(found_state):
            raise RuntimeError("LOCKED_CHECKPOINT_MODEL_STATE_KEYS_MISMATCH")
        for name in expected_state:
            left = expected_state[name]
            right = found_state[name]
            if hasattr(left, "detach") and hasattr(right, "detach"):
                if tuple(left.shape) != tuple(right.shape) or not bool(
                    torch.equal(left.detach().cpu(), right.detach().cpu())
                ):
                    raise RuntimeError("LOCKED_CHECKPOINT_MODEL_STATE_MISMATCH")
            elif left != right:
                raise RuntimeError("LOCKED_CHECKPOINT_MODEL_STATE_MISMATCH")
        if digest_mismatch:
            if not source_contract_migration_allowed(
                {"run_contract": existing.get("run_contract")}, payload.get("run_contract")
            ):
                raise RuntimeError("LOCKED_CHECKPOINT_IDENTITY_MISMATCH:run_contract_digest")
            # Existing locked bytes stay authoritative; nothing is overwritten.
        return
    atomic_torch_save(payload, path)


def preserve_precontinuation_checkpoint(path: Path, completed_epochs: int) -> Path:
    preserved = path.with_name(f"{path.stem}_precontinuation_epoch{completed_epochs}{path.suffix}")
    if preserved.exists():
        return preserved
    try:
        os.link(path, preserved)
    except OSError:
        shutil.copy2(path, preserved)
    return preserved


def load_resume_payload(path: Path, device):
    import torch

    try:
        # Resume metadata (especially RNG and DataLoader generator states) must
        # remain CPU tensors while it is validated.  Mapping the whole payload
        # to CUDA here can move byte-state tensors onto the device and makes a
        # subsequent exact replay depend on allocator/device timing.  Model
        # weights are moved explicitly by ``load_state_dict`` at restore time.
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot resume from {path}. The progress file may be from an interrupted non-atomic save. "
            "Move it aside or rerun with --no_resume to restart the current fold."
        ) from exc


def progress_payload(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    epoch: int,
    best_val: float,
    best_epoch: int | None,
    best_model_state: dict | None,
    patience: int,
    train_curve: list[float],
    val_curve: list[float],
    evaluation_curve: list[float | None],
    fold: int,
    fit_idx,
    fit_y,
    val_idx,
    val_y,
    test_idx,
    test_y,
    epoch_seconds: list[float],
    estimated_remaining_seconds: float | None,
    training_continuation: dict | None = None,
    ema_state: dict | None = None,
    swa_state: dict | None = None,
    swa_count: int = 0,
    # The fields below are intentionally optional.  Keeping defaults preserves
    # the legacy H-only checkpoint format while allowing the C staging backend
    # to bind every recovery cursor and digest explicitly.
    omit_best_state: bool = False,
    best_averaged_state: dict | None = None,
    best_averaged_source: str | None = None,
    optimizer_step_count: int = 0,
    checkpoint_generation: int = 0,
    epoch_batch_cursor: int = 0,
    sampler_cursor: int | None = None,
    runtime_resume_contract: Mapping[str, Any] | None = None,
    storage_policy_digest_value: str | None = None,
    fold_membership_digest: str | None = None,
    rng_state: Mapping[str, Any] | None = None,
    loader_generator_state: Any | None = None,
    train_generator: Any | None = None,
    sampler_epoch_generator_state: Any | None = None,
    fold_terminal: bool = False,
    epoch_loss_sum: float = 0.0,
    epoch_loss_count: int = 0,
    run_contract: Mapping[str, Any] | None = None,
) -> dict:
    # A negative cursor is never meaningful and, if serialized, can make a
    # resumed sampler silently skip or repeat data.  Reject it at the payload
    # boundary rather than relying on the staging layer to notice later.
    counters = {
        "optimizer_step_count": optimizer_step_count,
        "checkpoint_generation": checkpoint_generation,
        "epoch_batch_cursor": epoch_batch_cursor,
        "sampler_cursor": epoch_batch_cursor if sampler_cursor is None else sampler_cursor,
        "epoch_loss_count": epoch_loss_count,
    }
    for name, value in counters.items():
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    effective_sampler_cursor = int(counters["sampler_cursor"])
    contract_mapping = run_contract if isinstance(run_contract, Mapping) else None
    scientific_digest = (
        scientific_contract_digest(contract_mapping) if contract_mapping is not None else None
    )
    runtime_digest = (
        runtime_resume_contract.get("digest")
        if isinstance(runtime_resume_contract, Mapping)
        else None
    )
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "ema_state_dict": ema_state if float(getattr(args, "ema_decay", 0.0)) > 0.0 else None,
        "swa_state_dict": swa_state if bool(getattr(args, "swa", False)) else None,
        "swa_count": int(swa_count),
        "fusion": args.fusion,
        "backbone": args.backbone,
        "attention": getattr(args, "attention", "none"),
        "alignment_head": getattr(args, "alignment_head", "linear"),
        "alignment_head_mode": getattr(args, "alignment_head_mode", "linear"),
        "tab_encoder": getattr(args, "tab_encoder", "linear"),
        "causal_adjustment": getattr(args, "causal_adjustment", "none"),
        "q2_channel_gate": getattr(args, "q2_channel_gate", "none"),
        "hybrid_variant": getattr(args, "hybrid_variant", "baseline"),
        "hybrid_attention_heads": int(getattr(args, "hybrid_attention_heads", 2)),
        "hybrid_attention_window_size": int(getattr(args, "hybrid_attention_window_size", 4)),
        "input_mode": getattr(args, "input_mode", "mri_plus_tabular"),
        "drop_tabular_feature": str(getattr(args, "drop_tabular_feature", "none")),
        "validity_inner_only": bool(getattr(args, "validity_inner_only", False)),
        "hpo_selection_only": bool(getattr(args, "hpo_selection_only", False)),
        "pretrained_fold_manifest": getattr(args, "pretrained_fold_manifest", "") or None,
        "pretrained_load_reports": list(getattr(args, "_pretrained_load_reports", [])),
        "unfreeze_schedule": getattr(args, "unfreeze_schedule", "none"),
        "unfreeze_epoch": int(getattr(args, "unfreeze_epoch", 8)),
        "unfreeze_transitions": list(getattr(args, "_unfreeze_transitions", [])),
        "latent_dim": args.latent_dim,
        "tab_hidden_dim": args.tab_hidden_dim,
        "dropout": args.dropout,
        "bottleneck_dim": args.bottleneck_dim,
        "fusion_hidden_dim": args.fusion_hidden_dim,
        "n_experts": getattr(args, "n_experts", 4),
        "fusion_num_heads": getattr(args, "fusion_num_heads", 2),
        "moe_load_balance_lambda": float(getattr(args, "moe_load_balance_lambda", 0.0)),
        "restore_best_at_end": bool(getattr(args, "restore_best_at_end", False)),
        "task_loss": getattr(args, "task_loss", "bce"),
        "focal_gamma": float(getattr(args, "focal_gamma", 2.0)),
        "focal_alpha": float(getattr(args, "focal_alpha", -1.0)),
        "auc_margin": float(getattr(args, "auc_margin", 1.0)),
        "auc_loss_weight": float(getattr(args, "auc_loss_weight", 1.0)),
        "ema_decay": float(getattr(args, "ema_decay", 0.0)),
        "swa": bool(getattr(args, "swa", False)),
        "swa_start_fraction": float(getattr(args, "swa_start_fraction", 0.8)),
        "epoch": epoch,
        "resume_from_epoch": epoch,
        "best_val": best_val,
        "best_epoch": best_epoch,
        # Staged payloads must not contain a second copy of the model.  The
        # validation-best state is written to the H sidecar by the caller.
        "best_state_offloaded": bool(omit_best_state),
        "best_averaged_source": best_averaged_source,
        "patience": patience,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "evaluation_curve": evaluation_curve,
        "evaluation_loss_curve": evaluation_curve,
        "fold": fold,
        "fit_idx": list(fit_idx),
        "fit_y": list(fit_y),
        "val_idx": list(val_idx),
        "val_y": list(val_y),
        "test_idx": list(test_idx),
        "test_y": list(test_y),
        "epoch_seconds": epoch_seconds,
        "average_epoch_seconds": (sum(epoch_seconds) / len(epoch_seconds)) if epoch_seconds else None,
        "estimated_remaining_seconds": estimated_remaining_seconds,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_continuation": training_continuation,
        # Runtime/recovery metadata is deliberately outside ``run_contract``;
        # adding it here must not change the scientific contract digest.
        "run_contract": dict(contract_mapping) if contract_mapping is not None else None,
        "run_contract_digest": run_contract["digest"] if isinstance(run_contract, Mapping) and "digest" in run_contract else None,
        "scientific_contract_digest": scientific_digest,
        "runtime_resume_contract": (
            dict(runtime_resume_contract)
            if isinstance(runtime_resume_contract, Mapping)
            else None
        ),
        "runtime_resume_digest": runtime_digest,
        "storage_policy_digest": storage_policy_digest_value,
        "checkpoint_generation": int(checkpoint_generation),
        "last_committed_optimizer_step": int(optimizer_step_count),
        "optimizer_step_count": int(optimizer_step_count),
        "epoch_batch_cursor": int(epoch_batch_cursor),
        "sampler_cursor": effective_sampler_cursor,
        "split_membership_digest": fold_membership_digest,
        "rng_state": capture_rng_state() if rng_state is None else rng_state,
        "loader_generator_state": train_generator.get_state() if train_generator is not None else loader_generator_state,
        "sampler_epoch_generator_state": sampler_epoch_generator_state,
        "fold_terminal": bool(fold_terminal),
        "epoch_loss_sum": float(epoch_loss_sum),
        "epoch_loss_count": int(epoch_loss_count),
    }
    if omit_best_state:
        # Remove the keys entirely (None is not sufficient: the stager rejects
        # key presence to prevent accidental multi-hundred-MB duplication).
        payload.pop("best_model_state_dict", None)
        payload.pop("best_averaged_state_dict", None)
        payload.pop("best_model_state", None)
        payload.pop("best_averaged_state", None)
    else:
        payload["best_model_state_dict"] = best_model_state
        payload["best_averaged_state_dict"] = best_averaged_state
    return payload


def estimate_remaining_seconds(
    *,
    fold: int,
    completed_epoch_in_fold: int,
    folds: int,
    epochs: int,
    epoch_seconds: list[float],
) -> float | None:
    if not epoch_seconds:
        return None
    avg_epoch_seconds = sum(epoch_seconds) / len(epoch_seconds)
    remaining_current_fold = max(epochs - completed_epoch_in_fold, 0)
    remaining_future_folds = max(folds - fold - 1, 0) * epochs
    return float((remaining_current_fold + remaining_future_folds) * avg_epoch_seconds)


def requested_model_signature(args) -> dict[str, object]:
    signature = {
        "fusion": args.fusion,
        "backbone": args.backbone,
        "attention": getattr(args, "attention", "none"),
        "alignment_head": getattr(args, "alignment_head", "linear"),
        "tab_encoder": getattr(args, "tab_encoder", "linear"),
        "causal_adjustment": getattr(args, "causal_adjustment", "none"),
        "q2_channel_gate": getattr(args, "q2_channel_gate", "none"),
        "input_mode": getattr(args, "input_mode", "mri_plus_tabular"),
        "drop_tabular_feature": getattr(args, "drop_tabular_feature", "none"),
        "latent_dim": int(args.latent_dim),
        "tab_hidden_dim": int(args.tab_hidden_dim),
        "dropout": float(args.dropout),
        "bottleneck_dim": int(args.bottleneck_dim),
        "fusion_hidden_dim": args.fusion_hidden_dim,
        "n_experts": int(getattr(args, "n_experts", 4)),
        "fusion_num_heads": int(getattr(args, "fusion_num_heads", 2)),
        "moe_load_balance_lambda": float(getattr(args, "moe_load_balance_lambda", 0.0)),
        "task_loss": getattr(args, "task_loss", "bce"),
        "focal_gamma": float(getattr(args, "focal_gamma", 2.0)),
        "focal_alpha": float(getattr(args, "focal_alpha", -1.0)),
        "auc_margin": float(getattr(args, "auc_margin", 1.0)),
        "auc_loss_weight": float(getattr(args, "auc_loss_weight", 1.0)),
        "ema_decay": float(getattr(args, "ema_decay", 0.0)),
        "swa": bool(getattr(args, "swa", False)),
        "swa_start_fraction": float(getattr(args, "swa_start_fraction", 0.8)),
    }
    # MX-03 validity arms are a distinct, inner-validation-only execution
    # contract.  Keep the key absent for the historical battery so existing
    # finalized model contracts remain byte-compatible.
    if bool(getattr(args, "validity_inner_only", False)):
        signature["validity_inner_only"] = True
    if bool(getattr(args, "hpo_selection_only", False)):
        signature["hpo_selection_only"] = True
    return signature


def payload_model_signature(payload: dict) -> dict[str, object] | None:
    required = [
        "fusion",
        "backbone",
        "attention",
        "input_mode",
        "latent_dim",
        "tab_hidden_dim",
        "dropout",
        "bottleneck_dim",
        "fusion_hidden_dim",
    ]
    if any(key not in payload for key in required):
        return None
    try:
        return {
            "fusion": payload["fusion"],
            "backbone": payload["backbone"],
            "attention": payload["attention"],
            "alignment_head": payload.get("alignment_head", "linear"),
            "alignment_head_mode": payload.get("alignment_head_mode", payload.get("alignment_head", "linear")),
            "tab_encoder": payload.get("tab_encoder", "linear"),
            "causal_adjustment": payload.get("causal_adjustment", "none"),
            "q2_channel_gate": payload.get("q2_channel_gate", "none"),
            "input_mode": payload["input_mode"],
            "drop_tabular_feature": payload.get("drop_tabular_feature", "none"),
            "latent_dim": int(payload["latent_dim"]),
            "tab_hidden_dim": int(payload["tab_hidden_dim"]),
            "dropout": float(payload["dropout"]),
            "bottleneck_dim": int(payload["bottleneck_dim"]),
            "fusion_hidden_dim": payload["fusion_hidden_dim"],
            "n_experts": int(payload.get("n_experts", 4)),
            "fusion_num_heads": int(payload.get("fusion_num_heads", 2)),
            "moe_load_balance_lambda": float(payload.get("moe_load_balance_lambda", 0.0)),
            "task_loss": payload.get("task_loss", "bce"),
            "focal_gamma": float(payload.get("focal_gamma", 2.0)),
            "focal_alpha": float(payload.get("focal_alpha", -1.0)),
            "auc_margin": float(payload.get("auc_margin", 1.0)),
            "auc_loss_weight": float(payload.get("auc_loss_weight", 1.0)),
            "ema_decay": float(payload.get("ema_decay", 0.0)),
            "swa": bool(payload.get("swa", False)),
            "swa_start_fraction": float(payload.get("swa_start_fraction", 0.8)),
            # Historical payloads predate these keys; the defaults preserve
            # byte-compatibility with the finalized battery (validity off,
            # no feature dropped).
            "validity_inner_only": bool(payload.get("validity_inner_only", False)),
            "hpo_selection_only": bool(payload.get("hpo_selection_only", False)),
            "drop_tabular_feature": str(payload.get("drop_tabular_feature", "none")),
        }
    except (TypeError, ValueError):
        return None


def checkpoint_payload_compatible(payload: dict, args) -> tuple[bool, str, dict[str, object] | None]:
    found = payload_model_signature(payload)
    expected = requested_model_signature(args)
    if found is None:
        return False, "missing_model_signature_metadata", None
    # A key absent from a saved payload is a mismatch verdict, never an
    # exception: historical payloads predate newer signature keys (e.g. the
    # MX-03 validity flag) and must SKIP cleanly, not crash the run.
    mismatches = {
        key: {"expected": expected[key], "found": found.get(key)}
        for key in expected
        if expected[key] != found.get(key)
    }
    if mismatches:
        return False, "model_signature_mismatch:" + json.dumps(mismatches, sort_keys=True), found
    return True, "model_signature_match", found


def evaluate_saved_fold(
    ckpt_path: Path,
    *,
    fold: int,
    cache_path: Path,
    split_dir: Path,
    sample_to_idx: dict[str, int],
    ok: set[int],
    per_split_limit: int,
    device,
    args,
    epochs: int,
    include_supplementary: bool,
    include_global_supplementary: bool = False,
) -> tuple[dict, list[int], list[float]]:
    """Reuse a committed exactly-once result; never reopen the outer split."""

    import torch

    del (
        cache_path,
        split_dir,
        sample_to_idx,
        ok,
        per_split_limit,
        device,
        epochs,
        include_supplementary,
        include_global_supplementary,
    )
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    compatible, reason, _ = checkpoint_payload_compatible(payload, args)
    if not compatible:
        raise RuntimeError(f"Checkpoint is not compatible with requested model: {ckpt_path} ({reason})")
    embedded = payload.get("locked_outer_test_evaluation")
    locked_path = ckpt_path.with_name(f"method_x_3d_fold_{fold}_outer_eval.json")
    if not isinstance(embedded, Mapping) or not locked_path.is_file():
        raise RuntimeError("LOCKED_OUTER_TEST_RESULT_MISSING")
    try:
        locked_result = json.loads(locked_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("LOCKED_OUTER_TEST_RESULT_INVALID") from exc
    expected_digest = str(payload.get("run_contract_digest") or "")
    if not expected_digest:
        raise RuntimeError("LOCKED_OUTER_TEST_RESULT_INVALID")
    reused = reuse_committed_outer_result(
        payload,
        locked_result,
        expected_fold=fold,
        expected_contract_digest=expected_digest,
    )
    return dict(reused["metrics"]), list(reused["y_true"]), list(reused["y_prob"])


def dataloader_throughput_kwargs(args) -> dict:
    """Throughput-only DataLoader knobs shared by every Method X loader.

    Single call-site for the canonical, unit-tested resolver: sample order,
    augmentation, seeds, and fold membership are never affected. Raises
    fail-closed on invalid knob combinations.
    """
    return resolve_dataloader_kwargs(
        num_workers=int(getattr(args, "num_workers", 0) or 0),
        pin_memory=bool(getattr(args, "pin_memory", False)),
        persistent_workers=bool(getattr(args, "persistent_workers", False)),
        prefetch_factor=int(getattr(args, "prefetch_factor", 2) or 0),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cached/t1_volume_cache_128.h5")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out_dir", default="results/final_results/three_d")
    ap.add_argument("--phase", choices=["dev", "final"], default="dev")
    ap.add_argument(
        "--validity_inner_only",
        action="store_true",
        help=(
            "MX-03 validity task mode: use inner validation only and never "
            "construct or evaluate the outer/test loader."
        ),
    )
    ap.add_argument(
        "--hpo_selection_only",
        action="store_true",
        help=(
            "Fine-tuning/HPO selection mode: use fold-internal validation only; "
            "never read, construct, or evaluate an outer-test split."
        ),
    )
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--early_stopping_patience", type=int, default=5)
    ap.add_argument(
        "--restore_best_at_end",
        action="store_true",
        help="Load the best validation-loss epoch into the model before final fold prediction/checkpoint save.",
    )
    ap.add_argument("--fusion", default="cross_attention", choices=available_fusion_modes())
    ap.add_argument("--backbone", default="resnet18_3d", choices=available_3d_backbones())
    ap.add_argument(
        "--hybrid_variant", default="baseline",
        choices=("baseline", "plus_dense_skip", "plus_large_kernel", "plus_attention", "plus_a3", "plus_a4", "full_model", "amuse_x_hybrid_lite"),
    )
    ap.add_argument("--hybrid_attention_heads", type=int, default=2)
    ap.add_argument("--hybrid_attention_window_size", type=int, default=4)
    ap.add_argument("--parameter_budget_max", type=int, default=0)
    ap.add_argument("--attention", default="none", choices=AVAILABLE_ATTENTION)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--tab_hidden_dim", type=int, default=32)
    ap.add_argument("--alignment_head", choices=["linear", "linear_stack", "mlp"], default="linear")
    ap.add_argument("--alignment_head_mode", choices=["linear", "nonlinear"], default="linear")
    ap.add_argument(
        "--tab_encoder",
        choices=["linear", "bn_gelu"],
        default="linear",
        help="B3 optional tabular encoder. linear preserves prior behavior; bn_gelu uses batch-size-1-safe BN+GELU.",
    )
    ap.add_argument("--causal_adjustment", choices=["none", "channel", "roi"], default="none")
    ap.add_argument("--causal_intervention_enabled", action="store_true")
    ap.add_argument(
        "--q2_channel_gate",
        choices=["none", "latent_logistic", "phenotype"],
        default="none",
        help=(
            "AMUSE-X Q2 opt-in channel weighting. latent_logistic gates MRI latents only; "
            "phenotype gates with approved age/sex/education/apoe4/mmse fields."
        ),
    )
    ap.add_argument("--bottleneck_dim", type=int, default=32)
    ap.add_argument("--fusion_hidden_dim", type=int, default=0, help="0 means use latent_dim.")
    ap.add_argument("--n_experts", type=int, default=4, help="Number of MoE experts when --fusion moe is used.")
    ap.add_argument("--fusion_num_heads", type=int, default=2, help="Cross-attention heads when --fusion cross_attention is used.")
    ap.add_argument(
        "--moe_load_balance_lambda",
        type=float,
        default=0.0,
        help="Optional MoE load-balancing auxiliary loss weight; 0 preserves prior behavior.",
    )
    ap.add_argument("--latent_alignment_lambda", type=float, default=0.0)
    ap.add_argument("--contrastive_alignment", action="store_true", help="Use InfoNCE contrastive alignment instead of MSE latent alignment.")
    ap.add_argument(
        "--phenotypic_contrastive_alignment",
        action="store_true",
        help="Use AMUSE-X A2 soft phenotype-positive contrastive alignment with cohort features excluded.",
    )
    ap.add_argument("--phenotypic_contrastive_lambda", type=float, default=0.0)
    ap.add_argument("--phenotypic_temperature", type=float, default=0.1)
    ap.add_argument("--supervised_contrastive_alignment", action="store_true", help="Use supervised contrastive MRI-tabular alignment; off by default.")
    ap.add_argument("--contrastive_temperature", type=float, default=0.07, help="Temperature for contrastive alignment loss.")
    ap.add_argument("--tta", action="store_true", help="Apply test-time augmentation during evaluation.")
    ap.add_argument("--tta_augmentations", type=int, default=5, help="Number of TTA augmentations.")
    ap.add_argument(
        "--tta_mode",
        choices=["basic", "affine", "full"],
        default="basic",
        help="TTA mode: basic=flips+noise, affine=basic plus light affine, full=affine plus light intensity.",
    )
    ap.add_argument("--synib_lambda", type=float, default=0.0)
    ap.add_argument("--modality_dropout", type=float, default=0.5)
    ap.add_argument("--input_mode", default="mri_plus_tabular", choices=ALLOWED_METHOD_X_INPUT_MODES)
    ap.add_argument("--drop_tabular_feature", default="none",
                    choices=("none",) + tuple(METHOD_X_PHENOTYPE_FEATURES),
                    help="Feature-drop ablation: zero one canonical tabular column (fully removed before phenotype derivation).")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--require_gpu",
        action="store_true",
        help="Fail if CUDA is unavailable; training must run on GPU (recommended for final phase).",
    )
    ap.add_argument(
        "--validate_claim_strict",
        action="store_true",
        help="Block artifact publication when the callable AMUSE-X A6 claim policy reports blockers.",
    )
    ap.add_argument(
        "--claim_scope",
        default="amyloid biomarker / preclinical risk only",
        help="Reader-facing claim scope; diagnosis/dementia framing is blocked by AMUSE-X A6.",
    )
    ap.add_argument(
        "--framing_language",
        default="MRI-only amyloid biomarker / preclinical risk estimate",
        help="Reader-facing framing text validated before artifact publication.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional per-split cap for CPU smoke.")
    ap.add_argument(
        "--throughput_profile_only",
        action="store_true",
        help=(
            "Disposable one-fold inner-only throughput measurement. Skips the "
            "cohort probe and global run-manifest write; never valid selection evidence."
        ),
    )
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False,
                    help="Throughput-only: page-locked host staging for device transfer.")
    ap.add_argument("--no-pin_memory", dest="pin_memory", action="store_false",
                    help="Throughput-only: disable page-locked host staging.")
    ap.add_argument("--persistent_workers", dest="persistent_workers", action="store_true", default=False,
                    help="Throughput-only: keep DataLoader workers alive across epochs (effective only with num_workers>0).")
    ap.add_argument("--no-persistent_workers", dest="persistent_workers", action="store_false",
                    help="Throughput-only: recycle DataLoader workers every epoch.")
    ap.add_argument("--prefetch_factor", type=int, default=2,
                    help="Throughput-only: batches prefetched per worker (effective only with num_workers>0).")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps for larger effective batch size.")
    ap.add_argument("--warmup_epochs", type=int, default=5, help="Linear LR warmup epochs before cosine annealing.")
    ap.add_argument("--scheduler", choices=["cosine", "step", "onecycle", "none"], default="cosine")
    ap.add_argument("--step_size", type=int, default=10)
    ap.add_argument("--step_gamma", type=float, default=0.5)
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor (0.0 = no smoothing).")
    ap.add_argument(
        "--task_loss",
        choices=SUPPORTED_TASK_LOSSES,
        default="bce",
        help="Primary train/inner-validation task loss. Default bce preserves existing behavior.",
    )
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma for --task_loss focal.")
    ap.add_argument(
        "--focal_alpha",
        type=float,
        default=-1.0,
        help="Optional focal alpha in [0,1]; negative disables class alpha weighting.",
    )
    ap.add_argument(
        "--auc_margin",
        type=float,
        default=1.0,
        help="Margin/scale for dependency-free --task_loss auc_pairwise and bce_auc.",
    )
    ap.add_argument(
        "--auc_loss_weight",
        type=float,
        default=1.0,
        help="Pairwise AUC surrogate weight when --task_loss bce_auc.",
    )
    ap.add_argument("--pos_weight", default="auto", help="Positive-class weight: auto, none, or a numeric scalar.")
    ap.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="Enable EMA weight averaging when >0; default 0 disables EMA.",
    )
    ap.add_argument("--swa", action="store_true", help="Enable opt-in SWA over the last training fraction.")
    ap.add_argument(
        "--swa_start_fraction",
        type=float,
        default=0.8,
        help="Fraction of epochs after which SWA starts; 0.8 averages the last 20 percent.",
    )
    ap.add_argument("--checkpoint", action="store_true", help="Use gradient checkpointing to reduce VRAM usage.")
    ap.add_argument("--freeze_encoder", action="store_true", help="Freeze all MRI encoder parameters before training.")
    ap.add_argument("--freeze_until", default="", help="Freeze encoder parameters until a child index or name marker.")
    ap.add_argument("--gradual_unfreeze", action="store_true", help="Unfreeze encoder child blocks progressively across epochs.")
    ap.add_argument(
        "--unfreeze_schedule",
        choices=["none", "head_8_then_last_stage", "full_after_warmup"],
        default="none",
        help="Prespecified MX-04 encoder unfreeze schedule.",
    )
    ap.add_argument(
        "--unfreeze_epoch",
        type=int,
        default=8,
        help="Zero-based epoch at which the prespecified MX-04 unfreeze transition occurs.",
    )
    ap.add_argument("--layerwise_lr_decay", type=float, default=1.0)
    ap.add_argument("--discriminative_lr", action="store_true")
    ap.add_argument(
        "--wd_exclude_bias_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude bias and normalization parameters from AdamW weight decay.",
    )
    ap.add_argument("--monai_augment", action="store_true", help="Apply train-fold-only MONAI augmentation to T1 MRI.")
    ap.add_argument(
        "--aug_prob_scale",
        type=float,
        default=0.2,
        help="Base probability for train-only MONAI augmentation; 0.2 preserves the previous hard-coded defaults.",
    )
    ap.add_argument("--augmentation_exposures_per_real", type=int, default=4)
    ap.add_argument(
        "--no_resume",
        action="store_true",
        help="Retrain all folds even when method_x_3d_fold_*.pt checkpoints already exist.",
    )
    ap.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="Allow optimizer/scheduler reset when continuing from a checkpoint without optimizer state.",
    )
    ap.add_argument(
        "--sanity_random_labels",
        action="store_true",
        help="Randomize labels for sanity testing. Disallowed in final phase/reporting.",
    )
    ap.add_argument(
        "--sanity_random_noise_images",
        action="store_true",
        help="Replace MRI volumes with random noise for sanity testing. Disallowed in final phase/reporting.",
    )
    ap.add_argument(
        "--no_supplementary_train",
        action="store_true",
        help="Do not add synthetic/simulated supplementary samples to train folds.",
    )
    ap.add_argument(
        "--include_global_supplementary_train",
        action="store_true",
        help=(
            "Add all global synthetic/simulated samples to every train fold. "
            "Use only for explicit augmentation research runs; final claims are caveated."
        ),
    )
    ap.add_argument(
        "--pretrained_checkpoint",
        default="",
        help="Optional encoder checkpoint for fine-tuning (Method X fold checkpoint or compatible encoder weights).",
    )
    ap.add_argument(
        "--pretrained_fold_manifest",
        default="",
        help="MX-04 manifest mapping each fold to its same-fold baseline checkpoint and provenance sidecar.",
    )
    ap.add_argument(
        "--pretrained_manifest",
        default="",
        help="Provenance sidecar for a single --pretrained_checkpoint.",
    )
    ap.add_argument(
        "--pretrained_sha256",
        default="",
        help="Expected SHA-256 for a single --pretrained_checkpoint.",
    )
    ap.add_argument(
        "--pretrained_min_coverage",
        type=float,
        default=0.95,
        help="Minimum compatible encoder state-key coverage; MX-04 requires at least 0.95.",
    )
    ap.add_argument(
        "--require_pretrained_provenance",
        action="store_true",
        help="Fail closed unless pretrained weights have verified checkpoint, fold, backbone, hash, and license provenance.",
    )
    ap.add_argument(
        "--finetune_lr",
        type=float,
        default=5e-5,
        help="Learning rate when --pretrained_checkpoint is set; otherwise 1e-4.",
    )
    ap.add_argument(
        "--max_runtime_minutes",
        type=float,
        default=0.0,
        help=(
            "Stop gracefully after this many minutes, checked after each completed epoch. "
            "The next run resumes from the saved fold progress."
        ),
    )
    ap.add_argument("--hparam_manifest", default="results/final_results/three_d_hparam_search/three_d_hparam_search_summary.json")
    ap.add_argument("--use_hparam_manifest", action="store_true")
    ap.add_argument("--allow_hparam_backbone_override", action="store_true")
    ap.add_argument(
        "--cache_sha256",
        default="",
        help="Locked SHA-256 of the MRI cache bound into the exact-resume run contract.",
    )
    ap.add_argument(
        "--spm_snapshot_sha256",
        default="",
        help="Locked SPM snapshot digest bound into the exact-resume run contract.",
    )
    ap.add_argument(
        "--require_exact_resume_contract",
        action="store_true",
        help="Fail closed on any scientific contract drift before training I/O.",
    )
    ap.add_argument(
        "--contract_preflight_only",
        action="store_true",
        help="Read-only run-contract decision; never migrates, trains, or writes artifacts.",
    )
    # Resume/storage controls are runtime policy, not scientific knobs.  They
    # are registered in one shared helper so the runner and direct trainer
    # invocations cannot silently disagree about the allowlist.
    add_staging_arguments(ap)
    ap.add_argument(
        "--storage_policy",
        default="configs/policy/method_x_storage_locality_v1.yaml",
        help="Storage-locality policy used only when checkpoint staging is enabled.",
    )
    ap.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Rejected for the locked Method X protocol; use a new exploratory run.",
    )
    ap.add_argument(
        "--torch_compile",
        action="store_true",
        help="Rejected for the locked Method X protocol; use a new exploratory run.",
    )
    args = ap.parse_args()
    reject_method_x_nondeterministic_acceleration(args)
    if bool(args.validity_inner_only) and bool(args.hpo_selection_only):
        raise SystemExit("--validity_inner_only and --hpo_selection_only are mutually exclusive")
    if bool(args.validity_inner_only) and args.phase != "dev":
        raise SystemExit("--validity_inner_only is restricted to --phase dev")
    if bool(args.throughput_profile_only) and not (
        bool(args.hpo_selection_only)
        and args.phase == "dev"
        and int(args.folds) == 1
        and int(args.epochs) == 1
        and int(args.limit) > 0
    ):
        raise SystemExit(
            "--throughput_profile_only requires --hpo_selection_only --phase dev "
            "--folds 1 --epochs 1 and a positive --limit"
        )
    if int(args.unfreeze_epoch) < 0:
        raise SystemExit("--unfreeze_epoch must be non-negative")
    if args.backbone == "amuse_x_hybrid3d":
        if int(args.parameter_budget_max) != 20_000_000:
            raise SystemExit("Hybrid requires the locked 20M parameter ceiling")
        if args.hybrid_attention_heads != 2 or args.hybrid_attention_window_size != 4:
            raise SystemExit("Hybrid attention settings must match the locked protocol")
    elif args.hybrid_variant != "baseline" or args.parameter_budget_max:
        raise SystemExit("Hybrid-only flags require --backbone amuse_x_hybrid3d")
    if args.unfreeze_schedule != "none" and not args.freeze_encoder:
        raise SystemExit("--unfreeze_schedule requires --freeze_encoder")
    if args.pretrained_fold_manifest and args.pretrained_checkpoint:
        raise SystemExit("Use exactly one of --pretrained_fold_manifest or --pretrained_checkpoint")
    if args.require_pretrained_provenance and not (
        args.pretrained_fold_manifest or args.pretrained_checkpoint
    ):
        raise SystemExit("--require_pretrained_provenance requires pretrained weights")
    if not 0.0 <= float(args.pretrained_min_coverage) <= 1.0:
        raise SystemExit("--pretrained_min_coverage must be in [0, 1]")
    if args.pretrained_fold_manifest and float(args.pretrained_min_coverage) < 0.95:
        raise SystemExit("MX-04 fold-local pretrained coverage cannot be below 0.95")
    apply_hparam_manifest(args)
    if not 0.0 <= float(args.ema_decay) < 1.0:
        raise SystemExit("--ema_decay must be in [0, 1). Use 0 to disable EMA.")
    if bool(args.swa) and float(args.ema_decay) > 0.0:
        raise SystemExit("--swa and --ema_decay are mutually exclusive; choose one weight-averaging path.")
    if not 0.0 <= float(args.swa_start_fraction) <= 1.0:
        raise SystemExit("--swa_start_fraction must be between 0 and 1.")
    if not 0.0 <= float(args.auc_loss_weight):
        raise SystemExit("--auc_loss_weight must be non-negative.")
    if not 0.0 < float(args.auc_margin):
        raise SystemExit("--auc_margin must be positive.")
    if not (float(args.focal_alpha) < 0.0 or 0.0 <= float(args.focal_alpha) <= 1.0):
        raise SystemExit("--focal_alpha must be in [0,1] or negative to disable alpha weighting.")
    if float(args.moe_load_balance_lambda) < 0.0:
        raise SystemExit("--moe_load_balance_lambda must be non-negative.")
    if float(args.synib_lambda) > 0.0 and args.task_loss != "bce":
        raise SystemExit("--task_loss values other than bce are not combined with --synib_lambda in this conservative Phase 1 path.")
    if float(args.phenotypic_contrastive_lambda) < 0.0:
        raise SystemExit("--phenotypic_contrastive_lambda must be non-negative.")
    if float(args.phenotypic_temperature) <= 0.0:
        raise SystemExit("--phenotypic_temperature must be positive.")
    if args.q2_channel_gate == "phenotype":
        validate_phenotype_feature_names(("age", "sex", "education_years", "apoe4_carrier", "mmse_base"))
    if float(args.phenotypic_contrastive_lambda) > 0.0:
        validate_phenotype_feature_names(METHOD_X_PHENOTYPE_FEATURES)
    if args.phenotypic_contrastive_alignment:
        args.contrastive_alignment = True
        validate_phenotype_feature_names(METHOD_X_PHENOTYPE_FEATURES)
    if args.supervised_contrastive_alignment:
        args.contrastive_alignment = True
    if (args.sanity_random_labels or args.sanity_random_noise_images) and args.phase == "final":
        raise SystemExit("Sanity random-label/noise modes are disallowed in final reporting mode.")
    if args.backbone == "tiny" and args.phase == "final":
        raise SystemExit("Final 3D training must use a full backbone, not --backbone tiny.")
    if args.epochs < 40 and args.phase == "final":
        raise SystemExit("Final 3D training requires --epochs >= 40.")
    if args.fusion_hidden_dim <= 0:
        args.fusion_hidden_dim = None
    if args.alignment_head_mode == "nonlinear" and args.alignment_head == "linear":
        args.alignment_head = "mlp"
    elif args.alignment_head_mode == "linear":
        args.alignment_head = "linear"
    if args.causal_intervention_enabled:
        args.causal_adjustment = "channel"

    selection_only = is_selection_only_run(args)
    args._validated_pretrained_fold_manifest = None
    if args.pretrained_fold_manifest:
        args._validated_pretrained_fold_manifest = validate_pretrained_fold_manifest(
            args.pretrained_fold_manifest,
            folds=int(args.folds),
            expected_backbone=str(args.backbone),
            allowed_root=PROJECT_ROOT,
            minimum_coverage=max(float(args.pretrained_min_coverage), 0.95),
        )

    cache_path = PROJECT_ROOT / args.cache
    if not cache_path.exists():
        report = {"status": "BLOCKED", "blocker": "MRI_CACHE_MISSING", "cache": str(cache_path)}
        out_dir = PROJECT_ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "three_d_metrics.json", {"aggregate": report})
        print(json.dumps(report, indent=2))
        return 2
    args._input_policy = validate_method_x_input_policy(args, cache_path)

    # ---- Exact-resume run-contract gate (must precede any training I/O) ----
    seed_everything(int(args.seed))
    args._determinism_runtime = enforce_method_x_strict_determinism()
    device_obj = resolve_training_device(str(getattr(args, "device", "auto")))
    _mode_tuple = resolve_mode(device_obj, args.phase, args.limit, require_gpu=args.require_gpu)
    sample_to_idx = cache_index(cache_path)
    ok = ok_indices(cache_path)
    splits_dir = PROJECT_ROOT / args.splits
    split_digest = method_x_split_contract_digest(
        splits_dir,
        folds=int(args.folds),
        selection_only=selection_only,
    ) if selection_only else method_x_split_contract_digest(splits_dir)
    source_digest_value, runtime_versions = method_x_training_source_contract()
    loader_contract = {
        "num_workers": int(getattr(args, "num_workers", 0)),
        "batch_size": int(args.batch_size),
        "pin_memory": bool(getattr(args, "pin_memory", False)),
        "persistent_workers": bool(getattr(args, "persistent_workers", False)),
        "prefetch_factor": int(getattr(args, "prefetch_factor", 2) or 0),
        # These fields were part of the issued v3 contract.  They describe
        # deterministic loader plumbing and are retained for exact resume
        # compatibility; they are not user-controlled data or paths.
        "non_blocking_transfer": True,
        "train_generator": "explicit_per_fold_torch_generator",
        "worker_init_fn": "src.utils.reproducibility.seed_worker",
    }
    parameters_contract = build_method_x_contract_parameters(
        args,
        source_contract_digest=source_digest_value,
        runtime_versions=runtime_versions,
    )
    run_contract = build_run_contract(
        cache_sha256=str(args.cache_sha256 or ""),
        split_digest=split_digest,
        folds=int(args.folds),
        epochs=int(args.epochs),
        seed=int(args.seed),
        parameters=parameters_contract,
        loader=loader_contract,
        spm_snapshot_sha256=str(args.spm_snapshot_sha256 or "") or None,
    )
    # Source/runtime provenance is included *before* the digest is computed.
    # ``scientific_contract_digest`` projects these fields out, so a source-only
    # repair remains auditable without weakening scientific drift checks.
    out_dir = PROJECT_ROOT / args.out_dir
    if bool(args.contract_preflight_only):
        decision = inspect_exact_run_contract(
            out_dir,
            run_contract=run_contract,
            require_exact_resume_contract=bool(args.require_exact_resume_contract),
        )
        print(
            json.dumps(
                {
                    "status": "CONTRACT_PREFLIGHT_PASS",
                    "decision": decision,
                    "runtime": runtime_versions,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    ensure_exact_run_contract(
        out_dir,
        run_contract=run_contract,
        require_exact_resume_contract=bool(args.require_exact_resume_contract),
    )
    # Runtime resume policy is bound separately from the scientific contract.
    # This lets the official runner use the fast C staging path without
    # invalidating the already-issued Method X experiment digest.
    staging_enabled = bool(str(getattr(args, "checkpoint_staging_root", "") or "").strip())
    runtime_resume_contract = build_runtime_resume_contract(
        staging_root_class=(
            "C_ALLOWLISTED_CHECKPOINT_STAGING" if staging_enabled else "H_ONLY"
        ),
        full_cadence=(
            int(getattr(args, "checkpoint_full_cadence", 1)) if staging_enabled else 1
        ),
        h_mirror_policy=str(getattr(args, "checkpoint_h_mirror", "epoch")),
        journal_enabled=staging_enabled,
        replay_policy=str(getattr(args, "resume_replay_policy", "deterministic")),
    )
    storage_policy_path = Path(str(getattr(args, "storage_policy", "")))
    if not storage_policy_path.is_absolute():
        storage_policy_path = PROJECT_ROOT / storage_policy_path
    storage_policy_digest_value = (
        storage_policy_digest(storage_policy_path) if staging_enabled else None
    )
    # ---------------- end exact-resume contract gate ----------------

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception:
        report = {"status": "BLOCKED", "blocker": "PYTORCH_MISSING"}
        out_dir = PROJECT_ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "three_d_metrics.json", {"aggregate": report})
        print(json.dumps(report, indent=2))
        return 1

    device = resolve_method_x_training_device(args.device, require_gpu=args.require_gpu)
    mode, epoch_cap, fold_cap, default_limit = resolve_mode(
        device, args.phase, args.limit, require_gpu=args.require_gpu
    )
    epochs = epoch_cap or args.epochs
    folds = fold_cap or args.folds
    per_split_limit = args.limit or default_limit

    sample_to_idx = cache_index(cache_path)
    ok = ok_indices(cache_path)
    split_dir = PROJECT_ROOT / args.splits
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    claim_preflight = validate_claim(
        {
            "method": "X",
            "phase": args.phase,
            "resolution": args.resolution,
            "folds": args.folds,
            "uses_pet_as_input": False,
            "claim_scope": args.claim_scope,
            "framing_language": args.framing_language,
            "spm_complete": True,
            "validity_recommendation": "CLAIM_ALLOWED",
            "enforce_required_fields": False,
        },
        strict=False,
    )
    if not claim_preflight["ok"]:
        blocked = {
            "status": "BLOCKED",
            "blocker": "CLAIM_POLICY_PREFLIGHT_VIOLATION",
            "claim_policy_report": claim_preflight,
        }
        atomic_write_json(out_dir / "BLOCKED_RUN.json", blocked)
        print(json.dumps(blocked, indent=2))
        return 3

    synib_default = SynIBConfig()
    synib_cfg = SynIBConfig(
        lambda_syn=args.synib_lambda,
        mask_prob=args.modality_dropout,
        tabular_dim=5,
        mri_dim=synib_default.mri_dim,
    )
    use_synib = args.synib_lambda > 0.0
    use_amp = args.amp and device.type == "cuda"
    if args.supervised_contrastive_alignment:
        contrastive_loss_fn = SupervisedContrastiveAlignmentLoss(temperature=args.contrastive_temperature)
    else:
        contrastive_loss_fn = (
            ContrastiveAlignmentLoss(
                temperature=args.contrastive_temperature,
                phenotypic_soft_positives=args.phenotypic_contrastive_alignment,
            )
            if args.contrastive_alignment
            else None
        )
    include_supplementary = not args.no_supplementary_train and cache_has_supplementary(cache_path)
    supplementary_ok_n = 0
    if include_supplementary:
        supp_idx, _ = supplementary_context(cache_path, ok)
        supplementary_ok_n = len(supp_idx)
    pretrained_checkpoint = args.pretrained_checkpoint.strip() or None
    pretrained_enabled = bool(pretrained_checkpoint or args.pretrained_fold_manifest)
    base_lr = args.finetune_lr if pretrained_enabled else args.lr

    fold_metrics: list[dict] = []
    all_y: list[int] = []
    all_p: list[float] = []
    all_calibrated_p: list[float] = []
    temperature_rows: list[dict] = []
    checkpoints: list[str] = []
    threshold_rows: list[dict] = []
    selection_evidence_folds: list[dict[str, Any]] = []
    args._pretrained_load_reports = []
    args._unfreeze_transitions = []
    args._encoder_stage_map = None

    cache_resolution = args.resolution
    summary_path = PROJECT_ROOT / "results" / "tables" / "mri_volume_cache_summary.json"
    if summary_path.exists():
        cache_resolution = json.loads(summary_path.read_text(encoding="utf-8")).get("resolution", args.resolution)
    try:
        import h5py

        with h5py.File(cache_path, "r") as h5:
            if "resolution" in h5.attrs:
                cache_resolution = int(h5.attrs["resolution"])
    except Exception:
        pass

    run_started = time.monotonic()
    max_runtime_seconds = args.max_runtime_minutes * 60.0 if args.max_runtime_minutes > 0 else None
    # The locked Method X contract records a total cross-validation budget.
    # Exact-contract runs divide that budget evenly across folds while keeping
    # ``epochs`` unchanged in the run contract for digest compatibility.  Ad-hoc
    # runs without the exact-contract gate retain the historical per-fold CLI
    # interpretation.
    epochs_per_fold = (
        protocol_epochs_per_fold(epochs, folds)
        if bool(args.require_exact_resume_contract)
        else int(epochs)
    )

    def relative_or_str(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    def write_pause_marker(
        *,
        status: str,
        reason: str,
        current_fold: int,
        resume_from_epoch: int,
        progress_path: Path,
        estimated_remaining: float | None,
    ) -> None:
        partial_aggregate = build_aggregate(
            mode=mode,
            args=args,
            device=device,
            epochs=epochs,
            folds=folds,
            use_synib=use_synib,
            use_amp=use_amp,
            cache_path=cache_path,
            cache_resolution=cache_resolution,
            fold_metrics=fold_metrics,
            all_y=all_y,
            all_p=all_p,
            checkpoints=checkpoints,
            threshold_rows=threshold_rows,
            include_supplementary_train=include_supplementary,
            supplementary_ok_n=supplementary_ok_n,
        )
        partial_aggregate.update(
            {
                "status": status,
                "resume": {
                    "reason": reason,
                    "current_fold": current_fold,
                    "resume_from_epoch": resume_from_epoch,
                    "progress_path": relative_or_str(progress_path),
                    "completed_fold_count": len(checkpoints),
                    "rerun_same_command_without_no_resume": True,
                    "argv": sys.argv,
                },
                "estimated_remaining_seconds": estimated_remaining,
            }
        )
        write_metrics_payload(out_dir, partial_aggregate, fold_metrics, status=status)
        write_run_manifest(
            "results/logs/train_3d_mri_manifest.json",
            stage="train-3d",
            data_root=None,
            extra={"aggregate": partial_aggregate},
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "reason": reason,
                    "current_fold": current_fold,
                    "resume_from_epoch": resume_from_epoch,
                    "progress_path": relative_or_str(progress_path),
                    "estimated_remaining_seconds": estimated_remaining,
                },
                indent=2,
            ),
            flush=True,
        )

    for fold in range(folds):
        ckpt_path = out_dir / f"method_x_3d_fold_{fold}.pt"
        progress_path = out_dir / f"method_x_3d_fold_{fold}_progress.pt"
        fold_state_path = out_dir / f"method_x_3d_fold_{fold}_state.json"
        locked_outer_path = out_dir / f"method_x_3d_fold_{fold}_outer_eval.json"
        locked_checkpoint_path = out_dir / f"method_x_3d_fold_{fold}_locked.pt"
        progress = None
        completed_payload = None
        completed_epochs = 0
        fold_state = read_fold_state(fold_state_path)
        if ckpt_path.exists() and not args.no_resume:
            import torch

            # Keep resume metadata (including RNG byte tensors) on CPU.  Model
            # tensors are moved by ``load_state_dict`` only when a fold truly
            # needs continuation.
            completed_payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            compatible, reason, found_signature = checkpoint_payload_compatible(completed_payload, args)
            if not compatible:
                print(
                    json.dumps(
                        {
                            "status": "SKIP_INCOMPATIBLE_COMPLETED_CHECKPOINT",
                            "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
                            "reason": reason,
                            "found_signature": found_signature,
                            "requested_signature": requested_model_signature(args),
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                completed_payload = None
                completed_epochs = 0
            else:
                completed_epochs = int(completed_payload.get("epochs_run") or 0)
            # A terminal fold must never reopen the outer loader.  Prefer the
            # immutable embedded result and validate/reuse it; only legacy
            # checkpoints with no committed outer payload may use the old
            # evaluator, and only when the fold state proves that outer access
            # has not already happened.
            if completed_epochs > epochs_per_fold:
                raise RuntimeError(
                    "COMPLETED_CHECKPOINT_EPOCH_EXCEEDS_PROTOCOL_CAP:"
                    f"fold={fold}:completed={completed_epochs}:cap={epochs_per_fold}"
                )
            if completed_payload is not None and (
                completed_epochs >= epochs_per_fold
                or (fold_state is not None and fold_state.get("stage") == "FINALIZED")
            ):
                if selection_only:
                    embedded_selection = completed_payload.get("selection_only_evaluation")
                    if not isinstance(embedded_selection, Mapping):
                        raise RuntimeError("COMMITTED_SELECTION_EVALUATION_MISSING_FOR_TERMINAL_FOLD")
                    # A finalized checkpoint is immutable, so after a legitimate
                    # source-only migration it still carries the historical
                    # contract digest that produced it.  Accept that only when the
                    # append-only source_history chain proves a unique
                    # source-only path to the current contract; a scientific
                    # contract drift is never waived.
                    embedded_digest = str(embedded_selection.get("run_contract_digest") or "")
                    checkpoint_digest = str(completed_payload.get("run_contract_digest") or "")
                    state_split_digest = (
                        (fold_state.get("details") or {}).get("split_membership_digest")
                        if isinstance(fold_state, Mapping) else None
                    )
                    embedded_contract_bound = embedded_digest == run_contract["digest"] or (
                        source_contract_migration_allowed(completed_payload, run_contract)
                        and source_history_binds_digest(
                            history=(fold_state or {}).get("source_history"),
                            from_digest=embedded_digest,
                            to_digest=run_contract["digest"],
                        )
                    )
                    if (
                        embedded_selection.get("fold") != fold
                        or embedded_digest != checkpoint_digest
                        or not embedded_contract_bound
                        or embedded_selection.get("outer_test_access_count") != 0
                        or embedded_selection.get("outer_test_used_for_selection") is not False
                        or not isinstance(embedded_selection.get("metrics"), Mapping)
                        or not isinstance(embedded_selection.get("y_true"), list)
                        or not isinstance(embedded_selection.get("y_prob"), list)
                        or not isinstance(embedded_selection.get("logits"), list)
                        or not isinstance(embedded_selection.get("split_membership_digest"), str)
                        or embedded_selection.get("split_membership_digest") != state_split_digest
                    ):
                        raise RuntimeError("COMMITTED_SELECTION_EVALUATION_INVALID")
                    metrics = dict(embedded_selection["metrics"])
                    y_true = list(embedded_selection["y_true"])
                    y_prob = list(embedded_selection["y_prob"])
                    selection_evidence_folds.append(
                        {
                            "fold": fold,
                            "split_membership_digest": embedded_selection[
                                "split_membership_digest"
                            ],
                            "y_true": y_true,
                            "y_prob": y_prob,
                            "logits": list(embedded_selection["logits"]),
                        }
                    )
                else:
                    reused_outer = None
                    embedded_outer = completed_payload.get("locked_outer_test_evaluation")
                    if isinstance(embedded_outer, Mapping):
                        reused_outer = reuse_committed_outer_result(
                            completed_payload,
                            embedded_outer,
                            expected_fold=fold,
                            expected_contract_digest=run_contract["digest"],
                        )
                    elif fold_state is not None and fold_state.get("stage") in {
                        "FINALIZED",
                        "OUTER_EVAL_DONE",
                    }:
                        raise RuntimeError("COMMITTED_OUTER_EVALUATION_MISSING_FOR_TERMINAL_FOLD")
                    if reused_outer is not None:
                        metrics = dict(reused_outer["metrics"])
                        y_true = list(reused_outer["y_true"])
                        y_prob = list(reused_outer["y_prob"])
                    else:
                        metrics, y_true, y_prob = evaluate_saved_fold(
                            ckpt_path,
                            fold=fold,
                            cache_path=cache_path,
                            split_dir=split_dir,
                            sample_to_idx=sample_to_idx,
                            ok=ok,
                            per_split_limit=per_split_limit,
                            device=device,
                            args=args,
                            epochs=epochs_per_fold,
                            include_supplementary=include_supplementary,
                            include_global_supplementary=args.include_global_supplementary_train,
                        )
                saved_pretrained_report = completed_payload.get("pretrained_load_report")
                if isinstance(saved_pretrained_report, Mapping):
                    args._pretrained_load_reports.append(dict(saved_pretrained_report))
                for transition in completed_payload.get("unfreeze_transitions") or []:
                    if isinstance(transition, Mapping):
                        args._unfreeze_transitions.append(dict(transition))
                fold_metrics.append(metrics)
                all_y.extend(y_true)
                all_p.extend(y_prob)
                checkpoints.append(str(ckpt_path.relative_to(PROJECT_ROOT)))
                partial_aggregate = build_aggregate(
                    mode=mode,
                    args=args,
                    device=device,
                    epochs=epochs,
                    folds=folds,
                    use_synib=use_synib,
                    use_amp=use_amp,
                    cache_path=cache_path,
                    cache_resolution=cache_resolution,
                    fold_metrics=fold_metrics,
                    all_y=all_y,
                    all_p=all_p,
                    checkpoints=checkpoints,
                    threshold_rows=threshold_rows,
                    include_supplementary_train=include_supplementary,
                    supplementary_ok_n=supplementary_ok_n,
                )
                write_metrics_payload(
                    out_dir,
                    partial_aggregate,
                    fold_metrics,
                    # A fully reused run must report COMPLETED, otherwise the
                    # orchestrator's inner-metric gate rejects the artifact as
                    # HYBRID_INNER_METRICS_NOT_COMPLETED.  Mirrors the fresh path.
                    status="IN_PROGRESS" if len(checkpoints) < folds else "COMPLETED",
                )
                continue

        # The C staging source is authoritative for an interrupted fold.  It
        # must be inspected before the legacy H progress file; otherwise a
        # stale H copy can rewind the optimizer and create a divergent branch.
        checkpoint_stager = None
        staged_recovery_plan = None
        if staging_enabled:
            checkpoint_stager = build_stager_from_args(
                args,
                run_id=str(run_contract["digest"])[:16],
                backbone=str(args.backbone),
                fold=fold,
                h_output_dir=out_dir,
                policy_path=storage_policy_path,
            )
            checkpoint_stager.require_preflight()
            if not args.no_resume:
                staged_recovery_plan = checkpoint_stager.plan_recovery()
                if staged_recovery_plan.payload_path is not None:
                    staged_payload = load_resume_payload(
                        staged_recovery_plan.payload_path, device
                    )
                    if staged_recovery_plan.replay_steps != 0:
                        # The official runner currently authorizes cadence=1;
                        # silently ignoring a non-zero replay window would make
                        # the result scientifically non-reproducible.
                        raise RuntimeError(
                            "STAGED_RECOVERY_REPLAY_NOT_ZERO:"
                            f"{staged_recovery_plan.replay_steps}"
                        )
                    compatible, reason, found_signature = checkpoint_payload_compatible(
                        staged_payload, args
                    )
                    if not compatible:
                        raise RuntimeError(
                            "STAGED_RECOVERY_MODEL_SIGNATURE_MISMATCH:"
                            + str(reason)
                        )
                    try:
                        assert_resume_contract(staged_payload, run_contract)
                    except RuntimeError as exc:
                        # A source-only hop may leave the C payload carrying an
                        # older contract.  It is admissible only when the
                        # scientific projection is identical; no broad mismatch
                        # waiver is permitted here.
                        if not source_contract_migration_allowed(
                            staged_payload, run_contract
                        ):
                            raise
                    progress = staged_payload
                    progress["_staged_recovery_source"] = str(
                        staged_recovery_plan.source
                    )
                    progress["_staged_recovery_verified"] = False

        # If a valid C-staging payload was found above it is authoritative.  A
        # legacy H progress file may be older (or may reflect a different
        # generation after a source-only hop); loading it unconditionally would
        # silently rewind an exact-resume run.  Only consult H when no staged
        # candidate exists, and reject a contradictory H payload rather than
        # choosing one heuristically.
        if progress is None and progress_path.exists() and not args.no_resume:
            candidate_progress = load_resume_payload(progress_path, device)
            compatible, reason, found_signature = checkpoint_payload_compatible(candidate_progress, args)
            if not compatible:
                print(
                    json.dumps(
                        {
                            "status": "SKIP_INCOMPATIBLE_PROGRESS_CHECKPOINT",
                            "progress_checkpoint": str(progress_path.relative_to(PROJECT_ROOT)),
                            "reason": reason,
                            "found_signature": found_signature,
                            "requested_signature": requested_model_signature(args),
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                progress = None
            else:
                progress = candidate_progress
        if progress is not None:
            progress_resume_epoch = int(progress.get("epoch", 0))
            continuation = {
                "type": "exact_resume_from_progress_checkpoint",
                "source_progress": str(progress_path.relative_to(PROJECT_ROOT)),
                "resume_from_epoch": progress_resume_epoch,
                "optimizer_state_resumed": bool(progress.get("optimizer_state_dict")),
                "scheduler_state_resumed": bool(progress.get("scheduler_state_dict")),
                "exact_optimizer_resume": bool(
                    progress.get("optimizer_state_dict") and progress.get("scheduler_state_dict")
                ),
            }
            if completed_payload is not None and 0 < completed_epochs < progress_resume_epoch:
                continuation["parent_continuation"] = {
                    "type": "fine_tune_continuation_from_completed_checkpoint",
                    "source_checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
                    "source_checkpoint_epochs": completed_epochs,
                    "resume_from_epoch": completed_epochs,
                    "optimizer_state_resumed": bool(completed_payload.get("optimizer_state_dict")),
                    "scheduler_state_resumed": bool(completed_payload.get("scheduler_state_dict")),
                    "exact_optimizer_resume": bool(
                        completed_payload.get("optimizer_state_dict")
                        and completed_payload.get("scheduler_state_dict")
                    ),
                    "note": "Legacy completed fold checkpoint did not include full optimizer/scheduler state; subsequent progress checkpoint resumes exact state from the continuation run.",
                }
            progress.setdefault("training_continuation", continuation)
        elif completed_payload is not None:
            completed_has_optimizer = bool(completed_payload.get("optimizer_state_dict"))
            completed_has_scheduler = bool(completed_payload.get("scheduler_state_dict"))
            progress = {
                "model_state_dict": completed_payload["model_state_dict"],
                "optimizer_state_dict": completed_payload.get("optimizer_state_dict"),
                "scheduler_state_dict": completed_payload.get("scheduler_state_dict"),
                "scaler_state_dict": completed_payload.get("scaler_state_dict"),
                "epoch": completed_epochs,
                "resume_from_epoch": completed_epochs,
                "best_val": completed_payload.get("best_val_loss", float("inf")),
                "best_epoch": completed_payload.get("best_epoch"),
                "patience": 0,
                "train_curve": list(completed_payload.get("train_loss_curve", [])),
                "val_curve": list(completed_payload.get("val_loss_curve", [])),
                "evaluation_curve": list(
                    completed_payload.get("evaluation_loss_curve")
                    or completed_payload.get("evaluation_curve")
                    or []
                ),
                "epoch_seconds": [],
                "resumed_from_completed_checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
                "training_continuation": {
                    "type": "fine_tune_continuation_from_completed_checkpoint",
                    "source_checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
                    "preserved_source_checkpoint": str(
                        preserve_precontinuation_checkpoint(ckpt_path, completed_epochs).relative_to(PROJECT_ROOT)
                    ),
                    "source_checkpoint_epochs": completed_epochs,
                    "resume_from_epoch": completed_epochs,
                    "optimizer_state_resumed": completed_has_optimizer,
                    "scheduler_state_resumed": completed_has_scheduler,
                    "exact_optimizer_resume": completed_has_optimizer and completed_has_scheduler,
                    "note": "Completed fold checkpoint includes optimizer/scheduler state." if completed_has_optimizer and completed_has_scheduler else "Legacy completed fold checkpoint did not include optimizer/scheduler state; pass --reset_optimizer for explicit reset continuation.",
                },
            }

        train_idx, train_y = resolve_train_indices(
            split_dir / f"fold_{fold}_train.csv",
            sample_to_idx,
            ok,
            cache_path=cache_path,
            include_supplementary=include_supplementary,
            include_global_supplementary=args.include_global_supplementary_train,
            limit=per_split_limit,
        )
        validity_inner_only = bool(getattr(args, "validity_inner_only", False))
        hpo_selection_only = bool(getattr(args, "hpo_selection_only", False))
        selection_only = is_selection_only_run(args)
        if selection_only:
            # MX-03 validity and MX-04/HPO selection runs are both forbidden
            # from reading even the outer CSV labels. Represent the unused
            # split explicitly as empty so no loader can be constructed later.
            test_idx, test_y = (), ()
        if not selection_only:
            test_idx, test_y = split_indices(
                split_dir / f"fold_{fold}_test.csv",
                sample_to_idx,
                ok,
                limit=max(per_split_limit // 4, 8) if per_split_limit else 0,
            )
        if len(train_idx) < 8 or (not selection_only and len(test_idx) < 4):
            continue

        if progress and all(k in progress for k in ("fit_idx", "fit_y", "val_idx", "val_y", "test_idx", "test_y")):
            fit_idx = tuple(int(v) for v in progress["fit_idx"])
            fit_y = tuple(int(v) for v in progress["fit_y"])
            val_idx = tuple(int(v) for v in progress["val_idx"])
            val_y = tuple(int(v) for v in progress["val_y"])
            test_idx = tuple(int(v) for v in progress["test_idx"])
            test_y = tuple(int(v) for v in progress["test_y"])
            val_n = len(val_idx)
        else:
            fit_idx, fit_y, val_idx, val_y = split_fit_val_indices(
                train_idx,
                train_y,
                cache_path=cache_path,
                ok=ok,
                val_fraction=args.val_fraction,
                seed=args.seed,
                fold=fold,
            )
            val_n = len(val_idx)

        fold_membership_digest = split_membership_digest(
            fit=fit_idx,
            validation=val_idx,
            outer_test=test_idx,
        )
        if fold_state is None:
            fold_state = transition_fold_state(
                fold_state_path,
                fold=fold,
                target="TRAINING",
                contract_digest=run_contract["digest"],
                details={"training_started": True},
            )
        elif fold_state.get("stage") not in {"TRAINING"}:
            # A fold that already crossed the checkpoint/outer boundary must
            # be handled by the terminal/reuse path above; never rewind it to
            # TRAINING just because a stale progress file is present.
            if fold_state.get("stage") in {"CHECKPOINT_LOCKED", "OUTER_EVAL_IN_PROGRESS"}:
                outer_action = outer_evaluation_resume_action(
                    str(fold_state.get("stage")),
                    locked_result_exists=locked_outer_path.is_file(),
                )
                if outer_action.startswith("REUSE"):
                    raise RuntimeError("OUTER_EVALUATION_REQUIRES_COMMITTED_RESULT_REUSE")
        tab_mean, tab_std = tabular_stats(cache_path, fit_idx)
        # Keep the sampler stream explicit.  The generator state is stored in
        # every resume payload; worker seeding is likewise deterministic, so a
        # restarted fold cannot silently receive a different sample order.
        train_generator = seeded_torch_generator(int(args.seed) + int(fold))
        val_generator = seeded_torch_generator(int(args.seed) + int(fold) + 1000)
        test_generator = seeded_torch_generator(int(args.seed) + int(fold) + 2000)
        train_loader = DataLoader(
            maybe_wrap_train_dataset(
                H5MethodXDataset(
                    cache_path,
                    fit_idx,
                    fit_y,
                    sanity_random_labels=args.sanity_random_labels,
                    sanity_random_noise_images=args.sanity_random_noise_images,
                    return_phenotype_features=(
                        args.phenotypic_contrastive_alignment
                        or float(args.phenotypic_contrastive_lambda) > 0.0
                    ),
                    input_mode=args.input_mode,
                    drop_tabular_feature=getattr(args, "drop_tabular_feature", "none"),
                    seed=args.seed + fold,
                ),
                enabled=args.monai_augment,
                image_positions=(0,),
                prob=args.aug_prob_scale,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            generator=train_generator,
            worker_init_fn=seed_worker,
            **dataloader_throughput_kwargs(args),
        )
        val_loader = DataLoader(
            H5MethodXDataset(
                cache_path,
                val_idx,
                val_y,
                sanity_random_labels=False,
                sanity_random_noise_images=args.sanity_random_noise_images,
                input_mode=args.input_mode,
                drop_tabular_feature=getattr(args, "drop_tabular_feature", "none"),
                seed=args.seed + fold + 1000,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            generator=val_generator,
            worker_init_fn=seed_worker,
            **dataloader_throughput_kwargs(args),
        )
        # The outer/test loader is deliberately deferred until after the fold
        # checkpoint is locked.  Constructing it here (or evaluating its loss
        # every epoch) would consume the outer test split during training and
        # violate the exactly-once policy.
        test_loader = None

        model = MethodX3DClassifier(
            fusion=args.fusion,
            backbone=args.backbone,
            latent_dim=args.latent_dim,
            tab_hidden_dim=args.tab_hidden_dim,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            fusion_hidden_dim=args.fusion_hidden_dim,
            gradient_checkpointing=args.checkpoint,
            pretrained_checkpoint=None,
            attention=args.attention,
            alignment_head=args.alignment_head,
            alignment_head_mode=args.alignment_head_mode,
            tab_encoder=args.tab_encoder,
            causal_adjustment=args.causal_adjustment,
            n_experts=args.n_experts,
            fusion_num_heads=args.fusion_num_heads,
            moe_load_balance_lambda=args.moe_load_balance_lambda,
            channel_gate=args.q2_channel_gate,
            hybrid_variant=args.hybrid_variant,
            hybrid_attention_heads=args.hybrid_attention_heads,
            hybrid_attention_window_size=args.hybrid_attention_window_size,
        )
        pretrained_report = None
        if args._validated_pretrained_fold_manifest is not None:
            pretrained_report = load_fold_local_encoder_weights(
                model.encoder,
                args._validated_pretrained_fold_manifest,
                fold=fold,
            )
        elif pretrained_checkpoint is not None:
            from src.models.backbones.pretrained_loader import load_encoder_weights

            checkpoint_path = Path(pretrained_checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = PROJECT_ROOT / checkpoint_path
            provenance_path = Path(args.pretrained_manifest) if args.pretrained_manifest else None
            if provenance_path is not None and not provenance_path.is_absolute():
                provenance_path = PROJECT_ROOT / provenance_path
            pretrained_report = load_encoder_weights(
                model.encoder,
                checkpoint_path,
                device="cpu",
                provenance_manifest_path=provenance_path,
                expected_sha256=args.pretrained_sha256 or None,
                min_coverage=float(args.pretrained_min_coverage),
                require_provenance=bool(args.require_pretrained_provenance),
            )
            pretrained_report = {**pretrained_report, "fold": fold, "fold_local": False}
        model.pretrained_load_report = pretrained_report
        if pretrained_report is not None:
            args._pretrained_load_reports.append(dict(pretrained_report))
        model = model.to(device)
        args._frozen_layers = apply_finetuning_controls(model, args)
        if args.unfreeze_schedule != "none":
            args._encoder_stage_map = encoder_stage_map(args.backbone, model.encoder)
        counts = parameter_counts(model)
        if args.backbone == "amuse_x_hybrid3d" and counts["total_parameter_count"] > int(args.parameter_budget_max):
            raise SystemExit("HYBRID_PARAMETER_BUDGET_EXCEEDED")
        args._total_parameter_count = counts["total_parameter_count"]
        args._trainable_parameter_count = counts["trainable_parameter_count"]
        pos_weight = parse_pos_weight(args.pos_weight, fit_y, device)
        optimizer, args._optimizer_manifest = build_optimizer(model, base_lr=base_lr, args=args)
        # Preserve the issued v3 schedule horizon. The append-only epoch-budget
        # amendment changes only the loop cap (20 per fold), not the locked
        # ``epochs=100`` scheduler/run-contract semantics used by completed arms.
        scheduler = build_scheduler(
            optimizer,
            args=args,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            base_lr=base_lr,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_val = float("inf")
        best_epoch = None
        best_model_state = None
        patience = 0
        train_curve: list[float] = []
        val_curve: list[float] = []
        evaluation_curve: list[float | None] = []
        fold_epoch_seconds: list[float] = []
        start_epoch = 0
        if progress is not None:
            training_continuation = progress.get("training_continuation")
            model.load_state_dict(progress["model_state_dict"])
            # Preserve already-applied schedule transitions across a process
            # restart.  The schedule itself is re-applied idempotently below,
            # while this receipt prevents resume from losing provenance or
            # recording the same transition twice.
            for transition in progress.get("unfreeze_transitions") or []:
                if isinstance(transition, Mapping) and dict(transition) not in args._unfreeze_transitions:
                    args._unfreeze_transitions.append(dict(transition))
            start_epoch = int(progress.get("epoch", 0))
            missing_optimizer_state = start_epoch > 0 and not progress.get("optimizer_state_dict")
            missing_scheduler_state = start_epoch > 0 and scheduler is not None and not progress.get("scheduler_state_dict")
            if (missing_optimizer_state or missing_scheduler_state) and not args.reset_optimizer:
                raise RuntimeError(
                    "Refusing to resume with reset optimizer/scheduler state. "
                    "Pass --reset_optimizer only for an explicit reset run."
                )
            if progress.get("optimizer_state_dict"):
                optimizer.load_state_dict(progress["optimizer_state_dict"])
            if progress.get("rng_state"):
                restore_rng_state(progress["rng_state"])
            if progress.get("loader_generator_state") is not None:
                train_generator.set_state(progress["loader_generator_state"])
            if scheduler is not None and progress.get("scheduler_state_dict"):
                scheduler.load_state_dict(progress["scheduler_state_dict"])
            if scaler is not None and progress.get("scaler_state_dict"):
                scaler.load_state_dict(progress["scaler_state_dict"])
            best_val = float(progress.get("best_val", best_val))
            best_epoch = progress.get("best_epoch") or best_epoch_from_curve(list(progress.get("val_curve", [])))
            best_model_state = progress.get("best_model_state_dict")
            if best_model_state is None and checkpoint_stager is not None:
                best_sidecar = checkpoint_stager.load_best_state()
                if isinstance(best_sidecar, Mapping):
                    best_model_state = best_sidecar.get("best_model_state_dict")
            patience = int(progress.get("patience", 0))
            train_curve = list(progress.get("train_curve", []))
            val_curve = list(progress.get("val_curve", []))
            evaluation_curve = list(progress.get("evaluation_curve") or progress.get("evaluation_loss_curve") or [])
            fold_epoch_seconds = [float(v) for v in progress.get("epoch_seconds", [])]
        else:
            training_continuation = None
        if start_epoch > epochs_per_fold:
            raise RuntimeError(
                "RESUME_EPOCH_EXCEEDS_PROTOCOL_CAP:"
                f"fold={fold}:resume={start_epoch}:cap={epochs_per_fold}"
            )

        ema_state = progress.get("ema_state_dict") if progress is not None else None
        if float(args.ema_decay) > 0.0 and ema_state is None:
            ema_state = initialize_weight_average_state(model)
        swa_state = progress.get("swa_state_dict") if progress is not None else None
        swa_count = int(progress.get("swa_count", 0)) if progress is not None else 0
        swa_start_epoch = max(1, int(epochs * float(args.swa_start_fraction))) if args.swa else None

        # Recovery cursors are kept separately from the completed-epoch count.
        # A staged payload may represent a safe optimizer boundary in the
        # middle of an epoch; restoring only ``epoch`` would repeat or skip
        # batches.
        optimizer_step_count = int(progress.get("optimizer_step_count", 0)) if progress is not None else 0
        resume_batch_cursor = int(progress.get("epoch_batch_cursor", 0)) if progress is not None else 0
        resume_sampler_cursor = int(progress.get("sampler_cursor", resume_batch_cursor)) if progress is not None else 0
        resume_epoch_loss_sum = float(progress.get("epoch_loss_sum", 0.0)) if progress is not None else 0.0
        resume_epoch_loss_count = int(progress.get("epoch_loss_count", 0)) if progress is not None else 0
        fold_terminal = bool(progress.get("fold_terminal", False)) if progress is not None else False
        # The state saved at the start of an epoch is used to recreate the
        # same RandomSampler permutation before skipping committed batches.
        sampler_epoch_generator_state = (
            progress.get("sampler_epoch_generator_state") if progress is not None else None
        )
        if (
            progress is not None
            and resume_batch_cursor > 0
            and sampler_epoch_generator_state is not None
        ):
            train_generator.set_state(sampler_epoch_generator_state)

        best_averaged_state_at_best = None
        best_averaged_source_at_best = None
        current_epoch_for_payload = int(start_epoch)
        current_batch_cursor = int(resume_batch_cursor)
        current_sampler_cursor = int(resume_sampler_cursor)
        current_epoch_loss_sum = float(resume_epoch_loss_sum)
        current_epoch_loss_count = int(resume_epoch_loss_count)
        training_stop_request: dict[str, Any] = {"requested": False, "signal": None}
        # The signal handler only records intent.  All I/O and the exception
        # boundary happen after an optimizer step has been committed below.
        safe_stop_requested = training_stop_request
        previous_sigint_handler, previous_sigterm_handler = (
            _install_training_signal_handlers(training_stop_request)
        )

        def resume_payload_factory() -> dict[str, Any]:
            return progress_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                epoch=int(current_epoch_for_payload),
                best_val=best_val,
                best_epoch=best_epoch,
                best_model_state=best_model_state,
                best_averaged_state=best_averaged_state_at_best,
                best_averaged_source=best_averaged_source_at_best,
                patience=patience,
                train_curve=train_curve,
                val_curve=val_curve,
                evaluation_curve=evaluation_curve,
                fold=fold,
                fit_idx=fit_idx,
                fit_y=fit_y,
                val_idx=val_idx,
                val_y=val_y,
                test_idx=test_idx,
                test_y=test_y,
                epoch_seconds=fold_epoch_seconds,
                estimated_remaining_seconds=estimate_remaining_seconds(
                    fold=fold,
                    completed_epoch_in_fold=int(current_epoch_for_payload),
                    folds=folds,
                    epochs=epochs_per_fold,
                    epoch_seconds=fold_epoch_seconds,
                ),
                training_continuation=training_continuation,
                ema_state=ema_state,
                swa_state=swa_state,
                swa_count=swa_count,
                omit_best_state=checkpoint_stager is not None,
                optimizer_step_count=optimizer_step_count,
                checkpoint_generation=(
                    checkpoint_stager.generation if checkpoint_stager is not None else 0
                ),
                epoch_batch_cursor=current_batch_cursor,
                sampler_cursor=current_sampler_cursor,
                runtime_resume_contract=runtime_resume_contract,
                storage_policy_digest_value=storage_policy_digest_value,
                fold_membership_digest=fold_membership_digest,
                rng_state=capture_rng_state(),
                train_generator=train_generator,
                sampler_epoch_generator_state=sampler_epoch_generator_state,
                fold_terminal=fold_terminal,
                epoch_loss_sum=current_epoch_loss_sum,
                epoch_loss_count=current_epoch_loss_count,
                run_contract=run_contract,
            )

        def optimizer_step_commit_callback(**event: Any) -> None:
            nonlocal optimizer_step_count
            nonlocal current_batch_cursor, current_sampler_cursor
            nonlocal current_epoch_loss_sum, current_epoch_loss_count
            optimizer_step_count = int(event["optimizer_step"])
            current_batch_cursor = int(event["batch_cursor"])
            current_sampler_cursor = int(event["sampler_cursor"])
            current_epoch_loss_sum = float(event["epoch_loss_sum"])
            current_epoch_loss_count = int(event["epoch_loss_count"])
            if checkpoint_stager is not None:
                state_for_digest = capture_rng_state()
                state_for_digest["loader_generator_state"] = train_generator.get_state()
                checkpoint_stager.record_step(
                    payload_factory=resume_payload_factory,
                    optimizer_step=optimizer_step_count,
                    epoch=int(event["epoch"]),
                    epoch_batch_cursor=current_batch_cursor,
                    sampler_cursor=current_sampler_cursor,
                    loss_sum=current_epoch_loss_sum,
                    loss_count=current_epoch_loss_count,
                    rng_digest=rng_state_digest(state_for_digest),
                    force_full=bool(safe_stop_requested["requested"]),
                )
            if safe_stop_requested["requested"]:
                # This raise is reachable only after the callback has recorded
                # the optimizer boundary and (when staged) its journal entry.
                raise SafeOptimizerBoundaryStop

        last_completed_epoch = start_epoch
        try:
            for epoch in range(start_epoch, epochs_per_fold):
                if fold_terminal:
                    break
                current_epoch_for_payload = int(epoch)
                if epoch == start_epoch and resume_batch_cursor > 0:
                    # ``sampler_epoch_generator_state`` was captured before the
                    # original iterator was created; recreating that iterator
                    # gives the same permutation before committed batches are
                    # skipped by run_epoch.
                    if sampler_epoch_generator_state is not None:
                        train_generator.set_state(sampler_epoch_generator_state)
                else:
                    current_batch_cursor = 0
                    current_sampler_cursor = 0
                    current_epoch_loss_sum = 0.0
                    current_epoch_loss_count = 0
                    sampler_epoch_generator_state = train_generator.get_state()
                epoch_optimizer_step_offset = int(optimizer_step_count)
                epoch_started = time.monotonic()
                newly_unfrozen = gradual_unfreeze_step(
                    model,
                    epoch=epoch,
                    epochs=epochs,
                    enabled=args.gradual_unfreeze,
                )
                if args.unfreeze_schedule != "none":
                    schedule_report = apply_prespecified_unfreeze_schedule(
                        model,
                        backbone=args.backbone,
                        schedule=args.unfreeze_schedule,
                        unfreeze_epoch=args.unfreeze_epoch,
                        epoch=epoch,
                    )
                    newly_unfrozen.extend(schedule_report["newly_unfrozen"])
                    already_recorded = any(
                        row.get("fold") == fold
                        and row.get("schedule") == args.unfreeze_schedule
                        and row.get("transition_applied") is True
                        for row in args._unfreeze_transitions
                    )
                    if schedule_report["transition_applied"] and not already_recorded:
                        args._unfreeze_transitions.append(
                            {"fold": fold, **schedule_report}
                        )
                if newly_unfrozen:
                    newly_unfrozen_set = set(newly_unfrozen)
                    args._frozen_layers = [
                        name for name in args._frozen_layers if name not in newly_unfrozen_set
                    ]
                    counts = parameter_counts(model)
                    args._trainable_parameter_count = counts["trainable_parameter_count"]
                train_loss = run_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    pos_weight,
                    latent_alignment_lambda=args.latent_alignment_lambda,
                    synib_cfg=synib_cfg,
                    use_synib=use_synib,
                    tab_mean=tab_mean,
                    tab_std=tab_std,
                    use_amp=use_amp,
                    scaler=scaler,
                    grad_accum_steps=args.grad_accum,
                    label_smoothing=args.label_smoothing,
                    contrastive_alignment=args.contrastive_alignment,
                    contrastive_temperature=args.contrastive_temperature,
                    contrastive_loss_fn=contrastive_loss_fn,
                    supervised_contrastive_alignment=args.supervised_contrastive_alignment,
                    phenotypic_contrastive_lambda=args.phenotypic_contrastive_lambda,
                    phenotypic_temperature=args.phenotypic_temperature,
                    phenotype_feature_names=METHOD_X_PHENOTYPE_FEATURES,
                    use_checkpoint=args.checkpoint,
                    scheduler=scheduler,
                    scheduler_step_per_batch=args.scheduler == "onecycle",
                    task_loss_name=args.task_loss,
                    focal_gamma=args.focal_gamma,
                    focal_alpha=args.focal_alpha,
                    auc_margin=args.auc_margin,
                    auc_loss_weight=args.auc_loss_weight,
                    ema_state=ema_state,
                    ema_decay=args.ema_decay,
                    optimizer_step_commit_callback=optimizer_step_commit_callback,
                    start_batch_cursor=(
                        resume_batch_cursor if epoch == start_epoch else 0
                    ),
                    initial_loss_sum=(
                        resume_epoch_loss_sum if epoch == start_epoch else 0.0
                    ),
                    initial_loss_count=(
                        resume_epoch_loss_count if epoch == start_epoch else 0
                    ),
                    optimizer_step_offset=epoch_optimizer_step_offset,
                    epoch_index=epoch,
                )
                # Once the resumed epoch has consumed its remaining batches,
                # subsequent epochs start from a fresh sampler cursor.
                resume_batch_cursor = 0
                resume_sampler_cursor = 0
                resume_epoch_loss_sum = 0.0
                resume_epoch_loss_count = 0
                val_loss = eval_loss(
                    model,
                    val_loader,
                    device,
                    pos_weight,
                    tab_mean=tab_mean,
                    tab_std=tab_std,
                    task_loss_name=args.task_loss,
                    focal_gamma=args.focal_gamma,
                    focal_alpha=args.focal_alpha,
                    auc_margin=args.auc_margin,
                    auc_loss_weight=args.auc_loss_weight,
                )
                evaluation_loss = None
                # Outer/test loss is reporting-only and is computed once after
                # the checkpoint lock below; it is never part of early
                # stopping, scheduler decisions, or model selection.
                if scheduler is not None and args.scheduler != "onecycle":
                    scheduler.step()

                train_curve.append(train_loss)
                val_curve.append(val_loss)
                evaluation_curve.append(evaluation_loss)
                completed_epoch = epoch + 1
                if args.swa and swa_start_epoch is not None and completed_epoch >= swa_start_epoch:
                    swa_state, swa_count = update_swa_state(swa_state, model, swa_count)
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = completed_epoch
                    if args.restore_best_at_end or checkpoint_stager is not None:
                        best_model_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                    if checkpoint_stager is not None:
                        if ema_state is not None:
                            best_averaged_state_at_best = clone_state_dict(ema_state)
                            best_averaged_source_at_best = "ema_at_best_validation"
                        elif swa_state is not None:
                            best_averaged_state_at_best = clone_state_dict(swa_state)
                            best_averaged_source_at_best = "swa_at_best_validation"
                        seed_offloaded_best_state_sidecar(
                            checkpoint_stager,
                            best_model_state=best_model_state,
                            best_averaged_state=best_averaged_state_at_best,
                            best_averaged_source=best_averaged_source_at_best,
                            best_epoch=best_epoch,
                            best_val=best_val,
                        )
                    patience = 0
                else:
                    patience += 1

                last_completed_epoch = completed_epoch
                fold_epoch_seconds.append(time.monotonic() - epoch_started)
                estimated_remaining = estimate_remaining_seconds(
                    fold=fold,
                    completed_epoch_in_fold=completed_epoch,
                    folds=folds,
                    epochs=epochs_per_fold,
                    epoch_seconds=fold_epoch_seconds,
                )
                early_stop = patience >= args.early_stopping_patience
                # Epoch completion is a safe boundary.  A terminal early-stop
                # marker prevents a later invocation from running one extra
                # epoch, while the explicit reason remains ``EARLY_STOPPING``
                # in the progress/traceability record.
                fold_terminal = bool(early_stop)
                current_epoch_for_payload = int(completed_epoch)
                current_batch_cursor = 0
                current_sampler_cursor = 0
                current_epoch_loss_sum = 0.0
                current_epoch_loss_count = 0
                persist_resume_boundary(
                    checkpoint_stager=checkpoint_stager,
                    payload_factory=resume_payload_factory,
                    progress_path=progress_path,
                    optimizer_step=optimizer_step_count,
                    epoch=completed_epoch,
                    epoch_batch_cursor=0,
                    sampler_cursor=0,
                    loss_sum=0.0,
                    loss_count=0,
                    reason=("TERMINAL" if early_stop else "EPOCH_END"),
                )
                print(
                    json.dumps(
                        {
                            "status": "EPOCH_SAVED",
                            "fold": fold,
                            "epoch": completed_epoch,
                            "epochs": epochs_per_fold,
                            "total_epoch_budget": epochs,
                            "boundary_reason": "EARLY_STOPPING" if early_stop else "EPOCH_END",
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "evaluation_loss": evaluation_loss,
                            "evaluation_loss_source": (
                                "not_accessed_selection_only"
                                if selection_only
                                else "locked_fold_test_split_reporting_only"
                            ),
                            "progress_path": relative_or_str(progress_path),
                            "estimated_remaining_seconds": estimated_remaining,
                            "newly_unfrozen_layers": newly_unfrozen,
                        },
                        indent=2,
                    ),
                    flush=True,
                )

                if early_stop:
                    break
                if max_runtime_seconds is not None and time.monotonic() - run_started >= max_runtime_seconds:
                    write_pause_marker(
                        status="PAUSED_RESUMABLE",
                        reason="MAX_RUNTIME_MINUTES_REACHED",
                        current_fold=fold,
                        resume_from_epoch=completed_epoch,
                        progress_path=progress_path,
                        estimated_remaining=estimated_remaining,
                    )
                    return RESUMABLE_EXIT_CODE
        except (KeyboardInterrupt, SafeOptimizerBoundaryStop):
            estimated_remaining = estimate_remaining_seconds(
                fold=fold,
                completed_epoch_in_fold=last_completed_epoch,
                folds=folds,
                epochs=epochs_per_fold,
                epoch_seconds=fold_epoch_seconds,
            )
            if checkpoint_stager is not None:
                checkpoint_stager.finalize_clean_stop(reason="CONTROLLED_SIGNAL")
            write_pause_marker(
                status="INTERRUPTED_RESUMABLE",
                reason=(
                    "CONTROLLED_SIGNAL"
                    if safe_stop_requested["requested"]
                    else "KEYBOARD_INTERRUPT"
                ),
                current_fold=fold,
                resume_from_epoch=last_completed_epoch,
                progress_path=progress_path,
                estimated_remaining=estimated_remaining,
            )
            signal.signal(signal.SIGINT, previous_sigint_handler)
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
            return 75

        # Keep the selected terminal state explicit.  The assignment is also
        # the single source of truth used by the checkpoint-lock transaction
        # below; no state may be changed after the outer loader is opened.
        selected_state = model.state_dict()
        selected_state, restored_best_at_end = selected_state, False
        if args.restore_best_at_end and best_model_state is not None:
            model.load_state_dict(best_model_state)
            restored_best_at_end = True
        averaged_state_applied = False
        averaged_state_source = "raw_model"
        if float(args.ema_decay) > 0.0 and ema_state is not None:
            model.load_state_dict(ema_state)
            averaged_state_applied = True
            averaged_state_source = "ema_shadow_state"
        elif args.swa and swa_state is not None and swa_count > 0:
            model.load_state_dict(swa_state)
            averaged_state_applied = True
            averaged_state_source = "swa_shadow_state"

        # The state that is about to be evaluated is now immutable.  Capture a
        # CPU copy before opening the outer loader so a later device operation
        # cannot silently change the bytes represented by the lock checkpoint.
        selected_state = {
            key: value.detach().cpu().clone() if hasattr(value, "detach") else value
            for key, value in model.state_dict().items()
        }

        val_true, val_prob, val_logits = predict(
            model,
            val_loader,
            device,
            tab_mean,
            tab_std,
            use_tta=args.tta,
            tta_augmentations=args.tta_augmentations,
            tta_mode=args.tta_mode,
            return_logits=True,
        )

        # ---- checkpoint lock / exactly-once outer-evaluation transaction ----
        # No outer/test DataLoader may be constructed before this boundary.
        # The lock payload is immutable evidence of the selected model state;
        # the raw prediction artifact is written only after the in-progress
        # state transition and is then embedded in the final checkpoint.
        locked_outer_for_checkpoint: dict[str, Any] | None = None
        locked_outer_result: dict[str, Any] | None = None
        outer_test_access_count = 0
        lock_payload = resume_payload_factory()
        lock_payload["model_state_dict"] = selected_state
        lock_payload["checkpoint_kind"] = "CHECKPOINT_LOCKED"
        lock_payload["outer_test_access_count"] = 0
        lock_payload["locked_at_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        lock_payload["fold_terminal"] = True
        atomic_torch_save_immutable(lock_payload, locked_checkpoint_path)
        locked_checkpoint_sha256 = sha256_path(locked_checkpoint_path)

        current_stage = str(fold_state.get("stage")) if fold_state else None
        if current_stage in {None, "TRAINING"}:
            fold_state = transition_fold_state(
                fold_state_path,
                fold=fold,
                target="CHECKPOINT_LOCKED",
                contract_digest=run_contract["digest"],
                details={
                    "locked_checkpoint": str(
                        locked_checkpoint_path.relative_to(PROJECT_ROOT)
                    ),
                    "locked_checkpoint_sha256": locked_checkpoint_sha256,
                    "split_membership_digest": fold_membership_digest,
                    "outer_test_access_count": 0,
                },
            )
            current_stage = "CHECKPOINT_LOCKED"
        elif current_stage not in {
            "CHECKPOINT_LOCKED",
            "OUTER_EVAL_IN_PROGRESS",
            "OUTER_EVAL_DONE",
        }:
            raise RuntimeError(f"OUTER_EVALUATION_STATE_INCONSISTENT:{current_stage}")

        if selection_only:
            # The task metric is descriptive inner-validation performance.
            # Calibration and threshold estimation would reuse the same rows
            # and are therefore explicitly not run.
            y_true, y_prob, test_logits = val_true, val_prob, val_logits
            temperature_report = {
                "status": "NOT_RUN",
                "reason": "selection_only_task_has_no_independent_calibration_or_outer_split",
                "fit_scope": "not_applicable",
                "test_labels_used_for_calibration": False,
                "outer_test_access_count": 0,
            }
            calibrated_prob = []
            threshold_eval = {
                "status": "NOT_RUN",
                "reason": "selection_only_task_threshold_not_estimable_without_reuse",
                "test_labels_used_for_threshold_selection": False,
                "outer_test_access_count": 0,
            }
        else:
            if current_stage == "OUTER_EVAL_IN_PROGRESS":
                # A result already present is the only admissible recovery
                # path.  Reopening the loader would violate exactly-once.
                if not locked_outer_path.is_file():
                    raise RuntimeError("AMBIGUOUS_OUTER_EVALUATION_STATE")
                locked_outer_result = json.loads(
                    locked_outer_path.read_text(encoding="utf-8")
                )
                validated_outer = validate_locked_outer_result(
                    locked_outer_result,
                    expected_fold=fold,
                    expected_contract_digest=run_contract["digest"],
                    expected_split_membership_digest=fold_membership_digest,
                    expected_count=len(test_idx),
                )
                y_true = list(validated_outer["y_true"])
                y_prob = list(validated_outer["y_prob"])
                test_logits = list(validated_outer["test_logits"])
                outer_test_access_count = 1
            elif current_stage == "OUTER_EVAL_DONE":
                if not locked_outer_path.is_file():
                    raise RuntimeError("COMMITTED_OUTER_EVALUATION_MISSING")
                locked_outer_result = json.loads(
                    locked_outer_path.read_text(encoding="utf-8")
                )
                validated_outer = validate_locked_outer_result(
                    locked_outer_result,
                    expected_fold=fold,
                    expected_contract_digest=run_contract["digest"],
                    expected_split_membership_digest=fold_membership_digest,
                    expected_count=len(test_idx),
                )
                y_true = list(validated_outer["y_true"])
                y_prob = list(validated_outer["y_prob"])
                test_logits = list(validated_outer["test_logits"])
                outer_test_access_count = 1
            else:
                fold_state = transition_fold_state(
                    fold_state_path,
                    fold=fold,
                    target="OUTER_EVAL_IN_PROGRESS",
                    contract_digest=run_contract["digest"],
                    details={
                        "locked_checkpoint": str(
                            locked_checkpoint_path.relative_to(PROJECT_ROOT)
                        ),
                        "locked_checkpoint_sha256": locked_checkpoint_sha256,
                        "outer_test_access_count": 0,
                        "split_membership_digest": fold_membership_digest,
                    },
                )
                current_stage = "OUTER_EVAL_IN_PROGRESS"
                # This is the first and only construction of the outer loader.
                test_loader = DataLoader(
                    H5MethodXDataset(
                        cache_path,
                        test_idx,
                        test_y,
                        sanity_random_labels=False,
                        sanity_random_noise_images=False,
                        input_mode=args.input_mode,
                        drop_tabular_feature=getattr(
                            args, "drop_tabular_feature", "none"
                        ),
                        seed=args.seed + fold + 2000,
                    ),
                    batch_size=args.batch_size,
                    shuffle=False,
                    generator=test_generator,
                    worker_init_fn=seed_worker,
                    **dataloader_throughput_kwargs(args),
                )
                y_true, y_prob, test_logits = predict(
                    model,
                    test_loader,
                    device,
                    tab_mean,
                    tab_std,
                    use_tta=args.tta,
                    tta_augmentations=args.tta_augmentations,
                    tta_mode=args.tta_mode,
                    return_logits=True,
                )
                locked_outer_result = build_locked_outer_result(
                    fold=fold,
                    run_contract_digest=run_contract["digest"],
                    split_membership_digest=fold_membership_digest,
                    y_true=list(y_true),
                    y_prob=list(y_prob),
                    test_logits=list(test_logits),
                )
                atomic_write_json(locked_outer_path, locked_outer_result)
                locked_outer_result = json.loads(
                    locked_outer_path.read_text(encoding="utf-8")
                )
                validated_outer = validate_locked_outer_result(
                    locked_outer_result,
                    expected_fold=fold,
                    expected_contract_digest=run_contract["digest"],
                    expected_split_membership_digest=fold_membership_digest,
                    expected_count=len(test_idx),
                )
                y_true = list(validated_outer["y_true"])
                y_prob = list(validated_outer["y_prob"])
                test_logits = list(validated_outer["test_logits"])
                outer_test_access_count = 1
            temperature_report, calibrated_prob = temperature_scaling_report(
                validation_y=val_true,
                validation_logits=val_logits,
                test_y=y_true,
                test_logits=test_logits,
            )
            threshold_eval = evaluate_with_train_selected_threshold(
                validation_y=val_true,
                validation_prob=val_prob,
                test_y=y_true,
                test_prob=y_prob,
            )
            if current_stage == "OUTER_EVAL_IN_PROGRESS":
                fold_state = transition_fold_state(
                    fold_state_path,
                    fold=fold,
                    target="OUTER_EVAL_DONE",
                    contract_digest=run_contract["digest"],
                    details={
                        "locked_result": str(
                            locked_outer_path.relative_to(PROJECT_ROOT)
                        ),
                        "outer_result_digest": locked_outer_result.get(
                            "result_digest"
                        )
                        if isinstance(locked_outer_result, Mapping)
                        else None,
                        "outer_test_access_count": 1,
                        "split_membership_digest": fold_membership_digest,
                    },
                )
                current_stage = "OUTER_EVAL_DONE"
        if bool(args.throughput_profile_only):
            cohort_probe_report = {
                "status": "NOT_RUN",
                "reason": "throughput_profile_only_never_selection_evidence",
            }
        else:
            cohort_probe_report = {"status": "NOT_RUN", "reason": "cohort_labels_unavailable_or_single_class"}
            try:
                cohort_y = cohort_labels_for_indices(split_dir / f"fold_{fold}_train.csv", sample_to_idx, val_idx)
                if len(cohort_y) == len(val_idx) and len(set(cohort_y)) >= 2:
                    val_latents = collect_mri_latents(model, val_loader, device, tab_mean, tab_std)
                    cohort_probe_report = train_cohort_probe(val_latents, cohort_y, max_iter=200)
                else:
                    cohort_probe_report = {
                        "status": "NOT_RUN",
                        "reason": "validation_cohort_labels_missing_or_single_class",
                        "n_labels": len(cohort_y),
                        "n_validation": len(val_idx),
                        "n_classes": len(set(cohort_y)) if cohort_y else 0,
                    }
            except Exception as exc:
                cohort_probe_report = {
                    "status": "NOT_RUN",
                    "reason": f"cohort_probe_failed:{type(exc).__name__}:{exc}",
                }
        threshold_rows.append({"fold": fold, **threshold_eval})
        fold_epochs_run = reported_epochs_run(last_completed_epoch, train_curve)
        metrics = {
            "fold": fold,
            "epoch_target": epochs_per_fold,
            "n": len(y_true),
            "test_n": 0 if selection_only else len(y_true),
            "inner_validation_evaluation_n": len(y_true) if selection_only else None,
            "train_n": len(fit_idx),
            "val_n": len(val_idx),
            "roc_auc": roc_auc(y_true, y_prob),
            "pr_auc": average_precision(y_true, y_prob),
            "brier": brier(y_true, y_prob),
            "best_val_loss": best_val,
            "best_epoch": best_epoch or best_epoch_from_curve(val_curve),
            "restore_best_at_end": bool(args.restore_best_at_end),
            "restored_best_at_end": restored_best_at_end,
            "averaged_state_applied": averaged_state_applied,
            "averaged_state_source": averaged_state_source,
            "ema_decay": float(args.ema_decay),
            "swa_enabled": bool(args.swa),
            "swa_count": int(swa_count),
            "val_loss_last": val_curve[-1] if val_curve else None,
            "evaluation_loss_curve": evaluation_curve,
            "evaluation_loss_last": last_numeric(evaluation_curve),
            "best_evaluation_loss": min_numeric(evaluation_curve),
            "evaluation_scope": (
                "inner_validation_only" if selection_only else "locked_outer_test_reporting_only"
            ),
            "outer_test_access_count": 0 if selection_only else 1,
            "outer_test_evaluation_count": 0 if selection_only else 1,
            "outer_test_used_for_selection": False,
            "validity_inner_only": validity_inner_only,
            "hpo_selection_only": hpo_selection_only,
            "outer_test_evaluation_policy": outer_test_evaluation_policy(selection_only),
            "evaluation_loss_source": (
                "not_accessed_selection_only"
                if selection_only
                else "locked_fold_test_split_reporting_only"
            ),
            "evaluation_loss_used_for_early_stopping": False,
            "evaluation_loss_used_for_hyperparameter_selection": False,
            "evaluation_loss_used_for_threshold_selection": False,
            "evaluation_loss_used_for_model_selection": False,
            "loss_last": train_curve[-1] if train_curve else None,
            "loss_parts_last": getattr(model, "_last_epoch_loss_parts", {}),
            "phenotypic_contrastive_loss_last": getattr(model, "_last_epoch_loss_parts", {}).get(
                "phenotypic_contrastive_loss"
            ),
            "phenotypic_contrastive_weighted_loss_last": getattr(model, "_last_epoch_loss_parts", {}).get(
                "phenotypic_contrastive_weighted_loss"
            ),
            "epochs_run": fold_epochs_run,
            "epoch_seconds": [float(value) for value in fold_epoch_seconds],
            "average_epoch_seconds": (
                float(sum(fold_epoch_seconds) / len(fold_epoch_seconds))
                if fold_epoch_seconds
                else None
            ),
            "loss_curve_epochs_observed": len(train_curve),
            "early_stopped": fold_epochs_run < epochs,
            "train_loss_curve": train_curve,
            "val_loss_curve": val_curve,
            "task_loss": args.task_loss,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": args.focal_alpha,
            "auc_margin": args.auc_margin,
            "auc_loss_weight": args.auc_loss_weight,
            "moe_load_balance_lambda": float(args.moe_load_balance_lambda),
            "synib_lambda": args.synib_lambda,
            "modality_dropout": args.modality_dropout,
            "attention": args.attention,
            "scheduler": args.scheduler,
            "pos_weight": args.pos_weight,
            "tta": bool(args.tta),
            "tta_mode": args.tta_mode,
            "tta_augmentations": int(args.tta_augmentations),
            "temperature_scaling": temperature_report,
            "threshold_metrics": threshold_eval,
            "cohort_probe": cohort_probe_report,
            "cohort_probe_auc": cohort_probe_report.get("auc"),
            "training_continuation": training_continuation,
            "progress_checkpoint_preserved": str(progress_path.relative_to(PROJECT_ROOT))
            if progress_path.exists()
            else None,
        }
        fold_metrics.append(metrics)
        all_y.extend(y_true)
        all_p.extend(y_prob)
        if calibrated_prob:
            all_calibrated_p.extend(calibrated_prob)
        temperature_rows.append({"fold": fold, **temperature_report})

        selection_only_evaluation = None
        if selection_only:
            selection_only_evaluation = {
                "schema_version": 1,
                "fold": fold,
                "run_contract_digest": run_contract["digest"],
                "split_membership_digest": fold_membership_digest,
                "outer_test_access_count": 0,
                "outer_test_used_for_selection": False,
                "y_true": list(y_true),
                "y_prob": list(y_prob),
                "logits": list(test_logits),
                "metrics": dict(metrics),
            }
            selection_evidence_folds.append(
                {
                    "fold": fold,
                    "split_membership_digest": fold_membership_digest,
                    "y_true": list(y_true),
                    "y_prob": list(y_prob),
                    "logits": list(test_logits),
                }
            )
        elif isinstance(locked_outer_result, Mapping):
            locked_outer_for_checkpoint = {
                **dict(locked_outer_result),
                "metrics": dict(metrics),
                "calibrated_prob": list(calibrated_prob),
            }

        ckpt_path = out_dir / f"method_x_3d_fold_{fold}.pt"
        atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "fold": fold,
                "mode": mode,
                "fusion": args.fusion,
                "backbone": args.backbone,
                "attention": args.attention,
                "alignment_head": args.alignment_head,
                "alignment_head_mode": args.alignment_head_mode,
                "tab_encoder": args.tab_encoder,
                "causal_adjustment": args.causal_adjustment,
                "q2_channel_gate": args.q2_channel_gate,
                "hybrid_variant": args.hybrid_variant,
                "hybrid_attention_heads": int(args.hybrid_attention_heads),
                "hybrid_attention_window_size": int(args.hybrid_attention_window_size),
                "input_mode": args.input_mode,
                "drop_tabular_feature": str(getattr(args, "drop_tabular_feature", "none")),
                "validity_inner_only": bool(getattr(args, "validity_inner_only", False)),
                "hpo_selection_only": bool(getattr(args, "hpo_selection_only", False)),
                "pretrained_checkpoint": pretrained_checkpoint,
                "pretrained_fold_manifest": args.pretrained_fold_manifest or None,
                "pretrained_load_report": pretrained_report,
                "frozen_layers": getattr(args, "_frozen_layers", []),
                "unfreeze_schedule": args.unfreeze_schedule,
                "unfreeze_epoch": int(args.unfreeze_epoch),
                "unfreeze_transitions": [
                    row for row in args._unfreeze_transitions if row.get("fold") == fold
                ],
                "encoder_stage_map": args._encoder_stage_map,
                "layerwise_lr_decay": args.layerwise_lr_decay,
                "discriminative_lr": args.discriminative_lr,
                "trainable_parameter_count": getattr(args, "_trainable_parameter_count", None),
                "total_parameter_count": getattr(args, "_total_parameter_count", None),
                "lr": base_lr,
                "dropout": args.dropout,
                "latent_dim": args.latent_dim,
                "tab_hidden_dim": args.tab_hidden_dim,
                "bottleneck_dim": args.bottleneck_dim,
                "fusion_hidden_dim": args.fusion_hidden_dim,
                "n_experts": args.n_experts,
                "fusion_num_heads": args.fusion_num_heads,
                "moe_load_balance_lambda": float(args.moe_load_balance_lambda),
                "restore_best_at_end": bool(args.restore_best_at_end),
                "task_loss": args.task_loss,
                "focal_gamma": args.focal_gamma,
                "focal_alpha": args.focal_alpha,
                "auc_margin": args.auc_margin,
                "auc_loss_weight": args.auc_loss_weight,
                "ema_decay": float(args.ema_decay),
                "swa": bool(args.swa),
                "swa_start_fraction": float(args.swa_start_fraction),
                "averaged_state_applied": averaged_state_applied,
                "averaged_state_source": averaged_state_source,
                "swa_count": int(swa_count),
            "latent_alignment_lambda": args.latent_alignment_lambda,
            "phenotypic_contrastive_alignment": args.phenotypic_contrastive_alignment,
            "phenotypic_contrastive_lambda": args.phenotypic_contrastive_lambda,
            "phenotypic_temperature": args.phenotypic_temperature,
            "synib_lambda": args.synib_lambda,
                "modality_dropout": args.modality_dropout,
                "scheduler": args.scheduler,
                "scheduler_step_scope": "batch" if args.scheduler == "onecycle" else "epoch",
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "ema_state_dict": ema_state if float(args.ema_decay) > 0.0 else None,
                "swa_state_dict": swa_state if bool(args.swa) else None,
                "tab_mean": tab_mean,
                "tab_std": tab_std,
                "epochs_run": fold_epochs_run,
                "loss_curve_epochs_observed": len(train_curve),
                "requested_epochs": epochs,
                "best_val_loss": best_val,
                "best_epoch": best_epoch or best_epoch_from_curve(val_curve),
                "restore_best_at_end": bool(args.restore_best_at_end),
                "restored_best_at_end": restored_best_at_end,
                "final_val_loss": val_curve[-1] if val_curve else None,
                "evaluation_loss_curve": evaluation_curve,
                "evaluation_loss_last": last_numeric(evaluation_curve),
                "best_evaluation_loss": min_numeric(evaluation_curve),
                "evaluation_scope": (
                    "inner_validation_only"
                    if selection_only
                    else "locked_outer_test_reporting_only"
                ),
                "outer_test_access_count": 0 if selection_only else 1,
                "outer_test_evaluation_count": 0 if selection_only else 1,
                "outer_test_used_for_selection": False,
                "outer_test_evaluation_policy": outer_test_evaluation_policy(selection_only),
                "evaluation_loss_source": (
                    "not_accessed_selection_only"
                    if selection_only
                    else "locked_fold_test_split_reporting_only"
                ),
                "evaluation_loss_reporting_only": True,
                "final_train_loss": train_curve[-1] if train_curve else None,
                "train_loss_curve": train_curve,
                "val_loss_curve": val_curve,
                "augmentation_enabled": args.monai_augment,
                "aug_prob_scale": args.aug_prob_scale,
                "hparam_selection": hparam_selection_payload(args),
                "training_continuation": training_continuation,
                "selection_only_evaluation": selection_only_evaluation,
                "locked_outer_test_evaluation": locked_outer_for_checkpoint,
                "run_contract": dict(run_contract),
                "run_contract_digest": run_contract["digest"],
                "split_membership_digest": fold_membership_digest,
                "fold_terminal": True,
            },
            ckpt_path,
        )
        fold_state = transition_fold_state(
            fold_state_path,
            fold=fold,
            target="FINALIZED",
            contract_digest=run_contract["digest"],
            details={
                "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": sha256_path(ckpt_path),
                "split_membership_digest": fold_membership_digest,
                "outer_test_access_count": 0 if selection_only else 1,
                "selection_only": selection_only,
                "hpo_selection_only": hpo_selection_only,
            },
        )
        checkpoints.append(str(ckpt_path.relative_to(PROJECT_ROOT)))
        partial_aggregate = build_aggregate(
            mode=mode,
            args=args,
            device=device,
            epochs=epochs,
            folds=folds,
            use_synib=use_synib,
            use_amp=use_amp,
            cache_path=cache_path,
            cache_resolution=cache_resolution,
            fold_metrics=fold_metrics,
            all_y=all_y,
            all_p=all_p,
            checkpoints=checkpoints,
            threshold_rows=threshold_rows,
            include_supplementary_train=include_supplementary,
            supplementary_ok_n=supplementary_ok_n,
        )
        write_metrics_payload(
            out_dir,
            partial_aggregate,
            fold_metrics,
            status="IN_PROGRESS" if len(checkpoints) < folds else "COMPLETED",
        )

    aggregate = build_aggregate(
        mode=mode,
        args=args,
        device=device,
        epochs=epochs,
        folds=folds,
        use_synib=use_synib,
        use_amp=use_amp,
        cache_path=cache_path,
        cache_resolution=cache_resolution,
        fold_metrics=fold_metrics,
        all_y=all_y,
        all_p=all_p,
        checkpoints=checkpoints,
        threshold_rows=threshold_rows,
        include_supplementary_train=include_supplementary,
        supplementary_ok_n=supplementary_ok_n,
    )
    aggregate["status"] = "COMPLETED"
    # The claim gate reads ``training_status``; without this the
    # TRAINING_NOT_COMPLETED check is skipped on absence (a silent fail-open).
    aggregate["training_status"] = aggregate["status"]
    aggregate["calibration"] = calibration_analysis(all_y, all_p) if all_y else {"status": "NOT_RUN_NO_PREDICTIONS"}
    if all_calibrated_p and len(all_calibrated_p) == len(all_y):
        aggregate["temperature_scaling"] = {
            "status": "FITTED_ON_INNER_VALIDATION_ONLY",
            "folds": temperature_rows,
            "aggregate_test_uncalibrated": calibration_metric_summary(all_y, all_p),
            "aggregate_test_calibrated": calibration_metric_summary(all_y, all_calibrated_p),
            "aggregate_reliability_data": reliability_diagram_data(all_y, all_calibrated_p),
            "fit_scope": "inner_validation_only_by_fold",
            "test_labels_used_for_calibration": False,
            "subject_level_logits_or_predictions_exported": False,
        }
    else:
        aggregate["temperature_scaling"] = {
            "status": "NOT_RUN",
            "reason": (
                "selection_only_task_has_no_independent_calibration_or_outer_split"
                if selection_only
                else "no_completed_fold_temperature_scaling"
            ),
            "folds": temperature_rows,
            "fit_scope_required": "not_applicable"
            if selection_only
            else "inner_validation_only",
            "test_labels_used_for_calibration": False,
            "subject_level_logits_or_predictions_exported": False,
        }
    aggregate["memory_profile"] = memory_profile(device)
    if device.type == "cuda":
        aggregate["vram_peak_gb"] = float(torch.cuda.max_memory_allocated(device) / (1024**3))
        aggregate["vram_peak_allocated_gb"] = aggregate["vram_peak_gb"]
        aggregate["vram_peak_reserved_gb"] = float(
            torch.cuda.max_memory_reserved(device) / (1024**3)
        )

    if selection_only:
        ordered_selection_rows = sorted(selection_evidence_folds, key=lambda row: int(row["fold"]))
        if [int(row["fold"]) for row in ordered_selection_rows] != list(range(int(folds))):
            raise RuntimeError("SELECTION_EVIDENCE_FOLDS_INCOMPLETE")
        for row in ordered_selection_rows:
            labels = row.get("y_true")
            probabilities = row.get("y_prob")
            logits = row.get("logits")
            if (
                not isinstance(labels, list)
                or not isinstance(probabilities, list)
                or not isinstance(logits, list)
                or not labels
                or len(labels) != len(probabilities)
                or len(labels) != len(logits)
                or any(int(value) not in {0, 1} for value in labels)
                or any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in probabilities)
                or any(not math.isfinite(float(value)) for value in logits)
                or len(str(row.get("split_membership_digest") or "")) != 64
            ):
                raise RuntimeError(f"SELECTION_EVIDENCE_INVALID:FOLD_{row.get('fold')}")
        selection_body = {
            "schema_version": 1,
            "artifact": "method_x_private_selection_evidence",
            "backbone": args.backbone,
            "seed": int(args.seed),
            "folds": ordered_selection_rows,
            "fold_count": int(folds),
            "run_contract_digest": run_contract["digest"],
            "outer_test_access_count": 0,
            "outer_test_used_for_selection": False,
            "raw_identifiers_stored": False,
            "private_selection_evidence": True,
            "publication_allowed": False,
        }
        selection_payload = {
            **selection_body,
            "integrity_sha256": canonical_digest(selection_body),
        }
        selection_path = out_dir / "selection_evidence.json"
        atomic_write_json(selection_path, selection_payload)
        aggregate["selection_evidence"] = {
            "path": str(selection_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha256_path(selection_path),
            "integrity_sha256": selection_payload["integrity_sha256"],
            "raw_identifiers_stored": False,
            "publication_allowed": False,
        }

    final_claim_report = validate_claim(
        claim_from_three_d_aggregate(aggregate, strict=args.validate_claim_strict),
        strict=args.validate_claim_strict,
    )
    aggregate["claim_policy_report"] = final_claim_report
    if args.validate_claim_strict and not final_claim_report["ok"]:
        blocked = {
            "status": "BLOCKED",
            "blocker": "CLAIM_POLICY_VIOLATION",
            "blocking_reasons": final_claim_report["blocking_reasons"],
            "soft_warnings": final_claim_report["soft_warnings"],
            "claim_policy_report": final_claim_report,
            "blocked_artifact": "three_d_metrics.json",
        }
        atomic_write_json(out_dir / "BLOCKED_RUN.json", blocked)
        print(json.dumps(blocked, indent=2))
        return 3

    payload = {"aggregate": aggregate, "folds": fold_metrics}
    atomic_write_json(out_dir / "three_d_metrics.json", payload)
    md = [
        "# Method X 3D metrics\n\n",
        f"- model: `{aggregate['model']}`\n",
        f"- fusion: `{aggregate['fusion']}`\n",
        f"- backbone: `{aggregate['backbone']}`\n",
        f"- mode: `{aggregate['mode']}`\n",
        f"- task_loss: `{aggregate['task_loss']}`\n",
        f"- ema_enabled: `{aggregate['weight_averaging']['ema_enabled']}`\n",
        f"- swa_enabled: `{aggregate['weight_averaging']['swa_enabled']}`\n",
        f"- synib_lambda: `{aggregate['synib_lambda']}`\n",
        f"- roc_auc: `{aggregate['roc_auc']}`\n",
        f"- final_claim_allowed: `{aggregate['final_claim_allowed']}`\n",
    ]
    atomic_write_text(out_dir / "three_d_metrics.md", "".join(md))
    if not bool(args.throughput_profile_only):
        write_run_manifest(
            "results/logs/train_3d_mri_manifest.json",
            stage="train-3d",
            data_root=None,
            extra={"aggregate": aggregate},
        )
    print(json.dumps({"status": "COMPLETED", "roc_auc": aggregate["roc_auc"], "mode": mode}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
