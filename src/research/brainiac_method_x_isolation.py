"""Read-only Method X source isolation proof for BrainIAC preparation.

The preprocessing runner must prove that the completed five-CNN battery source
tree did not drift, but importing ``scripts.train_3d_mri`` would execute a very
large, unrelated import graph and incorrectly bind Method Y/PET modules into
every BrainIAC preprocessing receipt.  This module reproduces the *frozen*
Method X source inventory and digest algorithm without importing any trainer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Frozen from method_x_training_source_paths() at the terminal battery source
# digest 9b0f90a5...871b3.  The final glob intentionally captures any additional
# method_x_*.py file exactly as the canonical implementation does.
FROZEN_SCOPED_ROOTS = (
    "scripts/train_3d_mri.py",
    "scripts/train_baseline.py",
    "src/data/combined_mri_cache.py",
    "src/data/volume_augmentation.py",
    "src/evaluation/calibration.py",
    "src/evaluation/cohort_probe.py",
    "src/evaluation/threshold_metrics.py",
    "src/evaluation/tta.py",
    "src/models/__init__.py",
    "src/models/backbones/__init__.py",
    "src/models/backbones/amuse_x_hybrid3d.py",
    "src/models/backbones/attention.py",
    "src/models/backbones/backbone_factory.py",
    "src/models/backbones/deterministic_pool.py",
    "src/models/backbones/lightweight_3d.py",
    "src/models/backbones/modern_3d.py",
    "src/models/backbones/pretrained_loader.py",
    "src/models/causal_intervention_block.py",
    "src/models/fusion/__init__.py",
    "src/models/fusion/amplitude_phase_fusion.py",
    "src/models/fusion/bottleneck_cross_attention.py",
    "src/models/fusion/channel_gate.py",
    "src/models/fusion/conv_feature_fusion.py",
    "src/models/fusion/healnet_fusion.py",
    "src/models/fusion/moe_fusion.py",
    "src/models/fusion/registry.py",
    "src/models/losses/__init__.py",
    "src/models/losses/contrastive_alignment.py",
    "src/models/losses/distillation.py",
    "src/models/losses/evidential.py",
    "src/models/losses/latent_alignment.py",
    "src/models/losses/phase1_losses.py",
    "src/models/losses/phenotypic_contrastive.py",
    "src/models/losses/synib_synergy_loss.py",
    "src/training/__init__.py",
    "src/training/method_x_autosupervisor.py",
    "src/training/method_x_loop_guard.py",
    "src/training/method_x_outer_compatibility.py",
    "src/training/method_x_registry.py",
    "src/training/method_x_registry_lifecycle.py",
    "src/training/method_x_repair_gate.py",
    "src/training/method_x_resume.py",
    "src/training/method_x_source_compatibility.py",
    "src/training/method_x_spm_snapshot.py",
    "src/training/method_x_stage_maps.py",
    "src/training/method_x_stage_registry.py",
    "src/utils/dataloader_config.py",
    "src/utils/final_claim_policy.py",
    "src/utils/io_helpers.py",
    "src/utils/privacy.py",
    "src/utils/reproducibility.py",
    "src/utils/run_manifest.py",
)


def frozen_method_x_source_paths() -> list[Path]:
    roots = [PROJECT_ROOT / relative for relative in FROZEN_SCOPED_ROOTS]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.py")))
    files.extend(sorted((PROJECT_ROOT / "src/training").glob("method_x_*.py")))
    return sorted({path.resolve() for path in files if path.is_file()})


def _source_tree_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_frozen_method_x_source_contract(*, expected_digest: str) -> dict[str, Any]:
    """Rehash frozen Method X bytes without importing training/model modules."""

    if len(str(expected_digest)) != 64:
        raise RuntimeError("BRAINIAC_METHOD_X_EXPECTED_SOURCE_DIGEST_INVALID")
    paths = frozen_method_x_source_paths()
    if len(paths) < 40:
        raise RuntimeError("BRAINIAC_METHOD_X_SOURCE_INVENTORY_INCOMPLETE")
    observed = _source_tree_digest(paths)
    if observed != str(expected_digest).casefold():
        raise RuntimeError("BRAINIAC_METHOD_X_SOURCE_CONTRACT_DRIFT")
    return {
        "status": "PASS",
        "source_contract_digest": observed,
        "file_count": len(paths),
        "trainer_imported": False,
        "battery_files_modified": False,
    }
