"""Load the pinned *vendor* BrainIAC encoder in its isolated runtime.

This module deliberately does not reconstruct MONAI's ViT.  Reconstructing the
network in the Method X environment previously changed ``qkv_bias`` and used
``strict=False`` against MONAI 1.5.2.  The only accepted implementation is the
``ViTBackboneNet`` class from the pinned vendor snapshot, executed with the
vendor-declared MONAI 1.3.2 runtime.  Its constructor performs the real
``strict=True`` load.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
import monai

from src.research.brainiac_execution_gate import assert_brainiac_execution_allowed

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "models" / "pretrained" / "brainiac" / "BrainIAC.ckpt"
PINNED_CHECKPOINT_SHA256 = "f22bdbcae26823a9d9e8aee883c6f24386ba4617339c12269848b6666cc62693"
PINNED_VENDOR_COMMIT = "ba60f45bed832ee0e5683c678caeaeefe2072f0d"
PINNED_VENDOR_ROOT = ROOT / "vendor" / "BrainIAC" / PINNED_VENDOR_COMMIT
PINNED_MODEL_PATH = PINNED_VENDOR_ROOT / "src" / "model.py"
PINNED_VENDOR_MODEL_SHA256 = "f975f04e208f6d0c1727df5b00e99fb7b8451d011cde84c0025c4fb6b9a4570b"
PINNED_LICENSE_PATH = PINNED_VENDOR_ROOT / "LICENSE"
PINNED_LICENSE_SHA256 = "e862b209788af5c4afa80031f75bc0946a8c48464991bce327ab615e742c6690"
OFFICIAL_MONAI_VERSION = "1.3.2"
OFFICIAL_MODULE_PATH = "src/model.py"
OFFICIAL_CLASS_NAME = "ViTBackboneNet"
OFFICIAL_LOADER_SYMBOL = "load_brainiac"


def validate_official_runtime_identity(
    *,
    python_version: str,
    torch_version: str,
    monai_version: str,
    vendor_commit: str,
    module_path: str,
    class_name: str,
    loader_symbol: str,
) -> dict[str, Any]:
    """Fail closed unless execution matches the pinned public implementation."""

    if str(monai_version) != OFFICIAL_MONAI_VERSION:
        raise RuntimeError(
            f"BRAINIAC_MONAI_VERSION_MISMATCH:{monai_version}:"
            f"expected_{OFFICIAL_MONAI_VERSION}"
        )
    if not str(python_version).startswith("3.11"):
        raise RuntimeError(f"BRAINIAC_PYTHON_VERSION_UNSUPPORTED:{python_version}")
    if not str(torch_version).strip():
        raise RuntimeError("BRAINIAC_TORCH_VERSION_MISSING")
    identity = {
        "vendor_commit": str(vendor_commit),
        "module_path": str(module_path).replace("\\", "/"),
        "class_name": str(class_name),
        "loader_symbol": str(loader_symbol),
    }
    expected = {
        "vendor_commit": PINNED_VENDOR_COMMIT,
        "module_path": OFFICIAL_MODULE_PATH,
        "class_name": OFFICIAL_CLASS_NAME,
        "loader_symbol": OFFICIAL_LOADER_SYMBOL,
    }
    if identity != expected:
        raise RuntimeError(f"BRAINIAC_VENDOR_IDENTITY_MISMATCH:{identity}")
    return {
        "status": "PASS",
        "official_vendor_runtime": True,
        "python": str(python_version),
        "torch": str(torch_version),
        "monai": str(monai_version),
        **identity,
    }


def _load_pinned_vendor_encoder_class() -> type[nn.Module]:
    if not PINNED_MODEL_PATH.is_file():
        raise RuntimeError(f"BRAINIAC_VENDOR_MODEL_NOT_FOUND:{PINNED_MODEL_PATH}")
    module_name = "amuse_x_pinned_brainiac_vendor_model"
    spec = importlib.util.spec_from_file_location(module_name, PINNED_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("BRAINIAC_VENDOR_MODEL_IMPORT_SPEC_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    encoder_class = getattr(module, OFFICIAL_CLASS_NAME, None)
    if not isinstance(encoder_class, type) or not issubclass(encoder_class, nn.Module):
        raise RuntimeError("BRAINIAC_VENDOR_ENCODER_CLASS_MISSING")
    return encoder_class


class BrainIACViTEncoder(nn.Module):
    """Frozen 3D ViT-B SimCLR encoder for BrainIAC (96x96x96 input -> 768-dim embedding)."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        assert_brainiac_execution_allowed("EXTRACT_FROZEN_BRAINIAC_EMBEDDINGS")
        validate_official_runtime_identity(
            python_version=sys.version.split()[0],
            torch_version=torch.__version__,
            monai_version=monai.__version__,
            vendor_commit=PINNED_VENDOR_COMMIT,
            module_path=OFFICIAL_MODULE_PATH,
            class_name=OFFICIAL_CLASS_NAME,
            loader_symbol=OFFICIAL_LOADER_SYMBOL,
        )
        ckpt_file = DEFAULT_CKPT
        if not ckpt_file.is_file():
            raise RuntimeError(f"BRAINIAC_CHECKPOINT_NOT_FOUND:{ckpt_file}")

        checkpoint_sha256 = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()
        if checkpoint_sha256 != PINNED_CHECKPOINT_SHA256:
            raise RuntimeError("BRAINIAC_CHECKPOINT_SHA256_MISMATCH")
        if hashlib.sha256(PINNED_MODEL_PATH.read_bytes()).hexdigest() != PINNED_VENDOR_MODEL_SHA256:
            raise RuntimeError("BRAINIAC_VENDOR_MODEL_SHA256_MISMATCH")
        if (
            not PINNED_LICENSE_PATH.is_file()
            or hashlib.sha256(PINNED_LICENSE_PATH.read_bytes()).hexdigest()
            != PINNED_LICENSE_SHA256
        ):
            raise RuntimeError("BRAINIAC_VENDOR_LICENSE_SHA256_MISMATCH")

        vendor_encoder = _load_pinned_vendor_encoder_class()
        # The vendor constructor calls ``self.backbone.load_state_dict(...,
        # strict=True)``.  Do not replace this with a locally reconstructed ViT.
        self.vendor_encoder = vendor_encoder(str(ckpt_file))

        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_path = str(ckpt_file)
        self.vendor_model_sha256 = hashlib.sha256(
            PINNED_MODEL_PATH.read_bytes()
        ).hexdigest()
        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B, 1, 96, 96, 96) tensor. Output: (B, 768) CLS token embedding."""
        if x.ndim != 5 or tuple(x.shape[1:]) != (1, 96, 96, 96):
            raise RuntimeError(f"BRAINIAC_INPUT_SHAPE_MISMATCH:{tuple(x.shape)}")
        output = self.vendor_encoder(x)
        if output.ndim != 2 or tuple(output.shape[1:]) != (768,):
            raise RuntimeError(f"BRAINIAC_OUTPUT_SHAPE_MISMATCH:{tuple(output.shape)}")
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("BRAINIAC_OUTPUT_NONFINITE")
        return output


def load_brainiac_encoder(device: str = "cpu") -> BrainIACViTEncoder:
    """Instantiate and return the verified BrainIAC encoder in evaluation mode."""
    encoder = BrainIACViTEncoder(device=device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder
