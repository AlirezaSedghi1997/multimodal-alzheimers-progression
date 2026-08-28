"""Verify the pinned BrainIAC encoder with the exact public implementation.

This is an asset/runtime verification action, not study-data inference.  It is
therefore intentionally independent of the later R1 execution gate: no A4 or
LEARN image, label, split, or outer-test artifact is opened.  The resulting
receipts bind the isolated H-side environment, the read-only Torch/CUDA overlay,
the vendor source tree, the official checkpoint, an actual ``strict=True``
constructor call, and a real 96-cubed forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENDOR_COMMIT = "ba60f45bed832ee0e5683c678caeaeefe2072f0d"
VENDOR_ROOT = ROOT / "vendor" / "BrainIAC" / VENDOR_COMMIT
VENDOR_SOURCE = VENDOR_ROOT / "src"
MODEL_FILE = VENDOR_SOURCE / "model.py"
LOADER_FILE = VENDOR_SOURCE / "load_brainiac.py"
LICENSE_FILE = VENDOR_ROOT / "LICENSE"
CHECKPOINT = ROOT / "models" / "pretrained" / "brainiac" / "BrainIAC.ckpt"
RUNTIME_ROOT = ROOT / ".venvs" / "brainiac"
REPORT_ROOT = ROOT / "results" / "method_x_registry" / "brainiac_r1"
RUNTIME_LOCK = REPORT_ROOT / "brainiac_official_runtime_lock.json"
STRICT_RECEIPT = REPORT_ROOT / "brainiac_official_strict_load_receipt.json"
FORWARD_RECEIPT = REPORT_ROOT / "brainiac_official_forward_receipt.json"
EXPECTED_CHECKPOINT_SHA256 = (
    "f22bdbcae26823a9d9e8aee883c6f24386ba4617339c12269848b6666cc62693"
)
EXPECTED_CHECKPOINT_BYTES = 362119240


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _write_canonical_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_vendor_class() -> type[Any]:
    spec = importlib.util.spec_from_file_location(
        "amuse_x_verified_vendor_brainiac_model", MODEL_FILE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("BRAINIAC_VENDOR_IMPORT_SPEC_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    encoder = getattr(module, "ViTBackboneNet", None)
    if not isinstance(encoder, type):
        raise RuntimeError("BRAINIAC_VENDOR_ENCODER_CLASS_MISSING")
    return encoder


def _runtime_lock(torch: Any, monai: Any, numpy: Any) -> dict[str, Any]:
    from src.research.brainiac_model_loader import validate_official_runtime_identity

    identity = validate_official_runtime_identity(
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        monai_version=monai.__version__,
        vendor_commit=VENDOR_COMMIT,
        module_path="src/model.py",
        class_name="ViTBackboneNet",
        loader_symbol="load_brainiac",
    )
    prefix = Path(sys.prefix).resolve()
    if prefix != RUNTIME_ROOT.resolve():
        raise RuntimeError(f"BRAINIAC_RUNTIME_PREFIX_INVALID:{prefix}")
    monai_path = Path(monai.__file__).resolve()
    try:
        monai_path.relative_to(prefix)
    except ValueError as exc:
        raise RuntimeError("BRAINIAC_MONAI_NOT_ISOLATED") from exc
    torch_path = Path(torch.__file__).resolve()
    expected_torch_root = (ROOT / ".venv" / "Lib" / "site-packages").resolve()
    try:
        torch_path.relative_to(expected_torch_root)
    except ValueError as exc:
        raise RuntimeError("BRAINIAC_TORCH_OVERLAY_SOURCE_INVALID") from exc
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    freeze = sorted(line.strip() for line in freeze if line.strip())
    overlay = prefix / "Lib" / "site-packages" / "amuse_x_battery_torch_readonly.pth"
    if not overlay.is_file():
        raise RuntimeError("BRAINIAC_TORCH_OVERLAY_RECEIPT_MISSING")
    body = {
        "schema_version": 1,
        "status": "PASS",
        "receipt_type": "BRAINIAC_OFFICIAL_RUNTIME_LOCK",
        "runtime_prefix": str(prefix),
        "interpreter_path": str(Path(sys.executable).resolve()),
        "interpreter_sha256": _sha256_file(Path(sys.executable)),
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_path": str(torch_path),
        "torch_overlay_mode": "READ_ONLY_BATTERY_TORCH_BINARY_OVERLAY",
        "torch_overlay_path": str(overlay.resolve()),
        "torch_overlay_sha256": _sha256_file(overlay),
        "monai": str(monai.__version__),
        "monai_path": str(monai_path),
        "numpy": str(numpy.__version__),
        "pip_freeze": freeze,
        "pip_freeze_sha256": _canonical_sha(freeze),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "runtime_identity": identity,
        "scientific_environment_isolated_from_battery_mutation": True,
    }
    body["integrity_sha256"] = _canonical_sha(body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    for required in (MODEL_FILE, LOADER_FILE, LICENSE_FILE, CHECKPOINT):
        if not required.is_file():
            raise RuntimeError(f"BRAINIAC_REQUIRED_ASSET_MISSING:{required}")
    checkpoint_bytes = CHECKPOINT.stat().st_size
    checkpoint_sha = _sha256_file(CHECKPOINT)
    if (
        checkpoint_bytes != EXPECTED_CHECKPOINT_BYTES
        or checkpoint_sha != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("BRAINIAC_CHECKPOINT_IDENTITY_MISMATCH")

    import monai
    import numpy as np
    import torch

    runtime = _runtime_lock(torch, monai, np)
    _write_canonical_json(RUNTIME_LOCK, runtime)
    runtime_lock_file_sha = _sha256_file(RUNTIME_LOCK)
    from src.research.brainiac_adapter import stable_source_tree_sha256

    source_tree_sha = stable_source_tree_sha256(VENDOR_ROOT)
    if not source_tree_sha:
        raise RuntimeError("BRAINIAC_VENDOR_SOURCE_TREE_EMPTY")
    module_identity = {
        "commit": VENDOR_COMMIT,
        "source_tree_sha256": source_tree_sha,
        "encoder_module_path": "src/model.py",
        "encoder_class_name": "ViTBackboneNet",
        "loader_symbol": "load_brainiac",
    }
    module_identity_sha = _canonical_sha(module_identity)

    encoder_class = _load_vendor_class()
    model = encoder_class(str(CHECKPOINT))
    model.eval()
    model_state = model.backbone.state_dict()
    if not model_state:
        raise RuntimeError("BRAINIAC_STRICT_LOAD_EMPTY_STATE")

    checkpoint_payload = torch.load(
        str(CHECKPOINT), map_location="cpu", weights_only=False
    )
    raw_state = checkpoint_payload.get("state_dict", checkpoint_payload)
    if not isinstance(raw_state, dict):
        raise RuntimeError("BRAINIAC_CHECKPOINT_STATE_INVALID")
    checkpoint_state = {
        key[9:]: value
        for key, value in raw_state.items()
        if isinstance(key, str) and key.startswith("backbone.")
    }
    if not checkpoint_state:
        raise RuntimeError("BRAINIAC_CHECKPOINT_BACKBONE_EMPTY")
    model_keys = sorted(model_state)
    checkpoint_keys = sorted(checkpoint_state)
    if model_keys != checkpoint_keys:
        missing = sorted(set(model_keys) - set(checkpoint_keys))
        unexpected = sorted(set(checkpoint_keys) - set(model_keys))
        raise RuntimeError(
            f"BRAINIAC_ACTUAL_STRICT_KEY_MISMATCH:{len(missing)}:{len(unexpected)}"
        )
    shape_map = {key: list(model_state[key].shape) for key in model_keys}
    checkpoint_shape_map = {
        key: list(checkpoint_state[key].shape) for key in checkpoint_keys
    }
    if shape_map != checkpoint_shape_map:
        raise RuntimeError("BRAINIAC_ACTUAL_STRICT_SHAPE_MISMATCH")
    del checkpoint_payload, raw_state, checkpoint_state

    strict_body = {
        "schema_version": 1,
        "receipt_type": "ACTUAL_STRICT_LOAD",
        "status": "PASS",
        "strict": True,
        "matched_keys": len(model_keys),
        "missing_keys": 0,
        "unexpected_keys": 0,
        "shape_key_count": len(shape_map),
        "model_keyset_sha256": _canonical_sha(model_keys),
        "shape_map_sha256": _canonical_sha(shape_map),
        "checkpoint_path": str(CHECKPOINT.resolve()),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_sha256": checkpoint_sha,
        "source_tree_sha256": source_tree_sha,
        "model_file_sha256": _sha256_file(MODEL_FILE),
        "loader_file_sha256": _sha256_file(LOADER_FILE),
        "license_sha256": _sha256_file(LICENSE_FILE),
        "environment_lock_path": str(RUNTIME_LOCK.resolve()),
        "environment_lock_sha256": runtime_lock_file_sha,
        "environment_lock_integrity_sha256": runtime["integrity_sha256"],
        "module_identity_sha256": module_identity_sha,
        "monai": str(monai.__version__),
        "torch": str(torch.__version__),
        "constructor": "vendor.src.model.ViTBackboneNet",
        "constructor_load_semantics": "load_state_dict_strict_true",
    }
    strict_body["integrity_sha256"] = _canonical_sha(strict_body)
    _write_canonical_json(STRICT_RECEIPT, strict_body)

    requested = args.device
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BRAINIAC_CUDA_REQUESTED_BUT_UNAVAILABLE")
    torch.manual_seed(20260821)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260821)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    model = model.to(device)
    input_tensor = torch.linspace(
        -1.0,
        1.0,
        steps=96 * 96 * 96,
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 96, 96, 96)
    with torch.inference_mode():
        output = model(input_tensor)
    if tuple(output.shape) != (1, 768) or not bool(torch.isfinite(output).all()):
        raise RuntimeError(f"BRAINIAC_ACTUAL_FORWARD_INVALID:{tuple(output.shape)}")
    output_cpu = output.detach().to(dtype=torch.float32, device="cpu").contiguous()
    output_sha = _sha256_bytes(output_cpu.numpy().tobytes())
    forward_body = {
        "schema_version": 1,
        "receipt_type": "ACTUAL_FORWARD",
        "status": "PASS",
        "finite": True,
        "device": device,
        "input_kind": "DETERMINISTIC_SYNTHETIC_NO_STUDY_DATA",
        "input_shape": [1, 1, 96, 96, 96],
        "input_sha256": _sha256_bytes(
            input_tensor.detach().to(device="cpu").contiguous().numpy().tobytes()
        ),
        "output_shape": [1, 768],
        "output_sha256": output_sha,
        "output_mean": float(output_cpu.mean()),
        "output_std": float(output_cpu.std(unbiased=False)),
        "checkpoint_sha256": checkpoint_sha,
        "source_tree_sha256": source_tree_sha,
        "environment_lock_sha256": runtime_lock_file_sha,
        "module_identity_sha256": module_identity_sha,
        "strict_load_receipt_path": str(STRICT_RECEIPT.resolve()),
        "strict_load_receipt_file_sha256": _sha256_file(STRICT_RECEIPT),
        "strict_load_receipt_sha256": strict_body["integrity_sha256"],
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0
        ),
        "study_images_read": 0,
        "labels_read": 0,
        "outer_test_touched": False,
    }
    forward_body["integrity_sha256"] = _canonical_sha(forward_body)
    _write_canonical_json(FORWARD_RECEIPT, forward_body)
    print(
        json.dumps(
            {
                "status": "PASS",
                "strict_receipt": str(STRICT_RECEIPT),
                "forward_receipt": str(FORWARD_RECEIPT),
                "output_shape": list(output.shape),
                "device": device,
                "peak_vram_bytes": forward_body["peak_vram_bytes"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
