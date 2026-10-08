from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeVar


T = TypeVar("T")


def seed_everything(seed: int) -> None:
    """Seed supported RNGs without requiring optional training dependencies."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass
    try:
        from monai.utils import set_determinism

        set_determinism(seed=int(seed))
    except ImportError:
        pass


def canonical_digest(value: object) -> str:
    """Hash a JSON-compatible contract with stable key ordering."""

    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_tree_digest(paths: Iterable[str | Path], *, base: str | Path) -> str:
    """Hash source paths and bytes in a stable order for fail-closed resume."""

    root = Path(base).resolve()
    digest = hashlib.sha256()
    for item in sorted((Path(path).resolve() for path in paths), key=lambda path: path.as_posix()):
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_version_contract() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_runtime": None,
        "cudnn": None,
        "gpu_name": None,
        "gpu_compute_capability": None,
    }
    for package in ("torch", "numpy", "monai", "h5py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        import torch

        versions["cuda_runtime"] = str(torch.version.cuda) if torch.version.cuda else None
        cudnn_version = torch.backends.cudnn.version()
        versions["cudnn"] = str(cudnn_version) if cudnn_version is not None else None
        if torch.cuda.is_available():
            versions["gpu_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            versions["gpu_compute_capability"] = f"{major}.{minor}"
    except ImportError:
        pass
    return versions


def capture_rng_state() -> dict[str, object]:
    """Capture supported RNGs using only weights-only-safe primitives/tensors."""

    state: dict[str, object] = {"python": random.getstate(), "numpy": None, "torch_cpu": None, "torch_cuda": []}
    try:
        import numpy as np

        np_state = np.random.get_state()
        state["numpy"] = (
            str(np_state[0]),
            np_state[1].astype("uint32").tolist(),
            int(np_state[2]),
            int(np_state[3]),
            float(np_state[4]),
        )
    except ImportError:
        pass
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)  # type: ignore[arg-type]
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np

            name, keys, position, has_gauss, cached_gaussian = numpy_state  # type: ignore[misc]
            np.random.set_state(
                (
                    str(name),
                    np.asarray(keys, dtype="uint32"),
                    int(position),
                    int(has_gauss),
                    float(cached_gaussian),
                )
            )
        except ImportError:
            pass
    try:
        import torch

        torch_cpu = state.get("torch_cpu")
        if torch_cpu is not None:
            torch.set_rng_state(torch_cpu)  # type: ignore[arg-type]
        torch_cuda = state.get("torch_cuda") or []
        if torch.cuda.is_available() and torch_cuda:
            torch.cuda.set_rng_state_all(torch_cuda)  # type: ignore[arg-type]
    except ImportError:
        pass


def seeded_torch_generator(seed: int):
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""

    del worker_id
    import torch

    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


def seeded_shuffle_aligned(*columns: Sequence[T], seed: int) -> tuple[list[T], ...]:
    """Shuffle aligned columns reproducibly while preserving row alignment."""

    if not columns:
        return ()
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError("Aligned columns must have equal lengths")
    rows = list(zip(*columns, strict=True))
    random.Random(int(seed)).shuffle(rows)
    if not rows:
        return tuple([] for _ in columns)
    shuffled = zip(*rows, strict=True)
    return tuple(list(column) for column in shuffled)


def split_membership_digest(**splits: Iterable[object]) -> str:
    """Return an aggregate-only digest that is insensitive to member order."""

    normalized = {
        str(name): sorted(str(value) for value in values)
        for name, values in sorted(splits.items())
    }
    payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
