from __future__ import annotations

from typing import Any


def resolve_dataloader_kwargs(
    *,
    num_workers: int,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> dict[str, Any]:
    """Validate DataLoader runtime knobs without changing scientific inputs."""

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be at least 1")
    options: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    # PyTorch rejects worker-only options when num_workers == 0.
    if num_workers > 0:
        options.update(
            {
                "persistent_workers": bool(persistent_workers),
                "prefetch_factor": int(prefetch_factor),
            }
        )
    return options
