from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, Sequence):
        values = tuple(int(item) for item in value)
        if len(values) != 3:
            raise ValueError("3D pooling parameters must contain exactly three values")
        return values
    item = int(value)
    return item, item, item


class DeterministicMaxPool3d(nn.Module):
    """MaxPool3d equivalent whose CUDA backward is accepted in strict deterministic mode.

    PyTorch's native max_pool3d_with_indices_backward_cuda is marked
    nondeterministic.  Unfolding the windows and taking their first maximum
    preserves native forward and tie-gradient semantics for the supported
    dilation=1/floor-output contract while avoiding that CUDA kernel.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(kernel_size if stride is None else stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.return_indices = bool(return_indices)
        self.ceil_mode = bool(ceil_mode)
        if self.dilation != (1, 1, 1):
            raise ValueError("DeterministicMaxPool3d currently requires dilation=1")
        if self.return_indices:
            raise ValueError("DeterministicMaxPool3d does not expose pooling indices")
        if self.ceil_mode:
            raise ValueError("DeterministicMaxPool3d currently requires ceil_mode=False")

    @classmethod
    def from_native(cls, module: nn.MaxPool3d) -> "DeterministicMaxPool3d":
        return cls(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            return_indices=module.return_indices,
            ceil_mode=module.ceil_mode,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError(f"DeterministicMaxPool3d expects NCDHW input, got shape {tuple(value.shape)}")
        pd, ph, pw = self.padding
        if pd or ph or pw:
            value = F.pad(value, (pw, pw, ph, ph, pd, pd), value=float("-inf"))
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        windows = value.unfold(2, kd, sd).unfold(3, kh, sh).unfold(4, kw, sw)
        flattened = windows.contiguous().view(*windows.shape[:5], kd * kh * kw)
        return flattened.max(dim=-1).values


def replace_native_maxpool3d(module: nn.Module) -> int:
    """Recursively replace supported native pools and return replacement count."""

    replaced = 0
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.MaxPool3d):
            setattr(module, name, DeterministicMaxPool3d.from_native(child))
            replaced += 1
        else:
            replaced += replace_native_maxpool3d(child)
    return replaced
