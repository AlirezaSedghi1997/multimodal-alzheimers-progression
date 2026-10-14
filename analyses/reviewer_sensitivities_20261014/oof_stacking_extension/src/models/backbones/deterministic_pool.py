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


class DeterministicAvgPool3d(nn.Module):
    """AvgPool3d equivalent whose CUDA backward is accepted in strict deterministic mode.

    PyTorch's native avg_pool3d_backward_cuda is marked nondeterministic.
    Unfolding the windows and averaging them elementwise preserves the native
    forward contract (zero padding included in the denominator, matching
    ``count_include_pad=True``) while avoiding that CUDA kernel: unfold, mean
    and view all have deterministic backward implementations.
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(kernel_size if stride is None else stride)
        self.padding = _triple(padding)
        self.ceil_mode = bool(ceil_mode)
        self.count_include_pad = bool(count_include_pad)
        self.divisor_override = None if divisor_override is None else int(divisor_override)
        if self.ceil_mode:
            raise ValueError("DeterministicAvgPool3d currently requires ceil_mode=False")
        if not self.count_include_pad:
            raise ValueError("DeterministicAvgPool3d currently requires count_include_pad=True")
        if self.divisor_override is not None:
            raise ValueError("DeterministicAvgPool3d does not support divisor_override")

    @classmethod
    def from_native(cls, module: nn.AvgPool3d) -> "DeterministicAvgPool3d":
        return cls(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
            count_include_pad=module.count_include_pad,
            divisor_override=module.divisor_override,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError(f"DeterministicAvgPool3d expects NCDHW input, got shape {tuple(value.shape)}")
        pd, ph, pw = self.padding
        if pd or ph or pw:
            value = F.pad(value, (pw, pw, ph, ph, pd, pd), value=0.0)
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        windows = value.unfold(2, kd, sd).unfold(3, kh, sh).unfold(4, kw, sw)
        flattened = windows.contiguous().view(*windows.shape[:5], kd * kh * kw)
        return flattened.mean(dim=-1)

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"ceil_mode={self.ceil_mode}, count_include_pad={self.count_include_pad}"
        )


def replace_native_avgpool3d(module: nn.Module) -> int:
    """Recursively replace native AvgPool3d (not adaptive) and return replacement count."""

    replaced = 0
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.AvgPool3d):
            setattr(module, name, DeterministicAvgPool3d.from_native(child))
            replaced += 1
        else:
            replaced += replace_native_avgpool3d(child)
    return replaced


def _adaptive_output_size(output_size: int | Sequence[int]) -> tuple[int, int, int]:
    # _triple() already implements exactly this contract for int and sized
    # sequences; adaptive pooling differs only in that it validates positivity,
    # which the caller does with a message naming the output grid.
    return _triple(output_size)


def deterministic_adaptive_avg_pool3d(
    value: torch.Tensor, output_size: int | Sequence[int]
) -> torch.Tensor:
    """``F.adaptive_avg_pool3d`` whose CUDA backward is deterministic.

    PyTorch's ``adaptive_avg_pool3d_backward_cuda`` accumulates with atomics and
    is refused under the strict Method X protocol enforced by
    ``scripts/train_3d_mri.py::enforce_method_x_strict_determinism``.  Relaxing
    that protocol is not an option: ``warn_only=False`` is an asserted contract
    (``tests/test_method_x_mx04_trainer_contract.py``) because exact resume and
    same-seed model selection depend on it, and the native ``avg_pool3d``
    backward is refused for the same reason, so it is not a substitute either.

    When the bins are uniform (``in % out == 0`` on every axis) adaptive average
    pooling is exactly non-overlapping window averaging, so the
    already-verified :class:`DeterministicAvgPool3d` (``unfold`` + ``mean``) is
    reused.  That covers every ``out == 1`` reduction and the whole locked
    128^3 encoder pyramid, whose successive scales are related by a factor of
    two.  Forward values and gradients match the native op to float64 round-off,
    and ``unfold``/``mean``/``view`` all have deterministic backward kernels
    without depending on ``CUBLAS_WORKSPACE_CONFIG``.

    Non-uniform bins are refused rather than approximated: the only exact
    alternative routes the reduction through cuBLAS, whose determinism depends
    on an environment variable that must be set before the process starts.  A
    loud refusal at the first forward is preferable to a silent loss of the
    reproducibility guarantee deep inside a training run.
    """

    if value.ndim != 5:
        raise ValueError(
            f"deterministic_adaptive_avg_pool3d expects NCDHW input, got shape {tuple(value.shape)}"
        )
    target = _adaptive_output_size(output_size)
    source = tuple(int(size) for size in value.shape[-3:])
    if any(size <= 0 for size in target):
        raise ValueError(f"Adaptive 3D pooling output must be positive, got {target}")
    if any(size < out for size, out in zip(source, target)):
        raise ValueError(
            f"Adaptive 3D pooling cannot upsample: input {source} to output {target}"
        )
    if all(size % out == 0 for size, out in zip(source, target)):
        kernel = tuple(size // out for size, out in zip(source, target))
        return DeterministicAvgPool3d(kernel_size=kernel, stride=kernel)(value)
    raise ValueError(
        "DeterministicAdaptiveAvgPool3d currently requires uniform adaptive bins "
        f"(input {source} to output {target} is not divisible on every axis); "
        "use a resolution whose encoder scales divide the fusion grid, or an "
        "output size of 1"
    )


class DeterministicAdaptiveAvgPool3d(nn.Module):
    """Module form of :func:`deterministic_adaptive_avg_pool3d`."""

    def __init__(self, output_size: int | Sequence[int] = 1) -> None:
        super().__init__()
        self.output_size = _adaptive_output_size(output_size)

    @classmethod
    def from_native(cls, module: nn.AdaptiveAvgPool3d) -> "DeterministicAdaptiveAvgPool3d":
        return cls(module.output_size)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return deterministic_adaptive_avg_pool3d(value, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


def replace_native_adaptive_pool3d(module: nn.Module) -> int:
    """Recursively replace native adaptive 3-D pools and return replacement count."""

    replaced = 0
    for name, child in tuple(module.named_children()):
        if isinstance(child, (nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool3d)):
            if isinstance(child, nn.AdaptiveMaxPool3d):
                raise ValueError(
                    "DeterministicAdaptiveMaxPool3d is not implemented; refusing to "
                    "silently substitute an averaging operator"
                )
            setattr(module, name, DeterministicAdaptiveAvgPool3d.from_native(child))
            replaced += 1
        else:
            replaced += replace_native_adaptive_pool3d(child)
    return replaced
