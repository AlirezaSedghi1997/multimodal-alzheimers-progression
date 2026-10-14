"""FP32 forward and gradients, float64 quadrature; never rescale a map to pass."""
import time
import numpy as np
import torch


def compare_maps(previous, current):
    a, b = np.asarray(previous, dtype=float), np.asarray(current, dtype=float)
    norm = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    return {'cosine': float((a * b).sum() / norm) if norm else None,
            'relative_l1': float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-15))}


def quadrature(forward, image, baseline, steps, options):
    if image.shape != baseline.shape or image.shape[:2] != (1, 1) or image.ndim != 5:
        raise ValueError('SINGLE_VOLUME_REQUIRED')
    if image.dtype != torch.float32 or not torch.isfinite(image).all() or not torch.isfinite(baseline).all():
        raise ValueError('FINITE_FP32_INPUT_REQUIRED')
    started = time.perf_counter()
    x, b = image.detach(), baseline.detach()
    with torch.no_grad():
        fx, fb = float(forward(x)), float(forward(b))
    if not np.isfinite([fx, fb]).all():
        raise ValueError('NONFINITE_TARGET')
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    total = torch.zeros_like(x, dtype=torch.float64)
    for alpha, weight in zip((nodes + 1) / 2, weights / 2):
        point = (b + float(alpha) * (x - b)).requires_grad_(True)
        value = forward(point)
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError('NONFINITE_SCALAR_TARGET')
        gradient, = torch.autograd.grad(value, point)
        if not torch.isfinite(gradient).all():
            raise ValueError('NONFINITE_FP32_GRADIENT')
        total.add_(gradient.detach().double(), alpha=float(weight))
    # Check the exact stored float32 map, not a different high-precision intermediate.
    array = ((x - b).double() * total).detach().cpu().numpy().squeeze().astype('f4')
    signed_sum = float(array.astype(float).sum())
    residual = abs(signed_sum - (fx - fb))
    tolerance = options['ig_absolute_tolerance'] + options['ig_relative_tolerance'] * abs(fx - fb)
    nonzero = bool(np.max(np.abs(array)) > 1e-10)
    return array, {'target': fx, 'baseline_target': fb, 'target_difference': fx - fb,
        'signed_sum': signed_sum, 'steps': steps, 'absolute_completeness_error': residual,
        'tolerance': tolerance, 'nonzero': nonzero, 'seconds': time.perf_counter() - started,
        'numerical_status': ('NOT_ESTIMABLE_ZERO_MAP' if not nonzero else
                             'PASS' if residual <= tolerance else 'FAIL_COMPLETENESS'),
        'forward_precision': 'FP32_CHILD_NOT_AMP_PARENT', 'accumulation': 'FLOAT64',
        'attribution_rescaled_to_force_completeness': False}


def converged(history, similarity, options):
    return (len(history) >= 2 and all(h['numerical_status'] == 'PASS' for h in history[-2:])
            and similarity.get('cosine') is not None
            and similarity['cosine'] >= options['ig_stability_cosine']
            and similarity['relative_l1'] <= options['ig_stability_relative_l1'])
