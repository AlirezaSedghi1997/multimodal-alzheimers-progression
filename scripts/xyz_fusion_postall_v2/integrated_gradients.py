"""AMP-preserving IG with scaled input gradients and unscaled completeness."""
import numpy as np
import torch


def integrated_gradients(forward, image, baseline, *, steps=(16, 32, 64), gradient_scale=1024.):
    # https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/amp_examples.rst
    if image.shape != baseline.shape or image.ndim != 5 or image.shape[:2] != (1, 1):
        raise ValueError('SINGLE_VOLUME_SHAPE_REQUIRED')
    if not torch.isfinite(image).all() or not torch.isfinite(baseline).all():
        raise ValueError('NONFINITE_INPUT')
    if not np.isfinite(gradient_scale) or gradient_scale < 1:
        raise ValueError('INVALID_FIXED_GRADIENT_SCALE')
    x, b = image.detach(), baseline.detach()
    with torch.no_grad():
        fx, fb = float(forward(x)), float(forward(b))
    if not np.isfinite([fx, fb]).all():
        raise ValueError('NONFINITE_TARGET')
    delta = fx - fb
    history = []
    accepted_scales, rejected_scales = [], []
    for count in steps:
        nodes, weights = np.polynomial.legendre.leggauss(count)
        total = torch.zeros_like(x)
        for alpha, weight in zip((nodes + 1) / 2, weights / 2):
            # Numerical rescaling only: fixed quadrature, forward model, and
            # acceptance tolerance never change after an overflow.
            finite = False
            for scale in [gradient_scale / (4 ** k) for k in range(8)]:
                point = (b + float(alpha) * (x - b)).requires_grad_(True)
                value = forward(point)
                if value.numel() != 1 or not torch.isfinite(value).all():
                    raise ValueError('INVALID_SCALAR_TARGET')
                grad, = torch.autograd.grad(value * scale, point)
                grad = grad.float() / scale
                if torch.isfinite(grad).all():
                    accepted_scales.append(scale)
                    finite = True
                    break
                rejected_scales.append(scale)
            if not finite:
                raise ValueError('NONFINITE_SCALED_INPUT_GRADIENT')
            total += float(weight) * grad.detach()
        attribution = (x - b) * total
        residual = abs(float(attribution.double().sum()) - delta)
        tolerance = 1e-4 + .05 * abs(delta)
        history.append({'steps': count, 'absolute_completeness_error': residual, 'tolerance': tolerance})
        if residual <= tolerance:
            break
    array = attribution.detach().cpu().numpy().squeeze().astype('f4')
    nonzero = bool(np.max(np.abs(array)) > 1e-10)
    return array, {'target': fx, 'baseline_target': fb, 'target_difference': delta,
                   'quadrature_history': history, 'nonzero': nonzero,
                   'maximum_gradient_scale': gradient_scale, 'minimum_used_scale': min(accepted_scales),
                   'gradient_overflow_retries': len(rejected_scales),
                   'completeness_in_original_risk_units': True,
                   'forward_precision': 'PARENT_AMP_POLICY_NOT_FP32_SUBSTITUTE',
                   'numerical_status': 'PASS' if residual <= tolerance and nonzero else
                       'NOT_ESTIMABLE_ZERO_MAP' if not nonzero else 'FAIL_COMPLETENESS'}
