"""Signed input attribution with numerical and perturbation controls.

This explains a frozen model, not disease tissue. No outcome is accepted here.
"""
from __future__ import annotations
import numpy as np
import torch


def integrated_gradients(forward, image, baseline, *, steps=(16, 32, 64), progress=None):
    if image.shape != baseline.shape or image.ndim != 5 or image.shape[:2] != (1, 1):
        raise ValueError('SINGLE_VOLUME_SHAPE_REQUIRED')
    if not torch.isfinite(image).all() or not torch.isfinite(baseline).all():
        raise ValueError('NONFINITE_INPUT')
    x, b = image.detach(), baseline.detach()
    with torch.no_grad():
        fx, fb = float(forward(x).item()), float(forward(b).item())
    if not np.isfinite([fx, fb]).all():
        raise ValueError('NONFINITE_TARGET')
    delta = fx - fb
    history = []
    for n in steps:
        nodes, weights = np.polynomial.legendre.leggauss(n)
        total = torch.zeros_like(x)
        for i, (alpha, weight) in enumerate(zip((nodes + 1) / 2, weights / 2)):
            point = (b + float(alpha) * (x - b)).requires_grad_(True)
            value = forward(point)
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError('INVALID_SCALAR_TARGET')
            grad, = torch.autograd.grad(value.sum(), point)
            if not torch.isfinite(grad).all():
                raise ValueError('NONFINITE_GRADIENT')
            total += float(weight) * grad.detach()
            if progress and (i == 0 or (i + 1) % 8 == 0):
                progress(n, i + 1)
        attribution = (x - b) * total
        residual = abs(float(attribution.double().sum().item()) - delta)
        tolerance = 1e-4 + .05 * abs(delta)
        history.append({'steps': n, 'absolute_completeness_error': residual,
                        'tolerance': tolerance})
        if residual <= tolerance:
            break
    arr = attribution.detach().cpu().numpy().squeeze().astype('f4')
    nonzero = bool(np.max(np.abs(arr)) > 1e-10)
    return arr, {'target': fx, 'baseline_target': fb, 'target_difference': delta,
                 'quadrature_history': history, 'nonzero': nonzero,
                 'numerical_status': 'PASS' if residual <= tolerance and nonzero
                 else 'NOT_ESTIMABLE_ZERO_MAP' if not nonzero else 'FAIL_COMPLETENESS'}


def signed_similarity(a, b, mask):
    x, y = np.asarray(a)[mask].astype('f8'), np.asarray(b)[mask].astype('f8')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('NONFINITE_MAP')
    if min(float(x.std()), float(y.std())) < 1e-12:
        return {'status': 'NOT_ESTIMABLE_CONSTANT_MAP', 'pearson': None}
    return {'status': 'ESTIMATED', 'pearson': float(np.corrcoef(x, y)[0, 1])}


def perturbation_controls(forward, image, baseline, attribution, mask, *, seed=13,
                          repeats=20, fraction=.05):
    """Equal-volume voxel controls, not an anatomical ROI/localization test.

    Top positive and negative sets are disjoint. The fixed protocol also reports
    the entire random-control distribution; a passing direction is descriptive.
    """
    mask = np.asarray(mask, bool)
    a = np.asarray(attribution)
    if tuple(image.shape[2:]) != mask.shape or a.shape != mask.shape or not mask.any():
        raise ValueError('MASK_MAP_GEOMETRY')
    locations = np.flatnonzero(mask)
    budget = max(1, int(len(locations) * fraction))
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        original = float(forward(image).item())
        def drop(indices):
            altered = image.clone()
            idx = torch.as_tensor(indices, dtype=torch.long, device=image.device)
            altered.flatten()[idx] = baseline.flatten()[idx]
            value = original - float(forward(altered).item())
            if not np.isfinite(value):
                raise ValueError('NONFINITE_PERTURBATION')
            return value
        result = {}
        for direction, sign in [('positive', 1), ('negative', -1)]:
            eligible = locations[sign * a.ravel()[locations] > 1e-10]
            n = min(budget, len(eligible))
            if n == 0:
                result[direction] = {'status': 'NOT_ESTIMABLE_NO_SIGNED_SUPPORT'}
                continue
            top = eligible[np.argsort(sign * a.ravel()[eligible], kind='stable')[-n:]]
            top_drop = drop(top)
            random_drops = [drop(rng.choice(locations, n, replace=False)) for _ in range(repeats)]
            result[direction] = {'status': 'ESTIMATED', 'voxel_count': n,
                'target_drop': top_drop, 'random_target_drops': random_drops,
                'directional_margin_vs_random_mean': float(sign * (top_drop - np.mean(random_drops))),
                'descriptive_random_tail_fraction': float((1 + sum(sign*d >= sign*top_drop for d in random_drops))/(repeats+1))}
    return {'comparison_unit': 'EQUAL_VOLUME_VOXEL_SET_NOT_ANATOMICAL_ROI',
            'perturbations_may_be_out_of_distribution': True,
            'statistical_population_inference': False, **result}


def reset_model_in_memory(model, seed=13):
    """Use each module's own initialization, preserving GN/BN conventions."""
    torch.manual_seed(seed)
    count = 0
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            count += 1
    if not count:
        raise ValueError('MODEL_HAS_NO_RESETTABLE_MODULES')
    model.eval()
    return count
