"""Model-risk perturbation of anatomical labels, not lesion localization."""
import numpy as np
import torch


def region_controls(forward, image, baseline, attribution, mask, atlas, *, seed, repeats=99):
    mask = np.asarray(mask, bool)
    atlas = np.asarray(atlas, int)
    a = np.asarray(attribution)
    if a.shape != mask.shape or atlas.shape != a.shape or not mask.any():
        raise ValueError('ROI_FIDELITY_GEOMETRY')
    available = np.flatnonzero(mask)
    coordinates = np.column_stack(np.unravel_index(available, mask.shape)).astype(np.float32)
    labels = sorted(set(atlas[mask]) - {0})
    if not labels:
        return {'status': 'NOT_ESTIMABLE_NO_DOCUMENTED_LABELS'}
    rng = np.random.default_rng(seed)
    result = {'comparison_unit': 'ANATOMICAL_ROI_VS_EQUAL_VOXEL_COUNT_SPATIALLY_CLUSTERED_BRAIN_CONTROLS',
              'perturbations_may_be_out_of_distribution': True, 'statistical_population_inference': False}
    with torch.no_grad():
        original = float(forward(image).item())
        def drop(ids):
            changed = image.clone()
            idx = torch.as_tensor(ids, device=image.device, dtype=torch.long)
            changed.reshape(-1)[idx] = baseline.reshape(-1)[idx]
            value = original - float(forward(changed).item())
            if not np.isfinite(value):
                raise ValueError('NONFINITE_ROI_PERTURBATION')
            return value
        for name, sign in [('positive', 1), ('negative', -1)]:
            candidates = [(float(np.maximum(sign * a[(atlas == label) & mask], 0).mean()), label) for label in labels]
            score, label = max(candidates)
            if score <= 1e-10:
                result[name] = {'status': 'NOT_ESTIMABLE_NO_SIGNED_ROI_SUPPORT'}
                continue
            ids = np.flatnonzero((atlas == label) & mask)
            value = drop(ids)
            controls = []
            for _ in range(repeats):
                center = coordinates[rng.integers(len(coordinates))]
                distance = np.sum((coordinates - center) ** 2, axis=1)
                selected = np.argpartition(distance, len(ids) - 1)[:len(ids)]
                controls.append(drop(available[selected]))
            result[name] = {'status': 'ESTIMATED', 'atlas_label': int(label), 'voxel_count': len(ids),
                            'target_drop': value, 'random_target_drops': controls,
                            'selection': 'HIGHEST_MEAN_SIGNED_ATTRIBUTION_NO_OUTCOME_SELECTION',
                            'directional_margin_vs_random_mean': float(sign * (value - np.mean(controls))),
                            'descriptive_random_tail_fraction': float((1 + sum(sign * v >= sign * value for v in controls)) / (repeats + 1))}
    return result
