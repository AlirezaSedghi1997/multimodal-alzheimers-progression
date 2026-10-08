"""Finite signed ROI perturbation of the actual frozen ensemble, not lesions."""
import numpy as np
import torch
from .spatial_target import ensemble_mean
from .core import file_sha


@torch.no_grad()
def roi_effects(targets, image, baseline, labels, mask):
    labels, mask = np.asarray(labels, int), np.asarray(mask, bool)
    if labels.shape != tuple(image.shape[2:]) or mask.shape != labels.shape:
        raise ValueError('OCCLUSION_LABEL_GEOMETRY')
    original = float(ensemble_mean(targets, image))
    if not np.isfinite(original) or not 0 <= original <= 1:
        raise ValueError('INVALID_FULL_ENSEMBLE_PROBABILITY')
    rows, result = [], np.zeros(labels.shape, dtype=np.float32)
    for label in sorted(set(labels[mask]) - {0}):
        indices = np.flatnonzero((labels == label) & mask)
        ix = torch.as_tensor(indices, device=image.device, dtype=torch.long)
        changed = image.clone()
        changed.flatten()[ix] = baseline.flatten()[ix]
        perturbed = float(ensemble_mean(targets, changed))
        if not np.isfinite(perturbed) or not 0 <= perturbed <= 1:
            raise ValueError('INVALID_OCCLUDED_ENSEMBLE_PROBABILITY')
        delta = original - perturbed
        result.ravel()[indices] = delta
        rows.append({'atlas_label': int(label), 'voxel_count': len(indices),
                     'original_risk': original, 'perturbed_risk': perturbed,
                     'signed_risk_difference': delta})
    return result, rows


def run_baseline(ctx, name, baseline_name, targets, random_targets, image, baseline, atlas, mask, affine):
    from scripts.xyz_postprimary_v1.rendering import save_nifti
    from src.evaluation.signed_spatial_validation_v1 import signed_similarity
    values, regions = roi_effects(targets, image, baseline, atlas, mask)
    random_map, random_regions = roi_effects(random_targets, image, baseline, atlas, mask)
    path = ctx.out / 'private/spatial' / name / (baseline_name + '_roi_occlusion.nii')
    save_nifti(path, values, affine)
    control_path = ctx.out / 'private/spatial' / name / (baseline_name + '_roi_occlusion_random.nii')
    save_nifti(control_path, random_map, affine)
    if not regions:
        raise ValueError('NO_DOCUMENTED_ROI_FOR_OCCLUSION')
    relation = signed_similarity(values, random_map, mask)
    return {'status': 'DESCRIPTIVE_ROI_PERTURBATION_COMPLETED_NOT_DISEASE_LOCALIZATION',
            'regions': regions, 'random_control_regions': random_regions,
            'random_weight_similarity': relation,
            'map': path.relative_to(ctx.out).as_posix(),
            'outputs': {p.relative_to(ctx.out).as_posix(): file_sha(p) for p in (path, control_path)},
            'map_units': 'ABSOLUTE_PROBABILITY_CHANGE_FOR_WHOLE_ROI_NOT_PER_VOXEL_CAUSAL_EFFECT',
            'risk_differences_are_not_additive_across_interacting_regions': True,
            'roi_volume_and_out_of_distribution_baseline_effects_remain': True,
            'sanity_status': 'DESCRIPTIVE_MATCHED_LABEL_CONTROL_PENDING',
            'fully_validated_attribution': False,
            'completeness_identity_of_IG_is_not_applicable': True,
            'separate_method_not_a_passing_relabel_of_failed_ig': True}
