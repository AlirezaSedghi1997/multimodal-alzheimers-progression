"""Signed whole-ensemble risk maps with per-component resumable controls."""
from __future__ import annotations

import io
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import gaussian_filter

from .core import digest, file_sha, read_json
from .experts import load_expert, load_head
from .spatial_target import ComponentTarget, FixedInferenceBatch, ensemble_mean


def compressed_map(path, array):
    from scripts.xyz_postprimary_v1.rendering import atomic_bytes
    stream = io.BytesIO()
    np.savez_compressed(stream, attribution=np.asarray(array, dtype=np.float32))
    atomic_bytes(path, stream.getvalue())


def map_value(ctx, receipt):
    path = ctx.out / receipt['map']
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data['attribution'], dtype=np.float32)


def make_target(ctx, task, arm, modality, row, control='trained'):
    from scripts.xyz_fusion_v2.training import predict
    from scripts.xyz_fusion_v2.models import score_features
    lock = ctx.parent.job(task + '_lock')
    if row.iloc[0].patient_hash in set(lock['meta_subjects']) | set(lock['expert_subjects']):
        raise ValueError('EXPLAINED_SUBJECT_SEEN_IN_TRAINING')
    if control not in ('trained', 'random'):
        raise ValueError('NEW_CNN_TRAINING_IS_OUT_OF_SCOPE')
    models = {m: load_expert(ctx, task, m, randomized=control == 'random')[0] for m in lock['experts']}
    head, receipt = load_head(ctx, task, arm)
    features = receipt['features']
    ordered = ctx.parent.job(task + '_predictions')['arms'][arm]['subjects']
    position = ordered.index(row.iloc[0].patient_hash)
    microbatch = ctx.config['training']['microbatch_size']
    start = (position // microbatch) * microbatch
    source = ctx.frame if task.startswith('mri_') else ctx.pet
    batch_rows = source.set_index('patient_hash').loc[ordered[start:start + microbatch]].reset_index()
    local_position = position - start
    values = row.copy()
    for mod, model in models.items():
        policy = dict(ctx.config['training'], modality=mod)
        p = predict(model, ctx.banks()[mod], batch_rows[['patient_hash', 'cache_row']], policy,
                    next(model.parameters()).device, ctx.config['fusion']['ibs_grid_days'])
        values = score_features(values, p[local_position:local_position + 1], ctx.config['fusion']['ibs_grid_days'], ctx.config['fusion']['score_times_days'], mod)
    model = models[modality]
    batch = ctx.banks()[modality].batch(batch_rows.cache_row.to_numpy(), next(model.parameters()).device)
    target = ComponentTarget(FixedInferenceBatch(model, batch, local_position), head, values[features].to_numpy(dtype=float), features, modality,
                             ctx.config['training'], ctx.config['fusion']['ibs_grid_days'],
                             ctx.config['fusion']['score_times_days'])
    return target


def verify_prediction_parity(ctx, task, arm, key, target, image):
    pred = ctx.parent.job(task + '_predictions')['arms'][arm]
    position = pred['subjects'].index(key)
    expected = 1 - np.interp(1095.75, ctx.config['fusion']['ibs_grid_days'], pred['survival'][position])
    with torch.no_grad():
        observed = float(target(image).item())
    error = abs(observed - expected)
    tolerance = ctx.options['spatial_parity_absolute_tolerance']
    if not np.isfinite(error) or error > tolerance:
        raise ValueError(f'FULL_COMPONENT_RISK_PARITY_FAILED:{task}:error={error:.12g}')
    return {'absolute_error': error, 'tolerance': tolerance, 'target': observed, 'locked_prediction': float(expected)}


def geometry(ctx, row, modality):
    if modality == 'mri':
        folder = ctx.root / 'results/xyz_mri_baseline_fast_v1/private/mri_scans' / row.iloc[0].mri_sample_hash
        path = folder / 'registered_t1.nii.gz'
        image = nib.load(path)
        return image, {'background': 'ACTUAL_SUBJECT_REGISTERED_MRI', 'image_sha256': file_sha(path)}
    import h5py
    r = row.iloc[0]
    reference = ctx.root / 'data/templates/method_y_mni152nlin2009casym_v1/reference_128_1p5mm.nii.gz'
    with h5py.File(r.image_path_private, 'r') as h:
        # If the original cache lacks a documented geometry binding, do not
        # silently assign an affine merely because both arrays are 128 cubed.
        expected = h.attrs.get('reference_template_sha256')
        group = h[str(r.h5_group)]
        affine = group.attrs.get('affine')
        if expected is not None and str(expected) == file_sha(reference):
            matrix = nib.load(reference).affine
        elif affine is not None:
            matrix = np.asarray(affine, dtype=float).reshape(4, 4)
            if not np.allclose(matrix, nib.load(reference).affine, atol=1e-5):
                raise ValueError('PET_CACHE_GEOMETRY_DIFFERS_FROM_LOCKED_MODEL_SPACE')
        else:
            raise ValueError('PET_CACHE_GEOMETRY_PROVENANCE_REQUIRED')
        array = np.asarray(group['image'], dtype=np.float32).squeeze()
    return nib.Nifti1Image(array, matrix), {'background': 'ACTUAL_SUBJECT_PATTERN_PET_NOT_QUANTITATIVE_SUVR',
                                          'source_image_sha256': str(r.source_image_sha256)}


def case_maps(ctx, fold, case, key, track, modality):
    from src.evaluation.signed_spatial_validation_v1 import signed_similarity
    from .integrated_gradients import integrated_gradients
    from .roi_fidelity import region_controls
    from scripts.xyz_postprimary_v1.rendering import save_nifti, overlay
    from scripts.xyz_fusion_v2.atlas import roi_summary
    frame = ctx.frame if track == 'mri' else ctx.pet
    row = frame.loc[frame.patient_hash.eq(key)]
    name = f'f04_{track}_{modality}_o{fold}_case{case}'
    if row.empty:
        return {'status': 'NOT_ELIGIBLE_NO_LOCKED_PET_IMAGE', 'fold': fold, 'case': case,
                'replacement_selected': False}
    arm = 'clinical_mri_amyloid' if track == 'mri' else 'clinical_mri_amyloid_pet'
    device = torch.device('cuda')
    image = ctx.banks()[modality].batch([int(row.iloc[0].cache_row)], device)
    array = image.detach().cpu().numpy().squeeze()
    background, geo = geometry(ctx, row, modality)
    # MRI cache is normalized; background is the corresponding actual registered MRI.
    if background.shape != array.shape:
        raise ValueError('ACTUAL_BACKGROUND_SHAPE_MISMATCH')
    mask = array != 0
    atlas = ctx.parent.job('atlas_bridge')
    image_atlas = nib.load(ctx.parent.root / atlas['path'])
    atlas_eligible = (atlas['status'] == 'APPROXIMATE_ANATOMICAL_LABELING_ONLY' and image_atlas.shape == background.shape
                      and np.allclose(image_atlas.affine, background.affine, atol=1e-5))
    if not atlas_eligible:
        raise ValueError('DOCUMENTED_ATLAS_MODEL_SPACE_GEOMETRY_REQUIRED')
    atlas_array = image_atlas.get_fdata().astype(int)
    tasks = [f'{track}_s{s}_o{fold}_c{c}' for s in ctx.config['training_seeds'] for c in range(ctx.config['components'])]
    targets, parity = [], []
    for task in tasks:
        target = make_target(ctx, task, arm, modality, row)
        parity.append(verify_prediction_parity(ctx, task, arm, key, target, image))
        targets.append(target)
    random_targets = [make_target(ctx, task, arm, modality, row, 'random') for task in tasks]
    cells = ctx.parent.job(track + '_ensemble_cells')
    a = cells[fold]['arms'][arm]
    expected = 1 - np.interp(1095.75, ctx.config['fusion']['ibs_grid_days'], a['curves'][a['subjects'].index(key)])
    actual = float(np.mean([p['target'] for p in parity]))
    if abs(actual - expected) > ctx.options['spatial_parity_absolute_tolerance']:
        raise ValueError('NINE_COMPONENT_ENSEMBLE_PARITY_FAILED')
    ctx.units.job(name + '_parity', lambda: {'status': 'FULL_ENSEMBLE_PREDICTION_PARITY_VERIFIED',
        'components': parity, 'ensemble_error': abs(actual - expected),
        'patient_hash': key, 'arm': arm, 'horizon_days': 1095.75}, {'key': digest(key)})
    baselines = {'zero': np.zeros_like(array)}
    for sigma in ctx.config['spatial']['baseline_sigma']:
        b = gaussian_filter(array, sigma=sigma)
        b[~mask] = 0
        baselines['blur_' + str(sigma).replace('.', 'p')] = b
    maps, outcomes, occlusion_maps, occlusion_results = {}, {}, {}, {}
    for kind, baseline in baselines.items():
        base = torch.as_tensor(baseline, device=device).reshape_as(image)
        accumulated = {c: np.zeros_like(array, dtype=np.float64) for c in ('trained', 'random')}
        available = {c: True for c in accumulated}
        numerics = {c: [] for c in accumulated}
        component_inputs = []
        for index, task in enumerate(tasks):
            for control in accumulated:
                unit = name + '_' + kind + f'_component{index}_' + control
                def compute():
                    target = targets[index] if control == 'trained' else random_targets[index]
                    try:
                        attribution, numeric = integrated_gradients(target, image, base, steps=tuple(ctx.config['spatial']['ig_steps']),
                                                                    gradient_scale=ctx.options['spatial_gradient_scale'])
                    except ValueError as exc:
                        if str(exc) != 'NONFINITE_SCALED_INPUT_GRADIENT':
                            raise
                        # A numerical IG failure is not an all-zero attribution.
                        return {'numerical': {'numerical_status': 'NOT_ESTIMABLE_NONFINITE_GRADIENT',
                                             'reason': str(exc)}, 'map': None, 'outputs': {},
                                'target': 'COMPONENT_COMPLETE_THREE_YEAR_COX_RISK', 'control': control}
                    path = ctx.out / 'private/spatial' / name / (unit + '.npz')
                    compressed_map(path, attribution)
                    return {'numerical': numeric, 'map': path.relative_to(ctx.out).as_posix(),
                            'outputs': {path.relative_to(ctx.out).as_posix(): file_sha(path)},
                            'target': 'COMPONENT_COMPLETE_THREE_YEAR_COX_RISK', 'control': control}
                receipt = ctx.units.job(unit, compute, {'parent_lock': digest(ctx.parent.job(task + '_lock')), 'key': digest(key)})
                component_inputs.append(digest(receipt))
                if receipt['map'] is None:
                    available[control] = False
                else:
                    accumulated[control] += map_value(ctx, receipt) / len(tasks)
                numerics[control].append(receipt['numerical'])
        attribution = accumulated['trained'].astype(np.float32)
        def finish_baseline():
            fidelity = (region_controls(lambda x: ensemble_mean(targets, x), image, base, attribution, mask, atlas_array,
                seed=ctx.config['seed'] + fold, repeats=ctx.config['spatial']['random_controls']) if available['trained'] else
                        {'status': 'NOT_ESTIMABLE_INCOMPLETE_TRAINED_IG'})
            similarities = {'random': (signed_similarity(attribution, accumulated['random'], mask)
                                      if all(available.values()) else
                                      {'status': 'NOT_ESTIMABLE_INCOMPLETE_IG', 'pearson': None})}
            numerical = all(n['numerical_status'] == 'PASS' for values in numerics.values() for n in values)
            threshold = ctx.config['spatial']['max_randomized_correlation']
            sanity = all(v.get('pearson') is not None and abs(v['pearson']) < threshold for v in similarities.values())
            positive = fidelity.get('positive', {})
            faithful = (positive.get('directional_margin_vs_random_mean', -1) > 0 and
                        positive.get('descriptive_random_tail_fraction', 1) <= .05)
            status = ('DIAGNOSTIC_ONLY_MATCHED_LABEL_CONTROL_NOT_AVAILABLE' if numerical and sanity and faithful else
                      'DIAGNOSTIC_ONLY_NUMERICAL_FAILURE' if not numerical else
                      'DIAGNOSTIC_ONLY_RANDOMIZATION_FAILURE' if not sanity else 'DIAGNOSTIC_ONLY_FIDELITY_FAILURE')
            path = ctx.out / 'private/spatial' / name / (kind + '_ensemble.nii')
            if available['trained']:
                save_nifti(path, attribution, background.affine)
            return {'status': status, 'component_numerics': numerics, 'fidelity': fidelity,
                    'randomization': similarities, 'map': path.relative_to(ctx.out).as_posix() if available['trained'] else None,
                    'outputs': {path.relative_to(ctx.out).as_posix(): file_sha(path)} if available['trained'] else {},
                    'all_trained_components_available': available['trained'],
                    'explains_entire_ensemble': True, 'explains_disease_lesion': False,
                    'label_permutation_status': 'WHOLE_ENSEMBLE_CONTROL_REQUIRES_NEW_TRAINING_NOT_AUTHORIZED',
                    'fully_validated_attribution': False,
                    'validation_is_not_population_inference': True}
        outcome = ctx.units.job(name + '_' + kind + '_ensemble', finish_baseline, {'component_receipts': digest(component_inputs)})
        outcomes[kind] = outcome
        if outcome['all_trained_components_available']:
            maps[kind] = nib.Nifti1Image(attribution, background.affine)
        from .occlusion_maps import run_baseline
        occlusion = ctx.units.job(name + '_' + kind + '_roi_occlusion',
            lambda: run_baseline(ctx, name, kind, targets, random_targets, image, base, atlas_array, mask, background.affine),
            {'parity': digest(parity), 'key': digest(key)})
        occlusion_results[kind] = occlusion
        occlusion_maps[kind] = nib.load(ctx.out / occlusion['map'])
    directory = ctx.out / 'private/spatial' / name
    if maps:
        overlay(background, maps, directory / 'actual_subject_overlay.png',
                'Diagnostic IG of full ensemble risk; consult numerical status before interpretation')
    overlay(background, occlusion_maps, directory / 'actual_subject_roi_occlusion.png',
            'Whole-ROI replacement effect on ensemble risk; non-additive, not disease localization')
    regions = {}
    if atlas_eligible:
        labels = read_json(ctx.root / 'results/xyz_fusion_v2_assets/source_registry.json')['labels']
        regions = {k: roi_summary(v.get_fdata(), image_atlas.get_fdata().astype(int), labels) for k, v in maps.items()}
    return {'status': 'COMPLETED_BASELINE_SPECIFIC_WHOLE_ENSEMBLE_AUDIT', 'patient_hash': key,
            'fold': fold, 'case': case, 'track': track, 'modality': modality,
            'baselines': outcomes, 'geometry': geo, 'roi_attribution': regions,
            'roi_occlusion': occlusion_results,
            'native_space_status': 'NOT_CLAIMED_MODEL_SPACE_ACTUAL_SUBJECT_IMAGE',
            'all_failures_retained': True, 'disease_localization_allowed': False,
            'outputs': {p.relative_to(ctx.out).as_posix(): file_sha(p) for p in directory.glob('*') if p.is_file()}}


def run_spatial(ctx):
    if not torch.cuda.is_available():
        raise ValueError('CUDA_REQUIRED_FOR_MANUAL_SPATIAL_EXECUTION')
    ctx.ensure_inputs()
    results = []
    for fold, split in enumerate(ctx.splits):
        keys = sorted(split['test'], key=lambda k: digest({'seed': ctx.config['seed'], 'key': k}))[:ctx.config['spatial']['cases_per_fold']]
        for case, key in enumerate(keys):
            for track, modality in [('mri', 'mri'), ('pet', 'mri'), ('pet', 'pet')]:
                name = f'f04_{track}_{modality}_o{fold}_case{case}'
                result = ctx.units.job(name, lambda: case_maps(ctx, fold, case, key, track, modality), {'key': digest(key)})
                results.append(result)
    return {'status': 'COMPLETED_SPATIAL_DIAGNOSTICS_LABEL_CONTROL_PENDING', 'requested_units': len(results),
            'eligible_units': sum(r['status'] != 'NOT_ELIGIBLE_NO_LOCKED_PET_IMAGE' for r in results),
            'cases': results, 'aggregate_maps_public': False, 'individual_maps_private': True,
            'disease_localization_allowed': False}
