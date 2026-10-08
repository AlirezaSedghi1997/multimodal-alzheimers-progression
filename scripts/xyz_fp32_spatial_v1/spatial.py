"""Signed maps of the new FP32 ensemble, never mislabeled as parent explanations."""
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import gaussian_filter
from scripts.xyz_fusion_postall_v2.core import digest, file_sha, read_json
from scripts.xyz_fusion_postall_v2.spatial import compressed_map, geometry
from scripts.xyz_fusion_postall_v2.spatial_target import ComponentTarget, FixedInferenceBatch, ensemble_mean
from .pipeline import load_head, task_names, chosen_arms
from .numerics import quadrature, compare_maps, converged


def make_target(ctx, task, arm, modality, row, randomized=False):
    from scripts.xyz_fusion_postall_v2.experts import load_expert
    from scripts.xyz_fusion_v2.models import score_features
    from scripts.xyz_fusion_v2.training import predict
    lock = ctx.units.read(task + '_child_lock')
    key = row.iloc[0].patient_hash
    if key not in lock['outer_subjects'] or key in set(lock['meta_subjects']) | set(lock['expert_subjects']):
        raise ValueError('SPATIAL_SUBJECT_ANCESTRY_VIOLATION')
    head, receipt = load_head(ctx, task, arm)
    pred = ctx.units.read(task + '_child_predictions')['arms'][arm]
    order = pred['subjects']; position = order.index(key)
    batch_size = ctx.config['training']['microbatch_size']
    start = position // batch_size * batch_size
    source = ctx.frame if task.startswith('mri_') else ctx.pet
    batch_rows = source.set_index('patient_hash').loc[order[start:start + batch_size]].reset_index()
    local = position - start
    values = row.copy()
    for mod in lock['experts']:
        # Randomize all CNNs in the component, including the fixed other modality.
        if randomized:
            other, policy = load_expert(ctx, task, mod, randomized=True)
            curve = predict(other, ctx.banks()[mod], batch_rows[['patient_hash', 'cache_row']], policy,
                            next(other.parameters()).device, ctx.config['fusion']['ibs_grid_days'])[local:local + 1]
            del other
        else:
            saved = ctx.units.read(f'{task}_{mod}_outer_fp32')
            curve = [saved['survival'][saved['subjects'].index(key)]]
        values = score_features(values, curve, ctx.config['fusion']['ibs_grid_days'],
                                ctx.config['fusion']['score_times_days'], mod)
    model, policy = load_expert(ctx, task, modality, randomized=randomized)
    batch = ctx.banks()[modality].batch(batch_rows.cache_row.to_numpy(), next(model.parameters()).device)
    return ComponentTarget(FixedInferenceBatch(model, batch, local), head,
        values[receipt['features']].to_numpy(float), receipt['features'], modality, policy,
        ctx.config['fusion']['ibs_grid_days'], ctx.config['fusion']['score_times_days'])


def adaptive_map(ctx, name, target, image, baseline, dependency):
    previous = None; history = []; similarity = {}
    final = None
    for count in ctx.options['ig_steps']:
        unit = name + f'_q{count}'
        def compute():
            try:
                array, numeric = quadrature(target, image, baseline, count, ctx.options)
            except ValueError as exc:
                if str(exc) not in ('NONFINITE_FP32_GRADIENT', 'NONFINITE_SCALAR_TARGET', 'NONFINITE_TARGET'):
                    raise
                return {'numeric': {'numerical_status': 'NOT_ESTIMABLE_NONFINITE', 'reason': str(exc)}, 'map': None}
            path = ctx.out / 'private/spatial/components' / (unit + '.npz')
            compressed_map(path, array)
            return {'numeric': numeric, 'map': path.relative_to(ctx.out).as_posix(),
                    'outputs': {path.relative_to(ctx.out).as_posix(): file_sha(path)}}
        value = ctx.units.job(unit, compute, dependency)
        history.append(value['numeric'])
        if value['map'] is None:
            break
        with np.load(ctx.out / value['map'], allow_pickle=False) as data:
            current = np.asarray(data['attribution'])
        final = current
        if previous is not None:
            similarity = compare_maps(previous, current)
            if converged(history, similarity, ctx.options):
                break
        # Zero maps are a failed/undefined control, not evidence of passing sanity.
        if len(history) >= 2 and all(v['numerical_status'] == 'NOT_ESTIMABLE_ZERO_MAP' for v in history[-2:]):
            break
        previous = current
    valid = converged(history, similarity, ctx.options)
    receipt = {'status': 'PASS' if valid else 'NOT_VALIDATED_NUMERICALLY', 'history': history,
               'successive_map_stability': similarity, 'two_consecutive_quadratures_required': True}
    ctx.units.job(name + '_numeric', lambda: receipt, dependency)
    return final, receipt


def case_maps(ctx, fold, case, key, track, modality):
    from scripts.xyz_fusion_postall_v2.roi_fidelity import region_controls
    from scripts.xyz_fusion_postall_v2.occlusion_maps import run_baseline
    from src.evaluation.signed_spatial_validation_v1 import signed_similarity
    from scripts.xyz_postprimary_v1.rendering import save_nifti, overlay
    from scripts.xyz_fusion_v2.atlas import roi_summary
    frame = ctx.frame if track == 'mri' else ctx.pet
    row = frame.loc[frame.patient_hash.eq(key)]
    if row.empty:
        return {'status': 'NOT_ELIGIBLE_NO_LOCKED_PET_IMAGE', 'replacement_selected': False}
    arm = chosen_arms(track)[-1]
    name = f'spatial_{track}_{modality}_o{fold}_case{case}'
    image = ctx.banks()[modality].batch([int(row.iloc[0].cache_row)], torch.device('cuda'))
    array = image.cpu().numpy().squeeze(); mask = array != 0
    background, geo = geometry(ctx, row, modality)
    atlas_receipt = ctx.parent.job('atlas_bridge')
    atlas = nib.load(ctx.parent.root / atlas_receipt['path'])
    if (background.shape != array.shape or atlas.shape != array.shape or
        not np.allclose(atlas.affine, background.affine, atol=1e-5) or
        atlas_receipt['status'] != 'APPROXIMATE_ANATOMICAL_LABELING_ONLY'):
        raise ValueError('ATLAS_AND_ACTUAL_IMAGE_GEOMETRY_REQUIRED')
    labels = atlas.get_fdata().astype(int)
    tasks = task_names(ctx.config, track, fold)
    targets = [make_target(ctx, t, arm, modality, row) for t in tasks]
    random_targets = [make_target(ctx, t, arm, modality, row, randomized=True) for t in tasks]
    parity = []
    for t, target in zip(tasks, targets):
        prediction = ctx.units.read(t + '_child_predictions')['arms'][arm]
        expected = 1 - np.interp(1095.75, ctx.config['fusion']['ibs_grid_days'],
            prediction['survival'][prediction['subjects'].index(key)])
        with torch.no_grad(): observed = float(target(image))
        error = abs(expected - observed)
        if not np.isfinite(error) or error > ctx.options['spatial_parity_absolute_tolerance']:
            raise ValueError('FP32_COMPONENT_PREDICTION_PARITY_FAILED')
        parity.append({'target': observed, 'absolute_error': error})
    cell = ctx.units.read(track + '_child_cells')['cells'][fold]['arms'][arm]
    expected = 1 - np.interp(1095.75, ctx.config['fusion']['ibs_grid_days'], cell['curves'][cell['subjects'].index(key)])
    if abs(np.mean([p['target'] for p in parity]) - expected) > ctx.options['spatial_parity_absolute_tolerance']:
        raise ValueError('FP32_FULL_ENSEMBLE_PARITY_FAILED')
    baselines = {'zero': np.zeros_like(array)}
    for sigma in ctx.config['spatial']['baseline_sigma']:
        base = gaussian_filter(array, sigma=sigma); base[~mask] = 0
        baselines['blur_' + str(sigma).replace('.', 'p')] = base
    maps, occlusion_maps, outcomes = {}, {}, {}
    for kind, base_array in baselines.items():
        base = torch.as_tensor(base_array, device=image.device).reshape_as(image)
        accum = {c: np.zeros_like(array, dtype=float) for c in ('trained', 'random')}
        available = {'trained': True, 'random': True}; numeric = {'trained': [], 'random': []}
        for i, task in enumerate(tasks):
            for control, objects in [('trained', targets), ('random', random_targets)]:
                a, n = adaptive_map(ctx, f'{name}_{kind}_c{i}_{control}', objects[i], image, base,
                    {'child_lock': digest(ctx.units.read(task + '_child_lock')), 'key': digest(key)})
                numeric[control].append(n)
                if a is None: available[control] = False
                else: accum[control] += a / len(tasks)
        def finish():
            trained_ok = all(x['status'] == 'PASS' for x in numeric['trained'])
            random_ok = all(x['status'] == 'PASS' for x in numeric['random'])
            similarity = (signed_similarity(accum['trained'], accum['random'], mask) if all(available.values())
                          else {'pearson': None, 'status': 'NOT_ESTIMABLE'})
            fidelity = (region_controls(lambda x: ensemble_mean(targets, x), image, base, accum['trained'], mask, labels,
                         seed=ctx.config['seed'] + fold, repeats=ctx.options['random_controls']) if available['trained']
                        else {'status': 'NOT_ESTIMABLE_NO_MAP'})
            path = ctx.out / 'private/spatial' / name / (kind + '_ensemble.nii')
            if available['trained']: save_nifti(path, accum['trained'].astype('f4'), background.affine)
            return {'trained_ig_numeric': 'PASS' if trained_ok else 'NOT_VALIDATED_NUMERICALLY',
                'random_ig_numeric': 'PASS' if random_ok else 'NOT_ESTIMABLE_OR_FAILED',
                'component_numerics': numeric, 'random_weight_similarity': similarity, 'fidelity': fidelity,
                'map': path.relative_to(ctx.out).as_posix() if available['trained'] else None,
                'outputs': {path.relative_to(ctx.out).as_posix(): file_sha(path)} if available['trained'] else {},
                'label_permutation_status': 'NOT_RUN_REQUIRES_NEW_CNN_TRAINING',
                'fully_validated_attribution': False, 'clinical_or_disease_localization_allowed': False}
        dep = {'components': digest(numeric), 'parity': digest(parity)}
        outcome = ctx.units.job(name + '_' + kind + '_ensemble', finish, dep)
        outcomes[kind] = outcome
        if outcome['map']: maps[kind] = nib.load(ctx.out / outcome['map'])
        # Non-gradient perturbation remains an independent diagnostic even if IG fails.
        occlusion = ctx.units.job(name + '_' + kind + '_occlusion',
            lambda: run_baseline(ctx, name, kind, targets, random_targets, image, base, labels, mask, background.affine),
            {'child_parity': digest(parity), 'key': digest(key)})
        outcomes[kind]['occlusion'] = occlusion
        occlusion_maps[kind] = nib.load(ctx.out / occlusion['map'])
    directory = ctx.out / 'private/spatial' / name
    if maps:
        overlay(background, maps, directory / 'fp32_ig_actual_subject.png',
                'FP32 sensitivity ensemble: signed IG; numeric status is required; not disease localization')
    overlay(background, occlusion_maps, directory / 'fp32_roi_occlusion_actual_subject.png',
            'FP32 sensitivity: whole-ROI risk replacement effect, non-additive and non-causal')
    label_table = read_json(ctx.root / 'results/xyz_fusion_v2_assets/source_registry.json')['labels']
    regions = {k: roi_summary(v.get_fdata(), labels, label_table) for k, v in maps.items()}
    del targets, random_targets
    torch.cuda.empty_cache()
    return {'status': 'COMPLETED_DIAGNOSTICS_NOT_VALIDATED_DISEASE_LOCALIZATION',
        'track': track, 'modality': modality, 'fold': fold, 'case': case, 'patient_hash': key,
        'parity': parity, 'baselines': outcomes, 'geometry': geo, 'roi_attribution': regions,
        'same_model_as_amp_parent': False, 'native_space_status': 'NOT_CLAIMED',
        'individual_images_private': True, 'outputs': {p.relative_to(ctx.out).as_posix(): file_sha(p)
            for p in directory.glob('*') if p.is_file()}}


def run_spatial(ctx):
    cases = []
    for fold, split in enumerate(ctx.splits):
        # Preserve the original label-blind panel; never substitute successful maps.
        keys = sorted(split['test'], key=lambda k: digest({'seed': ctx.config['seed'], 'key': k}))[:ctx.config['spatial']['cases_per_fold']]
        for case, key in enumerate(keys):
            for track, mod in [('mri', 'mri'), ('pet', 'mri'), ('pet', 'pet')]:
                name = f'spatial_{track}_{mod}_o{fold}_case{case}'
                cases.append(ctx.units.job(name, lambda: case_maps(ctx, fold, case, key, track, mod), {'key': digest(key)}))
    eligible = [c for c in cases if 'baselines' in c]
    return {'requested_units': len(cases), 'eligible_units': len(eligible), 'cases': cases,
            'numerically_passing_baselines': sum(b['trained_ig_numeric'] == 'PASS' for c in eligible for b in c['baselines'].values()),
            'evaluated_baselines': sum(len(c['baselines']) for c in eligible),
            'fully_validated_attribution': False, 'disease_localization_allowed': False}
