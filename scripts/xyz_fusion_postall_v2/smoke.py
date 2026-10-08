"""Bounded locked-model smoke test; never a replacement for the full XAI audit."""
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-case', action='store_true')
    parser.add_argument('--roi-case', action='store_true')
    parser.add_argument('--parity-panel', action='store_true')
    parser.add_argument('--case-integration', action='store_true')
    args = parser.parse_args()
    root = Path('H:/New project')
    from scripts.xyz_fusion_v1.runtime import configure, gpu_guard
    configure(root, 4)
    gpu_guard()
    import numpy as np
    import pandas as pd
    import torch
    from .core import ReadOnlyParent, file_sha, digest, read_json, atomic_json, parent_exclusion
    from .spatial import make_target, verify_prediction_parity, geometry
    from scripts.xyz_fusion_v2.training import ImageBank, deterministic
    from scripts.xyz_fusion_v2.inference import PETBank
    from .integrated_gradients import integrated_gradients
    from scripts.xyz_postprimary_v1.rendering import overlay, save_nifti
    import nibabel as nib
    from scipy.ndimage import gaussian_filter
    out = root / ('results/xyz_fusion_postall_v2_review/case_integration_v1' if args.case_integration else
                 'results/xyz_fusion_postall_v2_review/parity_panel_v2' if args.parity_panel else
                 'results/xyz_fusion_postall_v2_review/roi_smoke_v1' if args.roi_case else
                 'results/xyz_fusion_postall_v2_review/bounded_full_case_v1' if args.full_case else
                  'results/xyz_fusion_postall_v2_review/gpu_smoke_amp_batch_v3')
    out.mkdir(parents=True, exist_ok=True)
    parent = ReadOnlyParent(root / 'results/xyz_fusion_v2')
    cfg = parent.contract['protocol']
    if args.case_integration:
        import copy
        cfg = copy.deepcopy(cfg)
        # A bounded execution fixture, never a production validation result.
        cfg['spatial']['ig_steps'] = [8]
        cfg['spatial']['random_controls'] = 3
    deterministic(cfg['seed'], cfg['training']['cpu_threads'])
    frame = pd.DataFrame(parent.contract['manifest'])
    membership = root / 'results/method_y_revival_v3_image_v2/private/amyloid_image_membership.parquet'
    if file_sha(membership) != parent.contract['input_sha256']['pet_membership']:
        raise ValueError('SMOKE_MEMBERSHIP_DRIFT')
    petkeys = [k for c in parent.job('pet_ensemble_cells') for k in c['arms']['clinical_mri_amyloid_pet']['subjects']]
    pet = frame.loc[frame.patient_hash.isin(petkeys)].merge(
        pd.read_parquet(membership)[['patient_hash', 'image_path_private', 'h5_group', 'source_image_sha256']],
        on='patient_hash', validate='one_to_one')
    banks = {'mri': ImageBank(root / 'results/xyz_mri_baseline_fast_v1/private/mri_cache.npy', 0),
             'pet': PETBank(pet, max_gib=0)}
    ctx = SimpleNamespace(root=root, parent=parent, config=cfg, frame=frame, pet=pet,
                          options=read_json(root / 'configs/xyz_fusion_postall_v2.json'), banks=lambda: banks)
    report = {'status': 'RUNNING', 'scope': 'FROZEN_INFERENCE_AND_BOUNDED_IG_NOT_PRODUCTION_RESULTS',
              'cnn_training': False, 'parent_binding': parent.binding, 'tests': []}
    try:
      with parent_exclusion(parent.root):
        if args.parity_panel:
            started = time.monotonic()
            records = []
            for fold, split in enumerate(parent.job('splits')):
                keys = sorted(split['test'], key=lambda k: digest({'seed': cfg['seed'], 'key': k}))[:cfg['spatial']['cases_per_fold']]
                for case, key in enumerate(keys):
                    for track, modality in [('mri', 'mri'), ('pet', 'mri'), ('pet', 'pet')]:
                        source = frame if track == 'mri' else pet
                        row = source.loc[source.patient_hash.eq(key)]
                        if row.empty:
                            continue
                        image = banks[modality].batch([int(row.iloc[0].cache_row)], torch.device('cuda'))
                        arm = 'clinical_mri_amyloid' if track == 'mri' else 'clinical_mri_amyloid_pet'
                        parity = []
                        for seed in cfg['training_seeds']:
                            for component in range(cfg['components']):
                                task = f'{track}_s{seed}_o{fold}_c{component}'
                                target = make_target(ctx, task, arm, modality, row)
                                parity.append(verify_prediction_parity(ctx, task, arm, key, target, image))
                                del target
                        records.append({'fold': fold, 'case': case, 'track': track, 'modality': modality,
                                        'component_parity': parity})
                        print(json.dumps({'event': 'PANEL_PARITY_VERIFIED', 'cases_completed': len(records),
                                          'max_error': max(p['absolute_error'] for p in parity)}), flush=True)
            report.update(status='ALL_SELECTED_COMPONENT_PARITY_PASSED', tests=records,
                          seconds=time.monotonic() - started)
            return
        if args.roi_case:
            from .occlusion_maps import run_baseline
            ctx.out = out
            row = frame.loc[frame.patient_hash.eq(min(set(parent.job('splits')[0]['test']),
                key=lambda k: digest({'seed': cfg['seed'], 'key': k})))]
            image = banks['mri'].batch([int(row.iloc[0].cache_row)], torch.device('cuda'))
            background, _ = geometry(ctx, row, 'mri')
            values = image.cpu().numpy().squeeze()
            b = gaussian_filter(values, 2); b[values == 0] = 0
            baseline = torch.as_tensor(b, device=image.device).reshape_as(image)
            atlas = nib.load(parent.root / parent.job('atlas_bridge')['path']).get_fdata().astype(int)
            tasks = [f'mri_s{s}_o0_c{c}' for s in cfg['training_seeds'] for c in range(cfg['components'])]
            trained = [make_target(ctx, t, 'clinical_mri_amyloid', 'mri', row) for t in tasks]
            randomized = [make_target(ctx, t, 'clinical_mri_amyloid', 'mri', row, 'random') for t in tasks]
            started = time.monotonic()
            value = run_baseline(ctx, 'bounded_roi', 'blur_2p0', trained, randomized,
                                 image, baseline, atlas, values != 0, background.affine)
            overlay(background, {'blur_2p0_occlusion': nib.load(out / value['map'])},
                    out / 'actual_subject_roi_occlusion.png',
                    'Signed whole-ROI effect on ensemble risk; not a disease lesion map')
            atomic_json(out / 'roi_effects_private.json', value)
            report['status'] = 'BOUNDED_ROI_EFFECTS_EXECUTED'
            report['seconds'] = time.monotonic() - started
            report['region_count'] = len(value['regions'])
            report['sanity'] = value['random_weight_similarity']
            print(json.dumps(report), flush=True)
            return
        if args.full_case or args.case_integration:
            from .core import Units
            from .spatial import case_maps
            ctx.out = out
            ctx.units = Units(out, {'parent': parent.binding, 'options': ctx.options, 'smoke_protocol': cfg,
                'source': {p.name: file_sha(p) for p in Path(__file__).parent.glob('*.py')}})
            key = min(set(parent.job('splits')[0]['test']), key=lambda k: digest({'seed': cfg['seed'], 'key': k}))
            started = time.monotonic()
            value = ctx.units.job('bounded_full_case', lambda: case_maps(ctx, 0, 0, key, 'mri', 'mri'))
            report['status'] = ('BOUNDED_CASE_INTEGRATION_EXECUTED_NOT_SCIENTIFIC_VALIDATION' if args.case_integration else
                                'FULL_SINGLE_CASE_AUDIT_EXECUTED_NOT_FINALIZATION')
            report['seconds'] = time.monotonic() - started
            report['baseline_status'] = {k: v['status'] for k, v in value['baselines'].items()}
            print(json.dumps({k: report[k] for k in ('status', 'seconds', 'baseline_status')}), flush=True)
            return
        for track, modality in [('mri', 'mri'), ('pet', 'mri'), ('pet', 'pet')]:
            source = frame if track == 'mri' else pet
            held = set(parent.job('splits')[0]['test'])
            key = min(set(source.patient_hash) & held, key=lambda k: digest({'seed': cfg['seed'], 'key': k}))
            row = source.loc[source.patient_hash.eq(key)]
            arm = 'clinical_mri_amyloid' if track == 'mri' else 'clinical_mri_amyloid_pet'
            image = banks[modality].batch([int(row.iloc[0].cache_row)], torch.device('cuda'))
            background, geometry_receipt = geometry(ctx, row, modality)
            if tuple(image.shape[2:]) != background.shape:
                raise ValueError('SMOKE_IMAGE_GEOMETRY_MISMATCH')
            array = image.detach().cpu().numpy().squeeze()
            baseline = gaussian_filter(array, 2)
            baseline[array == 0] = 0
            base = torch.as_tensor(baseline, device=image.device).reshape_as(image)
            maps, parities, numerics = [], [], []
            started = time.monotonic()
            targets = []
            for seed in cfg['training_seeds']:
                for component in range(cfg['components']):
                    task = f'{track}_s{seed}_o0_c{component}'
                    target = make_target(ctx, task, arm, modality, row)
                    targets.append(target)
                    try:
                        parities.append(verify_prediction_parity(ctx, task, arm, key, target, image))
                    except ValueError:
                        pred = parent.job(task + '_predictions')['arms'][arm]
                        expected = 1 - np.interp(1095.75, cfg['fusion']['ibs_grid_days'], pred['survival'][pred['subjects'].index(key)])
                        with torch.no_grad():
                            fp32 = float(target(image))
                            target.amp = True
                            amp = float(target(image))
                            target.amp = False
                        diagnostics = {'task': task, 'locked_risk': expected, 'fp32_risk': fp32, 'amp_risk': amp}
                        atomic_json(out / 'parity_failure.json', diagnostics)
                        print(json.dumps(diagnostics), flush=True)
                        raise
                    # Eight nodes exercise real gradients, not a validation relaxation.
                    attr, numeric = integrated_gradients(target, image, base, steps=(8,))
                    maps.append(attr)
                    numerics.append(numeric)
            combined = np.mean(maps, axis=0)
            output = out / (track + '_' + modality + '_diagnostic_smoke.nii.gz')
            save_nifti(output, combined, background.affine)
            overlay(background, {'blur_2_smoke': nib.Nifti1Image(combined, background.affine)},
                    out / (track + '_' + modality + '_diagnostic_overlay.png'),
                    'DIAGNOSTIC SMOKE ONLY - full ensemble risk, 8 nodes')
            a = parent.job(track + '_ensemble_cells')[0]['arms'][arm]
            expected = 1 - np.interp(1095.75, cfg['fusion']['ibs_grid_days'], a['curves'][a['subjects'].index(key)])
            ensemble_error = abs(np.mean([p['target'] for p in parities]) - expected)
            if ensemble_error > ctx.options['spatial_parity_absolute_tolerance']:
                raise ValueError('SMOKE_ENSEMBLE_PARITY_FAIL')
            entry = {'track': track, 'modality': modality, 'components': len(targets),
                     'max_component_parity_error': max(p['absolute_error'] for p in parities),
                     'ensemble_error': ensemble_error, 'seconds': time.monotonic() - started,
                     'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                     'numerical_checks': numerics, 'geometry': geometry_receipt,
                     'scope_not_fully_validated_xai': True}
            report['tests'].append(entry)
            print(json.dumps({k: v for k, v in entry.items() if k not in ('numerical_checks', 'geometry')}), flush=True)
            del target, targets, maps, image, base
            gc.collect()
            torch.cuda.empty_cache()
        report['status'] = 'PASS_REAL_ENSEMBLE_PARITY_AND_GRADIENT_EXECUTION_NOT_FULL_XAI_VALIDATION'
    finally:
        banks['pet'].close()
        atomic_json(out / 'smoke_report.json', report)


if __name__ == '__main__':
    main()
