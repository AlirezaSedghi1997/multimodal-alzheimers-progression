"""Non-production numerical review; never starts a training or image-inference job.

Use a fresh, timestamped review directory when source changes. These diagnostics
do not replace the production launcher's fresh full checkpoint/hash verification.
"""
from pathlib import Path
import argparse
import json
from .core import ReadOnlyParent, Units, parent_exclusion, digest, file_sha, read_json, atomic_json
from .context import Context, eligibility
from .audit import audit_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--pacc', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    from scripts.xyz_fusion_v1.runtime import configure
    configure(root, 4)
    out = Path(args.output).resolve()
    if not out.is_relative_to(root / 'results/xyz_fusion_postall_v2_review'):
        raise ValueError('REVIEW_OUTPUT_NAMESPACE_REQUIRED')
    parent = ReadOnlyParent(root / 'results/xyz_fusion_v2')
    options = read_json(root / 'configs/xyz_fusion_postall_v2.json')
    frozen = read_json(root / 'results/xyz_fusion_postall_v2_review/parent_review.json')
    contract = {'source': {p.name: file_sha(p) for p in Path(__file__).parent.glob('*.py')},
                'options': options, 'parent_binding': parent.binding, 'prior_full_parent_review': digest(frozen)}
    if frozen['parent_contract'] != parent.binding:
        raise ValueError('WRONG_PRIOR_PARENT_REVIEW')
    units = Units(out, contract)
    ctx = Context(root, parent, units, options)
    with parent_exclusion(parent.root):
        for path in (parent.root / 'private/jobs').glob('*.json'):
            if file_sha(path) != frozen['files'][path.relative_to(parent.root).as_posix()]:
                raise ValueError('PARENT_JOB_CHANGED_AFTER_FULL_VERIFICATION')
        print('EXACT_SOURCE_RECONSTRUCTION_STARTED', flush=True)
        proof = ctx.load_inputs()
        units.job('input_reconstruction', lambda: proof)
        e = units.job('f02', lambda: eligibility(ctx))
        print(json.dumps({'intersections': e['intersection_counts'], 'ancestry_tasks': e['tasks_verified']}), flush=True)
        from .contrasts import four_way_decision, common_support
        from .robustness import scanner_support
        from scripts.xyz_fusion_v2.protocol import CLINICAL, MRI
        f3 = units.job('f03_feasibility', lambda: common_support(ctx, 'mri', ctx.tau, CLINICAL + MRI + ['amyloid_centiloid', 'tau_temporal']))
        f7 = units.job('f07_feasibility', lambda: four_way_decision(ctx))
        f5 = units.job('f05_feasibility', lambda: scanner_support(ctx))
        result = audit_metrics(ctx, bootstrap=args.bootstrap)
        if args.bootstrap:
            units.job('f01', lambda: result)
        f6 = None
        if args.pacc:
            from .secondary import pacc_analysis
            f6 = units.job('f06', lambda: pacc_analysis(ctx))
        summary = {'status': 'CPU_REVIEW_COMPLETED_NOT_PRODUCTION_FINALIZATION',
                   'analysis_count': len(result['analyses']),
                   'metric_cells': sum(a['metrics']['arm_fold_cells'] for a in result['analyses'].values()),
                   'contrast_count': sum(len(a['paired_comparisons']) for a in result['analyses'].values()),
                   'intersections': e['intersection_counts'], 'tau_support': f3['status'],
                   'four_way_support': f7['status'], 'pet_two_groups_nested_support': f5['two_manufacturers_have_nested_support'],
                   'pacc': f6['status'] if f6 else 'NOT_RUN',
                   'fresh_cnn_training': False, 'parent_models_rehashed_in_prior_full_audit_not_this_diagnostic': True}
        atomic_json(out / 'review_summary.json', summary)
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
