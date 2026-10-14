"""One manual command; small content-bound units and truthful incomplete states."""
from __future__ import annotations

import json
from pathlib import Path
from .core import ReadOnlyParent, Units, parent_exclusion, writer_lock, read_json, make_contract, atomic_json, digest
from .audit import verify_parent, freeze_once, audit_metrics
from .context import Context, eligibility
from .release import closure, verify_final, write_release


def execute(root, mode, config_path):
    root = Path(root).resolve()
    if str(root).replace('\\', '/').lower() != 'h:/new project':
        raise ValueError('AUTHORITATIVE_H_PROJECT_REQUIRED')
    options = read_json(config_path)
    if options.get('allow_new_cnn_training') is not False:
        raise ValueError('POST_FUSION_RECALCULATION_MUST_NOT_TRAIN_CNNS')
    out = (root / options['output_root']).resolve()
    parent_root = (root / options['parent_root']).resolve()
    historical = [parent_root, root / 'results/xyz_fusion_postall_v1', root / 'results/xyz_fusion_postall_v1_maps']
    if (not out.is_relative_to(root / 'results') or
            any(out.is_relative_to(p.resolve()) or p.resolve().is_relative_to(out) for p in historical)):
        raise ValueError('SEPARATE_VERSIONED_OUTPUT_REQUIRED')
    parent = ReadOnlyParent(parent_root)
    with writer_lock(out), parent_exclusion(parent.root):
        print(json.dumps({'event': 'FULL_READ_ONLY_PARENT_VERIFICATION_STARTED'}), flush=True)
        freeze = verify_parent(root, parent)
        freeze_once(out, freeze)
        units = Units(out, make_contract(root, parent, freeze, options))
        ctx = Context(root, parent, units, options)
        units.job('f00', lambda: freeze)
        if mode in ('verify', 'promote'):
            ctx.load_inputs()
            value = verify_final(ctx)
            if mode == 'promote':
                from .supersession import promote
                value = promote(ctx)
                print(json.dumps(value), flush=True)
                return 0
            print(json.dumps(value), flush=True)
            return 0 if value['verifier'].startswith('PASS') else 2
        try:
            # Reconstruct and hash inputs even when resuming downstream receipts.
            proof = ctx.load_inputs()
            units.job('input_reconstruction', lambda: proof)
            units.job('f02', lambda: dict(eligibility(ctx), reconstructed_inputs=proof), {'inputs': proof['input_digest']})
            from .contrasts import four_way_decision, matched_extension
            from .robustness import scanner_support, scanner_probes, pet_lmo
            units.job('f07_feasibility', lambda: four_way_decision(ctx))
            units.job('f05_feasibility', lambda: scanner_support(ctx))
            if mode in ('audit', 'preflight'):
                if mode == 'audit':
                    audit_metrics(ctx, bootstrap=False)
            if mode in ('all', 'statistics'):
                units.job('f01', lambda: audit_metrics(ctx, bootstrap=True))
            if mode in ('all', 'pacc'):
                from .secondary import pacc_analysis
                units.job('f06', lambda: pacc_analysis(ctx))
            pending_gpu = any(units.read(k) is None for k in
                              ({'all': ['f03', 'f04', 'f05', 'f07'], 'tau': ['f03'],
                                'spatial': ['f04'], 'scanner': ['f05'], 'four_way': ['f07']}.get(mode, [])))
            if pending_gpu:
                from scripts.xyz_fusion_v1.runtime import gpu_guard
                from scripts.xyz_fusion_v2.training import deterministic
                gpu_guard()
                deterministic(ctx.config['seed'], ctx.config['training']['cpu_threads'])
            if mode in ('all', 'tau'):
                units.job('f03', lambda: matched_extension(ctx, 'tau'))
            if mode in ('all', 'four_way'):
                decision = units.job('f07_feasibility')
                units.job('f07', lambda: matched_extension(ctx, 'four_way')
                          if decision['action'] == 'RUN_MATCHED_FOUR_HEAD_CHAIN' else decision)
            if mode in ('all', 'scanner'):
                probes = units.job('f05_probes', lambda: scanner_probes(ctx))
                lmo = units.job('f05_lmo', lambda: pet_lmo(ctx))
                units.job('f05', lambda: {'status': 'PROBES_AND_FEASIBILITY_COMPLETED_NO_LMO_TRAINING', 'probes': probes, 'lmo': lmo})
            if mode in ('all', 'spatial'):
                from .spatial import run_spatial
                units.job('f04', lambda: run_spatial(ctx))
            current = closure(units)
            if not current['missing_tasks']:
                units.job('f08', lambda: write_release(ctx))
                current = verify_final(ctx)
                from .supersession import candidate
                atomic_json(out / 'candidate_reference.json', candidate(ctx))
            atomic_json(out / 'orchestration_state.json', dict(current, contract=units.binding))
            print(json.dumps(current), flush=True)
            return 0
        except BaseException as exc:
            atomic_json(out / 'orchestration_state.json', dict(closure(units), contract=units.binding,
                        execution_status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'BLOCKED',
                        terminal_reason=type(exc).__name__ + ':' + str(exc)))
            raise
        finally:
            ctx.close()
