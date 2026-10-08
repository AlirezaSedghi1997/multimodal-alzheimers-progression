"""Manual staged primary MRI/FBP/blood fusion; no retired BrainIAC or old OOF reuse."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.xyz_fusion_v1.runtime import configure,gpu_guard
configure(ROOT)
from scripts.xyz_fusion_v1.store import Store,writer_lock,atomic_json,digest,file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','benchmark','smoke-preprocess','run','verify'],default='run')
    p.add_argument('--config',default=str(ROOT/'configs/xyz_mri_baseline_fast_v1.json'))
    p.add_argument('--smoke-count',type=int,default=2)
    args=p.parse_args();config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    out=(ROOT/config['output_root']).resolve()
    if not out.is_relative_to((ROOT/'results').resolve()):raise ValueError('OUTPUT_MUST_STAY_ON_PROJECT_H')
    from scripts.xyz_fusion_v1.baseline import baseline_intake
    from scripts.xyz_fusion_v1.mri_cache import prepare_cache,validate_reference
    from scripts.xyz_fusion_v1.pipeline import validate_nested_support
    import torch
    frame,lineage,summary,paths=baseline_intake(ROOT)
    template=ROOT/config['template_dir'];asset=validate_reference(template)
    # Scope code hashes to the actual dependency closure, not unrelated Git edits or HEAD.
    sources=[Path(__file__),*sorted((ROOT/'scripts/xyz_fusion_v1').glob('*.py')),
        ROOT/'src/models/backbones/amuse_x_hybrid3d.py',ROOT/'src/models/backbones/deterministic_pool.py',
        ROOT/'src/utils/privacy.py',ROOT/'src/method_y_revival/privacy.py']
    contract={'config':config,'source_sha256':{str(x.relative_to(ROOT)):file_sha(x) for x in sources},
              'input_sha256':{k:file_sha(v) for k,v in paths.items()},'template':asset,
              'manifest':json.loads(frame.to_json(orient='records')),
              'runtime':{k:importlib.metadata.version(k) for k in ('torch','numpy','pandas','scipy','statsmodels','scikit-learn','scikit-survival','antspyx','nibabel')},
              'python':sys.version,'torch_cuda':torch.version.cuda,
              'cuda_device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'no_historical_weights_reused':True}
    if args.mode in ('preflight','smoke-preprocess','benchmark'):
        check=ROOT/'results/xyz_mri_baseline_fast_v1_checks'/digest(contract)[:16]
        support=validate_nested_support(frame,config)
        atomic_json(check/'readiness.json',{'status':'PREPROCESSING_READY_NOT_FUSION_COMPLETED','summary':summary,
                    'input_contract_sha256':digest(contract),'production_training_started':False,'nested_support':support})
        if args.mode=='benchmark':
            gpu_guard()
            from scripts.xyz_fusion_v1.benchmark import benchmark_microbatch
            r=benchmark_microbatch(config['training']);atomic_json(check/'gpu_benchmark.json',r)
            print(json.dumps(r,indent=2),flush=True)
        elif args.mode=='smoke-preprocess':
            with writer_lock(check):
                r=prepare_cache(frame,check,template,config['preprocessing'],digest(contract),limit=args.smoke_count)
                atomic_json(check/'preprocess_smoke.json',r)
                print(json.dumps({'status':'SMOKE_FINISHED','statuses':[x['status'] for x in r['rows']]}),flush=True)
        else:print(json.dumps({'status':'PREPROCESSING_READY_NOT_FUSION_COMPLETED','summary':summary},indent=2),flush=True)
        return 0
    with writer_lock(out):
        store=Store(out,contract)
        store.job('intake',lambda:{'summary':summary,'lineage':json.loads(__import__('pandas').DataFrame(lineage).to_json(orient='records'))})
        state_path=out/'orchestration_state.json'
        terminal=state_path.exists() and json.loads(state_path.read_text()).get('stage')=='PRIMARY_FINALIZED'
        if args.mode=='verify' or terminal:
            from scripts.xyz_fusion_v1.pipeline import verify_bound_outputs
            verify_bound_outputs(out,store,config)
            report=store.job('reports')
            if report['outer_folds_completed']!=5:raise ValueError('FINAL_FIVE_FOLD_GATE')
            for i in range(5):store.job(f'outer_{i}')
            from src.method_y_revival.privacy import scan_public_bundle
            if scan_public_bundle(out/'public'):raise ValueError('PUBLIC_PRIVACY_FAILED')
            if json.loads((out/'public/results.json').read_text())!=report:raise ValueError('PUBLIC_REPORT_HASH_DRIFT')
            print(json.dumps({'status':'PRIMARY_VERIFIED','not_completed_by_this_run':report['not_completed_by_this_run']}),flush=True)
            return 0
        # GPU benchmark occurs before any scientific fit; effective batch stays fixed.
        gpu_guard()
        from scripts.xyz_fusion_v1.benchmark import benchmark_microbatch
        perf=store.job('runtime_benchmark',lambda:benchmark_microbatch(config['training']))
        # Benchmark temporaries are dead, but PyTorch may still reserve their VRAM.
        # Release that unused reserve before the later free-VRAM preflight.
        torch.cuda.empty_cache()
        policy=dict(config['training']);policy['microbatch_size']=perf['selected_microbatch']
        live=dict(config);live['training']=policy
        selected,cache=prepare_cache(frame,out,template,config['preprocessing'],store.binding)
        store.job('cache',lambda:cache)
        store.job('post_qc_nested_support',lambda:validate_nested_support(selected,config))
        atomic_json(out/'orchestration_state.json',{'stage':'MRI_CACHE_COMMITTED','contract':store.binding,
                                                  'n':len(selected),'events':int(selected.event.sum())})
        gpu_guard()
        from scripts.xyz_fusion_v1.mri_training import ImageBank
        from scripts.xyz_fusion_v1.pipeline import crossfit,verify_bound_outputs
        from scripts.xyz_fusion_v1.report import make_report,render_report
        bank=ImageBank(out/'private/mri_cache.npy',policy['ram_cache_max_gib'])
        print(json.dumps({'event':'MRI_BANK_READY','mode':bank.mode}),flush=True)
        cells=crossfit(selected,bank,live,out,store)
        report=store.job('reports',lambda:make_report(cells,selected,live))
        atomic_json(out/'public/results.json',report)
        render_report(report,out/'public/REPORT.md')
        from src.method_y_revival.privacy import scan_public_bundle
        if scan_public_bundle(out/'public'):raise ValueError('PUBLIC_PRIVACY_FAILED')
        verification=verify_bound_outputs(out,store,live)
        atomic_json(out/'orchestration_state.json',{'stage':'PRIMARY_FINALIZED','contract':store.binding,
                    'report_sha256':digest(report),'claim_status':report['claim_status'],
                    'verification':verification,
                    'extensions':report['not_completed_by_this_run']})
        print(json.dumps({'status':report['status'],'n':len(selected),'claim':report['claim_status'],
                          'extensions':report['not_completed_by_this_run']}),flush=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({'status':'INTERRUPTED_RESUMABLE','instruction':'REPEAT_SAME_COMMAND'}),flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(json.dumps({'status':'BLOCKED','reason':f'{type(exc).__name__}:{exc}'}),flush=True)
        raise
