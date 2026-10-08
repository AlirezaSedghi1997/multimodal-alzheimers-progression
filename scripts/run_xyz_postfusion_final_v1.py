"""One manual command for corrected F00-F08 plus regional and manufacturer evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True


def steps(mode):
    names=['run_xyz_fusion_postall_v2.py','run_xyz_pet_manufacturer_audit_v1.py','run_xyz_regional_evidence_v1.py']
    if mode not in ('preflight','all','verify'):raise ValueError('UNKNOWN_FINAL_MODE')
    return [(name,mode) for name in names]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['all','preflight','verify'],default='preflight')
    args=parser.parse_args()
    if ROOT.as_posix().lower()!='h:/new project':raise ValueError('AUTHORITATIVE_H_ROOT_REQUIRED')
    from scripts.xyz_fusion_v1.runtime import configure
    configure(ROOT,threads=4)
    from scripts.xyz_fusion_postall_v2.core import writer_lock,Units,file_sha,atomic_json
    out=ROOT/'results/xyz_postfusion_final_v1_ram'
    with writer_lock(out):
        files=[Path(__file__),ROOT/'scripts/run_xyz_postfusion_final_v1.ps1']
        for folder in ('xyz_postfusion_final_v1','xyz_pet_manufacturer_audit_v1','xyz_regional_evidence_v1','xyz_fusion_postall_v2'):
            files+=list((ROOT/'scripts'/folder).glob('*.py'))
        files += [ROOT/'scripts'/name for name,_ in steps(args.mode)]
        files += [ROOT/'configs'/name for name in ('xyz_fusion_postall_v2.json','xyz_pet_manufacturer_audit_v1.json','xyz_regional_evidence_v1.json')]
        contract={'source_sha256':{p.relative_to(ROOT).as_posix():file_sha(p) for p in files},
                  'python':sys.version,'manual_long_execution':True,'parent_fusion_retraining':False}
        units=Units(out,contract)
        try:
            for name,mode in steps(args.mode):
                print(json.dumps({'event':'POSTFUSION_CHAIN_STAGE','script':name,'mode':mode}),flush=True)
                atomic_json(out/'orchestration_state.json',{'status':'RUNNING','script':name,'mode':mode,'contract':units.binding})
                code=subprocess.call([sys.executable,'-B','-u',str(ROOT/'scripts'/name),'--mode',mode],cwd=ROOT)
                if code!=0:
                    atomic_json(out/'orchestration_state.json',{'status':'BLOCKED_OR_INTERRUPTED','script':name,'exit_code':code,'contract':units.binding})
                    return code if code>0 else 130
            if args.mode=='preflight':
                result={'status':'FINAL_CHAIN_PREFLIGHT_PASSED_NO_LONG_EXECUTION_STARTED','contract':units.binding}
            else:
                from scripts.xyz_postfusion_final_v1.closure import build,verify
                if args.mode=='all':build(ROOT,units)
                result=verify(ROOT,units)
            atomic_json(out/'orchestration_state.json',result);print(json.dumps(result),flush=True)
            return 0
        except BaseException as exc:
            atomic_json(out/'orchestration_state.json',{'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'BLOCKED',
                'reason':type(exc).__name__+':'+str(exc),'contract':units.binding});raise


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:print('INTERRUPTED_RESUMABLE',flush=True);raise SystemExit(130)
    except Exception as exc:print(json.dumps({'status':'BLOCKED','reason':type(exc).__name__+':'+str(exc)}),flush=True);raise
