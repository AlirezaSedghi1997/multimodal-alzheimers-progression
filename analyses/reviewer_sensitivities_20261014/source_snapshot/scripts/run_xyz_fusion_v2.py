"""Manual, resumable six-reviewer fusion redesign."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.xyz_fusion_v1.runtime import configure
configure(ROOT)
import argparse
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['all','preflight','run','verify'],default='all')
    p.add_argument('--config',default=str(ROOT/'configs/xyz_fusion_v2.json'))
    args=p.parse_args()
    from scripts.xyz_fusion_v2.runner import execute
    return execute(ROOT,'run' if args.mode=='all' else args.mode,args.config)


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({'status':'INTERRUPTED_RESUMABLE','instruction':'REPEAT_SAME_COMMAND'}),flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(json.dumps({'status':'BLOCKED','reason':f'{type(exc).__name__}:{exc}'}),flush=True)
        raise
