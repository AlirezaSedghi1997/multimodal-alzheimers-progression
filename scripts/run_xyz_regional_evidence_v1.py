"""Source-region association supplement; no image-model fitting."""
from pathlib import Path
import sys
import argparse
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['all','associations','preflight','verify'],default='preflight')
    args=parser.parse_args()
    from scripts.xyz_fusion_v1.runtime import configure
    configure(ROOT,threads=4)
    try:
        from scripts.xyz_regional_evidence_v1.runner import execute
        raise SystemExit(execute(ROOT,args.mode))
    except KeyboardInterrupt:
        print(json.dumps({'status':'INTERRUPTED_RESUMABLE'}),flush=True);raise SystemExit(130)
    except Exception as exc:
        print(json.dumps({'status':'BLOCKED','reason':type(exc).__name__+':'+str(exc)}),flush=True);raise
