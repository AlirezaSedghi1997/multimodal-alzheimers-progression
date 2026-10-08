"""Manual entry point; library imports occur after H-only runtime routing."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all', 'preflight', 'audit', 'statistics', 'tau', 'pacc', 'scanner', 'spatial', 'four_way', 'verify', 'promote'], default='preflight')
    args = parser.parse_args()
    from scripts.xyz_fusion_v1.runtime import configure
    configure(ROOT, threads=4)
    from scripts.xyz_fusion_postall_v2.runner import execute
    return execute(ROOT, args.mode, ROOT / 'configs/xyz_fusion_postall_v2.json')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({'status': 'INTERRUPTED_RESUMABLE', 'committed_units_preserved': True}), flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print(json.dumps({'status': 'BLOCKED', 'reason': type(exc).__name__ + ':' + str(exc)}), flush=True)
        raise
