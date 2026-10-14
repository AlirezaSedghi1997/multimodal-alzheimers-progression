"""Run locally with authorized frozen research artifacts; no downloads or CNN fitting."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'smoke', 'fit', 'report', 'all', 'verify'], default='preflight')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/xyz_oof_stacking_v1.json')
    args = parser.parse_args()
    runtime = ROOT / '.runtime/xyz_oof_stacking_v1'
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ('TMP', 'TEMP', 'TMPDIR'):
        os.environ[name] = str(runtime)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(name, '2')
    from scripts.xyz_oof_stacking_v1.runner import execute
    return execute(ROOT, args.mode, args.config)


if __name__ == '__main__':
    raise SystemExit(main())
