"""Entry point; configure project-local temporary directories before scientific imports."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all', 'preflight', 'smoke', 'verify'], default='preflight')
    args = parser.parse_args()
    runtime = ROOT / '.runtime/xyz_fp32_spatial_v1'
    for key in ('TEMP', 'TMP', 'TMPDIR', 'MPLCONFIGDIR', 'TORCH_HOME', 'TORCHINDUCTOR_CACHE_DIR',
                'CUDA_CACHE_PATH', 'NUMBA_CACHE_DIR', 'JOBLIB_TEMP_FOLDER'):
        path = runtime / key; path.mkdir(parents=True, exist_ok=True); os.environ[key] = str(path)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'; sys.dont_write_bytecode = True
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '4'
    from scripts.xyz_fp32_spatial_v1.runner import execute
    return execute(ROOT, args.mode, ROOT / 'configs/xyz_fp32_spatial_sensitivity_v1.json')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({'status': 'INTERRUPTED_RESUMABLE'}), flush=True)
        raise SystemExit(130)
    except Exception as error:
        print(json.dumps({'status': 'BLOCKED', 'reason': type(error).__name__ + ':' + str(error)}), flush=True)
        raise
