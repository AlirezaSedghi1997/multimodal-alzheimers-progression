"""Versioned CPU-only postanalysis transactions. No model/outer inference."""
from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform

from .atomic_io import atomic_write_json, read_json
from .contracts import sha256_mapping


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def make_contract(inputs, sources, policy):
    return {'schema_version': 5, 'inputs': {k: file_sha(v) for k, v in inputs.items()},
            'sources': {k: file_sha(v) for k, v in sources.items()}, 'policy': policy,
            'runtime': {'python': platform.python_version(), **{
                p: version(p) for p in ('numpy', 'scipy', 'pandas', 'statsmodels', 'scikit-survival')
            }}, 'threads': {k: os.environ.get(k) for k in
                ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}


@contextmanager
def analysis_lock(root):
    """Kernel-owned lock: released on Ctrl-C/process death; no stale PID deletion."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.validation_v5.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('ACTIVE_LOCAL_VALIDATION_WRITER') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def committed_job(root, name, contract, compute, *, verify_only=False):
    """Caller holds analysis_lock. Recovery checks precede ALL computation."""
    if not name.replace('_', '').isalnum():
        raise ValueError('INVALID_JOB_NAME')
    path = Path(root) / 'jobs' / (name + '.json')
    digest = sha256_mapping(contract)
    if path.exists():
        result = read_json(path)
        if result.get('contract_sha256') != digest:
            raise RuntimeError('LOCAL_VALIDATION_CONTRACT_MISMATCH:' + name)
        if result.get('report_sha256') != sha256_mapping({k: v for k, v in result.items() if k != 'report_sha256'}):
            raise RuntimeError('LOCAL_VALIDATION_REPORT_HASH_MISMATCH:' + name)
        if result.get('job_status') != 'COMMITTED':
            raise RuntimeError('LOCAL_VALIDATION_JOB_NOT_COMMITTED:' + name)
        json.dumps(result, allow_nan=False)
        return result['payload']
    if verify_only:
        raise RuntimeError('LOCAL_VALIDATION_JOB_MISSING:' + name)
    payload = compute()
    json.dumps(payload, allow_nan=False)
    result = {'job_status': 'COMMITTED', 'contract_sha256': digest, 'payload': payload}
    result['report_sha256'] = sha256_mapping(result)
    atomic_write_json(path, result)
    return payload


def verified_hashed_report(path):
    report = read_json(path)
    if report.get('report_sha256') != sha256_mapping({k: v for k, v in report.items() if k != 'report_sha256'}):
        raise RuntimeError('PARENT_REPORT_HASH_MISMATCH:' + Path(path).name)
    json.dumps(report, allow_nan=False)
    return report
