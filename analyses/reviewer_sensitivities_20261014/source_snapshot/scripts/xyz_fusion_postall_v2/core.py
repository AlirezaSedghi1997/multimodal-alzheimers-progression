"""Contract-bound units and read-only access to the frozen parent."""
from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import types

from scripts.xyz_fusion_v1.store import atomic_json, atomic_text, digest, file_sha, writer_lock

CLAIM = 'POST_HOC_INTERNAL_EXPLORATORY'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def inside(root, relative):
    target = (Path(root) / relative).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError('OUTPUT_PATH_ESCAPES_ROOT')
    return target


class Hashes:
    """Per-invocation cache only; bytes are rehashed on every invocation."""
    def __init__(self):
        self.values = {}

    def __call__(self, path):
        path = Path(path).resolve()
        st = path.stat()
        signature = (st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if path not in self.values or self.values[path][0] != signature:
            sha = file_sha(path)
            after = path.stat()
            if signature != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('FILE_CHANGED_DURING_HASH')
            self.values[path] = (signature, sha)
        return self.values[path][1]


class ReadOnlyParent:
    def __init__(self, root, contract=None, hashes=None):
        self.root = Path(root).resolve()
        self.hashes = hashes or Hashes()
        self.contract = read_json(self.root / 'private/contract.json')
        self.binding = digest(self.contract)
        self.execution_binding = self.binding
        self.approved = {}
        if contract is not None and contract != self.contract:
            raise ValueError('UNEXPECTED_CHILD_CONTRACT')
        amendment = self.root / 'private/repairs/ph_ndarray_20261003/execution_contract.json'
        if amendment.exists():
            from scripts.xyz_fusion_v2.store import _ph_input_amendment
            executed = read_json(amendment)
            self.approved = _ph_input_amendment(self.root, self.contract, executed)
            self.execution_binding = digest(executed)
            self.execution_contract = executed
        else:
            self.execution_contract = self.contract

    def job(self, name, compute=None):
        if compute is not None:
            raise ValueError('PARENT_COMPUTATION_FORBIDDEN')
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
            raise ValueError('INVALID_JOB_ID')
        path = self.root / 'private/jobs' / (name + '.json')
        r = read_json(path)
        if r.get('contract') != self.binding or r.get('payload_sha256') != digest(r['payload']):
            raise ValueError('PARENT_RECEIPT_INVALID:' + name)
        executed = r.get('execution_contract_sha256', self.binding)
        if executed != self.execution_binding:
            if name not in self.approved or self.hashes(path) != self.approved[name]:
                raise ValueError('PARENT_EXECUTION_CONTRACT_MISMATCH:' + name)
        return r['payload']

    def verify(self):
        """Run the actual parent's full verifier with non-mutating dependencies.

        A private function namespace prevents monkeypatching the live module.
        checked_job's recovery writes are intentionally unavailable here: an
        incomplete journal is an error, not an opportunity to repair the parent.
        """
        from scripts.xyz_fusion_v2 import verification
        ns = dict(vars(verification))
        for name, function in vars(verification).items():
            if isinstance(function, types.FunctionType) and function.__module__ == verification.__name__:
                cloned = types.FunctionType(function.__code__, ns, name, function.__defaults__, function.__closure__)
                cloned.__kwdefaults__ = function.__kwdefaults__
                ns[name] = cloned
        ns['checked_job'] = lambda store, name: store.job(name)
        ns['file_sha'] = self.hashes
        ns['Store'] = lambda root, contract: ReadOnlyParent(root, contract, self.hashes)
        return ns['verify'](self, self.contract['protocol'], final=True)


@contextlib.contextmanager
def parent_exclusion(root):
    """Lock an EXISTING byte without writing or creating any parent file."""
    path = Path(root) / '.writer.lock'
    if not path.is_file() or path.stat().st_size < 1:
        raise ValueError('EXISTING_PARENT_WRITER_LOCK_REQUIRED')
    before = file_sha(path)
    with path.open('r+b') as stream:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    if file_sha(path) != before:
        raise ValueError('PARENT_LOCK_CHANGED')


class Units:
    def __init__(self, root, contract):
        self.root = Path(root).resolve()
        self.hashes = Hashes()
        self.binding = digest(contract)
        path = self.root / 'private/contract.json'
        if path.exists():
            if read_json(path) != contract:
                raise ValueError('POSTALL_CONTRACT_MISMATCH_NEW_VERSION_REQUIRED')
        else:
            atomic_json(path, contract)

    def path(self, name):
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
            raise ValueError('INVALID_UNIT_ID')
        return self.root / 'private/units' / (name + '.json')

    def read(self, name, dependencies=None):
        path = self.path(name)
        if not path.exists():
            return None
        doc = read_json(path)
        if doc['contract'] != self.binding or doc['payload_sha256'] != digest(doc['payload']):
            raise ValueError('POSTALL_RECEIPT_DRIFT:' + name)
        if dependencies is not None and doc['dependencies'] != dependencies:
            raise ValueError('POSTALL_DEPENDENCY_DRIFT:' + name)
        for relative, sha in doc['payload'].get('outputs', {}).items():
            if self.hashes(inside(self.root, relative)) != sha:
                raise ValueError('POSTALL_BINARY_DRIFT:' + name)
        if isinstance(doc['payload'], dict) and 'path' in doc['payload'] and 'sha256' in doc['payload']:
            if self.hashes(inside(self.root, doc['payload']['path'])) != doc['payload']['sha256']:
                raise ValueError('POSTALL_MODEL_DRIFT:' + name)
        return doc['payload']

    def job(self, name, compute=None, dependencies=None):
        dependencies = dependencies or {}
        value = self.read(name, dependencies)
        if value is not None:
            print(json.dumps({'event': 'RECOVERED', 'unit': name}), flush=True)
            return value
        if compute is None:
            raise ValueError('POSTALL_UNIT_MISSING:' + name)
        value = compute()
        # One JSON is the commit point; abandoned temporary outputs are never proof.
        atomic_json(self.path(name), {'contract': self.binding, 'dependencies': dependencies,
                                     'payload': value, 'payload_sha256': digest(value)})
        self.read(name, dependencies)
        print(json.dumps({'event': 'COMMITTED', 'unit': name}), flush=True)
        return value


def make_contract(root, parent, freeze, config):
    sources = sorted((root / 'scripts/xyz_fusion_postall_v2').glob('*.py'))
    sources += [root / 'scripts/run_xyz_fusion_postall_v2.py', root / 'scripts/run_xyz_fusion_postall_v2.ps1']
    return {'schema_version': 1, 'claim_status': CLAIM, 'parent_contract': parent.binding,
            'parent_freeze_sha256': digest(freeze), 'config': config,
            'source_sha256': {p.relative_to(root).as_posix(): file_sha(p) for p in sources},
            'runtime': {p: importlib.metadata.version(p) for p in
                        ('torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'scikit-survival',
                         'statsmodels', 'nibabel', 'h5py', 'pyarrow')},
            'python': sys.version, 'reuse_old_postall_results_as_validated': False}
