"""Small atomic, content-bound work units. Never modifies predecessor artifacts."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_text(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            f.write(value);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


@contextmanager
def writer_lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.writer.lock').open('a+b') as f:
        if f.tell()==0:
            f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            try:
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as e:
                raise RuntimeError('ACTIVE_FUSION_WRITER') from e
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name=='nt':
                msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


class Store:
    def __init__(self,root,contract):
        self.root=Path(root);self.binding=digest(contract)
        self.execution_binding=self.binding
        self.storage_amended=False
        self.approved_prior_jobs={}
        self.numerics_amended=False
        cp=self.root/'private/contract.json'
        if cp.exists():
            original=json.loads(cp.read_text())
            if original!=contract:
                if (self.root/'private/amp_gradient_compatibility_amendment.json').exists():
                    self._accept_amp_amendment(original,contract)
                else:self._accept_storage_amendment(original,contract)
        else:
            atomic_json(cp,contract)

    def _accept_storage_amendment(self,original,current):
        """Admit only the evidenced Windows fsync repair, not scientific drift."""
        mismatch='FUSION_CONTRACT_MISMATCH_NEW_VERSION_REQUIRED'
        execution_binding=digest(current)
        try:
            amendment=json.loads((self.root/'private/storage_compatibility_amendment.json').read_text())
            allowed={'scripts/xyz_fusion_v1/mri_cache.py','scripts/xyz_fusion_v1/store.py'}
            before=dict(original);after=dict(current)
            old_sources={k.replace('\\','/'):v for k,v in before.pop('source_sha256').items()}
            new_sources={k.replace('\\','/'):v for k,v in after.pop('source_sha256').items()}
            if before!=after or old_sources.keys()!=new_sources.keys():raise ValueError(mismatch)
            changed={k for k in old_sources if old_sources[k]!=new_sources[k]}
            if changed!=allowed or set(amendment['allowed_source_changes'])!=allowed:raise ValueError(mismatch)
            if (amendment['kind']!='WINDOWS_CACHE_FSYNC_WRITABLE_HANDLE_20261001'
                    or amendment['original_contract_sha256']!=digest(original)
                    or amendment['execution_contract_sha256']!=execution_binding
                    or amendment['scientific_parameters_unchanged'] is not True):raise ValueError(mismatch)
            receipt_path=self.root/'private/mri_cache.json'
            if file_sha(receipt_path)!=amendment['cache_receipt_sha256']:raise ValueError(mismatch)
            evidence='private/repairs/windows_cache_fsync_20261001/validation.json'
            if amendment['evidence']!=evidence or file_sha(self.root/evidence)!=amendment['evidence_sha256']:
                raise ValueError(mismatch)
            proof=json.loads((self.root/evidence).read_text())
            receipt=json.loads(receipt_path.read_text())
            if (proof['kind']!=amendment['kind']
                    or proof['status']!='CACHE_COMMITTED_READY_FOR_MANUAL_RESUME'
                    or proof['old_contract']!=digest(original)
                    or proof['new_contract']!=execution_binding
                    or proof['original_contract_file_sha256']!=file_sha(self.root/'private/contract.json')
                    or proof['original_scan_receipts_unchanged'] is not True
                    or proof['voxel_bitwise_comparison']!='PASS_ALL_SELECTED_ROWS'
                    or proof['registration_executions']!=0 or proof['training_executions']!=0
                    or proof['cache_sha256']!=receipt['sha256']
                    or proof['cache_rows']!=receipt['shape'][0]
                    or receipt['parent_contract']!=digest(original)):raise ValueError(mismatch)
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
            raise RuntimeError(mismatch) from exc
        # Keep committed scientific receipts immutable; every new job records
        # the actual execution code separately, bound by the amendment above.
        self.binding=digest(original)
        self.storage_amended=True

    def _accept_amp_amendment(self,original,current):
        """Verify a second append-only amendment without rewriting prior evidence."""
        mismatch='FUSION_CONTRACT_MISMATCH_NEW_VERSION_REQUIRED'
        allowed={'scripts/xyz_fusion_v1/mri_training.py','scripts/xyz_fusion_v1/store.py'}
        prior_jobs={'intake','runtime_benchmark','cache','post_qc_nested_support','splits'}
        folder=self.root/'private/repairs/amp_gradscaler_20261001'
        try:
            a=json.loads((self.root/'private/amp_gradient_compatibility_amendment.json').read_text())
            parent=json.loads((folder/'parent_execution_contract.json').read_text())
            if (a['kind']!='AMP_SAME_BATCH_REPLAY_NUMERICAL_REPAIR_20261001'
                    or a['original_contract_sha256']!=digest(original)
                    or a['parent_execution_contract_sha256']!=digest(parent)
                    or a['execution_contract_sha256']!=self.execution_binding
                    or a['scientific_parameters_unchanged'] is not True
                    or a['protected_artifacts_unchanged'] is not True
                    or file_sha(folder/'parent_execution_contract.json')!=a['parent_contract_file_sha256']
                    or file_sha(folder/'validation.json')!=a['evidence_sha256']):raise ValueError(mismatch)
            before=dict(parent);after=dict(current)
            sa={k.replace('\\','/'):v for k,v in before.pop('source_sha256').items()}
            sb={k.replace('\\','/'):v for k,v in after.pop('source_sha256').items()}
            if (before!=after or sa.keys()!=sb.keys() or {k for k in sa if sa[k]!=sb[k]}!=allowed
                    or set(a['allowed_source_changes'])!=allowed
                    or set(a['approved_prior_job_sha256'])!=prior_jobs):raise ValueError(mismatch)
            proof=json.loads((folder/'validation.json').read_text())
            if (proof['kind']!=a['kind'] or proof['status']!='REAL_BATCH_REPLAY_AND_RESUME_VERIFIED'
                    or proof['execution_contract_sha256']!=self.execution_binding
                    or proof['parent_execution_contract_sha256']!=digest(parent)
                    or proof['original_contract_sha256']!=digest(original)
                    or proof['protected_artifacts_unchanged'] is not True
                    or proof['old_checkpoint_rebound'] is not False
                    or proof['optimizer_nonfinite_updates']!=0 or proof['batch_drop_allowed'] is not False
                    or proof['production_training_executed_by_repair'] is not False):raise ValueError(mismatch)
            # The parent must itself pass the original fsync compatibility proof.
            self._accept_storage_amendment(original,parent)
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
            raise RuntimeError(mismatch) from exc
        self.approved_prior_jobs=a['approved_prior_job_sha256']
        self.numerics_amended=True

    def job(self,name,compute=None):
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
            raise ValueError('INVALID_JOB_ID')
        path=self.root/'private/jobs'/f'{name}.json'
        if path.exists():
            r=json.loads(path.read_text())
            if r.get('contract')!=self.binding or r.get('payload_sha256')!=digest(r['payload']):
                raise RuntimeError('FUSION_RECEIPT_INVALID')
            executed=r.get('execution_contract_sha256',self.binding)
            legacy_storage_job=(self.storage_amended and not self.numerics_amended
                                and name in {'intake','runtime_benchmark'} and executed==self.binding)
            evidenced_prior_job=name in self.approved_prior_jobs and file_sha(path)==self.approved_prior_jobs[name]
            if executed!=self.execution_binding and not legacy_storage_job and not evidenced_prior_job:
                raise RuntimeError('FUSION_JOB_EXECUTION_CONTRACT_MISMATCH')
            print(json.dumps({'event':'RECOVERED','job':name}),flush=True)
            return r['payload']
        if compute is None:
            raise RuntimeError('FUSION_JOB_INCOMPLETE:'+name)
        result=compute()
        atomic_json(path,{'contract':self.binding,'execution_contract_sha256':self.execution_binding,
                          'payload_sha256':digest(result),'payload':result})
        print(json.dumps({'event':'COMMITTED','job':name}),flush=True)
        return result
