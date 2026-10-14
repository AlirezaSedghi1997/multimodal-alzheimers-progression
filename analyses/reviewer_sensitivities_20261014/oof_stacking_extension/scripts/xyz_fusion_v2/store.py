"""Exact V2 contracts, with one evidenced PH input-type repair; no broad drift waiver."""
import json
from pathlib import Path
from scripts.xyz_fusion_v1.store import Store as BaseStore, digest, file_sha


PH_ARRAY_ADAPTER = (
    '    # PHReg inspects auxiliary .ndim before coercion; keep interval values unchanged.\n'
    '    stop=np.asarray(stop,dtype=np.float64)\n'
    '    entry=np.asarray(entry,dtype=np.float64)\n'
    '    event=np.asarray(event,dtype=np.int64)\n'
)


def _ph_input_amendment(root, original, current):
    """Validate a pinned source delta and the exact pre-PH receipt inventory."""
    mismatch = 'V2_RUN_CONTRACT_MISMATCH_NEW_VERSION_REQUIRED'
    folder = root / 'private/repairs/ph_ndarray_20261003'
    allowed = {'scripts/xyz_fusion_v2/sensitivity.py', 'scripts/xyz_fusion_v2/store.py'}
    try:
        a = json.loads((root / 'private/ph_input_array_compatibility_amendment.json').read_text())
        before, after = dict(original), dict(current)
        old_sources, new_sources = before.pop('source_sha256'), after.pop('source_sha256')
        changed = {k for k in old_sources if old_sources[k] != new_sources.get(k)}
        if before != after or old_sources.keys() != new_sources.keys() or changed != allowed:
            raise ValueError('UNAPPROVED_CONTRACT_DELTA')
        if (a['kind'] != 'PHREG_AUXILIARY_NDARRAY_INPUT_REPAIR_20261003'
                or a['original_contract_sha256'] != digest(original)
                or a['execution_contract_sha256'] != digest(current)
                or set(a['allowed_source_changes']) != allowed
                or a['scientific_parameters_unchanged'] is not True):
            raise ValueError('UNAPPROVED_AMENDMENT')
        for name, key in [('validation.json', 'evidence_sha256'),
                          ('protected_files_before.json', 'protected_manifest_sha256'),
                          ('execution_contract.json', 'execution_contract_file_sha256')]:
            if file_sha(folder / name) != a[key]:
                raise ValueError('AMENDMENT_EVIDENCE_HASH_MISMATCH')
        if json.loads((folder / 'execution_contract.json').read_text()) != current:
            raise ValueError('EXECUTION_CONTRACT_MISMATCH')
        for name in allowed:
            if file_sha(folder / 'before' / Path(name).name) != old_sources[name]:
                raise ValueError('SOURCE_BEFORE_HASH_MISMATCH')
        anchor = '    a=np.asarray(xx);b=np.column_stack([xx,zz])\n'
        old_text = (folder / 'before/sensitivity.py').read_text(encoding='utf-8')
        new_path = Path(__file__).with_name('sensitivity.py')
        new_text = new_path.read_text(encoding='utf-8')
        if (old_text.count(anchor) != 1 or new_text != old_text.replace(anchor, anchor + PH_ARRAY_ADAPTER)
                or file_sha(new_path) != new_sources['scripts/xyz_fusion_v2/sensitivity.py']
                or file_sha(__file__) != new_sources['scripts/xyz_fusion_v2/store.py']):
            raise ValueError('PH_DATATYPE_ONLY_SOURCE_PROOF_FAILED')
        proof = json.loads((folder / 'validation.json').read_text())
        protected = json.loads((folder / 'protected_files_before.json').read_text())
        if (proof['kind'] != a['kind']
                or proof['status'] != 'DATATYPE_ONLY_REPAIR_AND_TRAIN_PH_REPLAY_VERIFIED'
                or proof['original_contract_sha256'] != digest(original)
                or proof['execution_contract_sha256'] != digest(current)
                or proof['original_contract_file_sha256'] != file_sha(root / 'private/contract.json')
                or proof['datatype_only_source_comparison'] is not True
                or proof['protected_artifacts_unchanged'] is not True
                or proof['new_production_training_executed'] is not False
                or proof['outer_prediction_or_evaluation_reexecuted'] is not False
                or proof['tests_passed'] < 1
                or {r['fold'] for r in proof['train_only_ph_replays']} != set(range(5))):
            raise ValueError('PH_REPAIR_PROOF_INVALID')
        approved = {Path(p).stem: sha for p, sha in protected.items()
                    if p.startswith('private/jobs/') and p.endswith('.json')}
        if (not approved or approved != a['approved_prior_job_sha256']
                or protected['private/contract.json'] != file_sha(root / 'private/contract.json')
                or any(n not in {'intake', 'analysis_plan', 'splits', 'blood_selection'}
                       and not n.startswith('scalar_') for n in approved)):
            raise ValueError('PH_REPAIR_PRIOR_JOB_SCOPE_INVALID')
        return approved
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError(mismatch) from exc


class Store(BaseStore):
    def __init__(self, root, contract):
        path = Path(root)/'private/contract.json'
        if path.exists():
            original = json.loads(path.read_text())
            if original != contract:
                approved = _ph_input_amendment(Path(root), original, contract)
                # Scientific receipt identity stays fixed; new jobs bind the repaired source.
                super().__init__(root, original)
                self.execution_binding = digest(contract)
                self.approved_prior_jobs = approved
                return
        super().__init__(root, contract)


def checked_job(store, name):
    """Recover journal after a receipt commit without repeating inference."""
    from scripts.xyz_fusion_v1.store import atomic_json
    result = store.job(name)
    path = store.root/'private/access_journal'/f'{name}.json'
    if path.exists():
        journal = json.loads(path.read_text())
        if journal.get('state') not in ('IN_PROGRESS', 'COMMITTED'):
            raise ValueError('UNKNOWN_PREDICTION_JOURNAL_STATE')
        if journal.get('state') == 'COMMITTED' and journal.get('payload_sha256') != digest(result):
            raise ValueError('PREDICTION_RECEIPT_JOURNAL_MISMATCH')
        journal.update(state='COMMITTED', payload_sha256=digest(result))
        atomic_json(path, journal)
    return result
