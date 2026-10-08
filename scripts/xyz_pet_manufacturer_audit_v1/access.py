"""A durable evaluation result is recoverable; an uncertain exposure is not rerun."""
from scripts.xyz_fusion_postall_v2.core import read_json, atomic_json, digest


def evaluated_once(store, name, dependencies, compute):
    store.path(name)  # Validate the identifier before constructing paths.
    journal=store.root/'private/access'/f'{name}.json'
    result=store.root/'private/access'/f'{name}.result.json'
    binding={'binding':store.binding,'dependencies':dependencies}
    old=read_json(journal) if journal.exists() else None
    if old is not None and (any(old.get(k)!=v for k,v in binding.items()) or old.get('access_count')!=1):
        raise ValueError('OUTER_ACCESS_CONTRACT_DRIFT')
    committed=store.read(name,dependencies)
    if old is None and (committed is not None or result.exists()):
        raise ValueError('RESULT_WITHOUT_OUTER_ACCESS_JOURNAL')
    if result.exists():
        doc=read_json(result)
        if (any(doc.get(k)!=v for k,v in binding.items()) or doc['payload_sha256']!=digest(doc['payload'])
                or committed is not None and committed!=doc['payload']):
            raise ValueError('OUTER_RESULT_DRIFT')
        value=store.job(name,lambda:doc['payload'],dependencies)
    else:
        if old is not None or committed is not None:
            raise ValueError('INCOMPLETE_OUTER_ACCESS_REQUIRES_REVIEW_NO_AUTOMATIC_REEVALUATION')
        atomic_json(journal,dict(binding,state='IN_PROGRESS',access_count=1))
        value=compute()
        atomic_json(result,dict(binding,payload=value,payload_sha256=digest(value)))
        value=store.job(name,lambda:value,dependencies)
    atomic_json(journal,dict(binding,state='COMMITTED',access_count=1,payload_sha256=digest(value)))
    return value


def verify_access(store,name,dependencies):
    value=store.read(name,dependencies)
    if value is None:raise ValueError('LMO_EVALUATION_MISSING')
    record=read_json(store.root/'private/access'/f'{name}.json')
    result=read_json(store.root/'private/access'/f'{name}.result.json')
    expected={'binding':store.binding,'dependencies':dependencies}
    if (any(record.get(k)!=v or result.get(k)!=v for k,v in expected.items()) or
            record.get('state')!='COMMITTED' or record.get('access_count')!=1 or
            record.get('payload_sha256')!=digest(value) or result.get('payload')!=value or
            result.get('payload_sha256')!=digest(value)):
        raise ValueError('LMO_ACCESS_VERIFICATION_FAILED')
    return value
