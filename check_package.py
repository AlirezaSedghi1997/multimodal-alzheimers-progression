"""Offline integrity check; never launches training or reads study data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'release_manifest.json').read_text(encoding='utf-8'))
    expected={r['path']:r for r in manifest['files']}
    present={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
             and '.git' not in p.relative_to(root).parts
             and p.name not in {'release_manifest.json','package_verification.json'}}
    if present!=set(expected):raise ValueError('FILESET_MISMATCH')
    for rel,row in expected.items():
        path=root/rel
        if not path.resolve().is_relative_to(root) or path.is_symlink(): raise ValueError('UNSAFE_PATH')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError('HASH_MISMATCH:'+rel)
    return {'status':'PASS_PACKAGE_INTEGRITY_ONLY','files':len(expected),
            'scientific_approval':False,'data_transfer_permission':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    try: print(json.dumps(verify(args.root)))
    except Exception as error:
        print(json.dumps({'status':'FAIL','reason':str(error)}));sys.exit(1)
