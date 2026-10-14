"""Fail-closed aggregate release, including inspectable vector plots."""
import csv
import io
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

PRIVATE={'event','time_days','original_event','original_time_days','patient_hash','subject_key',
         'censoring_reference','censor_train','predictions','embeddings','outcomes','risk_score',
         'source_subject_id','raw_path','sidecar_path','train_subjects','test_subjects'}


def scan_public_bundle(root):
    findings=[]
    def private(value):
        if isinstance(value,dict):return any(str(k).lower() in PRIVATE or private(v) for k,v in value.items())
        return isinstance(value,list) and any(private(v) for v in value)
    for p in sorted(Path(root).rglob('*')):
        if not p.is_file():continue
        rules=[]
        if p.suffix.lower() not in {'.json','.md','.svg','.csv','.txt'}:rules.append('UNAPPROVED_PUBLIC_FILE_TYPE')
        try:
            text=p.read_text(encoding='utf-8')
            if re.search(r'(?i)(?<![A-Za-z0-9])[A-Z]:[\\/](?!/)|A4[ _]Learn',text):rules.append('PRIVATE_PATH')
            if re.search(r'(?i)\b(?:pt|sub|scan)_[a-z0-9][a-z0-9_-]*\b',text):rules.append('SUBJECT_IDENTIFIER')
            if re.search(r'(?i)source_subject_id|raw_path|sidecar_path|\.(?:ckpt|h5|hdf5|pkl|nii)(?:\b|\.)',text):rules.append('PRIVATE_REFERENCE')
            if p.suffix=='.json' and private(json.loads(text)):rules.append('ROW_LEVEL_DATA')
            if p.suffix=='.csv' and any(k.lower() in PRIVATE for k in next(csv.reader(io.StringIO(text)),[])):
                rules.append('ROW_LEVEL_DATA')
            if p.suffix=='.svg':
                if '<!ENTITY' in text:raise ValueError('ENTITY_NOT_ALLOWED')
                xml=ET.fromstring(text)
                for e in xml.iter():
                    if e.tag.rsplit('}',1)[-1] in {'image','script','foreignObject','a'}:raise ValueError('SVG_EMBED_NOT_ALLOWED')
                    if any(k.rsplit('}',1)[-1]=='href' and not v.startswith('#') for k,v in e.attrib.items()):
                        raise ValueError('SVG_EXTERNAL_REFERENCE_NOT_ALLOWED')
        except (UnicodeError,OSError,ValueError,ET.ParseError):rules.append('UNINSPECTABLE_PUBLIC_PAYLOAD')
        findings.extend({'rule':r,'path':p.relative_to(root).as_posix()} for r in rules)
    return findings
