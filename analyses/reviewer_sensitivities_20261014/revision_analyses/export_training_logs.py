"""Export aggregate epoch histories only; never copy participant membership or weights."""
from pathlib import Path
import argparse, hashlib, json, re

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.repo);out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    records=[];inputs={}
    pattern=re.compile(r'(mri|pet)_s(13|29|47)_o[0-4]_c[0-2]_(mri|pet)_(expert|tune[01])')
    for src in sorted((root/'results/xyz_fusion_v2/private/models').glob('*/locked.json')):
        if not pattern.fullmatch(src.parent.name): continue
        raw=src.read_bytes();v=json.loads(raw)
        assert v['terminal'] is True
        rec={'model_label':src.parent.name,'best_epoch':v['best_epoch'],
             'fit_n':len(v['train_subjects']),'selection_n':len(v['selection_subjects']),
             'selected_inner_ipcw_ibs':v['inner_ibs'],
             'history':[{k:row[k] for k in ['epoch','train_nll','inner_ipcw_ibs']} for row in v['history']]}
        records.append(rec);inputs[str(src)]={'sha256':hashlib.sha256(raw).hexdigest()}
    assert len(records)==405
    obj={'scope':'Original completed training and selection histories, exported without participant-level information. Logs are execution records, not independent reproduction or proof of learned prognostic signal.',
         'new_training':False,'record_count':len(records),'records':records}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
    private=out.parent.parent/'private';private.mkdir(exist_ok=True)
    (private/'training_log_inputs.json').write_text(json.dumps(inputs,indent=2),encoding='utf8')
    print(json.dumps({'exported_histories':len(records),'participant_records_exported':False}))

if __name__=='__main__': main()
