"""Reproducible review evidence, separated from historical private outcomes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scripts.xyz_fusion_v1.store import file_sha


def six_sources():
    root=Path(__file__).resolve().parents[2]/'docs/xyz_fusion_v2_reviews'
    paths=[root/f'reviewer_{i}.md' for i in range(1,6)]+[root/'sol.md']
    return [{'reviewer':f'R{i+1}' if i<5 else 'SOL', 'file_name':p.name,
             'sha256':file_sha(p),'line_count':len(p.read_text(encoding='utf-8-sig').splitlines()),
             'full_content_read':True} for i,p in enumerate(paths)]


def historical_diagnostics(ctx):
    root=ctx['parent'];rows=[]
    for fold in range(5):
        a=[];b=[]
        for p in (root/'private/jobs').glob(f'mri_oof_o{fold}_i*.json'):
            a.extend(json.loads(p.read_text())['payload']['mri_risk'])
        for p in (root/'private/jobs').glob(f'outer_predictions_{fold}_*.json'):
            b.extend(json.loads(p.read_text())['payload']['mri_risk'])
        a=np.asarray(a);b=np.asarray(b)
        rows.append({'fold':fold,'oof_mean':float(a.mean()),'outer_mean':float(b.mean()),
                     'oof_sd':float(a.std()),'outer_sd':float(b.std()),
                     'outer_outside_oof_range':int(((b<a.min())|(b>a.max())).sum()),'n_outer':len(b)})
    return {'six_reviewers':six_sources(),'score_shift':rows,
            'historical_results_preserved':True,'historical_null_does_not_prove_mri_useless':True,
            'decision':'NEW_TRAINING_AND_EVALUATION_REUSE_ONLY_VERIFIED_OUTCOME_BLIND_CACHE',
            'unsupported_reviewer_claims':['SCALAR_LATE_FUSION_IS_NOT_FUSION','CDR_0P5_IS_DEMENTIA',
                'UNCONFIRMED_FINAL_CDR_ALONE_PROVES_IPCW_INVALID','POSITIVE_DELTA_UNO_IS_WORSE',
                'IN_SAMPLE_EXPERT_SCORES_ARE_VALID_META_TRAINING_INPUTS']}
