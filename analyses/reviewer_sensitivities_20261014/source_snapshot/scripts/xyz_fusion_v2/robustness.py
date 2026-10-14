"""MRI manufacturer/site transport sensitivities with explicit estimability limits."""
from __future__ import annotations
import numpy as np
from scripts.xyz_fusion_v1.store import digest
from scripts.xyz_fusion_v1.pipeline import partition
from scripts.xyz_fusion_v1.statistics import ibs_score
from .sensitivity import scanner_splits
from .tabular import run_tabular
from .pipeline import paired_crossfit
from .feasibility import optional_support


def run_robustness(frame, bank, config, store):
    result={}
    for column in ('mri_manufacturer','mri_site'):
        splits, excluded=scanner_splits(frame,column,config)
        # Bounded site image sensitivity, selected by size alone. All supported
        # sites still receive tabular evaluation; no outcome-based winner choice.
        image_groups={s['group'] for s in sorted(splits,key=lambda x:(-len(x['test']),x['group']))[:3]}
        evaluated=[]
        for split in splits:
            name=column+'_'+digest(split['group'])[:12]
            tr=frame.loc[frame.patient_hash.isin(split['train'])]
            te=frame.loc[frame.patient_hash.isin(split['test'])]
            support=optional_support(frame,[split],config,'scalar',require_five=False)
            if support['status']!='SUPPORTED':
                evaluated.append({'group':split['group'],**support})
                continue
            output=run_tabular(frame,config,store,'lmo_'+name,'scalar',[split],families=['cox_ridge'])
            row={'group':split['group'],'status':'TABULAR_TRANSPORT_COMPLETED',
                 'n':len(te),'events':int(te.event.sum()),
                 'tabular_metrics':{k:v['metrics'] for k,v in output['cox_ridge'][0]['arms'].items()}}
            if column=='mri_manufacturer' or split['group'] in image_groups:
                support=optional_support(frame,[split],config,'mri',image=True,require_five=False)
                if support['status']!='SUPPORTED':
                    row['image_status']=support; evaluated.append(row);continue
                # One fixed seed for bounded scanner sensitivity, not seed selection.
                local=dict(config,training_seeds=[config['training_seeds'][0]])
                # Use a distinct namespace/output directory with a derived contract.
                # This avoids training paths colliding with the main MRI models.
                from .store import Store
                child=Store(store.root/'private/robustness'/name,{'parent':store.binding,'split':split,'config':local})
                runs=paired_crossfit(frame,{'mri':bank},local,child,'mri',[split])
                row['image_metrics']={k:v['metrics'] for k,v in runs[0]['cells'][0]['arms'].items()}
                row['status']='MATCHED_PAIRED_EXPERT_SCANNER_SENSITIVITY_COMPLETED'
                row['image_seed']=local['training_seeds'][0]
                row['child_root']=(child.root.relative_to(store.root)).as_posix()
            else:row['image_status']='NOT_RUN_PRESPECIFIED_TOP_THREE_SITE_BUDGET'
            evaluated.append(row)
        result[column]={'evaluated':evaluated,'excluded':excluded,
                        'two_supported_independent_groups':len(splits)>=2,
                        'unknown_not_imputed':True,'ge_philips_not_merged':True}
    return result
