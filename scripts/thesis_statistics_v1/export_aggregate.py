"""Allowlisted aggregate projection. Never copy private predictions or event-time rows."""
from pathlib import Path
import argparse, bisect, hashlib, json, math, re

def load(root,name):
    return json.loads((Path(root)/name).read_text(encoding='utf8'))

def pick(value,keys):
    return {key:value[key] for key in keys if key in value}

def performance(value):
    out=pick(value,['n','events','ibs','uno_c','ipa'])
    out['fold_metrics']=[]
    for i,fold in enumerate(value.get('fold_metrics',[])):
        row=pick(fold,['n','events','ibs','uno_c_index'])
        row['outer_fold']=i
        row['calibration']={h:pick(v,['status','n_at_risk','events_by_horizon',
            'calibration_intercept','calibration_slope','calibration_is_descriptive_not_refit'])
            for h,v in fold.get('horizons',{}).items()}
        out['fold_metrics'].append(row)
    return out

def contrast(value,family):
    result=pick(value,['base','augmented','delta_ibs_mean','delta_uno_mean',
        'delta_ibs_ci95','delta_uno_ci95','delta_ibs_bonferroni_ci','delta_uno_bonferroni_ci',
        'valid_ibs_resamples','valid_uno_resamples','resamples','resampling','uncertainty','multiplicity'])
    result.update(comparisons_per_metric=family,family_coverage=0.95,
                  joint_coverage_across_metrics_claimed=False,seed=20261011)
    return result

def assert_aggregate(value):
    forbidden={'patient_hash','sample_hash','mri_sample_hash','BID','private_rows',
               'censor_train','test_subjects','train_subjects','curves','raw_path','signature'}
    def visit(v):
        if isinstance(v,dict):
            if forbidden.intersection(v):raise ValueError('PARTICIPANT_LEVEL_KEY_IN_EXPORT')
            for item in v.values():visit(item)
        elif isinstance(v,list):
            for item in v:visit(item)
        elif isinstance(v,str):
            if re.search(r'(?i)(?:\b[A-Z]:[\\/]|(?:pt|sample)_[0-9a-f]{12,}|gh[pousr]_[A-Za-z0-9]{20,})',v):
                raise ValueError('IDENTIFIER_OR_LOCAL_PATH_IN_EXPORT')
        elif isinstance(v,float) and not math.isfinite(v):raise ValueError('NONFINITE_AGGREGATE')
    visit(value)

def project(source):
    source=Path(source)
    audit=load(source,'data_audit.json')
    extra=load(source,'extra_demographics.json')
    demographic={k:v for k,v in {**audit['demographics'],**extra}.items() if v['n']>=20}
    attr=audit['attrition']
    cohort={'demographics':demographic,'standardized_mean_differences':audit['smd'],
            'excluded_small_group_details':['no_volumes','no_pacc'],
            'exclusion_detail_scope':'Counts retained in flow; small-group covariate summaries omitted.',
            'tau_chronology':pick(audit['tau_chronology'],['n','after_origin','same_day','min_days_before','median_days_before','max_days_before']),
            'flow':{'a4_source':attr['a4_source'],'initial_endpoint':attr['initial_endpoint'],
                    'no_initial_followup':attr['outside_initial_endpoint'],
                    'common_landmark':attr['intake']['common_landmark_subjects'],
                    'linkage_exclusions':attr['intake']['exclusions'],
                    'automated_anatomical_qc_excluded':attr['qc_excluded'],'final_n':1059},
            'kaplan_meier_fixed_grid':{}}
    for name,km in audit['km'].items():
        points=[]
        for year in [0,1,2,3,4]:
            idx=bisect.bisect_right(km['days'],365.25*year)-1
            points.append({'year':year,'survival':1.0 if idx<0 else km['survival'][idx],
                           'at_risk':km['at_risk'][str(year)]})
        cohort['kaplan_meier_fixed_grid'][name]=points
    perf=load(source,'performance_supplement.json')
    baseline={'models':{k:performance(v) for k,v in perf['models'].items()},
              'comparisons':[contrast(v,2) for v in perf['comparisons']],
              'null_fitted_on_outer_train':perf['null_fitted_on_outer_train'],
              'precision_scope':'ORIGINAL_AMP_PARENT_NOT_FP32_CHILD',
              'ipa_aggregation':'1 - fold-size-weighted IBS(model) / fold-size-weighted IBS(null)'}
    tau={k:{'difference':contrast(v['difference'],3),
             'models':{a:performance(m) for a,m in v['models'].items()}}
         for k,v in load(source,'tau_multiplicity.json').items()}
    nc=load(source,'nocognitive_pacc.json')
    longitudinal=pick(nc,['n_people','n_observations','risk_features_exclude',
                         'longitudinal_adjustment_includes_baseline_cognition','new_fits','frozen_cnn','risk_scale'])
    longitudinal['prediction']={k:performance(v) for k,v in nc['prediction'].items()}
    longitudinal['models']={}
    for k,v in nc['models'].items():
        row=pick(v,['status','formula','converged','random_effect_covariance_eigenvalues','coefficients',
                    'aic','n','random_effects','equivalent_random_time_basis_center_years','selected_by','unregularized_ML_same_rows'])
        row['optimizer_attempts']=[pick(a,['optimizer','converged','log_likelihood','gradient_per_observation','valid']) for a in v['optimizer_attempts']]
        longitudinal['models'][k]=row
    additional=load(source,'additional_audit.json')
    disposition=pick(additional,['disposition','APOE','source_mri_safety_reads'])
    out={'cohort_summary.json':cohort,'baseline_and_ipa.json':baseline,
         'tau_multiplicity.json':tau,'noncognitive_pacc.json':longitudinal,
         'disposition_and_qc.json':disposition,
         'riley_scenarios.json':load(source,'riley_full_scenarios.json'),
         'tau_training_support.json':load(source,'tau_training_support.json')}
    sources=['data_audit.json','extra_demographics.json','performance_supplement.json','tau_multiplicity.json',
             'nocognitive_pacc.json','additional_audit.json','riley_full_scenarios.json','tau_training_support.json']
    out['provenance.json']={'version':'THESIS_STATISTICS_V1_20261011',
        'scope':'POST_HOC_INTERNAL_EXPLORATORY_SUPPLEMENT',
        'source_summary_hashes':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in sources},
        'precision_relationship':'Retains original AMP parent identity; FP32 sensitivity remains a distinct analysis.',
        'excluded_material':['participant rows','exact individual event times','identifiers','predictions',
                             'MRI/PET images','model weights','signed thesis','data dictionaries'],
        'does_not_establish_data_redistribution_permission':True}
    for result in out.values():assert_aggregate(result)
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    if args.source.resolve()==args.destination.resolve():raise ValueError('SOURCE_AND_DESTINATION_MUST_DIFFER')
    values=project(args.source)
    args.destination.mkdir(parents=True,exist_ok=True)
    for name,value in values.items():
        (args.destination/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'aggregate_files':len(values),'participant_files_copied':0}))

if __name__=='__main__':main()
