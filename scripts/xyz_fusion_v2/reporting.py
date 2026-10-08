"""Complete matched contrasts, conditional intervals, seed variability and aggregate release."""
from __future__ import annotations
import numpy as np
from sksurv.metrics import concordance_index_ipcw
from scripts.xyz_fusion_v1.report import cell_frame, individual_bs
from scripts.xyz_fusion_v1.store import atomic_json, atomic_text, file_sha
from .protocol import contrasts, CLAIM

def paired_ci(cells,base,augmented,grid,seed,resamples,family_size):
    prepared=[]
    for fold in cells:
        a,b=fold['arms'][base],fold['arms'][augmented]
        if a['subjects']!=b['subjects'] or a['event']!=b['event'] or a['time']!=b['time'] or a['censor_train']!=b['censor_train']:
            raise ValueError('UNMATCHED_PAIRED_CONTRAST')
        train,test=cell_frame(a)
        ba,tr,te=individual_bs(train,test,np.asarray(a['curves']),grid)
        bb,_,_=individual_bs(train,test,np.asarray(b['curves']),grid)
        prepared.append((bb-ba,tr,te,1-np.asarray(a['curves'])[:,-1],1-np.asarray(b['curves'])[:,-1]))
    rng=np.random.default_rng(seed);bs=[];cs=[]
    for _ in range(resamples):
        sums=0.;n=0;deltas=[];weights=[]
        for delta,tr,te,ra,rb in prepared:
            idx=rng.integers(0,len(te),len(te));sums+=delta[idx].sum();n+=len(idx)
            try:
                ca=concordance_index_ipcw(tr,te[idx],ra[idx],tau=grid[-1])[0]
                cb=concordance_index_ipcw(tr,te[idx],rb[idx],tau=grid[-1])[0]
                if np.isfinite(ca) and np.isfinite(cb):deltas.append(cb-ca);weights.append(len(idx))
            except ValueError:pass
        bs.append(sums/n)
        if len(deltas)==len(prepared):cs.append(float(np.average(deltas,weights=weights)))
    def ci(values):
        return list(map(float,np.quantile(values,[.025,.975]))) if len(values)>=.95*resamples else None
    def simultaneous_ci(values):
        tail=.05/(2*family_size)
        return list(map(float,np.quantile(values,[tail,1-tail]))) if len(values)>=.95*resamples else None
    point_uno=[];weights=[]
    for delta,tr,te,ra,rb in prepared:
        try:
            point_uno.append(concordance_index_ipcw(tr,te,rb,tau=grid[-1])[0]-concordance_index_ipcw(tr,te,ra,tau=grid[-1])[0]);weights.append(len(te))
        except ValueError:pass
    return {'base':base,'augmented':augmented,'delta_ibs_mean':float(np.mean(np.concatenate([x[0] for x in prepared]))),
            'delta_uno_mean':float(np.average(point_uno,weights=weights)) if len(point_uno)==len(prepared) else None,
            'delta_ibs_ci95':ci(bs),'delta_uno_ci95':ci(cs),'valid_ibs_resamples':len(bs),'valid_uno_resamples':len(cs),
            'delta_ibs_bonferroni_ci':simultaneous_ci(bs),'delta_uno_bonferroni_ci':simultaneous_ci(cs),
            'resamples':resamples,'resampling':'PAIRED_SUBJECT_WITHIN_OUTER_FOLD',
            'uncertainty':'CONDITIONAL_ON_FITTED_MODELS_AND_TRAIN_CENSORING_NOT_SELECTION_UNCERTAINTY',
            'multiplicity':'EXPLORATORY_NO_CONFIRMATORY_SIGNIFICANCE_CLAIM'}



def summarize(cells, config, track, runs=None):
    if len(cells)!=5 or {c['fold'] for c in cells}!=set(range(5)):
        raise ValueError('REPORT_REQUIRES_ALL_FIVE_OUTER_FOLDS')
    seen=set()
    for c in cells:
        if c['status']!='OUTER_EVAL_COMMITTED' or seen & set(c['test_subjects']):
            raise ValueError('REPORT_INCOMPLETE_OR_OVERLAPPING_SUBJECTS')
        seen.update(c['test_subjects'])
    result={};rng=np.random.default_rng(config['seed'])
    for arm in cells[0]['arms']:
        ms=[c['arms'][arm]['metrics'] for c in cells]
        losses=[]
        for c in cells:
            v=c['arms'][arm];tr,te=cell_frame(v)
            losses.extend(individual_bs(tr,te,np.asarray(v['curves']),config['fusion']['ibs_grid_days'])[0])
        losses=np.asarray(losses)
        draws=[float(losses[rng.integers(0,len(losses),len(losses))].mean()) for _ in range(config['fusion']['bootstrap_resamples'])]
        cs=[m['uno_c_index'] for m in ms]
        result[arm]={'n':sum(m['n'] for m in ms),'events':sum(m['events'] for m in ms),
                     'ibs':float(losses.mean()),'ibs_conditional_ci95':np.quantile(draws,[.025,.975]).tolist(),
                     'uno_c':float(np.average(cs,weights=[m['n'] for m in ms])) if all(x is not None for x in cs) else None,
                     'fold_metrics':ms}
    pairs=contrasts(track)
    comparisons=[paired_ci(cells,a,b,config['fusion']['ibs_grid_days'],config['seed']+j,
                  config['fusion']['bootstrap_resamples'],len(pairs)) for j,(a,b) in enumerate(pairs)]
    seed_variation={}
    if runs:
        for arm in result:
            scores=[float(np.average([c['arms'][arm]['metrics']['ibs'] for c in run['cells']],
                                    weights=[c['arms'][arm]['metrics']['n'] for c in run['cells']])) for run in runs]
            seed_variation[arm]={'values':scores,'standard_deviation':float(np.std(scores,ddof=1)),
                                 'seeds_are_not_independent_subjects':True,'selection_of_best_seed':False}
    return {'status':'COMPLETED_INTERNAL_ANALYSIS','track':track,'claim_status':CLAIM,'subjects':len(seen),
            'outer_folds_completed':5,'arms':result,'paired_comparisons':comparisons,
            'seed_variability':seed_variation,'conditional_ci_covers_full_training_uncertainty':False,
            'clinical_claim_allowed':False,'external_claim_allowed':False,'disease_localization_allowed':False,
            'quantitative_pet_equivalence_allowed':False,
            'outer_based_model_selection':False,'external_status':'OUT_OF_SCOPE_NO_EXTERNAL_DATA'}


def write_report(store, reports, diagnostics):
    public=store.root/'public';public.mkdir(parents=True,exist_ok=True)
    report={'protocol':'XYZ_PAIRED_EXPERT_SURVIVAL_FUSION_V2','claim_status':CLAIM,
            'analyses':reports,'diagnostics':public_diagnostics(diagnostics),
            'completion_is_not_positive_scientific_claim':True,'clinical_claim_allowed':False,
            'external_claim_allowed':False,'disease_localization_allowed':False}
    atomic_json(public/'results.json',report)
    lines=['# XYZ fusion v2','',CLAIM,'',
           'Repeated paired-expert late fusion; frozen image experts are shared by meta training and outer inference.',
           'External, clinical, causal and disease-localization claims remain unavailable.','',
           '| Analysis | Arm | Subjects | Events | IBS | Uno C |','|---|---|---:|---:|---:|---:|']
    for name,r in reports.items():
        for arm,v in r.get('arms',{}).items():
            lines.append(f"| {name} | {arm} | {v['n']} | {v['events']} | {v['ibs']:.5f} | {v['uno_c']} |")
    lines+=['','## Uncertainty','',
            'Subject bootstrap intervals are conditional on fitted models and training censoring distributions.',
            'Repeated seed variability is reported separately; it is not a confidence interval for the whole development process.',
            'All planned contrasts include clinical-to-amyloid and matched clinical-to-blood comparisons.',
            'Changing the endpoint or model after historical outcome exposure does not create an untouched confirmatory cohort.',
            '', '## Scientific limits','',
            'Two-visit-confirmed CDR progression is not a diagnosis of Alzheimer disease.',
            'Selection into the MRI/follow-up cohort, blood missingness and independent censoring remain assumptions.',
            'Source regional PET values retain their documented source names; no Centiloid/CenTauR image equivalence is asserted.',
            'Every NOT_ESTIMABLE or BLOCKED diagnostic remains visible in results.json.']
    atomic_text(public/'REPORT.md','\n'.join(lines)+'\n')
    alignment=['# Reporting evidence map — no certification claim','',
        '| Framework / domain | Evidence | Limit |','|---|---|---|',
        '| TRIPOD+AI participants/predictors/outcome | locked input contract, dates, endpoint views, missingness | selected A4 cohort |',
        '| TRIPOD+AI development/evaluation | paired-expert ancestry, nested tabular HPO, five folds, locked seeds | reused historically exposed cohort |',
        '| PROBAST+AI analysis | IPCW, PH/non-PH sensitivity, calibration, matched contrasts | censoring/MNAR assumptions cannot be proven |',
        '| CLAIM acquisition/processing | exact MRI sidecars, parent QC/cache hashes, exact PET scan dates | automated QC, source-mask quantitative limitations |',
        '| CLAIM interpretability | baseline-specific numerical/fidelity/randomization evidence | no lesion truth; failures remain diagnostic |',
        '| Reproducibility/privacy | contracts, unit receipts, source/runtime hashes, aggregate release | no independent external validation |']
    atomic_text(public/'REPORTING_ALIGNMENT.md','\n'.join(alignment)+'\n')
    draw_figures(public,reports)
    from .privacy import scan_public_bundle
    if scan_public_bundle(public):raise ValueError('V2_PUBLIC_PRIVACY_FAILED')
    return {'status':'REPORTS_AND_PRIVACY_VERIFIED','files':{p.name:file_sha(p) for p in public.iterdir() if p.is_file()}}


def draw_figures(public,reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=reports.get('mri',{})
    if not r.get('arms'):return
    names=list(r['arms']);values=[r['arms'][k]['ibs'] for k in names]
    fig,ax=plt.subplots(figsize=(9,6));ax.barh(names,values);ax.set_xlabel('IPCW integrated Brier score (lower is better)')
    ax.set_title('Internal exploratory paired-expert MRI fusion');fig.tight_layout()
    # Text SVG permits recursive privacy inspection; no unidentified binary payload.
    fig.savefig(public/'fusion_ibs.svg');plt.close(fig)
    for track in ('mri','pet'):
        value=reports.get(track,{})
        if not value.get('arms'):continue
        preferred=['clinical','clinical_amyloid','clinical_mri_amyloid','clinical_mri_amyloid_pet']
        chosen=[a for a in preferred if a in value['arms']]
        horizons=sorted({h for a in chosen for m in value['arms'][a].get('fold_metrics',[]) for h in m.get('horizons',{})},key=float)
        if horizons:
            fig,axes=plt.subplots(1,len(horizons),figsize=(6*len(horizons),5),squeeze=False)
            for j,h in enumerate(horizons):
                ax=axes[0,j];ax.plot([0,1],[0,1],'k--',linewidth=1)
                for k,arm in enumerate(chosen):
                    labelled=False
                    for metrics in value['arms'][arm]['fold_metrics']:
                        bins=metrics.get('horizons',{}).get(h,{}).get('calibration_bins',[])
                        if not bins:continue
                        ax.plot([b['predicted_risk'] for b in bins],[b['ipcw_observed_fraction'] for b in bins],
                                marker='o',alpha=.5,color=f'C{k}',label=None if labelled else arm)
                        labelled=True
                ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted risk',ylabel='IPCW observed fraction',
                       title=f'{float(h)/365.25:.1f} years; separate outer-fold curves')
                handles,_=ax.get_legend_handles_labels()
                if handles:ax.legend(fontsize=7)
            fig.suptitle(f'{track.upper()}: aggregate calibration bins; estimable folds only')
            fig.tight_layout();fig.savefig(public/f'{track}_calibration.svg');plt.close(fig)
        pairs=value.get('paired_comparisons',[])
        if pairs:
            fig,ax=plt.subplots(figsize=(10,max(4,len(pairs)*.5)))
            for i,p in enumerate(pairs):
                ax.plot(p['delta_ibs_mean'],i,'o',color='C0')
                if p.get('delta_ibs_ci95'):ax.plot(p['delta_ibs_ci95'],[i,i],color='C0')
            ax.axvline(0,color='black',linestyle='--');ax.set_yticks(range(len(pairs)),[p['augmented']+' vs '+p['base'] for p in pairs])
            ax.set_xlabel('Delta IPCW IBS with conditional paired 95% CI (negative favors addition)')
            ax.set_title(f'{track.upper()}: exploratory contrasts; selection uncertainty excluded')
            fig.tight_layout();fig.savefig(public/f'{track}_paired_ibs.svg');plt.close(fig)


def public_diagnostics(diagnostics):
    """Whitelist aggregate PET intake fields; never publish HDF paths or row data."""
    import copy
    result=copy.deepcopy(diagnostics)
    for key in ('pet_feasibility','tau_feasibility'):
        if key in result:
            allowed={'status','subjects','n','events','events_primary','regions','reference_column',
                     'centaur_equivalence','quantitative_equivalence','exclusions',
                     'changed_landmarks','legacy_manifest_date_disagreements',
                     'exact_scan_date_subjects','date_basis','outer_fold_support',
                     'legacy_tau_beyond_amyloid_negative_result_preserved'}
            result[key]={k:v for k,v in result[key].items() if k in allowed}
    for category in result.get('scanner',{}).values():
        for row in category.get('evaluated',[]):row.pop('child_root',None)
    return result
