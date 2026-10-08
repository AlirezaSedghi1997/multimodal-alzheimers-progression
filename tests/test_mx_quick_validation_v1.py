import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.mx_quick_validation_v1.common import Store, digest


def test_actual_terminal_entrypoint_help_from_unrelated_cwd(tmp_path):
    import subprocess,sys
    script = Path(__file__).resolve().parents[1]/'scripts/mx_quick_validation_v1/supervise.py'
    result = subprocess.run([sys.executable,'-B',str(script),'--help'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert 'preflight' in result.stdout


def test_coordinator_lock_excludes_second_writer_and_releases_after_exception(tmp_path):
    from scripts.mx_fasttrack_v2_1.supervise_v21 import coordinator_lock
    with pytest.raises(KeyboardInterrupt):
        with coordinator_lock(tmp_path):
            with pytest.raises(RuntimeError,match='ALREADY_ACTIVE'):
                with coordinator_lock(tmp_path):
                    pass
            raise KeyboardInterrupt()
    with coordinator_lock(tmp_path):
        pass


def test_receipt_restart_skips_committed_unit_and_retries_interrupted(tmp_path):
    s = Store(tmp_path, {'output': 'results/final_results/test'}, {'source': 'a'})
    calls = []
    def work(i):
        calls.append(i)
        if i == 1 and calls == [0, 1]:
            raise KeyboardInterrupt()
        return {'value': i}
    with pytest.raises(KeyboardInterrupt):
        for i in range(3):
            s.receipt(f'fold/{i}', lambda: work(i))
    restarted = Store(tmp_path, s.config, s.protocol)
    for i in range(3):
        restarted.receipt(f'fold/{i}', lambda: work(i))
    assert calls == [0, 1, 1, 2]


def test_receipt_nan_source_drift_and_payload_tampering_fail_closed(tmp_path):
    s = Store(tmp_path, {'output': 'results/final_results/test'}, {'source': 'a'})
    with pytest.raises(ValueError):
        s.receipt('bad', lambda: {'p': float('nan')})
    assert not (s.private / 'bad.json').exists()
    s.receipt('good', lambda: {'p': .3})
    changed = Store(tmp_path, s.config, {'source': 'b'})
    with pytest.raises(RuntimeError, match='DRIFT'):
        changed.receipt('good')
    p = s.private / 'good.json'
    data = json.loads(p.read_text())
    data['payload']['p'] = .4
    p.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='DRIFT'):
        s.receipt('good')


def test_receipt_cannot_be_swapped_between_folds(tmp_path):
    s=Store(tmp_path,{'output':'results/final_results/test'},{'source':'a'})
    s.receipt('fold_0',lambda:{'p':.3})
    (s.private/'fold_1.json').write_bytes((s.private/'fold_0.json').read_bytes())
    with pytest.raises(RuntimeError,match='DRIFT'):
        s.receipt('fold_1')


def test_torn_atomic_write_is_retryable(tmp_path, monkeypatch):
    from scripts.mx_fasttrack_v2_1 import supervise_v21
    s = Store(tmp_path, {'output': 'results/final_results/test'}, {})
    replace = supervise_v21.os.replace
    def fail(*args):
        raise OSError('simulated power loss before commit')
    monkeypatch.setattr(supervise_v21.os, 'replace', fail)
    with pytest.raises(OSError):
        s.receipt('unit', lambda: {'ok': True})
    assert not (s.private / 'unit.json').exists()
    monkeypatch.setattr(supervise_v21.os, 'replace', replace)
    assert s.receipt('unit', lambda: {'ok': True}) == {'ok': True}


def test_same_voxels_with_different_ids_detected_and_nan_rejected():
    from scripts.mx_quick_validation_v1.audit import image_signature
    a = np.random.default_rng(1).normal(size=(1, 16, 16, 16)).astype('f4')
    assert image_signature(a)['voxel_sha256'] == image_signature(a.copy())['voxel_sha256']
    b = a.copy(); b.flat[0] += 1
    assert image_signature(a)['voxel_sha256'] != image_signature(b)['voxel_sha256']
    b.flat[0] = np.nan
    with pytest.raises(RuntimeError):
        image_signature(b)


def test_lineage_uses_original_source_index_not_compacted_cache_row():
    from scripts.mx_quick_validation_v1.audit import check_lineage
    parent=dict(row='0',source_base_index='30',is_synthetic='0',sample_hash='s',patient_hash='p')
    derivative=dict(row='1',source_base_index='30',is_synthetic='1',source_sample_hash='s',
                    source_patient_hash='p',patient_hash='p',source_fold='0',source_train_only='1')
    fit={(f,s):{'p'} for f in range(5) for s in (13,29,47)}
    assert check_lineage([parent,derivative],fit)['materialized_derivatives']==1
    derivative['source_patient_hash']='different'
    with pytest.raises(RuntimeError,match='LINEAGE'):
        check_lineage([parent,derivative],fit)
    derivative['source_patient_hash']='p'; fit[0,13]=set()
    with pytest.raises(RuntimeError,match='OUTSIDE'):
        check_lineage([parent,derivative],fit)


def test_tabular_selection_does_not_read_outer_labels_or_forbidden_features():
    from scripts.mx_quick_validation_v1.tabular import fit_nonlinear
    def rows(n):
        return [dict(age=65+i%20, education_years=12+i%4, apoe4_carrier=i%2,
                     mmse_base=29-i%3, sex=1+i%2, ab_label=str(i%2)) for i in range(n)]
    fit, val, outer = rows(70), rows(30), rows(10)
    cfg = {'depths': [2,3], 'learning_rate': .05, 'max_iter': 4,
           'min_samples_leaf': 10, 'l2_regularization': 1., 'early_stopping': False}
    a = fit_nonlinear(fit, val, outer, 13, cfg)
    for r in outer:
        r.pop('ab_label'); r.update(substudy='FORBIDDEN', scanner='FORBIDDEN', pet='FORBIDDEN')
    b = fit_nonlinear(fit, val, outer, 13, cfg)
    assert a == b
    assert a['feature_order'] == ['age','education_years','apoe4_carrier','mmse_base','sex']


def test_matching_no_replacement_no_prediction_dependence():
    from scripts.mx_quick_validation_v1.paired_analysis import match_pairs
    rows = [dict(patient_hash=str(i), fold=0, label=i%2, age=70+i/4, sex=1,
                 predictions={'ignored': i}) for i in range(20)]
    pairs = match_pairs(rows, 3)
    changed = copy.deepcopy(rows)
    for r in changed:
        r['predictions'] = {'ignored': -100}
    assert pairs == match_pairs(changed, 3)
    assert len(set(sum((list(p) for p in pairs), []))) == 2*len(pairs)


def test_bootstrap_identical_comparators_and_seed_replication():
    from scripts.mx_quick_validation_v1.paired_analysis import paired
    y = np.array([0,1]*30)
    p = np.linspace(.1,.9,60)[:,None]
    a = paired(y, p, p, resamples=25, seed=1)
    b = paired(y, np.repeat(p,3,axis=1), np.repeat(p,3,axis=1), resamples=25, seed=1)
    assert a['n'] == b['n'] == 60
    assert a['effects'] == b['effects']
    assert a['effects']['brier']['ci95'] == [0.,0.]


@pytest.mark.parametrize('condition', ['noise02','noise05','bias10','blur05','brain_only','background_only'])
def test_stress_shape_determinism_and_mask_scope(condition):
    from scripts.mx_quick_validation_v1.stress import perturb
    x = np.random.default_rng(4).normal(size=(16,16,16)).astype('f4')
    mask = np.zeros_like(x, dtype=bool); mask[2:-2,2:-2,2:-2] = True
    a = perturb(x, mask, condition, 17)
    assert a.shape == x.shape and np.isfinite(a).all()
    assert np.array_equal(a, perturb(x, mask, condition, 17))
    if condition in ('noise02','noise05','bias10'):
        assert np.array_equal(a[~mask], x[~mask])
    if condition == 'brain_only':
        assert (a[~mask] == 0).all()
    if condition == 'background_only':
        assert (a[mask] == 0).all()


def test_stratified_selection_weights_recover_population_counts():
    from scripts.mx_quick_validation_v1.stress import select_sample
    rows = [dict(patient_hash=str(i), fold=i//50, label=int(i%5!=0)) for i in range(250)]
    selected = select_sample(rows, 20, 7)
    assert len(selected) == 100
    assert sum(r['weight'] for r in selected) == pytest.approx(250)
    assert selected == select_sample(list(reversed(rows)), 20, 7)


def test_weighted_metrics_match_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
    from scripts.mx_quick_validation_v1.paired_analysis import metrics
    y=np.array([0,1,0,1,1,0]); p=np.array([.2,.2,.5,.5,.8,.8]); w=np.array([2,1,1,3,1,4])
    assert metrics(y,p,w)==pytest.approx([roc_auc_score(y,p,sample_weight=w),
        average_precision_score(y,p,sample_weight=w), brier_score_loss(y,p,sample_weight=w)])


def test_single_class_and_duplicated_matching_pair_are_not_validated():
    from scripts.mx_quick_validation_v1.paired_analysis import paired
    assert paired(np.zeros(5),np.ones((5,1))*.3,np.ones((5,1))*.4,resamples=10,seed=1)['status'].startswith('NOT_ESTIMABLE')
    with pytest.raises(RuntimeError,match='MATCHING'):
        paired(np.array([0,1,0,1]),np.ones((4,1))*.3,np.ones((4,1))*.4,
               resamples=10,seed=1,pairs=[[0,1],[0,3]])


def test_two_complete_pipeline_passes_after_mid_stress_interruption(tmp_path,monkeypatch):
    """Real CSV/H5, HGB, bootstrap, receipts/report; replace only external parent/GPU."""
    import csv,h5py,torch
    from scripts.mx_science_extension_v1.audit_and_lock import _digest
    from scripts.mx_quick_validation_v1 import audit,tabular,stress,paired_analysis,supervise
    from scripts.mx_quick_validation_v1.common import EXT,PANEL,INDEX,CACHE,immutable_json
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/method_x_quick_validation_v1.json').read_text())
    config.update(output='results/final_results/fixture',bootstrap_resamples=15,subgroup_min_per_class=2,stress_patients_per_fold=4)
    config['nonlinear']['max_iter']=3
    patients=[]
    for i in range(100):
        patients.append(dict(patient_hash=f'p{i:02d}',sample_hash=f's{i:02d}',ab_label=str(i%2),
            age=str(68+i%12),sex=str(1+(i//2)%2),education_years='15',apoe4_carrier=str(i%2),mmse_base='28',
            pet_day_t0='100',t1_day_t0='110',abs_pet_t1_gap_days='10',t1_file_hash=f'unique{i}'))
    def csv_file(path,rows,fields):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    membership=[]
    import random
    for fold in range(5):
        test=patients[fold*20:(fold+1)*20]; train=[r for r in patients if r not in test]
        for phase,rows in [('train',train),('test',test)]:
            csv_file(tmp_path/'splits'/f'fold_{fold}_{phase}.csv',rows,list(patients[0]))
        inner=[]
        for seed in (13,29,47):
            rows=copy.deepcopy(train);random.Random(seed+fold).shuffle(rows);n=int(len(rows)*.15)
            val,fit=rows[:n],rows[n:]
            inner.append({'membership_sha256':_digest({'fit':sorted(r['patient_hash'] for r in fit),
                'validation':sorted(r['patient_hash'] for r in val)}),'fit_n':len(fit),'inner_validation_n':len(val)})
        membership.append({'inner':inner})
    csv_file(tmp_path/'splits/external_holdout.csv',[],list(patients[0]))
    idx=[dict(row=str(i),source_base_index=str(i),is_synthetic='0',patient_hash=r['patient_hash'],sample_hash=r['sample_hash']) for i,r in enumerate(patients)]
    csv_file(tmp_path/INDEX,idx,list(idx[0]))
    immutable_json(tmp_path/EXT/'reports/day1_patient_provenance_audit.json',{'folds':membership})
    panel=[dict(patient_hash=r['patient_hash'],label=int(r['ab_label']),fold=i//20,gap_days=10,
                predictions={'hybrid_selected':{str(s):.5 for s in (13,29,47)}}) for i,r in enumerate(patients)]
    immutable_json(tmp_path/PANEL,{'rows':panel})
    (tmp_path/CACHE).parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(tmp_path/CACHE,'w') as h:
        h['image']=np.random.default_rng(10).normal(size=(100,1,16,16,16)).astype('f4')
        h['tabular']=np.ones((100,5),'f4');h['sample_hash']=np.array([r['sample_hash'].encode() for r in patients])
    parent={'fasttrack':{},'mx5_outer_n':100,'extension_outer_n':100,'xai_computations':'COMPLETED',
            'xai_scientific_gate':'BLOCKED_EXPLORATORY_GATE'}
    protocol={'parent_audit':parent,'stress_sample':stress.select_sample(panel,4,config['bootstrap_seed'])}
    store=Store(tmp_path,config,protocol)
    monkeypatch.setattr(audit,'build_audit',lambda root:{'status':'READY_INTERNAL_ONLY'})
    monkeypatch.setattr(stress,'checkpoint_specs',lambda root:{(f,a):(Path('fixture'), 'fixedhash') for f in range(5) for a in ('parent_final','hybrid_selected_seed13')})
    monkeypatch.setattr(stress,'load_model',lambda *args:({},torch.zeros(5),torch.ones(5)))
    calls=[]
    def predict(*args):
        calls.append(1)
        if len(calls)==10: raise KeyboardInterrupt()
        return .5
    monkeypatch.setattr(stress,'probability',predict)
    mask=np.zeros((16,16,16),bool);mask[2:-2,2:-2,2:-2]=True
    audit.run_audit(store);tabular.run_tabular(store);paired_analysis.run_statistics(store)
    with pytest.raises(KeyboardInterrupt): stress.run_stress(store,mask)
    assert len(list((store.private/'stress').rglob('*.json')))==9
    stress.run_stress(store,mask)
    assert len(calls)==281  # 280 committed forward units + precisely one interrupted unit
    first=supervise.report(store)
    # Full restart validates/reuses every unit and leaves identical final manifest.
    before={str(p):p.read_bytes() for p in store.out.rglob('*.json')}
    audit.run_audit(store);tabular.run_tabular(store);paired_analysis.run_statistics(store);stress.run_stress(store,mask)
    assert len(calls)==281
    assert supervise.report(store)==first
    assert all(Path(p).read_bytes()==value for p,value in before.items())
    assert first['xai_scientific_gate']=='BLOCKED_EXPLORATORY_GATE'
