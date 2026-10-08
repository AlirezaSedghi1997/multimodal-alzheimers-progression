"""A stronger nonlinear comparator; selections and preprocessing are fit/inner only."""
import random
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from scripts.mx_fasttrack_v2.audit_inputs import read_csv
from scripts.mx_science_extension_v1.baselines import feature_matrix, _labels, PHYSICAL
from scripts.mx_science_extension_v1.audit_and_lock import _digest
from .common import EXT, PANEL, read, require, progress


def partitions(root, fold, seed, real):
    train = [r for r in read_csv(root/'splits'/f'fold_{fold}_train.csv') if r['sample_hash'] in real]
    random.Random(seed+fold).shuffle(train)
    n = min(max(int(len(train)*.15),1),len(train)-1)
    val,fit = train[:n],train[n:]
    audit = read(root/EXT/'reports/day1_patient_provenance_audit.json')['folds'][fold]['inner'][(13,29,47).index(seed)]
    membership = _digest({'fit':sorted(r['patient_hash'] for r in fit),'validation':sorted(r['patient_hash'] for r in val)})
    require(audit['membership_sha256']==membership and audit['fit_n']==len(fit) and audit['inner_validation_n']==len(val), 'NONLINEAR_INNER_MEMBERSHIP')
    return fit,val


def fit_nonlinear(fit, val, outer_covariates, seed, config):
    # Outer labels are not accepted/used by either preprocessing or model selection.
    x, y = feature_matrix(fit), _labels(fit)
    xv, yv = feature_matrix(val), _labels(val)
    require(set(y)=={0,1} and set(yv)=={0,1}, 'INNER_CLASSES_REQUIRED')
    require(not np.isnan(x).all(axis=0).any(), 'FIT_FEATURE_ALL_MISSING')
    imputer = SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
    x,xv = imputer.transform(x),imputer.transform(xv)
    models,losses = [],[]
    with threadpool_limits(limits=2):
        for depth in config['depths']:
            model = HistGradientBoostingClassifier(max_depth=depth,random_state=seed,
                       **{k:v for k,v in config.items() if k!='depths'})
            model.fit(x,y)
            loss = float(log_loss(yv,model.predict_proba(xv)[:,1],labels=[0,1]))
            require(np.isfinite(loss),'NONLINEAR_LOSS_INVALID')
            models.append(model); losses.append(loss)
        best = min(range(len(models)),key=lambda i:(losses[i],config['depths'][i]))
        p = models[best].predict_proba(imputer.transform(feature_matrix(outer_covariates)))[:,1]
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'NONLINEAR_PROBABILITIES')
    return {'probabilities':p.tolist(),'selected_depth':config['depths'][best],
            'inner_bce':losses,'feature_order':list(PHYSICAL),'imputer_statistics':imputer.statistics_.tolist(),
            'selection_scope':'INNER_ONLY','preprocessing_scope':'FIT_ONLY'}


def run_tabular(store):
    from .audit import data_rows
    _,real,tests = data_rows(store.root)
    panel = read(store.root/PANEL)['rows']
    by_patient = {r['patient_hash']:r for r in panel}
    result = {}
    for fold in range(5):
        outer = tests[fold]
        require(all(by_patient[r['patient_hash']]['fold']==fold and by_patient[r['patient_hash']]['label']==int(r['ab_label']) for r in outer),'TABULAR_OUTER_BINDING')
        for seed in store.config['seeds']:
            progress('nonlinear',fold=fold,seed=seed,status='RUNNING_OR_REUSING')
            fit,val = partitions(store.root,fold,seed,real)
            # Strip outer labels/IDs before entering model code.
            covariates = [{k:r[k] for k in PHYSICAL} for r in outer]
            saved = store.receipt(f'nonlinear/fold_{fold}_seed_{seed}',lambda:fit_nonlinear(fit,val,covariates,seed,store.config['nonlinear']))
            p = np.asarray(saved['probabilities'])
            require(len(p)==len(outer) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'SAVED_NONLINEAR_PROBABILITIES')
            for row,prob in zip(outer,p):
                result.setdefault(row['patient_hash'],{})[str(seed)] = float(prob)
    require(set(result)==set(by_patient),'NONLINEAR_PATIENT_COVERAGE')
    return store.receipt('nonlinear_panel',lambda:result)
