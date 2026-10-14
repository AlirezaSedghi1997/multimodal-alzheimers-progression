"""Training-only tabular transforms and explicitly paired image-score features."""
from __future__ import annotations
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from scripts.xyz_fusion_v1.statistics import outcome


def fit_model(frame, features, family, parameter, config, seed):
    if set(features) & {'event', 'time_days', 'cohort', 'site', 'manufacturer',
                         'mri_site', 'mri_model', 'mri_manufacturer', 'pet_manufacturer',
                         'mri_acquisition_center_proxy', 'patient_hash', 'source_subject_id'}:
        raise ValueError('FORBIDDEN_MODEL_FEATURE')
    if frame[features].notna().sum().eq(0).any():
        raise ValueError('ENTIRELY_MISSING_TRAINING_FEATURE')
    if family == 'cox_ridge':
        estimator = CoxPHSurvivalAnalysis(alpha=float(parameter), ties='breslow', n_iter=500)
    elif family == 'rsf':
        estimator = RandomSurvivalForest(n_estimators=config['secondary']['rsf_trees'],
            min_samples_leaf=int(parameter), max_features='sqrt', max_depth=8,
            random_state=seed, n_jobs=config['training']['cpu_threads'])
    else:
        raise ValueError('UNKNOWN_SURVIVAL_FAMILY')
    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                          StandardScaler(), estimator)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(frame[features], outcome(frame))
    return model


def curves(model, frame, features, grid):
    values = np.asarray([f(np.asarray(grid)) for f in model.predict_survival_function(frame[features])])
    from scripts.xyz_fusion_v1.statistics import check_curves
    return check_curves(values, len(frame), grid)


def score_features(frame, survival, grid, times, modality):
    """Retain three time-specific signals; all produced by ONE locked expert."""
    from scripts.xyz_fusion_v1.statistics import check_curves
    s = check_curves(survival, len(frame), grid)
    out = frame.copy()
    for i, t in enumerate(times, 1):
        p = np.array([np.interp(t, grid, row) for row in s])
        out[f'{modality}_risk_{i}'] = -np.log(np.clip(p, 1e-6, 1.))
    return out


def assert_pair(expert_receipt, meta_subjects, test_subjects):
    ancestors = set(expert_receipt['train_subjects']) | set(expert_receipt['selection_subjects'])
    if ancestors & (set(meta_subjects) | set(test_subjects)) or set(meta_subjects) & set(test_subjects):
        raise ValueError('PAIRED_EXPERT_META_TEST_ANCESTRY_OVERLAP')


def score_diagnostic(meta, held, columns, checkpoint):
    rows = {}
    for c in columns:
        a, b = np.asarray(meta[c], float), np.asarray(held[c], float)
        rows[c] = {'meta_mean': float(a.mean()), 'meta_sd': float(a.std()),
                   'outer_mean': float(b.mean()), 'outer_sd': float(b.std()),
                   'outer_outside_meta_range_fraction': float(((b < a.min()) | (b > a.max())).mean()),
                   'near_constant_meta': bool(a.std() < 1e-6),
                   'near_constant_outer': bool(b.std() < 1e-6)}
    return {'checkpoint_for_meta_and_outer': checkpoint, 'same_expert_verified': True,
            'distribution': rows, 'uses_outer_outcomes': False,
            'automatic_model_reselection': False}
