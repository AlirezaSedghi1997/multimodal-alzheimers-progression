"""Leakage-resistant fit/inner-only control-model tests."""
from __future__ import annotations

import copy

import pytest

from scripts.mx_science_extension_v1.baselines import feature_matrix, fit_controls


def _rows(n: int, *, start: int = 0):
    return [{"patient_hash": f"patient-{i}", "sample_hash": f"sample-{i}",
             "ab_label": str(i % 2), "substudy": "DO_NOT_USE",
             "age": str(55 + i % 15), "education_years": str(8 + i % 7),
             "apoe4_carrier": str(i % 2), "mmse_base": str(25 + i % 5),
             "sex": "1" if i % 2 else "2", "pet_day_t0": str(i * 4),
             "t1_day_t0": str(i * 3)} for i in range(start, start + n)]


def test_locked_sex_encoding_matches_cache_semantics():
    matrix = feature_matrix(_rows(2))
    assert matrix[:, 4].tolist() == [0.0, 1.0]


def test_outer_labels_and_cohort_cannot_select_c_or_fit_preprocessing():
    fit, inner, outer = _rows(24), _rows(8, start=24), _rows(6, start=32)
    first = fit_controls(fit, inner, outer)
    changed = copy.deepcopy(outer)
    for row in changed:
        row["ab_label"] = str(1 - int(row["ab_label"]))
        row["substudy"] = "OUTER_LABEL_CHANGED"
        row["pet_day_t0"] = "99999999"
    second = fit_controls(fit, inner, changed)
    assert first == second
    assert first["selection_scope"] == "fit_to_inner_only_no_outer_label_access"
    assert first["preprocessing_fit_scope"] == "fit_partition_only"


def test_outer_feature_outlier_does_not_affect_other_patient_probability():
    fit, inner, outer = _rows(24), _rows(8, start=24), _rows(6, start=32)
    first = fit_controls(fit, inner, outer)
    changed = copy.deepcopy(outer)
    changed[0]["age"] = "100000"
    second = fit_controls(fit, inner, changed)
    assert first["selected_c"] == second["selected_c"]
    assert first["tabular_outer_probabilities"][1:] == second["tabular_outer_probabilities"][1:]


def test_one_class_fit_or_inner_blocks():
    fit, inner, outer = _rows(24), _rows(8, start=24), _rows(6, start=32)
    for row in fit:
        row["ab_label"] = "1"
    with pytest.raises(ValueError, match="INNER_CLASS"):
        fit_controls(fit, inner, outer)
