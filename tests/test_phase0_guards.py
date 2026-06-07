from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


check_matrix = load_module("check_method_x_feature_matrix", ROOT / "scripts" / "check_method_x_feature_matrix.py")


def write_csv(path: Path, header: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([""] * len(header))


def test_method_x_matrix_allows_only_fixed_tabular_plus_metadata(tmp_path: Path) -> None:
    p = tmp_path / "features.csv"
    write_csv(p, ["patient_hash", "sample_hash", "age", "sex", "education_years", "apoe4_carrier", "mmse_base", "ab_label"])
    assert check_matrix.read_header(p) == ["patient_hash", "sample_hash", "age", "sex", "education_years", "apoe4_carrier", "mmse_base", "ab_label"]
    assert check_matrix.classify_forbidden(check_matrix.read_header(p)) == {}


def test_method_x_matrix_rejects_pet_and_cognitive_predictors(tmp_path: Path) -> None:
    p = tmp_path / "features.csv"
    write_csv(p, ["age", "mmse_base", "centiloid", "PET_SUVR", "pacc_score", "pTau217"])
    violations = check_matrix.classify_forbidden(check_matrix.read_header(p))
    assert "centiloid_predictor" in violations
    assert "pet" in violations or "suvr" in violations
    assert "cognitive_beyond_fixed" in violations
    assert "blood_biomarker" in violations


def test_method_x_matrix_rejects_leaky_mri_surrogates(tmp_path: Path) -> None:
    p = tmp_path / "features.csv"
    write_csv(p, ["age", "voxel_spacing_x", "intensity_percentile_95"])
    violations = check_matrix.classify_forbidden(check_matrix.read_header(p))
    assert "leaky_mri_surrogate" in violations
