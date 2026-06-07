from pathlib import Path
import csv
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.pet_manifest import PetCandidate, CatalogRow, tier_for_gap, link_candidates_to_catalog, canonical_tracer


def policy():
    return yaml.safe_load((ROOT / "configs/model/method_y_spatial_preprocess_policy.yaml").read_text(encoding="utf-8"))


def test_tier_for_gap_includes_gold30_and_primary90():
    p = policy()
    assert tier_for_gap(10, p) == "gold_standard_30"
    assert tier_for_gap(80, p) == "primary_strict"
    assert tier_for_gap(110, p) == "secondary_window"
    assert tier_for_gap(300, p) == "relaxed_exploratory"
    assert tier_for_gap(500, p) == "excluded_from_training"


def test_tracer_aliases_are_canonicalized():
    assert canonical_tracer("florbetapir") == "fbp"
    assert canonical_tracer("Florbetaben") == "fbb"
    assert canonical_tracer("AV1451") == "ftp"


def test_linking_selects_one_pet_per_patient_and_excludes_duplicates():
    p = policy()
    catalog = [
        CatalogRow(sample_hash="s1", patient_hash="p1", substudy="A4", ab_label="1", abs_pet_t1_gap_days=50, t1_path="t1a.nii.gz"),
        CatalogRow(sample_hash="s2", patient_hash="p1", substudy="A4", ab_label="1", abs_pet_t1_gap_days=70, t1_path="t1b.nii.gz"),
        CatalogRow(sample_hash="s3", patient_hash="p2", substudy="LEARN", ab_label="0", abs_pet_t1_gap_days=95, t1_path="t1c.nii.gz"),
    ]
    candidates = [
        PetCandidate(raw_path="/private/B10001_FBP.nii.gz", tracer="fbp", substudy="A4", bid="B10001", patient_hash="p1", file_hash="a"),
        PetCandidate(raw_path="/private/B20001_FBP.nii.gz", tracer="fbp", substudy="LEARN", bid="B20001", patient_hash="p2", file_hash="b"),
    ]
    work, excluded, summary = link_candidates_to_catalog(catalog, candidates, p, requested_tier="primary_strict", one_per_patient=True)
    assert len(work) == 1
    assert work[0].catalog.patient_hash == "p1"
    assert any(e["reason"] == "DUPLICATE_PATIENT_NON_SELECTED" for e in excluded)
    assert any(e["reason"] == "GAP_NOT_IN_REQUESTED_TIER" for e in excluded)
