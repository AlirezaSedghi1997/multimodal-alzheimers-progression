from pathlib import Path
from src.utils.pet_manifest import parse_pet_filename, write_private_manifest, read_private_manifest, PetCandidate, link_candidates_to_catalog, CatalogRow


def test_filename_parser_fallback():
    c = parse_pet_filename("A4_B12345_FBP_pet.nii.gz", tracer="fbp")
    assert c is not None
    assert c.substudy == "A4"
    assert c.bid == "B12345"


def test_private_manifest_round_trip(tmp_path):
    p = tmp_path / "manifest.csv"
    rows = [PetCandidate(raw_path="/private/A4_B12345_FBP.nii.gz", tracer="fbp", substudy="A4", bid="B12345", file_hash="abc")]
    write_private_manifest(p, rows)
    loaded = read_private_manifest(p, tracer="fbp")
    assert len(loaded) == 1
    assert loaded[0].bid == "B12345"


def test_one_pet_per_patient_selection():
    candidates = [
        PetCandidate(raw_path="/private/A4_B12345_1.nii.gz", tracer="fbp", substudy="A4", bid="B12345", file_hash="aaa"),
        PetCandidate(raw_path="/private/A4_B12345_2.nii.gz", tracer="fbp", substudy="A4", bid="B12345", file_hash="bbb"),
    ]
    catalog = [
        CatalogRow(sample_hash="s1", patient_hash="p1", substudy="A4", ab_label="1", abs_pet_t1_gap_days=80, t1_path="t1a.nii.gz"),
        CatalogRow(sample_hash="s2", patient_hash="p1", substudy="A4", ab_label="1", abs_pet_t1_gap_days=70, t1_path="t1b.nii.gz"),
    ]
    policy = {"temporal_alignment": {"eligibility_tiers": {"primary_strict": {"max_abs_gap_days": 90}}},
              "sample_selection": {"exclude_substudies_from_mainline": []}}
    work, excluded, summary = link_candidates_to_catalog(catalog, candidates, policy, requested_tier="primary_strict", one_per_patient=True)
    assert len(work) == 1
    assert any(row["reason"] == "DUPLICATE_PATIENT_NON_SELECTED" for row in excluded)
    assert summary["one_pet_per_patient"] is True


def test_private_manifest_candidate_t1_path_can_satisfy_t1_requirement():
    policy = {"temporal_alignment": {"eligibility_tiers": {"primary_strict": {"max_abs_gap_days": 90}}},
              "sample_selection": {"exclude_substudies_from_mainline": []}}
    catalog = [
        CatalogRow(sample_hash="s1", patient_hash="p1", substudy="A4", ab_label="1", abs_pet_t1_gap_days=10),
    ]
    candidates = [
        PetCandidate(
            raw_path="/private/A4_B12345_FBP.nii.gz",
            tracer="fbp",
            substudy="A4",
            bid="B12345",
            patient_hash="p1",
            file_hash="abc",
            t1_path="/private/t1_for_s1.nii.gz",
        )
    ]

    work, excluded, summary = link_candidates_to_catalog(catalog, candidates, policy, requested_tier="primary_strict")

    assert len(work) == 1
    assert not any(row["reason"] == "MISSING_T1" for row in excluded)
    assert summary["linked_work_items"] == 1


def test_missing_catalog_and_candidate_t1_still_blocks_pet_linking():
    policy = {"temporal_alignment": {"eligibility_tiers": {"primary_strict": {"max_abs_gap_days": 90}}},
              "sample_selection": {"exclude_substudies_from_mainline": []}}
    catalog = [
        CatalogRow(sample_hash="s1", patient_hash="p1", substudy="A4", ab_label="1", abs_pet_t1_gap_days=10),
    ]
    candidates = [
        PetCandidate(
            raw_path="/private/A4_B12345_FBP.nii.gz",
            tracer="fbp",
            substudy="A4",
            bid="B12345",
            patient_hash="p1",
            file_hash="abc",
        )
    ]

    work, excluded, summary = link_candidates_to_catalog(catalog, candidates, policy, requested_tier="primary_strict")

    assert work == []
    assert any(row["reason"] == "MISSING_T1" for row in excluded)
    assert summary["linked_work_items"] == 0
