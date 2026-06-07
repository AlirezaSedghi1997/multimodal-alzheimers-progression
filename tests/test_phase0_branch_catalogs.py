from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


def test_branch_catalog_manifest_exists_and_locked():
    manifest_path = Path("data/processed/catalogs/branch_catalog_manifest.json")
    assert manifest_path.exists(), "Run scripts/build_branch_catalogs.py first"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "BRANCH_CATALOGS_LOCKED"
    assert set(manifest["catalogs"]) == {
        "method_x_strict",
        "method_x_cog_sf",
        "mri_roi_wmh_reads",
        "bio_cog_method_z",
        "clinical_endpoint",
    }


def test_branch_catalogs_are_hash_level_without_raw_ids():
    catalog_dir = Path("data/processed/catalogs")
    forbidden = {"bid", "subject_id", "raw_path", "filepath", "file_path"}
    for name in [
        "method_x_strict",
        "method_x_cog_sf",
        "mri_roi_wmh_reads",
        "bio_cog_method_z",
        "clinical_endpoint",
    ]:
        path = catalog_dir / f"{name}_catalog.csv"
        assert path.exists(), f"Missing catalog: {path}"
        with path.open("r", encoding="utf-8", newline="") as f:
            header = [c.lower() for c in next(csv.reader(f))]
        assert "patient_hash" in header
        assert "sample_hash" in header
        assert not forbidden.intersection(header)


def test_cohort_label_coupling_audit_before_after_sf():
    path = Path("results/governance/cohort_label_coupling_audit.json")
    assert path.exists(), "Run scripts/run_cohort_label_coupling_audit.py first"
    report = json.loads(path.read_text(encoding="utf-8"))
    before = report["before_sf_inclusion"]["cohort_label_cramers_v"]
    after = report["after_sf_inclusion"]["cohort_label_cramers_v"]
    assert before >= 0.99
    assert after < before
    assert report["sf_only_subset"]["rows"] > 0


def test_leakage_review_report_passes():
    path = Path("results/governance/leakage_review_report.json")
    assert path.exists(), "Run scripts/write_leakage_review_report.py first"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] in {"PASS", "PARTIAL"}
    assert report["pet_as_input_for_method_x"] is False


def test_sf_governance_policy_forbids_external_wording():
    text = Path("governance/sf_internal_decoupling_policy.md").read_text(encoding="utf-8").lower()
    assert "internal decoupling" in text
    assert "never use" in text
    assert "external validation" in text
