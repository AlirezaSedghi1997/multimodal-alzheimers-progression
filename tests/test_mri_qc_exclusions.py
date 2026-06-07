import csv

from scripts.build_master_catalog import apply_mri_qc_exclusions, read_mri_qc_exclusions


def test_mri_qc_exclusion_filters_active_sample_without_raw_ids(tmp_path):
    exclusions = tmp_path / "mri_qc_exclusions.csv"
    with exclusions.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_hash", "t1_file_hash", "reason", "decision", "active", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_hash": "s_bad",
                "t1_file_hash": "t1_bad",
                "reason": "RAW_T1_EMPTY_BINARY_QC_INVALID",
                "decision": "exclude_from_method_x_final_mri_cache_splits",
                "active": "true",
                "notes": "hashed only",
            }
        )
    rows = [
        {"sample_hash": "s_keep", "patient_hash": "p1", "t1_file_hash": "t1_keep"},
        {"sample_hash": "s_bad", "patient_hash": "p2", "t1_file_hash": "t1_bad"},
    ]

    loaded = read_mri_qc_exclusions(exclusions)
    kept, excluded = apply_mri_qc_exclusions(rows, loaded)

    assert [row["sample_hash"] for row in kept] == ["s_keep"]
    assert excluded == [{"sample_hash": "s_bad", "patient_hash": "p2", "t1_file_hash": "t1_bad", "reason": "RAW_T1_EMPTY_BINARY_QC_INVALID", "decision": "exclude_from_method_x_final_mri_cache_splits"}]
