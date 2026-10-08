"""Fail-closed, aggregate-only scientific add-on day-1 audit tests."""
from __future__ import annotations

import copy
import json

from scripts.mx_science_extension_v1.audit_and_lock import partition_audit
from scripts.mx_science_extension_v1.plan_arms import make_arm_commands


def _fixture():
    rows = []
    for i in range(6):
        rows.append({
            "patient_hash": f"patient-{i}", "sample_hash": f"sample-{i}",
            "substudy": "A4" if i % 2 else "LEARN",
            "ab_label": "1" if i % 2 else "0", "pet_day_t0": str(i + 10),
            "t1_day_t0": str(i + 8), "abs_pet_t1_gap_days": "2",
            "t1_file_hash": f"mri-file-{i}",
        })
    tests = [[rows[i]] for i in range(5)]
    trains = [[rows[j] for j in range(5) if j != i] for i in range(5)]
    held = [rows[5]]
    index = [{"patient_hash": row["patient_hash"], "sample_hash": row["sample_hash"],
              "label": row["ab_label"], "is_synthetic": "0"} for row in rows]
    return index, tests, trains, held


def test_patient_boundaries_and_no_subject_export():
    report = partition_audit(*_fixture())
    assert report["status"] == "READY_INTERNAL_ONLY"
    assert report["cohort_label_perfectly_confounding"] is True
    assert report["holdout_labels_used_for_this_audit"] is False
    assert len(report["folds"]) == 5
    assert all(len(row["inner"]) == 3 for row in report["folds"])
    assert "patient-" not in json.dumps(report)
    assert "sample-" not in json.dumps(report)


def test_holdout_patient_alias_blocks():
    index, tests, trains, held = copy.deepcopy(_fixture())
    held[0]["patient_hash"] = tests[0][0]["patient_hash"]
    report = partition_audit(index, tests, trains, held)
    assert report["status"] == "BLOCKED_LEAKAGE_OR_PROVENANCE"
    assert "OUTER_HOLDOUT_PATIENT_OVERLAP" in report["blockers"]


def test_train_fold_or_mri_provenance_drift_blocks():
    index, tests, trains, held = copy.deepcopy(_fixture())
    trains[2].pop()
    tests[1][0]["t1_file_hash"] = ""
    report = partition_audit(index, tests, trains, held)
    assert "TRAIN_SPLIT_MEMBERSHIP_INVALID:2" in report["blockers"]
    assert "SPLIT_REQUIRED_MRI_PET_PROVENANCE_FIELD_MISSING" in report["blockers"]
    assert "MRI_FILE_HASH_MISSING" in report["blockers"]


def test_temporal_inconsistency_blocks():
    index, tests, trains, held = copy.deepcopy(_fixture())
    tests[3][0]["abs_pet_t1_gap_days"] = "500"
    report = partition_audit(index, tests, trains, held)
    assert "PET_MRI_GAP_INCONSISTENT" in report["blockers"]


def test_real_arm_plan_keeps_negative_control_out_of_final_mode():
    from scripts.mx_science_extension_v1.audit_and_lock import ROOT

    arms = make_arm_commands(ROOT)
    assert len(arms) == 6
    for name, command in arms.items():
        assert "--hpo_selection_only" in command
        assert "--require_exact_resume_contract" in command
        assert "--no_resume" not in command
        phase = command[command.index("--phase") + 1]
        if name == "random_label_baseline_seed13":
            assert phase == "dev"
            assert "--sanity_random_labels" in command
        else:
            assert phase == "final"
            assert "--sanity_random_labels" not in command
