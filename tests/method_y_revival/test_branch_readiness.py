from __future__ import annotations

from pathlib import Path

import pytest


ALL_BRANCHES = {
    "amyloid_source_regional",
    "amyloid_image",
    "tau_source_regional",
    "tau_image",
    "paired_source_regional",
    "external_transport",
}


def _passed(*names: str) -> dict[str, dict[str, str]]:
    return {
        name: {
            "artifact_id": name,
            "status": "PASS",
            "sha256": "a" * 64,
        }
        for name in names
    }


def _source_ready_evidence(branch: str) -> dict[str, dict[str, str]]:
    return _passed(
        f"{branch}_feature_contract",
        f"{branch}_endpoint_contract",
        f"{branch}_split_contract",
        "method_y_predictor_temporality_contract",
    )


def test_readiness_has_exact_branch_set_and_is_fail_closed_by_default():
    from src.method_y_revival.branch_readiness import build_branch_readiness

    report = build_branch_readiness({})

    assert set(report["branches"]) == ALL_BRANCHES
    for record in report["branches"].values():
        assert record["development_allowed"] is False
        assert record["outer_evaluation_allowed"] is False
        assert record["claim_allowed"] is False
        assert record["external_claim_allowed"] is False
        assert record["blockers"]
        assert record["evidence"] == []


def test_image_blockers_do_not_leak_into_amyloid_source_regional_development():
    from src.method_y_revival.branch_readiness import build_branch_readiness

    report = build_branch_readiness(_source_ready_evidence("amyloid_source_regional"))
    source = report["branches"]["amyloid_source_regional"]
    image = report["branches"]["amyloid_image"]

    assert source["development_allowed"] is True
    assert source["outer_evaluation_allowed"] is False
    assert "AMYLOID_IMAGE_IDENTITY_NOT_LOCKED" not in source["blockers"]
    assert "AMYLOID_IMAGE_CALIBRATION_NOT_LOCKED" not in source["blockers"]
    assert image["development_allowed"] is False
    assert "AMYLOID_IMAGE_IDENTITY_NOT_LOCKED" in image["blockers"]
    assert "AMYLOID_IMAGE_CALIBRATION_NOT_LOCKED" in image["blockers"]


def test_tau_manufacturer_limit_blocks_claim_not_source_regional_development():
    from src.method_y_revival.branch_readiness import build_branch_readiness

    report = build_branch_readiness(_source_ready_evidence("tau_source_regional"))
    tau = report["branches"]["tau_source_regional"]

    assert tau["development_allowed"] is True
    assert "TAU_SECOND_MANUFACTURER_EVENT_SUPPORT_INSUFFICIENT" not in tau["blockers"]
    assert tau["claim_allowed"] is False
    assert "TAU_SECOND_MANUFACTURER_EVENT_SUPPORT_INSUFFICIENT" in tau["claim_blockers"]


def test_malformed_evidence_fails_closed_and_is_not_published():
    from src.method_y_revival.branch_readiness import build_branch_readiness

    evidence = _source_ready_evidence("amyloid_source_regional")
    evidence["amyloid_source_regional_feature_contract"]["sha256"] = "not-a-digest"
    report = build_branch_readiness(evidence)
    source = report["branches"]["amyloid_source_regional"]

    assert source["development_allowed"] is False
    assert "INVALID_EVIDENCE:amyloid_source_regional_feature_contract" in source["blockers"]
    assert source["evidence"] == [
        item
        for item in source["evidence"]
        if item["artifact_id"] != "amyloid_source_regional_feature_contract"
    ]


def test_mismatched_evidence_artifact_id_cannot_open_a_gate():
    from src.method_y_revival.branch_readiness import build_branch_readiness

    evidence = _source_ready_evidence("amyloid_source_regional")
    evidence["amyloid_source_regional_feature_contract"]["artifact_id"] = "tau_source_feature_contract"
    record = build_branch_readiness(evidence)["branches"]["amyloid_source_regional"]

    assert record["development_allowed"] is False
    assert "INVALID_EVIDENCE:amyloid_source_regional_feature_contract" in record["blockers"]


def test_outer_and_claim_permissions_obey_implication_invariants():
    from src.method_y_revival.branch_readiness import build_branch_readiness

    evidence = _source_ready_evidence("amyloid_source_regional") | _passed(
        "amyloid_source_regional_model_contract",
        "amyloid_source_regional_outer_evaluation",
        "amyloid_source_regional_science_gate",
        "amyloid_source_regional_scanner_generalization",
    )
    report = build_branch_readiness(evidence)
    record = report["branches"]["amyloid_source_regional"]

    assert record["development_allowed"] is True
    assert record["outer_evaluation_allowed"] is True
    assert record["claim_allowed"] is True
    assert record["external_claim_allowed"] is False


def test_amyloid_internal_claim_needs_branch_scanner_sensitivity_evidence():
    from src.method_y_revival.branch_readiness import build_branch_readiness

    evidence = _source_ready_evidence("amyloid_source_regional") | _passed(
        "amyloid_source_regional_model_contract",
        "amyloid_source_regional_outer_evaluation",
        "amyloid_source_regional_science_gate",
    )
    blocked = build_branch_readiness(evidence)["branches"]["amyloid_source_regional"]
    assert blocked["claim_allowed"] is False
    assert "AMYLOID_SCANNER_SENSITIVITY_NOT_PASSED" in blocked["claim_blockers"]

    evidence.update(_passed("amyloid_source_regional_scanner_generalization"))
    assert build_branch_readiness(evidence)["branches"]["amyloid_source_regional"]["claim_allowed"] is True


def test_source_regional_collector_never_reads_image_or_external_sentinels(tmp_path, monkeypatch):
    from src.method_y_revival.branch_readiness import collect_project_evidence

    result_root = tmp_path / "results" / "method_y_revival_v2"
    (result_root / "feature_contracts").mkdir(parents=True)
    (result_root / "endpoints").mkdir()
    (result_root / "splits").mkdir()
    (result_root / "feature_contracts" / "amyloid_source_regional_feature_contract.json").write_text(
        '{"locked": true, "feature_digest": "' + "a" * 64 + '"}', encoding="utf-8"
    )
    import hashlib
    feature_path = result_root / "feature_contracts" / "amyloid_source_regional_feature_contract.json"
    (result_root / "feature_contracts" / "source_regional_validation_state.json").write_text(
        '{"status":"COMMITTED","development_allowed":true,"artifact_sha256":{"amyloid_source_regional_feature_contract.json":"'
        + hashlib.sha256(feature_path.read_bytes()).hexdigest()
        + '"}}', encoding="utf-8"
    )
    (result_root / "endpoints" / "amyloid_endpoint_contract.json").write_text(
        '{"locked": true, "contract_sha256": "' + "b" * 64 + '"}', encoding="utf-8"
    )
    (result_root / "splits" / "amyloid_nested_splits.json").write_text(
        '{"group_unit": "subject_key", "outer_fold_count": 5, "inner_fold_count": 3, "split_digest": "'
        + "c" * 64
        + '"}',
        encoding="utf-8",
    )
    (result_root / "image-cache-sentinel.json").write_text("{}", encoding="utf-8")
    (result_root / "external-outcomes-sentinel.json").write_text("{}", encoding="utf-8")

    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path, *args, **kwargs):
        if path.name in {"image-cache-sentinel.json", "external-outcomes-sentinel.json"}:
            raise AssertionError(f"unexpected source-regional access: {path}")
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    evidence = collect_project_evidence(tmp_path, branches=("amyloid_source_regional",))
    assert set(evidence) == {"amyloid_source_regional_feature_contract"}


def test_source_feature_state_hash_rejects_tampered_contract(tmp_path):
    import hashlib
    from src.method_y_revival.branch_readiness import collect_project_evidence

    root = tmp_path / "results" / "method_y_revival_v2"
    (root / "feature_contracts").mkdir(parents=True)
    feature = root / "feature_contracts" / "amyloid_source_regional_feature_contract.json"
    feature.write_text('{"locked":true,"feature_digest":"' + "a" * 64 + '"}', encoding="utf-8")
    (root / "feature_contracts" / "source_regional_validation_state.json").write_text(
        '{"status":"COMMITTED","development_allowed":true,"artifact_sha256":{"amyloid_source_regional_feature_contract.json":"'
        + hashlib.sha256(feature.read_bytes()).hexdigest() + '"}}', encoding="utf-8")
    feature.write_text('{"locked":true,"feature_digest":"' + "b" * 64 + '"}', encoding="utf-8")
    evidence = collect_project_evidence(tmp_path, branches=("amyloid_source_regional",))
    assert "amyloid_source_regional_feature_contract" not in evidence


def test_collector_binds_one_byte_snapshot_if_file_changes_between_old_reads(tmp_path, monkeypatch):
    from src.method_y_revival.branch_readiness import _locked_json_evidence

    artifact = tmp_path / "contract.json"
    original = b'{"locked": true, "contract_sha256": "' + b"a" * 64 + b'"}'
    artifact.write_bytes(original)
    original_read_bytes = Path.read_bytes
    reads = []

    def mutate_after_snapshot(path):
        reads.append(path)
        payload = original_read_bytes(path)
        path.write_text("[]", encoding="utf-8")
        return payload

    monkeypatch.setattr(Path, "read_bytes", mutate_after_snapshot)
    evidence = _locked_json_evidence(
        artifact,
        "test_contract",
        predicate=lambda payload: payload.get("locked") is True,
    )

    assert len(reads) == 1
    assert evidence == {
        "artifact_id": "test_contract",
        "status": "PASS",
        "sha256": __import__("hashlib").sha256(original).hexdigest(),
    }


def test_split_collector_requires_locked_digest_for_source_and_image_branches(tmp_path):
    from src.method_y_revival.branch_readiness import collect_project_evidence

    root = tmp_path / "results" / "method_y_revival_v2"
    (root / "splits").mkdir(parents=True)
    for stem in ("amyloid", "tau"):
        (root / "splits" / f"{stem}_nested_splits.json").write_text(
            '{"group_unit": "subject_key", "outer_fold_count": 5, "inner_fold_count": 3}',
            encoding="utf-8",
        )
    evidence = collect_project_evidence(
        tmp_path, branches=("amyloid_source_regional", "amyloid_image", "tau_image")
    )
    assert not {key for key in evidence if key.endswith("_split_contract")}

    (root / "splits" / "amyloid_nested_splits.json").write_text(
        '{"group_unit": "subject_key", "outer_fold_count": 5, "inner_fold_count": 3, "split_digest": "bad"}',
        encoding="utf-8",
    )
    assert "amyloid_source_regional_split_contract" not in collect_project_evidence(
        tmp_path, branches=("amyloid_source_regional",)
    )


def test_incomplete_committed_outer_folds_cannot_open_claim(tmp_path):
    from src.method_y_revival.branch_readiness import (
        build_branch_readiness,
        collect_project_evidence,
    )

    root = tmp_path / "results" / "method_y_revival_v2"
    (root / "feature_contracts").mkdir(parents=True)
    (root / "endpoints").mkdir()
    (root / "splits").mkdir()
    (root / "locked").mkdir()
    (root / "outer_evaluation" / "amyloid_source_regional").mkdir(parents=True)
    (root / "reports").mkdir()
    def locked(digest: str) -> str:
        return '{"locked": true, "contract_sha256": "' + digest * 64 + '"}'

    (root / "feature_contracts" / "amyloid_source_regional_feature_contract.json").write_text(locked("a"), encoding="utf-8")
    (root / "endpoints" / "amyloid_endpoint_contract.json").write_text(locked("b"), encoding="utf-8")
    (root / "locked" / "amyloid_source_regional_model_contract.json").write_text(locked("c"), encoding="utf-8")
    (root / "splits" / "amyloid_nested_splits.json").write_text(
        '{"group_unit":"subject_key","outer_fold_count":5,"inner_fold_count":3,"split_digest":"' + "d" * 64 + '"}', encoding="utf-8"
    )
    (root / "outer_evaluation" / "amyloid_source_regional" / "committed.json").write_text(
        '{"status":"COMMITTED","exactly_once":true,"outer_fold_count":5,"outer_folds_completed":4,"split_digest":"'
        + "d" * 64
        + '","model_contract_sha256":"'
        + "c" * 64
        + '"}',
        encoding="utf-8",
    )
    (root / "reports" / "amyloid_source_regional_science_gate.json").write_text(
        '{"science_gate_status":"PASS"}', encoding="utf-8"
    )
    (root / "reports" / "scanner_generalization_amyloid.json").write_text(
        '{"science_gate_status":"PASS"}', encoding="utf-8"
    )

    report = build_branch_readiness(
        collect_project_evidence(tmp_path, branches=("amyloid_source_regional",))
    )
    assert report["branches"]["amyloid_source_regional"]["claim_allowed"] is False
    assert "MISSING_EVIDENCE:amyloid_source_regional_outer_evaluation" in report["branches"]["amyloid_source_regional"]["claim_blockers"]


def test_collector_rejects_malformed_json_schema_without_crashing(tmp_path):
    from src.method_y_revival.branch_readiness import collect_project_evidence

    result_root = tmp_path / "results" / "method_y_revival_v2"
    (result_root / "feature_contracts").mkdir(parents=True)
    (result_root / "endpoints").mkdir()
    (result_root / "splits").mkdir()
    (result_root / "feature_contracts" / "amyloid_source_regional_feature_contract.json").write_text(
        '{"locked": "false", "feature_digest": "' + "a" * 64 + '"}', encoding="utf-8"
    )
    (result_root / "endpoints" / "amyloid_endpoint_contract.json").write_text("[]", encoding="utf-8")
    (result_root / "splits" / "amyloid_nested_splits.json").write_text("[]", encoding="utf-8")

    evidence = collect_project_evidence(tmp_path, branches=("amyloid_source_regional",))
    assert evidence == {}


def test_external_claim_needs_mapping_model_and_committed_external_evaluation():
    from src.method_y_revival.branch_readiness import build_branch_readiness

    report = build_branch_readiness(
        _passed(
            "external_dua",
            "external_data",
            "external_mapping_contract",
            "external_locked_model_contract",
            "external_science_gate",
        )
    )
    external = report["branches"]["external_transport"]

    assert external["development_allowed"] is True
    assert external["outer_evaluation_allowed"] is True
    assert external["external_claim_allowed"] is False
    assert "EXTERNAL_EVALUATION_NOT_COMMITTED" in external["external_claim_blockers"]


def test_external_claim_implies_claim_permission_for_every_branch():
    from src.method_y_revival.branch_readiness import build_branch_readiness

    evidence = _passed(
        "external_dua",
        "external_data",
        "external_mapping_contract",
        "external_locked_model_contract",
        "external_evaluation",
        "external_science_gate",
    )
    external = build_branch_readiness(evidence)["branches"]["external_transport"]

    assert external["external_claim_allowed"] is True
    assert external["claim_allowed"] is True


def test_readiness_report_has_path_free_deterministic_evidence_snapshot():
    from src.method_y_revival.branch_readiness import build_branch_readiness

    report = build_branch_readiness(
        _source_ready_evidence("amyloid_source_regional"),
        source_role="read_only_project_aggregate_contracts",
    )

    snapshot = report["evidence_snapshot"]
    assert snapshot["source_role"] == "read_only_project_aggregate_contracts"
    assert len(snapshot["sha256"]) == 64
    assert "H:\\" not in __import__("json").dumps(report)


def test_external_readiness_roles_share_one_cached_snapshot(tmp_path, monkeypatch):
    from src.method_y_revival.branch_readiness import collect_project_evidence

    readiness = tmp_path / "results" / "method_y_revival_v2" / "external_validation" / "readiness.json"
    readiness.parent.mkdir(parents=True)
    approved_only = b'{"dua_status":"APPROVED","data_status":"MISSING"}'
    data_only = b'{"dua_status":"MISSING","data_status":"AVAILABLE"}'
    readiness.write_bytes(approved_only)
    original_read_bytes = Path.read_bytes
    reads = []

    def replace_after_first_snapshot(path):
        reads.append(path)
        payload = original_read_bytes(path)
        path.write_bytes(data_only)
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_first_snapshot)
    evidence = collect_project_evidence(tmp_path, branches=("external_transport",))

    assert reads == [readiness]
    assert set(evidence) == {"external_dua"}


def test_path_like_source_role_is_rejected_before_public_report_construction():
    from src.method_y_revival.branch_readiness import BranchReadinessError, build_branch_readiness

    with pytest.raises(BranchReadinessError, match="INVALID_EVIDENCE_SOURCE_ROLE"):
        build_branch_readiness({}, source_role=r"H:\A4 Learn\private_subject")


def test_public_writer_rejects_forged_path_like_source_role(tmp_path):
    from src.method_y_revival.branch_readiness import (
        BranchReadinessError,
        build_branch_readiness,
        write_branch_readiness,
    )

    report = build_branch_readiness({})
    report["evidence_snapshot"]["source_role"] = r"H:\A4 Learn\private_subject"
    with pytest.raises(BranchReadinessError, match="INVALID_EVIDENCE_SOURCE_ROLE"):
        write_branch_readiness(report, tmp_path)
    assert not list(tmp_path.iterdir())


def test_shared_pytest_bootstrap_imports_worktree_method_y_modules():
    import src.method_y_revival.branch_readiness as readiness
    import src.method_y_revival.tau_scan_provenance_audit as tau_audit

    root = Path(__file__).resolve().parents[2]
    assert Path(readiness.__file__).resolve().is_relative_to(root)
    assert Path(tau_audit.__file__).resolve().is_relative_to(root)
