from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest


def test_public_execution_api_has_no_authority_override_parameters() -> None:
    from src.research.brainiac_execution_gate import (
        assess_brainiac_execution_gate,
        assert_brainiac_execution_allowed,
        brainiac_action_lease,
    )

    assert tuple(inspect.signature(assess_brainiac_execution_gate).parameters) == (
        "action",
    )
    assert tuple(inspect.signature(assert_brainiac_execution_allowed).parameters) == (
        "action",
    )
    assert tuple(inspect.signature(brainiac_action_lease).parameters) == ("action",)


def test_legacy_cache128_embedding_entrypoint_is_permanently_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.research import extract_brainiac_embeddings as legacy

    monkeypatch.setattr(legacy, "CACHE_PATH", tmp_path / "would-be-read.h5")
    monkeypatch.setattr(legacy, "DERIVED_DIR", tmp_path / "must-not-exist")
    with pytest.raises(
        RuntimeError, match="BRAINIAC_LEGACY_CACHE128_EXTRACTOR_PERMANENTLY_DISABLED"
    ):
        legacy.extract_all_embeddings(limit=1)
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize(
    "role",
    [
        "preprocessing_v2",
        "brainiac_embeddings",
        "split_registry",
        "resnet34_representation_contract",
        "resnet34_inner_features",
        "brainiac_extractor_contract",
        "caft_trainer_contract",
        "outer_head_calibration_lock",
        "promotion_evidence",
        "external_authorization",
    ],
)
def test_future_authorities_cannot_pass_with_self_attested_json(
    role: str,
) -> None:
    from src.research.brainiac_readiness import validate_future_authority

    fake = {
        "schema_version": 999,
        "status": "PASS",
        "ready": True,
        "rows": 1669,
        "expected_rows": 1669,
        "shape": [1669, 768],
        "finite": True,
        "real_only": True,
        "label_read": False,
        "locked": True,
        "dev_subjects": 1336,
        "sealed_holdout_subjects": 333,
        "overlap_count": 0,
        "backbone": "resnet34_3d",
        "local_embedding_dim": 128,
        "strict_state_load_required": True,
        "outer_train_only_preparation": True,
        "encoders_frozen": True,
        "lora_unlocked": False,
        "arm_count": 6,
        "outer_test_touched": False,
        "head_locked": True,
        "calibration_locked": True,
        "sanity_pass": True,
        "all_metrics_finite": True,
        "dua": True,
        "lineage": True,
        "no_refit": True,
        "exactly_once": True,
    }
    report = validate_future_authority(role, fake)
    assert report["ready"] is False
    assert report["status"] == "BLOCKED_UNTIL_ROLE_SPECIFIC_VALIDATOR"


def test_complete_status_strings_without_fold_evidence_are_not_terminal(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_battery_terminal import verify_battery_terminal

    effective = (
        "resnet10_3d",
        "resnet18_3d",
        "resnet34_3d",
        "densenet121_3d",
        "mednext_3d_s",
    )
    excluded = ("resnet50_3d", "efficientnet3d_b0")
    manifest = tmp_path / "battery_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "backbone": name,
                        "status": "COMPLETED",
                        "output_dir": f"models/{name}",
                    }
                    for name in effective
                ]
                + [
                    {
                        "backbone": name,
                        "status": "PROTOCOL_EXCLUDED_NOT_STARTED",
                        "output_dir": f"models/{name}",
                    }
                    for name in excluded
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = verify_battery_terminal(
        manifest_path=manifest,
        battery_root=tmp_path,
        project_root=tmp_path,
        hash_checkpoints=False,
    )
    assert report["terminal"] is False
    assert report["blockers"]
    assert any(
        code.startswith(("FIVE_MODEL_AMENDMENT_INVALID", "FINALIZED_FOLD_EVIDENCE_MISSING"))
        for code in report["blockers"]
    )


def test_in_root_reparse_alias_is_rejected(tmp_path: Path) -> None:
    from src.research.brainiac_readiness import bind_file

    real = tmp_path / "real"
    real.mkdir()
    (real / "authority.json").write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    with pytest.raises(RuntimeError, match="REPARSE_AUTHORITY_FORBIDDEN"):
        bind_file(alias / "authority.json", project_root=tmp_path)


@pytest.mark.parametrize(
    "status",
    [
        "UNKNOWN_PRETRAINING_OVERLAP",
        "NOT_MEASURABLE",
        "UNKNOWN",
        "",
        "claimed_disjoint",
    ],
)
def test_only_verified_disjoint_enables_confirmatory_overlap_claim(status: str) -> None:
    from src.research.brainiac_readiness import derive_readiness_payload

    payload = derive_readiness_payload(
        battery={"models": []},
        evidence={"pretraining_overlap_status": status},
        writer_state={"inspection_status": "PASS", "active_writers": [], "recorded_owners": []},
        input_bindings={"x": {"path": "x", "bytes": 1, "sha256": "a" * 64}},
        battery_terminal_report={"terminal": False, "identity_valid": False, "blockers": ["TEST"]},
    )
    assert payload["claim_limits"]["confirmatory_pretraining_independence"] is False


def test_verified_disjoint_can_enable_only_the_overlap_claim_flag() -> None:
    from src.research.brainiac_readiness import derive_readiness_payload

    payload = derive_readiness_payload(
        battery={"models": []},
        evidence={"pretraining_overlap_status": "VERIFIED_DISJOINT"},
        writer_state={"inspection_status": "PASS", "active_writers": [], "recorded_owners": []},
        input_bindings={"x": {"path": "x", "bytes": 1, "sha256": "a" * 64}},
        battery_terminal_report={"terminal": False, "identity_valid": False, "blockers": ["TEST"]},
    )
    assert payload["claim_limits"]["confirmatory_pretraining_independence"] is True
    assert payload["claim_limits"]["external_generalization"] is False
    assert payload["lora_unlocked"] is False


@pytest.mark.parametrize(
    "forbidden",
    [
        "age",
        "sex",
        "education_years",
        "apoe4_carrier",
        "mmse_base",
        "tabular",
        "predictors",
        "subject_rows",
    ],
)
def test_readiness_artifacts_reject_predictor_or_subject_values(
    tmp_path: Path, forbidden: str
) -> None:
    from src.research.brainiac_readiness import write_readiness_artifact

    with pytest.raises(RuntimeError, match="READINESS_PRIVACY_FIELD_FORBIDDEN"):
        write_readiness_artifact(
            tmp_path / f"{forbidden}.json",
            {"schema_version": 4, forbidden: [1]},
        )


def test_production_loader_cannot_select_an_arbitrary_checkpoint() -> None:
    from src.research.brainiac_model_loader import (
        BrainIACViTEncoder,
        load_brainiac_encoder,
    )

    assert tuple(inspect.signature(load_brainiac_encoder).parameters) == ("device",)
    assert tuple(inspect.signature(BrainIACViTEncoder.__init__).parameters) == (
        "self",
        "device",
    )


def test_identifier_url_variants_normalize_to_the_same_canonical_token() -> None:
    from src.research.brainiac_readiness import _source_identifiers

    assert _source_identifiers({"doi": "10.1000/ABC"}) == _source_identifiers(
        {"url": "https://doi.org/10.1000/abc"}
    )
    assert _source_identifiers({"pmid": "12345"}) == _source_identifiers(
        {"url": "https://pubmed.ncbi.nlm.nih.gov/12345/"}
    )
    assert _source_identifiers({"pmc": "PMC987"}) == _source_identifiers(
        {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC987/"}
    )


def test_cross_identifier_row_requires_bound_equivalence_evidence() -> None:
    from src.research.brainiac_readiness import _validate_cross_identifier_consistency

    with pytest.raises(RuntimeError, match="R0_CROSS_IDENTIFIER_EVIDENCE_MISSING"):
        _validate_cross_identifier_consistency(
            {"id": "x", "doi": "10.1000/a", "pmid": "123"},
            project_root=Path.cwd(),
        )


def test_contradictory_identifier_url_is_not_ignored() -> None:
    from src.research.brainiac_readiness import _validate_cross_identifier_consistency

    with pytest.raises(RuntimeError, match="R0_CROSS_IDENTIFIER_EVIDENCE_MISSING"):
        _validate_cross_identifier_consistency(
            {
                "id": "x",
                "doi": "10.1000/correct",
                "url": "https://doi.org/10.1000/different",
            },
            project_root=Path.cwd(),
        )


def test_crosswalk_binding_cannot_attest_identifiers_absent_from_evidence(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_readiness import (
        _validate_cross_identifier_consistency,
        bind_file,
    )

    evidence_path = tmp_path / "crosswalk.md"
    evidence_path.write_text(
        "This unrelated file contains neither the row identity nor its identifiers.\n",
        encoding="utf-8",
    )
    binding = bind_file(evidence_path, project_root=tmp_path)
    source = {
        "id": "s-test",
        "doi": "10.1000/correct",
        "pmid": "12345",
        "identifier_crosswalk_evidence": {
            "source_path": "crosswalk.md",
            "source_bytes": binding["bytes"],
            "source_sha256": binding["sha256"],
            "canonical_identifiers": ["doi:10.1000/correct", "pmid:12345"],
        },
    }
    with pytest.raises(RuntimeError, match="R0_CROSS_IDENTIFIER_EVIDENCE_MISMATCH"):
        _validate_cross_identifier_consistency(source, project_root=tmp_path)


def test_legacy_schema1_preprocessing_entrypoint_is_disabled_before_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.research import preprocess_brainiac_raw_t1 as legacy

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("legacy preprocessing attempted I/O before fail-closed stop")

    monkeypatch.setattr(legacy, "_assert_no_cnn_writer", forbidden)
    monkeypatch.setattr(legacy, "_validate_template", forbidden)
    monkeypatch.setattr(legacy, "_atomic_write_json", forbidden)
    monkeypatch.setattr(legacy, "_acquire_lock", forbidden)
    with pytest.raises(
        RuntimeError,
        match="BRAINIAC_LEGACY_SCHEMA1_PREPROCESSOR_PERMANENTLY_DISABLED",
    ):
        legacy.preflight()
    with pytest.raises(
        RuntimeError,
        match="BRAINIAC_LEGACY_SCHEMA1_PREPROCESSOR_PERMANENTLY_DISABLED",
    ):
        legacy.execute(mode="smoke", limit=1, device="cpu")


@pytest.mark.parametrize("command_line", [None, ""])
def test_unknown_python_command_line_blocks_writer_inspection(
    command_line: object,
) -> None:
    from src.research.brainiac_readiness import inspect_writer_state

    report = inspect_writer_state(
        {"models": []},
        process_provider=lambda: [
            {
                "ProcessId": 987654,
                "Name": "python.exe",
                "CommandLine": command_line,
                "CreationDate": "20260822000000.000000+000",
            }
        ],
    )
    assert report["inspection_status"] == "UNKNOWN"
    assert report["active_writers"] == []


def test_readiness_transition_ledger_advances_without_a_source_hash_edit(
    tmp_path: Path,
) -> None:
    from src.research.brainiac_readiness_ledger import (
        append_readiness_transition,
        validate_readiness_ledger,
    )

    ledger = tmp_path / "ledger"
    first = append_readiness_transition(
        ledger,
        {"schema_version": 4, "stage": "B1", "authorized_actions": ["PREPARE_RAW_T1"]},
        project_root=tmp_path,
    )
    second = append_readiness_transition(
        ledger,
        {"schema_version": 4, "stage": "B2", "authorized_actions": ["PREPARE_RAW_T1"]},
        project_root=tmp_path,
    )
    assert first["event_sha256"] != second["event_sha256"]
    report = validate_readiness_ledger(ledger, project_root=tmp_path)
    assert report["event_count"] == 2
    assert report["head_sha256"] == second["event_sha256"]
