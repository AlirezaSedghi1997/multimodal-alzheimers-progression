"""Task 5 — BrainIAC R0 evidence registry and H-side provenance scaffold.

Two things are being enforced here, and they matter more than the file format:

* the upstream boundary cannot be *guessed*. Weight filenames, hashes, sizes and
  module paths stay ``UNRESOLVED_PENDING_VENDOR`` until the pinned commit is
  actually vendored. The playbook is explicit about this ("Never copy an import
  path from a secondary document without checking the live repo").
* claims cannot outrun evidence. Pretraining overlap stays ``UNKNOWN`` without
  subject-level proof, preprints never merge into the peer-reviewed tier, and the
  novelty ceiling stays ``NOVELTY_CANDIDATE`` until the review is complete.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.brainiac_provenance import (  # noqa: E402
    FORBIDDEN_CLAIM_PHRASES,
    NOVELTY_CEILING,
    R0_MIN_PEER_REVIEWED,
    R0_TARGET_SOURCE_RANGE,
    UNRESOLVED,
    UPSTREAM_PINNED_COMMIT,
    assert_claim_language,
    build_upstream_manifest,
    evidence_registry_status,
    validate_evidence_registry,
    validate_upstream_manifest,
)

REGISTRY_ROOT = PROJECT_ROOT / "research" / "brainiac_caft_20260808"
REGISTRY_PATH = REGISTRY_ROOT / "evidence_registry_v2.json"


# --- Upstream boundary --------------------------------------------------------


def test_upstream_manifest_pins_a_commit_not_a_floating_branch() -> None:
    manifest = build_upstream_manifest()
    assert manifest["commit"] == UPSTREAM_PINNED_COMMIT
    assert len(manifest["commit"]) == 40
    assert manifest["ref_type"] == "PINNED_COMMIT"
    assert "main" not in json.dumps(manifest.get("source_url", ""))


def test_upstream_manifest_records_the_published_input_contract() -> None:
    manifest = build_upstream_manifest()
    assert manifest["input_shape"] == [96, 96, 96]
    assert manifest["embedding_dim"] == 768
    assert manifest["backbone"] == "simclr_vit_b_3d"
    assert manifest["strict_load_required"] is True
    assert manifest["expected_preprocessing"] == [
        "n4_bias_field_correction",
        "resample_1mm_isotropic",
        "rigid_mni_registration",
        "brain_extraction",
        "standardize_96_cubed",
    ]


def test_weight_identity_is_unresolved_until_the_checkpoint_file_exists() -> None:
    """Encoder paths may come from the vendored tree; weight hashes only after the file exists."""

    manifest = build_upstream_manifest()
    if manifest["weight_filename"] == UNRESOLVED:
        for field in ("weight_filename", "weight_sha256", "weight_size_bytes"):
            assert manifest[field] == UNRESOLVED, field
    else:
        assert manifest["weight_filename"] == "BrainIAC.ckpt"
        assert manifest["weight_sha256"] == (
            "f22bdbcae26823a9d9e8aee883c6f24386ba4617339c12269848b6666cc62693"
        )
        assert int(manifest["weight_size_bytes"]) == 362119240
    if manifest["vendored"]:
        assert manifest["encoder_class_name"] == "ViTBackboneNet"
        assert manifest["loader_symbol"] == "load_brainiac"
        assert manifest["encoder_module_path"] == "src/model.py"
    else:
        for field in ("encoder_module_path", "encoder_class_name", "loader_symbol"):
            assert manifest[field] == UNRESOLVED, field


def test_manifest_validation_rejects_a_fabricated_weight_hash() -> None:
    manifest = build_upstream_manifest()
    manifest["weight_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="UNVERIFIED_WEIGHT_PROVENANCE"):
        validate_upstream_manifest(manifest)


def test_manifest_validation_rejects_a_guessed_module_path() -> None:
    manifest = build_upstream_manifest()
    manifest["encoder_module_path"] = "src.models.brainiac"
    with pytest.raises(RuntimeError, match="UNVERIFIED_MODULE_PATH"):
        validate_upstream_manifest(manifest)


def test_manifest_validation_accepts_the_honest_placeholder_or_vendored_state() -> None:
    report = validate_upstream_manifest(build_upstream_manifest())
    if report["vendored"]:
        assert report["status"] == "VENDORED"
    else:
        assert report["status"] == "PENDING_VENDOR"


def test_license_boundary_is_non_commercial_academic() -> None:
    manifest = build_upstream_manifest()
    assert manifest["license_boundary"] == "NON_COMMERCIAL_ACADEMIC_RESEARCH_ONLY"
    assert manifest["commercial_use_permitted"] is False


def test_pretraining_overlap_is_unknown_without_subject_level_evidence() -> None:
    """BrainIAC pretrained on ADNI/OASIS; absence of overlap cannot be inferred."""

    manifest = build_upstream_manifest()
    assert manifest["pretraining_overlap_status"] == "UNKNOWN_PRETRAINING_OVERLAP"
    assert manifest["overlap_evidence"] == UNRESOLVED


def test_overlap_cannot_be_cleared_without_a_subject_list() -> None:
    manifest = build_upstream_manifest()
    manifest["pretraining_overlap_status"] = "NO_OVERLAP"
    with pytest.raises(RuntimeError, match="OVERLAP_CLAIM_WITHOUT_EVIDENCE"):
        validate_upstream_manifest(manifest)


# --- Evidence registry --------------------------------------------------------


def _source(**overrides: object) -> dict[str, object]:
    source = {
        "id": "src_001",
        "title": "A generalizable foundation model for analysis of human brain MRI",
        "year": 2026,
        "tier": "peer_reviewed",
        "url": "https://www.nature.com/articles/s41593-026-02202-6",
        "endpoint": "multiple_downstream_tasks",
        "cohort": "multi_cohort",
        "modality": "T1_MRI",
        "preprocessing": "rigid_mni_skull_stripped_96",
        "split": "reported",
        "external_validation": "partial",
        "confound_audit": "not_reported",
        "uncertainty": "not_reported",
        "code_weights": "public",
        "citation_status": "IDENTIFIER_VERIFIED",
    }
    source.update(overrides)
    return source


def test_preprints_stay_in_a_separate_tier() -> None:
    registry = {
        "sources": [
            _source(id="a", tier="peer_reviewed"),
            _source(id="b", tier="preprint"),
        ]
    }
    status = evidence_registry_status(registry)
    assert status["peer_reviewed_count"] == 1
    assert status["preprint_count"] == 1
    assert status["total_count"] == 2


def test_unknown_tier_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="UNKNOWN_EVIDENCE_TIER"):
        validate_evidence_registry({"sources": [_source(tier="blog_post")]})


def test_inconclusive_citation_status_is_preserved_not_upgraded() -> None:
    """citeprobe returned INCONCLUSIVE; a partial tool result cannot mean verified."""

    registry = {
        "sources": [
            _source(id="a", citation_status="INCONCLUSIVE"),
            _source(id="b", citation_status="IDENTIFIER_VERIFIED"),
        ]
    }
    status = evidence_registry_status(registry)
    assert status["inconclusive_count"] == 1
    assert status["fully_verified"] is False


def test_registry_below_the_target_range_is_reported_incomplete() -> None:
    """The program asks for 24-30 sources; fewer must not read as complete."""

    registry = {"sources": [_source(id=f"s{i}") for i in range(19)]}
    status = evidence_registry_status(registry)
    assert status["total_count"] == 19
    assert status["meets_target_range"] is False
    assert status["status"] == "INCOMPLETE_PENDING_R0_COMPLETION"


def test_registry_within_range_and_peer_reviewed_floor_is_complete() -> None:
    """COMPLETE requires the count, the peer-reviewed floor, and full-text depth."""

    sources = [
        _source(
            id=f"p{i}",
            tier="peer_reviewed",
            screening_stage="FULL_TEXT_REVIEWED",
            review_evidence={
                "source_path": f"research/notes/p{i}.md",
                "source_sha256": "a" * 64,
                "source_bytes": 100,
                "reviewed_utc": "2026-08-21T00:00:00+00:00",
                "review_basis": "FULL_TEXT_READ_AND_STRUCTURED_EXTRACTION",
            },
        )
        for i in range(12)
    ]
    sources += [
        _source(
            id=f"q{i}",
            tier="preprint",
            screening_stage="FULL_TEXT_REVIEWED",
            review_evidence={
                "source_path": f"research/notes/q{i}.md",
                "source_sha256": "b" * 64,
                "source_bytes": 100,
                "reviewed_utc": "2026-08-21T00:00:00+00:00",
                "review_basis": "FULL_TEXT_READ_AND_STRUCTURED_EXTRACTION",
            },
        )
        for i in range(12)
    ]
    status = evidence_registry_status({"sources": sources})
    assert status["total_count"] == 24
    assert status["meets_target_range"] is True
    assert status["peer_reviewed_count"] >= R0_MIN_PEER_REVIEWED
    assert status["full_text_reviewed_count"] == 24
    assert status["status"] == "COMPLETE"


def test_screened_but_unread_sources_do_not_complete_the_review() -> None:
    """Finding a paper in a search is screening, not reading it.

    Counting search hits as reviewed evidence is the single easiest way to
    inflate a scoping review, so the depth blocker is independent of the count.
    """

    sources = [_source(id=f"p{i}", tier="peer_reviewed") for i in range(12)]
    sources += [_source(id=f"q{i}", tier="preprint") for i in range(12)]
    status = evidence_registry_status({"sources": sources})
    assert status["total_count"] == 24
    assert status["meets_target_range"] is True
    assert status["full_text_reviewed_count"] == 0
    assert status["screened_not_read_count"] == 24
    assert status["status"] == "INCOMPLETE_PENDING_R0_COMPLETION"
    assert "FULL_TEXT_REVIEW_BELOW_TARGET" in status["blockers"]


def test_missing_screening_stage_assumes_the_weaker_claim() -> None:
    """An unstamped source must not be silently credited as fully reviewed."""

    source = _source(id="unstamped")
    source.pop("screening_stage", None)
    status = evidence_registry_status({"sources": [source]})
    assert status["full_text_reviewed_count"] == 0


def test_full_text_stamp_without_bound_review_evidence_is_not_counted() -> None:
    """A mutable stage label is not proof that an agent read the source."""

    sources = [
        _source(id=f"p{i}", screening_stage="FULL_TEXT_REVIEWED")
        for i in range(24)
    ]
    status = evidence_registry_status({"schema_version": 2, "sources": sources})
    assert status["full_text_reviewed_count"] == 0
    assert status["status"] == "INCOMPLETE_PENDING_R0_COMPLETION"
    assert "FULL_TEXT_REVIEW_EVIDENCE_INVALID" in status["blockers"]


def test_full_text_available_pending_review_is_not_rounded_up() -> None:
    sources = [
        _source(
            id=f"p{i}",
            screening_stage="FULL_TEXT_AVAILABLE_PENDING_REVIEW",
            full_text_evidence={
                "source_path": f"research/notes/p{i}.md",
                "source_sha256": "c" * 64,
                "source_bytes": 100,
            },
        )
        for i in range(24)
    ]
    status = evidence_registry_status({"schema_version": 2, "sources": sources})
    assert status["full_text_available_pending_review_count"] == 24
    assert status["full_text_reviewed_count"] == 0
    assert status["status"] == "INCOMPLETE_PENDING_R0_COMPLETION"


def test_peer_reviewed_floor_is_enforced_independently_of_the_total() -> None:
    sources = [_source(id=f"p{i}", tier="peer_reviewed") for i in range(5)]
    sources += [_source(id=f"q{i}", tier="preprint") for i in range(20)]
    status = evidence_registry_status({"sources": sources})
    assert status["total_count"] == 25
    assert status["meets_target_range"] is True
    assert status["status"] == "INCOMPLETE_PENDING_R0_COMPLETION"
    assert "PEER_REVIEWED_FLOOR" in status["blockers"]


def test_every_source_must_carry_the_extraction_fields() -> None:
    incomplete = _source()
    del incomplete["confound_audit"]
    with pytest.raises(RuntimeError, match="MISSING_EXTRACTION_FIELD"):
        validate_evidence_registry({"sources": [incomplete]})


def test_duplicate_source_ids_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="DUPLICATE_SOURCE_ID"):
        validate_evidence_registry({"sources": [_source(id="x"), _source(id="x")]})


def test_sources_outside_the_review_window_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="SOURCE_OUTSIDE_REVIEW_WINDOW"):
        validate_evidence_registry({"sources": [_source(year=2019)]})


# --- Claim language -----------------------------------------------------------


def test_novelty_ceiling_is_candidate_until_the_review_completes() -> None:
    assert NOVELTY_CEILING == "NOVELTY_CANDIDATE"


@pytest.mark.parametrize("phrase", sorted(FORBIDDEN_CLAIM_PHRASES))
def test_forbidden_claim_phrases_are_rejected(phrase: str) -> None:
    with pytest.raises(RuntimeError, match="FORBIDDEN_CLAIM_LANGUAGE"):
        assert_claim_language(f"The model shows {phrase} in this cohort.")


def test_attribution_language_is_accepted() -> None:
    assert_claim_language(
        "Regions with higher attribution in the model decision are reported."
    )


def test_claim_language_check_is_case_insensitive() -> None:
    with pytest.raises(RuntimeError, match="FORBIDDEN_CLAIM_LANGUAGE"):
        assert_claim_language("This enables ALZHEIMER DIAGNOSIS from MRI.")


# --- Materialized artifacts ---------------------------------------------------


def test_registry_root_lives_on_the_authoritative_volume() -> None:
    assert REGISTRY_ROOT.is_dir(), REGISTRY_ROOT
    assert str(REGISTRY_ROOT).upper().startswith("H:")


@pytest.mark.parametrize(
    "name",
    [
        "upstream_boundary_manifest.json",
        "evidence_registry.json",
        "review_protocol.md",
        "novelty_matrix.json",
        "claim_language_policy.md",
        "environment_spec.json",
        "search_log.md",
    ],
)
def test_required_artifact_exists(name: str) -> None:
    assert (REGISTRY_ROOT / name).is_file(), name


def test_corrected_r0_registry_exists() -> None:
    assert REGISTRY_PATH.is_file()


def test_superseded_v1_writers_are_rejected_once_v2_exists() -> None:
    """A historical generator must not silently replace the corrected registry."""

    from src.research.brainiac_provenance import assert_current_r0_write_target

    with pytest.raises(RuntimeError, match="BRAINIAC_R0_V1_WRITER_SUPERSEDED"):
        assert_current_r0_write_target(
            REGISTRY_ROOT / "evidence_registry.json",
            current_registry_path=REGISTRY_PATH,
        )
    with pytest.raises(RuntimeError, match="BRAINIAC_R0_V1_WRITER_SUPERSEDED"):
        assert_current_r0_write_target(REGISTRY_ROOT / "evidence_registry.json")
    assert_current_r0_write_target(
        REGISTRY_PATH,
        current_registry_path=REGISTRY_PATH,
    )


def test_materialized_upstream_manifest_validates() -> None:
    manifest = json.loads(
        (REGISTRY_ROOT / "upstream_boundary_manifest.json").read_text(encoding="utf-8")
    )
    report = validate_upstream_manifest(manifest)
    assert report["status"] in {"PENDING_VENDOR", "VENDORED"}
    assert report["vendored"] is bool(manifest.get("vendored"))


def test_materialized_evidence_registry_validates_and_is_honest() -> None:
    registry = json.loads(
        REGISTRY_PATH.read_text(encoding="utf-8")
    )
    validate_evidence_registry(registry)
    status = evidence_registry_status(registry)
    # Window count stays in 24-30 after the s22/s23 merge. The 2024
    # publication stays background, not padding.
    assert status["total_count"] >= R0_TARGET_SOURCE_RANGE[0]
    assert status["total_count"] <= R0_TARGET_SOURCE_RANGE[1]
    assert len(registry["background_sources_outside_window"]) >= 1
    assert status["peer_reviewed_count"] >= R0_MIN_PEER_REVIEWED
    assert status["status"] == registry["status"]


def test_materialized_registry_refuses_to_call_retrieval_a_completed_review() -> None:
    """Only six sources have an auditable read event; retrieval is not review."""

    registry = json.loads(
        REGISTRY_PATH.read_text(encoding="utf-8")
    )
    status = evidence_registry_status(registry)
    assert status["meets_target_range"] is True
    assert status["full_text_reviewed_count"] == 6
    assert status["full_text_available_pending_review_count"] == 19
    assert status["status"] == "INCOMPLETE_PENDING_R0_COMPLETION"
    assert "FULL_TEXT_REVIEW_BELOW_TARGET" in status["blockers"]
    assert registry["citation_corroboration"] == "INCONCLUSIVE"
    assert status["fully_verified"] is False
    assert status["claim_ceiling"] == NOVELTY_CEILING
    assert registry["status"] == status["status"]


def test_background_sources_are_excluded_with_a_stated_reason() -> None:
    registry = json.loads(
        (REGISTRY_ROOT / "evidence_registry.json").read_text(encoding="utf-8")
    )
    for source in registry["background_sources_outside_window"]:
        assert source["excluded_reason"]
        assert int(source["year"]) < 2025


def test_vendor_snapshot_is_on_h_and_does_not_bundle_weights() -> None:
    """Code is vendored on H; pretrained weights live outside the GitHub snapshot."""

    vendor = PROJECT_ROOT / "vendor" / "BrainIAC" / UPSTREAM_PINNED_COMMIT
    assert vendor.is_dir(), vendor
    assert (vendor / "LICENSE").is_file()
    assert (vendor / "src" / "model.py").is_file()
    assert not (Path(r"C:\Users\USER\.venvs\brainiac")).exists()
    for pattern in ("*.pth", "*.ckpt", "*.pt", "*.safetensors"):
        assert not list(vendor.rglob(pattern)), pattern
        assert not list(REGISTRY_ROOT.rglob(pattern)), pattern


def test_environment_spec_targets_h_and_records_creation_honestly() -> None:
    spec = json.loads((REGISTRY_ROOT / "environment_spec.json").read_text(encoding="utf-8"))
    assert spec["target_path"].replace("\\", "/").startswith("H:/")
    venv = PROJECT_ROOT / ".venvs" / "brainiac"
    assert spec["created"] is venv.exists()
    if spec["created"]:
        assert venv.is_dir()


def test_novelty_matrix_holds_the_five_component_gap() -> None:
    matrix = json.loads((REGISTRY_ROOT / "novelty_matrix.json").read_text(encoding="utf-8"))
    assert matrix["claim_ceiling"] == NOVELTY_CEILING
    assert len(matrix["required_components"]) == 5
    assert matrix["prior_art_exact_match_found"] is False


def test_claim_language_policy_is_itself_compliant() -> None:
    """The policy quotes forbidden phrases, so it must mark them as forbidden."""

    text = (REGISTRY_ROOT / "claim_language_policy.md").read_text(encoding="utf-8")
    assert "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE" in text
    assert NOVELTY_CEILING in text
