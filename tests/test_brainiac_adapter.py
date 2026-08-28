"""Task 6 — adapter, geometry gate, strict load, audit separation, frozen probes."""

from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.combined_mri_cache import METHOD_X_PHENOTYPE_FEATURES  # noqa: E402
from src.research.brainiac_adapter import (  # noqa: E402
    AUDIT_ONLY_FIELDS,
    BRAINIAC_RESOLUTION,
    GEOMETRY_SHIFT_STATUS,
    GEOMETRY_UNKNOWN_STATUS,
    PROBE_C_GRID,
    PROJECT_RESOLUTION,
    adapt_volume,
    adapter_provenance,
    assert_no_audit_fields_in_predictors,
    assert_predictive_feature_order,
    audit_metadata_status,
    evaluate_geometry,
    fit_frozen_probe,
    geometry_manifest,
    mask_edge_perturbation,
    nonzero_normalize,
    sensitivity_pair,
    validate_strict_load,
    validate_vendor_equivalent_strict_load,
)


@pytest.fixture()
def brain_like() -> np.ndarray:
    """A 128-cubed volume with a zero background, like the locked cache."""

    rng = np.random.default_rng(13)
    volume = np.zeros((128, 128, 128), dtype=np.float32)
    grid = np.ogrid[:128, :128, :128]
    radius = sum((axis - 64) ** 2 for axis in grid)
    mask = radius < 55**2
    volume[mask] = rng.normal(1.0, 0.2, size=int(mask.sum())).astype(np.float32)
    return volume


def _probe_contract(n: int, train: list[int], validation: list[int], *, tabular: bool):
    roles = ["unused"] * n
    for index in train:
        roles[index] = "inner_train"
    for index in validation:
        roles[index] = "inner_validation"
    digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    subject_hashes = [digest(f"probe-subject-{i}") for i in range(n)]
    real_flags = [True] * n
    parent_hashes = [None] * n
    registry_body = {
        "schema_version": 1,
        "locked": True,
        "split_hash": digest("probe-split"),
        "group_hash": digest("probe-groups"),
        "subjects": [
            {
                "subject_hash": subject_hashes[index],
                "role": roles[index],
                "real": real_flags[index],
                "parent_hash": parent_hashes[index],
            }
            for index in range(n)
        ],
    }
    registry = {
        **registry_body,
        "integrity_sha256": hashlib.sha256(
            json.dumps(registry_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return {
        "subject_hashes": subject_hashes,
        "split_roles": roles,
        "real_flags": real_flags,
        "parent_hashes": parent_hashes,
        "feature_names": METHOD_X_PHENOTYPE_FEATURES if tabular else (),
        "split_provenance": {
            "selection_role": "inner_validation",
            "outer_test_touched": False,
            "split_hash": digest("probe-split"),
            "group_hash": digest("probe-groups"),
            "embedding_artifact_sha256": digest("probe-embeddings"),
            "split_registry_sha256": registry["integrity_sha256"],
        },
        "locked_split_registry": registry,
        "expected_split_registry_sha256": registry["integrity_sha256"],
    }


# --- Geometry gate ------------------------------------------------------------


def test_geometry_gate_reports_the_documented_domain_shift() -> None:
    report = evaluate_geometry()
    assert report.status == GEOMETRY_UNKNOWN_STATUS
    assert "VENDOR_TEMPLATE_AFFINE_UNVERIFIED" in report.findings
    # 128 * 1.5 mm = 192 mm of physical extent.
    assert report.source_fov_mm == pytest.approx(192.0)
    # A direct resize preserves extent, so spacing lands at 2.0 mm, not 1.0 mm.
    assert report.effective_voxel_mm == pytest.approx(2.0)
    assert report.voxel_ratio == pytest.approx(2.0)
    assert "EFFECTIVE_SPACING_DIFFERS_FROM_DECLARED" in report.findings


def test_unverified_96_cubed_shape_is_not_reported_as_a_physical_fov() -> None:
    """The vendor applies ``Resized(96³)``; array shape alone is not 96 mm."""

    report = evaluate_geometry()
    assert report.target_fov_mm is None
    assert "TARGET_PHYSICAL_FOV_NOT_MEASURABLE" in report.findings
    assert "FIELD_OF_VIEW_MISMATCH" not in report.findings


def test_geometry_gate_flags_the_registration_family_mismatch() -> None:
    report = evaluate_geometry()
    assert report.registration_match is False
    assert "REGISTRATION_FAMILY_MISMATCH" in report.findings


def test_geometry_gate_passes_only_when_everything_reconciles() -> None:
    report = evaluate_geometry(
        source_resolution=96,
        source_voxel_mm=2.0,
        target_resolution=96,
        declared_target_voxel_mm=2.0,
        source_registration="rigid_mni",
        target_registration="rigid_mni",
        target_geometry_verified=True,
    )
    assert report.status == "PASS"
    assert report.findings == ()


def test_geometry_manifest_leaves_upstream_interpretation_unresolved() -> None:
    manifest = geometry_manifest()
    assert manifest["target"]["fov_interpretation"] == "UNRESOLVED_PENDING_VENDOR"
    assert manifest["status"] == GEOMETRY_UNKNOWN_STATUS
    assert manifest["target_geometry_verified"] is False
    assert manifest["target"]["declared_fov_mm"] is None
    assert manifest["target"]["array_shape"] == [96, 96, 96]
    assert manifest["target"]["physical_fov_status"] == "NOT_MEASURABLE"
    assert manifest["branch_scope"] == "BRAINIAC_ONLY_MAIN_BATTERY_UNAFFECTED"
    assert manifest["direct_resize"]["effective_voxel_mm"] == pytest.approx(2.0)


# --- Adapter ------------------------------------------------------------------


def test_adapter_produces_the_target_geometry(brain_like: np.ndarray) -> None:
    out = adapt_volume(brain_like, purpose="sensitivity_only")
    assert out.shape == (BRAINIAC_RESOLUTION,) * 3
    assert out.dtype == np.float32


def test_adapter_is_deterministic(brain_like: np.ndarray) -> None:
    assert np.array_equal(
        adapt_volume(brain_like, purpose="sensitivity_only"),
        adapt_volume(brain_like, purpose="sensitivity_only"),
    )


def test_adapter_rejects_a_non_cubic_volume() -> None:
    with pytest.raises(RuntimeError, match="ADAPTER_EXPECTS_CUBIC_VOLUME"):
        adapt_volume(
            np.zeros((128, 128, 64), dtype=np.float32), purpose="sensitivity_only"
        )


def test_nonzero_normalization_uses_foreground_statistics_only() -> None:
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    volume[2:6, 2:6, 2:6] = 5.0
    out = nonzero_normalize(volume)
    # Background stays exactly zero; a constant foreground has zero variance.
    assert float(np.abs(out[out == 0]).sum()) == 0.0
    assert np.count_nonzero(out) == 0


def test_nonzero_normalization_standardizes_the_foreground() -> None:
    rng = np.random.default_rng(7)
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[4:12, 4:12, 4:12] = rng.normal(3.0, 2.0, size=(8, 8, 8))
    out = nonzero_normalize(volume)
    foreground = out[out != 0]
    assert float(foreground.mean()) == pytest.approx(0.0, abs=1e-4)
    assert float(foreground.std()) == pytest.approx(1.0, abs=1e-3)


def test_all_zero_volume_does_not_divide_by_zero() -> None:
    out = nonzero_normalize(np.zeros((8, 8, 8), dtype=np.float32))
    assert np.array_equal(out, np.zeros((8, 8, 8), dtype=np.float32))


def test_adapter_provenance_stores_no_identifier_or_path(
    brain_like: np.ndarray,
) -> None:
    record = adapter_provenance(
        adapt_volume(brain_like, purpose="sensitivity_only"), sample_hash="a" * 64
    )
    assert record["label_read"] is False
    assert record["raw_identifier_stored"] is False
    assert record["raw_path_stored"] is False
    serialized = str(record).lower()
    for fragment in ("h:/", "h:\\", "c:/", "c:\\", ".nii", "subject"):
        assert fragment not in serialized


def test_sensitivity_pair_compares_both_routes_label_blind(
    brain_like: np.ndarray,
) -> None:
    result = sensitivity_pair(brain_like)
    assert result["label_read"] is False
    assert result["intermediate_resolution"] == int(round(PROJECT_RESOLUTION * 1.5))
    assert result["mean_absolute_difference"] >= 0.0
    assert result["direct_resize_sha256"] != result["staged_resample_sha256"]


# --- Mask-edge shortcut probes -------------------------------------------------


@pytest.mark.parametrize("mode", ["erode", "dilate", "foreground_only"])
def test_mask_perturbations_preserve_shape(brain_like: np.ndarray, mode: str) -> None:
    assert mask_edge_perturbation(brain_like, mode=mode).shape == brain_like.shape


def test_erosion_shrinks_and_dilation_grows_the_foreground(
    brain_like: np.ndarray,
) -> None:
    baseline = int(np.count_nonzero(brain_like))
    eroded = int(np.count_nonzero(mask_edge_perturbation(brain_like, mode="erode")))
    dilated = int(np.count_nonzero(mask_edge_perturbation(brain_like, mode="dilate")))
    assert eroded < baseline < dilated


def test_unknown_perturbation_is_rejected(brain_like: np.ndarray) -> None:
    with pytest.raises(RuntimeError, match="UNKNOWN_MASK_PERTURBATION"):
        mask_edge_perturbation(brain_like, mode="blur")


# --- Strict load --------------------------------------------------------------


def test_strict_load_requires_full_coverage() -> None:
    result = validate_strict_load(["a", "b"], ["a", "b"])
    assert result["coverage"] == 1.0
    assert result["strict"] is True


def test_strict_load_rejects_missing_keys() -> None:
    with pytest.raises(RuntimeError, match="STRICT_LOAD_MISSING_KEYS"):
        validate_strict_load(["a"], ["a", "b"])


def test_strict_load_rejects_unexpected_keys() -> None:
    with pytest.raises(RuntimeError, match="STRICT_LOAD_UNEXPECTED_KEYS"):
        validate_strict_load(["a", "b", "c"], ["a", "b"])


def test_strict_load_rejects_a_shape_mismatch() -> None:
    with pytest.raises(RuntimeError, match="STRICT_LOAD_SHAPE_MISMATCH"):
        validate_strict_load(
            ["w"], ["w"], shapes={"w": (768, 512)}, expected_shapes={"w": (768, 768)}
        )


def test_vendor_equivalent_load_does_not_round_naive_monai15_mismatch_to_pass() -> None:
    checkpoint = ["patch_embedding.weight", "blocks.0.attn.qkv.weight"]
    battery_vit = [
        *checkpoint,
        "blocks.0.cross_attn.to_q.weight",
        "blocks.0.norm_cross_attn.weight",
    ]
    with pytest.raises(RuntimeError, match="STRICT_LOAD_MISSING_KEYS"):
        validate_strict_load(checkpoint, battery_vit)
    result = validate_vendor_equivalent_strict_load(
        checkpoint,
        battery_vit,
        shapes={key: (1,) for key in checkpoint},
        expected_shapes={
            **{key: (1,) for key in checkpoint},
            "blocks.0.cross_attn.to_q.weight": (1,),
            "blocks.0.norm_cross_attn.weight": (1,),
        },
    )
    assert result["status"] == "PASS_VENDOR_EQUIVALENT_STRICT"
    assert result["naive_strict_status"] == "FAIL"
    assert result["coverage"] == 1.0
    assert result["matched_keys"] == 2
    assert result["battery_monai_unused_cross_attn_keys"] == 2


def test_vendor_equivalent_load_rejects_a_checkpoint_that_carries_cross_attn() -> None:
    with pytest.raises(RuntimeError, match="CHECKPOINT_CONTAINS_UNUSED_CROSS_ATTN"):
        validate_vendor_equivalent_strict_load(
            ["w", "blocks.0.cross_attn.to_q.weight"],
            ["w", "blocks.0.cross_attn.to_q.weight"],
        )


# --- Audit separation ---------------------------------------------------------


def test_canonical_feature_order_is_enforced() -> None:
    assert_predictive_feature_order(METHOD_X_PHENOTYPE_FEATURES)
    with pytest.raises(RuntimeError, match="FEATURE_ORDER_VIOLATION"):
        assert_predictive_feature_order(
            ("sex", "age", "education_years", "apoe4_carrier", "mmse_base")
        )


@pytest.mark.parametrize("field", sorted(AUDIT_ONLY_FIELDS))
def test_audit_fields_can_never_enter_predictors(field: str) -> None:
    with pytest.raises(RuntimeError, match="AUDIT_FIELD_IN_PREDICTORS"):
        assert_no_audit_fields_in_predictors([*METHOD_X_PHENOTYPE_FEATURES, field])


def test_allowed_features_pass_the_audit_check() -> None:
    assert_no_audit_fields_in_predictors(METHOD_X_PHENOTYPE_FEATURES)


def test_missing_audit_metadata_is_not_measurable_not_a_pass() -> None:
    assert audit_metadata_status(None)["status"] == "NOT_MEASURABLE"
    assert audit_metadata_status({})["status"] == "NOT_MEASURABLE"
    assert audit_metadata_status({"age": 70})["status"] == "NOT_MEASURABLE"
    assert audit_metadata_status({"scanner_id": "x"})["status"] == "MEASURABLE"


# --- Frozen probe -------------------------------------------------------------


def test_frozen_probe_selects_c_on_inner_validation_only() -> None:
    rng = np.random.default_rng(29)
    labels = np.array([0] * 40 + [1] * 40)
    embeddings = rng.normal(0.0, 1.0, size=(80, 12))
    embeddings[labels == 1, 0] += 1.5
    train = list(range(0, 30)) + list(range(40, 70))
    validation = list(range(30, 40)) + list(range(70, 80))
    result = fit_frozen_probe(
        embeddings,
        labels,
        inner_train=train,
        inner_validation=validation,
        **_probe_contract(80, train, validation, tabular=False),
    )
    assert result["C"] in PROBE_C_GRID
    assert result["outer_test_touched"] is False
    assert result["selected_on"] == "inner_validation_only"


def test_frozen_probe_accepts_the_tabular_branch() -> None:
    rng = np.random.default_rng(47)
    labels = np.array([0] * 30 + [1] * 30)
    embeddings = rng.normal(size=(60, 8))
    tabular = rng.normal(size=(60, len(METHOD_X_PHENOTYPE_FEATURES)))
    train = list(range(0, 22)) + list(range(30, 52))
    validation = list(range(22, 30)) + list(range(52, 60))
    result = fit_frozen_probe(
        embeddings,
        labels,
        inner_train=train,
        inner_validation=validation,
        tabular=tabular,
        **_probe_contract(60, train, validation, tabular=True),
    )
    assert 0.0 <= result["inner_validation_auc"] <= 1.0
