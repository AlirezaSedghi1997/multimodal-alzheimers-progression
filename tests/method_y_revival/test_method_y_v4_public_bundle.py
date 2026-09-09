from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_method_y_image_factorial_v4_public_bundle import (
    BundleError,
    build_bundle,
)
from src.method_y_revival.privacy import scan_public_bundle


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_source(root: Path) -> None:
    _write(
        root / "reports/factorial_benchmark_summary.sanitized.json",
        {
            "schema_version": "4.0",
            "status": "COMPLETED",
            "positive_claim_allowed": False,
            "external_claim_allowed": False,
            "pair_reports": [],
        },
    )
    _write(
        root / "post_eval/post_eval_report.json",
        {
            "status": "PASS",
            "artifact_status": "COMPLETED",
            "science_gate_status": "BLOCKED",
            "claim_status": "INTERNAL_ONLY",
            "positive_claim_allowed": False,
            "external_claim_allowed": False,
            "ceiling": "POST_HOC_INTERNAL_EXPLORATORY",
            "compiled": True,
            "claim_blockers": ["EXTERNAL_PENDING"],
            "contrasts": [
                {
                    "branch": "amyloid",
                    "reference_arm": "A0",
                    "challenger_arm": "A1",
                    "delta_ibs_mean": 0.01,
                    "delta_ibs_95ci": [-0.1, 0.1],
                }
            ],
        },
    )
    _write(
        root / "attribution/xai_science_gate_v4.json",
        {
            "status": "FAIL",
            "science_gate_status": "FAIL",
            "claim_status": "NO_CLAIM",
            "positive_claim_allowed": False,
            "external_claim_allowed": False,
            "blockers": ["AGREEMENT_NOT_PASS"],
            "details": {"agreement": {"status": "RED_FLAG_INVESTIGATE"}},
        },
    )
    for branch in ("amyloid", "tau"):
        for arm in range(5):
            for fold in range(5):
                _write(
                    root / f"reports/{branch}_A{arm}_fold_{fold}_public.json",
                    {
                        "schema_version": "4.0",
                        "claim_status": "POST_HOC_INTERNAL_EXPLORATORY",
                        "aggregate": {
                            "integrated_brier_score": 0.2,
                            "test_events": 2,
                            "evaluation_source": "LOCKED_MODEL_WEIGHTS",
                        },
                    },
                )
    _write(
        root / "analysis_pairs/A0_vs_A1/reports/amyloid_factorial_acceptance.json",
        {
            "branch": "amyloid",
            "reference_arm": "A0",
            "challenger_arm": "A1",
            "positive_claim_allowed": False,
            "external_claim_allowed": False,
            "blockers": ["SCANNER_GATE_PENDING"],
        },
    )


def test_builder_creates_whitelisted_private_safe_bundle(tmp_path: Path) -> None:
    source = tmp_path / "run"
    output = tmp_path / "release"
    _minimal_source(source)

    result = build_bundle(source, output)

    assert result["status"] == "PASS"
    assert len(list((output / "folds").glob("*.json"))) == 50
    assert json.loads((output / "privacy_scan.json").read_text())["status"] == "PASS"
    assert scan_public_bundle(output) == []
    assert not (output / "private").exists()


def test_builder_rejects_nonempty_destination_without_replace(tmp_path: Path) -> None:
    source = tmp_path / "run"
    output = tmp_path / "release"
    _minimal_source(source)
    output.mkdir()
    (output / "old.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(BundleError, match="non-empty"):
        build_bundle(source, output)


def test_builder_does_not_allow_row_level_identifier_in_source(tmp_path: Path) -> None:
    source = tmp_path / "run"
    output = tmp_path / "release"
    _minimal_source(source)
    path = source / "reports/amyloid_A0_fold_0_public.json"
    payload = json.loads(path.read_text())
    payload["subject_id"] = "person-1"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BundleError, match="private key"):
        build_bundle(source, output)
