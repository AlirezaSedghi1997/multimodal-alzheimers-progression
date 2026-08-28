"""Additional fail-closed profile-schema checks retained from Plan 016 RED work."""

import pytest


def test_profile_rejects_missing_fresh_run():
    from scripts.research import prepare_brainiac_raw_t1_v2 as runner

    with pytest.raises(RuntimeError, match="FRESH_RUN_EVIDENCE_MISSING"):
        runner._validate_execution_leg_summary(
            {},
            expected_role="sequential",
            expected_contract_sha256="c" * 64,
            expected_samples=set(),
        )


def test_profile_integrity_is_checked_before_live_recomputation(monkeypatch):
    from scripts.research import prepare_brainiac_raw_t1_v2 as runner

    monkeypatch.setattr(
        runner,
        "recompute_execution_leg",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("live roots touched before profile integrity validation")
        ),
    )
    with pytest.raises(RuntimeError, match="PROFILE_INTEGRITY_INVALID"):
        runner.validate_execution_profile(
            {
                "schema_version": 2,
                "artifact": "BRAINIAC_PREPROCESSING_EXECUTION_PROFILE_V2",
                "legs": {"sequential": {}, "parallel": [{}, {}]},
                "integrity_sha256": "0" * 64,
            },
            expected_contract_sha256="c" * 64,
        )
