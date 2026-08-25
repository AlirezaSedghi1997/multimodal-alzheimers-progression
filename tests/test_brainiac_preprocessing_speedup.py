from __future__ import annotations

import json

import pytest


def test_pin_itk_threads_respects_env(monkeypatch):
    from scripts.research.prepare_brainiac_raw_t1_v2 import _pin_itk_threads

    calls: list[int] = []

    class FakeSitk:
        class ProcessObject:
            @staticmethod
            def SetGlobalDefaultNumberOfThreads(n: int) -> None:
                calls.append(int(n))

    import sys
    import types

    fake = types.ModuleType("SimpleITK")
    fake.ProcessObject = FakeSitk.ProcessObject
    monkeypatch.setitem(sys.modules, "SimpleITK", fake)
    monkeypatch.setenv("BRAINIAC_V2_ITK_THREADS", "1")
    assert _pin_itk_threads() == 1
    monkeypatch.setenv("BRAINIAC_V2_ITK_THREADS", "0")
    assert _pin_itk_threads() == 1  # clamped to >=1
    assert calls == [1, 1]


def test_scratch_registered_volume_is_uncompressed_nii(tmp_path):
    from scripts.research.prepare_brainiac_raw_t1_v2 import _scratch_paths

    reg, brain = _scratch_paths(tmp_path, "abc")
    assert reg.suffix == ".nii" and not reg.name.endswith(".gz")
    assert brain.name.endswith(".nii.gz")


def test_equivalence_comparator_flags_scientific_field_drift():
    from scripts.research.prepare_brainiac_raw_t1_v2 import _equivalence_mismatches

    base = {
        "sample_hash": "a" * 64,
        "output_sha256": "x",
        "transform_sha256": "y",
        "foreground_fraction": 0.2,
        "rigid_determinant": 1.0,
        "orthogonality_error": 0.0,
        "rotation_deg": 3.0,
        "translation_norm_mm": 10.0,
        "normalized_nonzero_std": 1.0,
    }
    assert _equivalence_mismatches([base], [dict(base)]) == []
    drifted = dict(base)
    drifted["rotation_deg"] = 4.2
    out = _equivalence_mismatches([base], [drifted])
    assert out == [f"{'a' * 10}:rotation_deg"]


def test_parallel_runner_returns_identical_receipts(monkeypatch):
    from src.research import brainiac_preprocessing_v2 as v2

    seen: list[str] = []

    def fake_tx(item, root, *, pipeline, bindings):
        seen.append(item["sample_hash"])
        return {"sample_hash": item["sample_hash"], "ok": True}

    monkeypatch.setattr(v2, "run_sample_transaction", fake_tx)
    items = [{"sample_hash": f"{i:064x}"} for i in range(6)]
    par = v2.run_orchestrated(
        items, root=None, pipeline_factory=lambda: None,
        bindings={}, cpu_workers=2, gpu_workers=1,
    )
    par_seen = list(seen)
    seen.clear()
    seq = v2.run_batch(items, root=None, pipeline_factory=lambda: None, bindings={})
    assert sorted(par_seen) == sorted(i["sample_hash"] for i in items)
    assert [r["sample_hash"] for r in par] == [r["sample_hash"] for r in seq["receipts"]]


def test_policy_keeps_sequential_only_until_equivalence_pass():
    from src.research.brainiac_preprocessing_v2 import POLICY_PATH

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    orch = policy["orchestration"]
    assert orch["sequential_only"] is True
    assert orch["parallel_activation"].startswith("REQUIRES_12_SAMPLE_EQUIVALENCE_PASS")
    assert int(orch.get("itk_threads_default", 0)) == 1
