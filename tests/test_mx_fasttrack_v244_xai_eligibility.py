"""Post-hoc atlas-only selection must not alter committed XAI folds or controls."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.mx_fasttrack_v2_2.xai_crossfit import choose_size_matched_controls
from scripts.mx_fasttrack_v2_4_4.control_eligibility import (
    eligible_atlas_ids, eligible_ranked_regions, run_with_eligibility,
)
from scripts.mx_fasttrack_v2_4_4 import entry_v244, supervise_v244


def test_atlas_only_filter_excludes_unmatchable_top_without_widening_control_ratio(monkeypatch):
    from src.evaluation import method_x_xai_forward as forward

    atlas = np.asarray([1] * 100 + [2] * 10 + [3] * 10 + [4] * 10 + [5] * 10)
    assert eligible_atlas_ids(atlas) == (2, 3, 4, 5)
    with pytest.raises(ValueError, match="THREE_SIZE_MATCHED"):
        choose_size_matched_controls(atlas, 1)
    ranked = [{"atlas_id": i, "region": str(i)} for i in (1, 2, 3, 4, 5)]
    original = forward._aggregate_regions
    monkeypatch.setattr(forward, "_aggregate_regions", lambda *a: ranked)
    stub = forward._aggregate_regions

    def fake_run(*, atlas_data, **kwargs):
        choices = []
        for _ in range(2):
            rows = forward._aggregate_regions(None, atlas_data, None)
            top = rows[0]["atlas_id"]
            controls = choose_size_matched_controls(atlas_data, top)
            assert len(controls) == 3
            choices.append({"top_region_id": top})
        return {"crossfit_directions": choices}

    result = run_with_eligibility(original_run=fake_run, atlas_data=atlas,
                                  amendment_sha256="a" * 64)
    assert [row["top_region_id"] for row in result["crossfit_directions"]] == [2, 2]
    assert [row["unrestricted_top_id"] for row in result["selection_trace"]] == [1, 1]
    assert forward._aggregate_regions is stub
    monkeypatch.setattr(forward, "_aggregate_regions", original)


def test_filter_preserves_prior_eligible_top_and_never_accepts_empty():
    rows = [{"atlas_id": i} for i in (2, 1, 3)]
    assert eligible_ranked_regions(rows, frozenset({2, 3}))[0]["atlas_id"] == 2
    with pytest.raises(ValueError, match="NO_ASSESSABLE"):
        eligible_ranked_regions(rows, frozenset())
    with pytest.raises(ValueError, match="DUPLICATE"):
        eligible_ranked_regions([{"atlas_id": 2}, {"atlas_id": 2}], frozenset({2}))


def test_coordinator_guard_rejects_spoofed_parent(monkeypatch, tmp_path: Path):
    coordinator = {"ProcessId": 100, "ParentProcessId": 1,
                   "CommandLine": r"python H:\New project\scripts\mx_fasttrack_v2_4_4\supervise_v244.py all"}
    launcher = {"ProcessId": 200, "ParentProcessId": 100,
                "CommandLine": r"python H:\New project\scripts\mx_fasttrack_v2_4_4\entry_v244.py xai-run"}
    monkeypatch.setenv(entry_v244.INTERNAL_PARENT_ENV, "100")
    monkeypatch.setattr(entry_v244.os, "getppid", lambda: 200)
    monkeypatch.setattr(entry_v244, "windows_python_processes",
                        lambda **kwargs: [coordinator, launcher])
    entry_v244.require_v244_coordinator(tmp_path)
    launcher["ParentProcessId"] = 300
    with pytest.raises(RuntimeError, match="ACTIVE_COORDINATOR_REQUIRED"):
        entry_v244.require_v244_coordinator(tmp_path)


def test_real_atlas_eligibility_does_not_change_four_committed_folds():
    import nibabel as nib
    from scripts.mx_fasttrack_v2.run import read
    root = Path(__file__).resolve().parents[1]
    out = root / supervise_v244.OUTPUT
    if not (out / "reports/xai_v22_folds/fold_3.json").is_file():
        pytest.skip("historical XAI receipts unavailable")
    registry = read(root / "results/spatial_attribution_v1/provenance/atlas_registry.json")
    atlas = np.asarray(nib.load(registry["output_atlas_path"]).dataobj)
    eligible = set(eligible_atlas_ids(atlas))
    assert len(eligible) == 85
    assert all(direction["top_region_id"] in eligible
               for fold in range(4)
               for direction in read(out / "reports/xai_v22_folds" / f"fold_{fold}.json")
               ["crossfit_directions"])
