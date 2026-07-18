import json
import sqlite3
from pathlib import Path

import pytest

from core import (
    atomic_write_json,
    canonical_json,
    compute_sha256,
    load_or_init_state,
    transition_state,
)
from inventory import deduplicate_sources, inventory_file
from scoring import Candidate, score_candidate


def test_atomic_json_roundtrip_and_state_transition(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"state": "INIT", "completed": []})
    state = load_or_init_state(path)
    assert state["state"] == "INIT"
    updated = transition_state(state, "SAFETY_SNAPSHOTTED")
    assert updated["state"] == "SAFETY_SNAPSHOTTED"
    atomic_write_json(path, updated)
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == "SAFETY_SNAPSHOTTED"


def test_canonical_json_is_stable_and_hash_is_content_based():
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
    assert compute_sha256(b"abc") == compute_sha256(b"abc")
    assert compute_sha256(b"abc") != compute_sha256(b"abd")


def test_inventory_deduplicates_identical_sources_without_double_evidence(tmp_path):
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    first.write_text("# Same\nclaim", encoding="utf-8")
    second.write_text("# Same\nclaim", encoding="utf-8")
    rows = [inventory_file(first, "agent_01"), inventory_file(second, "agent_02")]
    canonical, duplicates = deduplicate_sources(rows)
    assert len(canonical) == 1
    assert len(duplicates) == 1
    assert duplicates[0]["canonical_source_id"] == canonical[0]["source_id"]


def test_hard_failed_candidate_cannot_be_primary():
    candidate = Candidate(
        idea_id="x",
        normalized_family="raw_3d",
        gates={g: "PASS" for g in ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]},
        scores={"novelty": 18, "clinical": 16, "data": 15, "reuse": 15, "time": 12,
                "compute": 8, "evaluation": 8, "story": 5, "interpretability": 3},
        penalties=[("raw_3d_pipeline_new", 12)],
    )
    candidate.gates["G3"] = "FAIL"
    result = score_candidate(candidate)
    assert result["cannot_be_primary"] is True
    assert result["verdict"] != "PRIMARY"
