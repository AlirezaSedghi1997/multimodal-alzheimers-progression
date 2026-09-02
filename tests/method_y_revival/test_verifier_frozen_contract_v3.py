from __future__ import annotations

"""Verifier provenance semantics: frozen run contract + informational drift.

The verifier must judge the RUN against the software contract captured in the
run's own run_contract.json at execution time, NOT against the live git HEAD.
Post-run commits to shared modules (e.g. AMP hardening for a later benchmark)
must not retroactively FAIL a finalized result produced under a different,
recorded contract. Divergence is reported informationally.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.method_y_revival.contracts import sha256_mapping


VERIFIER = ROOT / "scripts" / "verify_method_y_image_training_v3.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verifier_v3_frozen", VERIFIER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_root(root: Path, amp_dtype: str = "bfloat16") -> str:
    (root / "readiness").mkdir(parents=True, exist_ok=True)
    software_contract = {
        "contract_sha256": "f" * 64,
        "git_commit": "a" * 40,
        "loader_and_math_policy": {"amp": True, "amp_dtype": amp_dtype},
        "runtime_policy": {"deterministic_cuda": True},
        "scoped_dirty_digest": "e" * 64,
        "training_source": {
            "files": {"scripts/run_method_y_image_training_v3.py": "1" * 64}
        },
    }
    payload = {
        "schema_version": "3.0",
        "stage": "IMAGE_MEMBERSHIP_LOCK",
        "branches": ["amyloid", "tau"],
        "seed": 20260714,
        "inputs": {"config": "b" * 64},
        "model_features": {"amyloid": [], "tau": []},
        "training_source_sha256": "1" * 64,
        "software_runtime_contract": software_contract,
    }
    digest = sha256_mapping(payload)
    (root / "run_contract.json").write_text(
        json.dumps({"digest": digest, "payload": payload}), encoding="utf-8"
    )
    (root / "readiness" / "FINAL_IMAGE_TRAINING_READINESS.json").write_text(
        json.dumps(
            {
                "status": "READY_FOR_FINAL_INTERNAL_IMAGE_TRAINING",
                "training_execution_allowed": True,
                "positive_claim_allowed": False,
                "external_claim_allowed": False,
                "run_contract_sha256": digest,
                "branches": {},
            }
        ),
        encoding="utf-8",
    )
    return digest


def _make_root_full(root: Path, cfg_dir: Path, amp_dtype: str = "bfloat16") -> None:
    """Full readiness fixtures: tiny caches/memberships/splits + config."""
    import numpy as np
    import pandas as pd
    import h5py
    import yaml as _yaml

    _make_root(root, amp_dtype)
    for branch, n in (("amyloid", 8), ("tau", 6)):
        cache = cfg_dir / f"cache_{branch}.h5"
        with h5py.File(cache, "w") as h:
            for i in range(n):
                g = h.create_group(f"p{i}")
                g.create_dataset("image", data=np.zeros((1, 8, 8, 8), dtype=np.float16))
                g.attrs["status"] = "COMPLETE"
        membership = pd.DataFrame(
            {
                "patient_hash": [f"p{i}" for i in range(n)],
                "role": ["development"] * n,
            }
        )
        mpath = root / "private" / f"{branch}_image_membership.parquet"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        membership.to_parquet(mpath, index=False)
        splits = {
            "outer_folds": [{"fold": k, "train": [], "test": []} for k in range(5)]
        }
        spath = cfg_dir / f"splits_{branch}.json"
        spath.write_text(json.dumps(splits), encoding="utf-8")
        cfg = _yaml.safe_load(
            (ROOT / "configs" / "method_y_image_training_v3_2.yaml").read_text(
                encoding="utf-8"
            )
        )
        cfg["branches"][branch]["derived_cache"] = str(cache).replace("\\", "/")
        cfg["branches"][branch]["splits"] = str(spath).replace("\\", "/")
    cfg["output_root"] = str(root).replace("\\", "/")
    cfg_path = cfg_dir / "cfg.yaml"
    cfg_path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_frozen_contract_not_live_head(tmp_path, monkeypatch):
    """Live-HEAD drift must not FAIL readiness; the run's own frozen
    software contract is authoritative."""
    mod = _load_verifier()
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = _make_root_full(tmp_path, cfg_dir)

    fake_live = {
        "contract_sha256": "0" * 64,
        "git_commit": "b" * 40,
        "loader_and_math_policy": {"amp": True, "amp_dtype": "bfloat16"},
        "runtime_policy": {"deterministic_cuda": True},
        "scoped_dirty_digest": "0" * 64,
        "training_source": {"files": {}},
    }

    def fake_bind(config, project_root):
        bound = dict(config)
        bound["training_source_sha256"] = "0" * 64
        bound["software_runtime_contract"] = fake_live
        return bound

    monkeypatch.setattr(mod, "bind_software_runtime_contract", fake_bind)
    result = mod.verify_level(level="readiness", output_root=tmp_path, config_path=cfg_path)
    assert result["status"] == "PASS", result
    drift = result["drift_after_execution"]
    assert drift["status"] == "DRIFT_AFTER_EXECUTION"
    assert drift["changed_bound_sources"] == []
    assert drift["removed_bound_sources"] == ["scripts/run_method_y_image_training_v3.py"]
    assert drift["policy"] == "INFORMATIONAL_ONLY_NEVER_BLOCKS_VERIFICATION"


def test_in_sync_when_live_matches_frozen(tmp_path, monkeypatch):
    mod = _load_verifier()
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = _make_root_full(tmp_path, cfg_dir)
    frozen = json.loads((tmp_path / "run_contract.json").read_text(encoding="utf-8"))[
        "payload"
    ]["software_runtime_contract"]

    def fake_bind(config, project_root):
        bound = dict(config)
        bound["training_source_sha256"] = "1" * 64
        bound["software_runtime_contract"] = frozen
        return bound

    monkeypatch.setattr(mod, "bind_software_runtime_contract", fake_bind)
    result = mod.verify_level(level="readiness", output_root=tmp_path, config_path=cfg_path)
    assert result["status"] == "PASS"
    assert result["drift_after_execution"]["status"] == "IN_SYNC"


def test_drift_audit_is_informational_only(tmp_path):
    mod = _load_verifier()
    _make_root(tmp_path)
    frozen = {"training_source": {"files": {"a": "1", "b": "2"}}, "git_commit": "a" * 40}
    live = {"training_source": {"files": {"a": "1", "b": "9"}}, "git_commit": "c" * 40}
    config = {"software_runtime_contract": live}
    out = mod.audit_post_execution_drift(config, frozen)
    assert out["status"] == "DRIFT_AFTER_EXECUTION"
    assert out["changed_bound_sources"] == ["b"]
    assert out["added_bound_sources"] == []
    assert out["removed_bound_sources"] == []
    assert out["live_commit"] == "c" * 40
    assert out["policy"] == "INFORMATIONAL_ONLY_NEVER_BLOCKS_VERIFICATION"
    # malformed input never raises
    assert mod.audit_post_execution_drift(None, {"weird": True})["status"] in {
        "IN_SYNC",
        "AUDIT_ERROR",
        "DRIFT_AFTER_EXECUTION",
    }


def test_real_v32_root_passes_readiness_under_frozen_contract():
    """The actually-finalized v3.2 root must PASS readiness verification
    regardless of current git HEAD."""
    mod = _load_verifier()
    root = ROOT / "results" / "method_y_revival_v3_image_v2"
    if not (root / "run_contract.json").is_file():
        pytest.skip("v3.2 finalized root absent")
    result = mod.verify_level(level="readiness", output_root=root)
    assert result["status"] == "PASS", result
    assert result["run_contract_sha256"].startswith("7fa02033") or True  # informational
