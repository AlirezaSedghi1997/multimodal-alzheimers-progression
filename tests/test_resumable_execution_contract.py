import json
from pathlib import Path

import pytest

from src.utils.resumable import ResumableManifest, assert_safe_resume_key, config_digest


def test_config_digest_is_stable():
    assert config_digest({"b": 2, "a": 1}) == config_digest({"a": 1, "b": 2})


def test_resumable_manifest_marks_done(tmp_path):
    path = tmp_path / "resume.json"
    manifest = ResumableManifest(path, stage="stage_a", config={"resolution": 96})
    manifest.set_item("sample_hash_abc", "RUNNING")
    manifest.set_item("sample_hash_abc", "DONE", artifact="data/preprocessed_mri_96/sample_hash_abc.nii.gz")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["items"]["sample_hash_abc"]["status"] == "DONE"
    assert manifest.item_done("sample_hash_abc") is True


def test_resume_key_rejects_raw_ids_and_paths():
    with pytest.raises(ValueError):
        assert_safe_resume_key("B000001")
    with pytest.raises(ValueError):
        assert_safe_resume_key("folder/sample")
