from __future__ import annotations

from pathlib import Path

from src.utils.pet_preprocess_progress import summarize_pet_preprocess_progress


def test_preprocess_progress_counts(tmp_path, monkeypatch):
    resume_dir = tmp_path / "results" / "logs"
    resume_dir.mkdir(parents=True)
    resume_path = resume_dir / "method_y_pet_preprocess_resume_fbp_spm_pet_t1_mni.json"
    resume_path.write_text(
        """
{
  "config_digest": "abc",
  "config": {"spatial_mode": "spm_pet_t1_mni"},
  "items": {
    "a": {"status": "PASS"},
    "b": {"status": "FAIL"}
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    out = summarize_pet_preprocess_progress(linked_work_items=10, project_root=tmp_path)
    assert out["resume_done"] == 1
    assert out["resume_fail"] == 1
    assert out["resume_pending"] == 9
    assert out["preprocessing_started"] is True
