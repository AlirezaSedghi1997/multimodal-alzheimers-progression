from pathlib import Path


def test_spm_toolchain_blocks_without_paths(monkeypatch, tmp_path):
    from scripts.check_spm_toolchain import check_spm_toolchain

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    report = check_spm_toolchain(None, None)
    assert report["status"] == "BLOCKED"
    assert "matlab" in report["missing"]
    assert "spm_dir" in report["missing"]
    assert report["contains_pet_input"] is False
    assert report["contains_raw_identifiers"] is False


def test_spm_toolchain_report_is_privacy_safe(monkeypatch, tmp_path):
    from scripts.check_spm_toolchain import write_spm_toolchain_report

    monkeypatch.chdir(tmp_path)
    write_spm_toolchain_report(
        {
            "status": "BLOCKED",
            "backend": "spm",
            "missing": ["matlab"],
            "matlab": None,
            "spm_dir": None,
        }
    )
    text = Path("results/tables/spm_preprocess_toolchain_report.md").read_text(encoding="utf-8")
    assert "contains_pet_input: `False`" in text
    assert "contains_raw_identifiers: `False`" in text


def test_spm_limited_execute_is_smoke_not_completed(monkeypatch, tmp_path):
    from scripts import preprocess_mri_spm

    monkeypatch.chdir(tmp_path)
    catalog = tmp_path / "data/processed/master_catalog.csv"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("sample_hash,t1_file_hash\nsample_a,t1_a\n", encoding="utf-8")
    source = tmp_path / "source.nii.gz"
    source.write_bytes(b"source")
    spm_dir = tmp_path / "spm12"
    spm_dir.mkdir()

    monkeypatch.setattr(
        preprocess_mri_spm,
        "check_spm_toolchain",
        lambda *_: {
            "status": "READY",
            "backend": "spm",
            "matlab": "/bin/matlab",
            "spm_dir": str(spm_dir),
            "missing": [],
            "contains_pet_input": False,
            "contains_raw_identifiers": False,
        },
    )
    monkeypatch.setattr(preprocess_mri_spm, "resolve_data_root", lambda _: tmp_path)
    monkeypatch.setattr(preprocess_mri_spm, "build_t1_hash_index", lambda _: {"t1_a": source})
    monkeypatch.setattr(preprocess_mri_spm, "preprocess_one_spm", lambda **_: (True, "out/sample_a.nii.gz"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "preprocess_mri_spm.py",
            "--catalog",
            str(catalog),
            "--spm_dir",
            str(spm_dir),
            "--matlab",
            "/bin/matlab",
            "--execute",
            "--limit",
            "1",
        ],
    )

    assert preprocess_mri_spm.main() == 0
    text = Path("results/tables/full_mri_preprocess_toolchain_report.json").read_text(encoding="utf-8")
    assert '"backend": "spm"' in text
    assert '"status": "SMOKE_TEST_COMPLETED"' in text
    assert '"preprocessing_version": "spm_v2_mni_brain_robust_intensity"' in text
    assert "brain_masked_zscore_intensity_normalization" in text


def test_spm_batches_only_pending_rows(monkeypatch, tmp_path):
    from scripts import preprocess_mri_spm

    monkeypatch.chdir(tmp_path)
    catalog = tmp_path / "data/processed/master_catalog.csv"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        "sample_hash,t1_file_hash\nsample_a,t1_a\nsample_b,t1_b\nsample_c,t1_c\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.nii.gz"
    source.write_bytes(b"source")
    spm_dir = tmp_path / "spm12"
    spm_dir.mkdir()
    calls = []

    monkeypatch.setattr(
        preprocess_mri_spm,
        "check_spm_toolchain",
        lambda *_: {
            "status": "READY",
            "backend": "spm",
            "matlab": "/bin/matlab",
            "spm_dir": str(spm_dir),
            "missing": [],
            "contains_pet_input": False,
            "contains_raw_identifiers": False,
        },
    )
    monkeypatch.setattr(preprocess_mri_spm, "resolve_data_root", lambda _: tmp_path)
    monkeypatch.setattr(preprocess_mri_spm, "build_t1_hash_index", lambda _: {"t1_a": source, "t1_b": source, "t1_c": source})

    def fake_preprocess_one_spm(**kwargs):
        calls.append(kwargs["sample_hash"])
        return True, f"out/{kwargs['sample_hash']}.nii.gz"

    monkeypatch.setattr(preprocess_mri_spm, "preprocess_one_spm", fake_preprocess_one_spm)
    monkeypatch.setattr(
        "sys.argv",
        [
            "preprocess_mri_spm.py",
            "--catalog",
            str(catalog),
            "--spm_dir",
            str(spm_dir),
            "--matlab",
            "/bin/matlab",
            "--execute",
            "--batch_size",
            "2",
        ],
    )

    assert preprocess_mri_spm.main() == 0
    assert calls == ["sample_a", "sample_b"]
    report = Path("results/tables/full_mri_preprocess_toolchain_report.json").read_text(encoding="utf-8")
    assert '"status": "BATCH_COMPLETED"' in report
    assert '"deferred_pending": 1' in report


def test_spm_auto_batches_continue_until_done(monkeypatch, tmp_path):
    from scripts import preprocess_mri_spm

    monkeypatch.chdir(tmp_path)
    catalog = tmp_path / "data/processed/master_catalog.csv"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        "sample_hash,t1_file_hash\nsample_a,t1_a\nsample_b,t1_b\nsample_c,t1_c\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.nii.gz"
    source.write_bytes(b"source")
    spm_dir = tmp_path / "spm12"
    spm_dir.mkdir()
    calls = []

    monkeypatch.setattr(
        preprocess_mri_spm,
        "check_spm_toolchain",
        lambda *_: {
            "status": "READY",
            "backend": "spm",
            "matlab": "/bin/matlab",
            "spm_dir": str(spm_dir),
            "missing": [],
            "contains_pet_input": False,
            "contains_raw_identifiers": False,
        },
    )
    monkeypatch.setattr(preprocess_mri_spm, "resolve_data_root", lambda _: tmp_path)
    monkeypatch.setattr(preprocess_mri_spm, "build_t1_hash_index", lambda _: {"t1_a": source, "t1_b": source, "t1_c": source})

    def fake_preprocess_one_spm(**kwargs):
        calls.append(kwargs["sample_hash"])
        return True, f"out/{kwargs['sample_hash']}.nii.gz"

    monkeypatch.setattr(preprocess_mri_spm, "preprocess_one_spm", fake_preprocess_one_spm)
    monkeypatch.setattr(
        "sys.argv",
        [
            "preprocess_mri_spm.py",
            "--catalog",
            str(catalog),
            "--spm_dir",
            str(spm_dir),
            "--matlab",
            "/bin/matlab",
            "--execute",
            "--batch_size",
            "2",
            "--auto_batches",
        ],
    )

    assert preprocess_mri_spm.main() == 0
    assert calls == ["sample_a", "sample_b", "sample_c"]
    report = Path("results/tables/full_mri_preprocess_toolchain_report.json").read_text(encoding="utf-8")
    assert '"status": "COMPLETED"' in report
    assert '"batches_completed": 2' in report
    assert '"done_total": 3' in report


def test_mismatched_spm_manifest_is_archived(monkeypatch, tmp_path):
    from scripts.preprocess_mri_spm import build_resume_manifest

    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "resume.json"
    manifest.write_text(
        '{"schema_version":1,"stage":"preprocess_full_mri_spm","config_hash":"old","items":{}}',
        encoding="utf-8",
    )

    resume = build_resume_manifest(
        str(manifest),
        stage="preprocess_full_mri_spm",
        config={"preprocessing_version": "current"},
    )

    assert resume.load()["items"] == {}
    archived = list(tmp_path.glob("resume.mismatch.*.json"))
    assert len(archived) == 1


def test_spm_scratch_allocation_does_not_delete_existing_locked_dir(tmp_path):
    from scripts.preprocess_mri_spm import make_spm_scratch_dir

    old_scratch = tmp_path / "_scratch_spm" / "sample_a"
    old_scratch.mkdir(parents=True)
    sentinel = old_scratch / "keep.txt"
    sentinel.write_text("do not delete", encoding="utf-8")

    scratch = make_spm_scratch_dir(tmp_path, "sample_a")

    assert scratch.exists()
    assert scratch.name.startswith("sample_a_")
    assert sentinel.read_text(encoding="utf-8") == "do not delete"


def test_spm_batch_requests_forward_deformation(tmp_path):
    from scripts.spm_preprocess_backend import write_spm_batch

    spm_dir = tmp_path / "spm"
    (spm_dir / "tpm").mkdir(parents=True)
    batch_path = tmp_path / "run_spm.m"
    write_spm_batch(
        batch_path=batch_path,
        spm_dir=spm_dir,
        input_nii=tmp_path / "input.nii",
        output_nii=tmp_path / "out.nii",
        windows_matlab=False,
    )
    text = batch_path.read_text(encoding="utf-8")
    assert "warp.write = [0 1]" in text
    assert "y_' name ext" in text
    assert "brain_mask = abs(Yw) > 0" in text
    assert "round(0.005 * nvals)" in text
    assert "Yw(brain_mask) = single((clipped(brain_mask) - mu) ./ sigma)" in text
    assert "qc.sample_hash = sample_hash" in text
    assert "qc.brain_volume_ml = brain_volume_ml" in text
    assert "jsonencode(qc)" in text


def test_spm_numeric_qc_is_hash_level_and_privacy_safe():
    import numpy as np

    from scripts.preprocess_mri_spm import summarize_spm_numeric_qc

    corrected = np.zeros((4, 4, 4), dtype=np.float32)
    corrected[1:3, 1:3, 1:3] = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
    c1 = np.zeros_like(corrected)
    c2 = np.zeros_like(corrected)
    c3 = np.zeros_like(corrected)
    c1[1, 1:3, 1:3] = 0.8
    c2[2, 1:3, 1:3] = 0.8
    c3[1:3, 1, 1] = 0.4
    normalized = np.zeros_like(corrected)
    normalized[corrected > 0] = (corrected[corrected > 0] - corrected[corrected > 0].mean()) / corrected[corrected > 0].std()

    qc = summarize_spm_numeric_qc(
        sample_hash="s_hash",
        bias_corrected=corrected,
        c1=c1,
        c2=c2,
        c3=c3,
        normalized_brain=normalized,
        voxel_volume_ml=1.0,
    )

    assert qc["sample_hash"] == "s_hash"
    assert qc["contains_pet_input"] is False
    assert qc["contains_raw_identifiers"] is False
    assert "brain_volume_ml" in qc
    assert "tissue_fraction_gm" in qc
    assert "gm_wm_cnr" in qc
    assert abs(qc["normalized_brain_mean"]) < 1e-6
