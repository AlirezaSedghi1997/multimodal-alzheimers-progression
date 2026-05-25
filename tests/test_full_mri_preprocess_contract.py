from pathlib import Path

from scripts.preprocess_mri_full import check_toolchain, write_toolchain_report


def test_full_mri_preprocess_blocks_without_specialized_toolchain(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "")
    report = check_toolchain(tmp_path / "missing_template.nii.gz")
    assert report["status"] == "BLOCKED"
    assert "template" in report["missing"]
    assert "bias_correction" in report["missing"]


def test_full_mri_toolchain_report_is_privacy_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_toolchain_report(
        {
            "status": "BLOCKED",
            "missing": ["template"],
            "contains_pet_input": False,
            "contains_raw_identifiers": False,
        }
    )
    text = Path("results/tables/full_mri_preprocess_toolchain_report.md").read_text(encoding="utf-8")
    assert "contains_pet_input: `False`" in text
    assert "contains_raw_identifiers: `False`" in text


def test_limited_execute_is_smoke_not_completed(monkeypatch, tmp_path):
    from scripts import preprocess_mri_full

    monkeypatch.chdir(tmp_path)
    catalog = tmp_path / "data/processed/master_catalog.csv"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("sample_hash,t1_file_hash\nsample_a,t1_a\n", encoding="utf-8")
    template = tmp_path / "template.nii.gz"
    template.write_bytes(b"template")
    source = tmp_path / "source.nii.gz"
    source.write_bytes(b"source")

    monkeypatch.setattr(
        preprocess_mri_full,
        "check_toolchain",
        lambda _: {
            "status": "READY",
            "available": {
                "bias_correction": "/bin/N4BiasFieldCorrection",
                "registration": "/bin/antsRegistrationSyNQuick.sh",
                "brain_extraction": "/bin/bet",
            },
            "missing": [],
            "required": preprocess_mri_full.TOOL_GROUPS,
            "template": str(template),
            "template_exists": True,
        },
    )
    monkeypatch.setattr(preprocess_mri_full, "resolve_data_root", lambda _: tmp_path)
    monkeypatch.setattr(preprocess_mri_full, "build_t1_hash_index", lambda _: {"t1_a": source})
    monkeypatch.setattr(preprocess_mri_full, "preprocess_one", lambda **_: (True, "out/sample_a.nii.gz"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "preprocess_mri_full.py",
            "--catalog",
            str(catalog),
            "--template",
            str(template),
            "--execute",
            "--limit",
            "1",
        ],
    )

    assert preprocess_mri_full.main() == 0
    assert '"status": "SMOKE_TEST_COMPLETED"' in Path("results/tables/full_mri_preprocess_toolchain_report.json").read_text(
        encoding="utf-8"
    )
