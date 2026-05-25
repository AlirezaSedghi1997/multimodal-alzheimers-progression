from pathlib import Path


def test_mri_qc_summary_if_generated():
    p = Path("results/tables/mri_preprocess_qc_summary.md")
    if not p.exists():
        return
    text = p.read_text(encoding="utf-8")
    assert "contains_pet_input: `False`" in text
    assert "contains_raw_identifiers: `False`" in text


def test_t1_header_cache_name_if_generated():
    cache = Path("data/cached/t1_header_cache.h5")
    if not cache.exists():
        return
    assert "pet" not in cache.name.lower()
    assert "tau" not in cache.name.lower()

