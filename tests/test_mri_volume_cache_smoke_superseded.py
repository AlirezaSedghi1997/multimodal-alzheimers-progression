from pathlib import Path


def test_mri_volume_cache_smoke_superseded_when_full_cache_exists():
    smoke = Path("results/tables/mri_volume_cache_smoke_summary.json")
    full = Path("results/tables/mri_volume_cache_summary.json")
    if not smoke.exists() or not full.exists():
        return
    import json

    full_obj = json.loads(full.read_text(encoding="utf-8"))
    if int(full_obj.get("ok") or 0) < 1600:
        return
    smoke_obj = json.loads(smoke.read_text(encoding="utf-8"))
    assert smoke_obj["status"] == "SUPERSEDED_BY_FULL_CACHE"
    assert smoke_obj.get("full_cache_rows", 0) >= 1600
