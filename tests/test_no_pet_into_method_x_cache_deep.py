from pathlib import Path

FORBIDDEN = ["pet", "suvr", "centiloid", "fbp", "fbb", "ftp", "amyloid", "tau", "tracer"]

def test_no_method_x_cache_files_in_patch_contain_pet_features():
    root = Path(__file__).resolve().parents[1]
    cache_roots = [root / "data" / "cached", root / "data" / "processed"]
    for cr in cache_roots:
        if not cr.exists():
            continue
        for p in cr.rglob("*"):
            name = p.name.lower()
            assert not any(term in name for term in FORBIDDEN), f"Forbidden PET term in Method X cache path: {p}"
