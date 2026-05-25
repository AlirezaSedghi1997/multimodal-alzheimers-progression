import json
from pathlib import Path


def test_public_bundle_manifest_if_generated():
    p = Path("public_bundle/PUBLIC_BUNDLE_MANIFEST.json")
    if not p.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["contains_raw_data"] is False
    assert obj["contains_hdf5"] is False
    assert obj["contains_images"] is False
    assert obj["contains_raw_identifiers"] is False
    assert obj["violations"] == []

