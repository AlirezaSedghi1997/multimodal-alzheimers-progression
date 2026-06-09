import json
import re
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
    assert obj.get("contains_raw_paths") is True
    assert obj.get("contains_subject_level_vectors") is True
    assert obj["violations"] == []


def test_public_bundle_has_no_local_paths_or_subject_vectors():
    root = Path("public_bundle")
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "PUBLIC_BUNDLE_MANIFEST.json":
            continue
        text = path.read_text(encoding="utf-8")
        assert "H:\\" not in text, path
        assert not re.search(r"/mnt/[a-z]/", text), path
        if path.suffix.lower() == ".json":
            assert "y_prob" not in text, path
            assert "y_true" not in text, path
