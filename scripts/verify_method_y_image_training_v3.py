from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival.atomic_io import read_json


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    config_path = ROOT / "configs/method_y_image_training_v3_1.yaml"
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = ROOT / config["output_root"]
    # verify readiness
    readiness = output_root / "readiness" / "FINAL_IMAGE_TRAINING_READINESS.json"
    if not readiness.exists():
        print(json.dumps({"status": "FAIL", "reason": "READINESS_MISSING"}, indent=2))
        return 2
    readiness_json = read_json(readiness)
    # verify caches
    for branch in ["amyloid", "tau"]:
        cache = ROOT / config["branches"][branch]["derived_cache"]
        if not cache.exists():
            print(json.dumps({"status": "FAIL", "reason": f"CACHE_MISSING:{branch}"}, indent=2))
            return 2
        # check at least one group exists
        import h5py

        with h5py.File(cache, "r") as f:
            groups = [k for k in f.keys() if not k.startswith("__tmp__")]
            if not groups:
                print(json.dumps({"status": "FAIL", "reason": f"CACHE_EMPTY:{branch}"}, indent=2))
                return 2
    # verify orchestration state not required for now
    print(json.dumps({"status": "PASS", "readiness": readiness_json, "caches": "VERIFIED"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
