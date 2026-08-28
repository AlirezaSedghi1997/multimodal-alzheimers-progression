"""Deprecated pre-B3 embedding entry point.

The historical implementation read the Method X 128-cubed cache, outcome and
tabular fields, and synthetic rows.  That route is scientifically ineligible
for primary BrainIAC embeddings and is intentionally non-executable.  B3 must
replace this file with the schema-v2 NIfTI-only extractor plus a pinned source
contract before the action policy can authorize extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(r"H:\New project")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DERIVED_DIR = ROOT / "data" / "derived" / "method_x_brainiac"
# Kept only as inert compatibility names for tests and downstream imports that
# prove the retired route cannot perform I/O.  ``extract_all_embeddings`` never
# reads or writes any of these paths.
CACHE_PATH = ROOT / "data" / "cached" / "t1_volume_cache_128.h5"
OUT_H5_PATH = DERIVED_DIR / "INVALID_LEGACY_EMBEDDINGS.h5"
MANIFEST_PATH = DERIVED_DIR / "INVALID_LEGACY_EMBEDDINGS.json"


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def validate_primary_embedding_source_manifest(
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Authorize only real, affine-aware raw-T1 outputs for primary embeddings.

    The locked 128-cubed Method X cache is scientifically useful for a
    sensitivity analysis but is not the BrainIAC preprocessing route.  This
    validator prevents a sensitivity cache from being silently relabelled as a
    primary BrainIAC input.
    """

    record = dict(manifest or {})
    allowed_routes = {
        "CANONICAL_UPSTREAM_REPRODUCTION_PRIMARY",
        "PROTOCOL_COMPATIBLE_INTERNAL_POSTHOC_PRIMARY",
    }
    valid = (
        int(record.get("schema_version", 0)) >= 2
        and record.get("status") == "COMPLETED"
        and record.get("route_role") in allowed_routes
        and record.get("source_role") == "raw_t1"
        and record.get("label_read") is False
        and record.get("direct_128_to_96_used") is False
        and record.get("real_only") is True
        and isinstance(record.get("rows"), int)
        and not isinstance(record.get("rows"), bool)
        and int(record.get("rows", 0)) > 0
        and tuple(record.get("output_shape", ())) == (96, 96, 96)
        and record.get("affine_aware") is True
        and _is_sha256(record.get("manifest_sha256"))
    )
    if not valid:
        raise RuntimeError("PRIMARY_EMBEDDING_SOURCE_ROUTE_INVALID")
    return {
        "status": "PASS",
        "route_role": str(record["route_role"]),
        "rows": int(record["rows"]),
        "manifest_sha256": str(record["manifest_sha256"]),
    }


def extract_all_embeddings(
    batch_size: int = 4,
    device: str = "cpu",
    limit: int | None = None
) -> dict[str, Any]:
    """Fail before any source, model, or output I/O."""

    del batch_size, device, limit
    raise RuntimeError(
        "BRAINIAC_LEGACY_CACHE128_EXTRACTOR_PERMANENTLY_DISABLED"
    )


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Extract BrainIAC frozen embeddings")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for extraction")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for testing")
    args = parser.parse_args()

    manifest = extract_all_embeddings(batch_size=args.batch_size, device=args.device, limit=args.limit)
    print(json.dumps(manifest, indent=2))
