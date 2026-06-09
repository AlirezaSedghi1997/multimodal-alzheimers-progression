#!/usr/bin/env python3
"""Build human visual QC review manifest for Method Y PET overlays."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.visual_qc import build_visual_review_manifest


def render_md(manifest: dict) -> str:
    lines = [
        "# Visual QC Review Manifest",
        "",
        f"- Generated: `{manifest.get('generated_at')}`",
        f"- Review items: **{manifest.get('total_review_items')}**",
        f"- Pending sign-off: **{manifest.get('pending_signoff_count')}**",
        f"- All signed: **{manifest.get('all_signed_off')}**",
        "",
        "## Checklist",
        "",
    ]
    for item in manifest.get("checklist") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Items", ""])
    for item in manifest.get("items") or []:
        lines.append(
            f"- `{item['sample_hash']}` reason=`{item.get('review_required_reason')}` "
            f"signed=`{item.get('visual_qc_signoff')}` ncc=`{item.get('ncc_pet_t1')}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pet_qc_dir", default="data/preprocessed_pet/fbp/qc")
    ap.add_argument("--out_json", default="results/qc/visual_review_manifest.json")
    ap.add_argument("--out_md", default="results/qc/visual_review_manifest.md")
    ap.add_argument("--pass_fraction", type=float, default=0.05)
    args = ap.parse_args()

    manifest = build_visual_review_manifest(
        PROJECT_ROOT / args.pet_qc_dir,
        project_root=PROJECT_ROOT,
        pass_sample_fraction=args.pass_fraction,
    )
    out_json = PROJECT_ROOT / args.out_json
    out_md = PROJECT_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_json, manifest)
    atomic_write_text(out_md, render_md(manifest))
    print(json.dumps({"status": "OK", "total_review_items": manifest["total_review_items"], "pending": manifest["pending_signoff_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
