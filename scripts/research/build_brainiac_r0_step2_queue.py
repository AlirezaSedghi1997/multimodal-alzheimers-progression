#!/usr/bin/env python3
"""Build the deterministic HyperResearch Step-2 URL queue for BrainIAC R0.

The two discovery files are intentionally independent.  This script performs
only mechanical URL deduplication and explicit identity merging; it does not
turn candidates into included studies or full-text reviews.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "research" / "runs" / "brainiac-method-x-gap-a5644d"
TEMP = RUN / "temp"
SOURCES = (TEMP / "candidate-urls-methods.md", TEMP / "candidate-urls-mri-models.md")

# These URLs are metadata mirrors of a higher-authority URL already in the
# queue.  The merge is explicit so that a future source edit cannot silently
# change study counts.
IDENTITY_MIRRORS = {
    "https://pubmed.ncbi.nlm.nih.gov/41644653/": "https://doi.org/10.1038/s41593-026-02202-6",
    "https://api.openalex.org/works/https://doi.org/10.1038/s41593-026-02202-6": "https://doi.org/10.1038/s41593-026-02202-6",
    "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41593-026-02202-6?fields=title,year,externalIds,publicationDate,venue": "https://doi.org/10.1038/s41593-026-02202-6",
    "https://pubmed.ncbi.nlm.nih.gov/40450379/": "https://doi.org/10.1186/s13195-025-01773-z",
    "https://pubmed.ncbi.nlm.nih.gov/40789853/": "https://doi.org/10.1038/s41467-025-62590-4",
}

# Lower-scored but methodologically necessary sources are retained even if the
# top-utility cut would otherwise omit them.
MANDATORY_URLS = {
    "https://pubmed.ncbi.nlm.nih.gov/40438162/",  # conformal task output
    "https://pubmed.ncbi.nlm.nih.gov/42338036/",  # conformal imaging
    "https://pubmed.ncbi.nlm.nih.gov/38765672/",  # subject leakage background
    "https://doi.org/10.1007/s10278-025-01747-5",  # frozen embeddings
    "https://www.medrxiv.org/content/10.1101/2024.12.02.24317992v1.full",  # version lineage
    "https://doi.org/10.1148/radiol.259008",  # correction record
}


def parse_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.startswith("|") or "http" not in line:
                continue
            match = re.search(r"https?://[^)|\s]+", line)
            if not match:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            score = None
            for cell in reversed(cells):
                score_match = re.fullmatch(r"(?:\*\*)?(\d{1,2})(?:\*\*)?", cell)
                if score_match:
                    score = int(score_match.group(1))
                    break
            if score is None or not 0 <= score <= 18:
                raise RuntimeError(f"R0_SCORE_MISSING:{source.name}:{line_no}")
            rows.append(
                {
                    "url": match.group(0),
                    "utility_score": score,
                    "source": source.name,
                    "source_line": line_no,
                    "description": " | ".join(cells[2:5])[:600],
                }
            )
    return rows


def main() -> int:
    exact: dict[str, dict[str, Any]] = {}
    for row in parse_rows():
        current = exact.get(row["url"])
        if current is None or row["utility_score"] > current["utility_score"]:
            exact[row["url"]] = row

    merged: dict[str, list[str]] = {}
    for mirror, canonical in IDENTITY_MIRRORS.items():
        if mirror in exact:
            merged.setdefault(canonical, []).append(mirror)
            exact.pop(mirror)

    ranked = sorted(
        exact.values(), key=lambda row: (-int(row["utility_score"]), str(row["url"]))
    )
    mandatory = [row for row in ranked if row["url"] in MANDATORY_URLS]
    selected = ranked[:80]
    selected_urls = {row["url"] for row in selected}
    for row in mandatory:
        if row["url"] not in selected_urls:
            replace = next(
                candidate
                for candidate in reversed(selected)
                if candidate["url"] not in MANDATORY_URLS
            )
            selected.remove(replace)
            selected.append(row)
            selected_urls.add(row["url"])
    selected.sort(key=lambda row: (-int(row["utility_score"]), str(row["url"])))
    if len(selected) != 80 or len({row["url"] for row in selected}) != 80:
        raise RuntimeError("R0_QUEUE_NOT_EXACTLY_80_UNIQUE_URLS")

    lines = [
        "# R0 Step 2 — scored URL queue\n\n",
        "Discovery candidates only. Selection here does not imply study inclusion or full-text review.\n\n",
        "| Rank | URL | Utility | Discovery source | Identity mirrors merged |\n",
        "|---:|---|---:|---|---|\n",
    ]
    for rank, row in enumerate(selected, 1):
        mirrors = ", ".join(merged.get(row["url"], [])) or "—"
        lines.append(
            f"| {rank} | {row['url']} | {row['utility_score']} | "
            f"{row['source']}:{row['source_line']} | {mirrors} |\n"
        )
    (TEMP / "scored-urls.md").write_text("".join(lines), encoding="utf-8")

    batches_dir = TEMP / "url-batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    for index in range(10):
        batch = selected[index * 8 : (index + 1) * 8]
        payload = {
            "schema_version": 1,
            "vault_tag": "brainiac-method-x-gap-a5644d",
            "batch_id": index + 1,
            "urls": [
                {"url": row["url"], "utility_score": row["utility_score"]}
                for row in batch
            ],
            "full_text_review_claimed": False,
        }
        (batches_dir / f"batch-{index + 1:02d}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    manifest = {
        "schema_version": 1,
        "status": "CANDIDATE_QUEUE_READY",
        "candidate_rows": len(parse_rows()),
        "exact_unique_urls": len(exact) + sum(len(value) for value in merged.values()),
        "identity_mirrors_excluded": IDENTITY_MIRRORS,
        "selected_unique_urls": len(selected),
        "batches": 10,
        "urls_per_batch": 8,
        "full_text_review_claimed": False,
    }
    (TEMP / "url-queue-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
