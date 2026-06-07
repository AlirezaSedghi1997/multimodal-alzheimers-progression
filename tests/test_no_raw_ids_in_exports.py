from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SAFE_DIRS_TO_SCAN = ["results", "public_bundle", "docs"]
FORBIDDEN_ID_PATTERNS = [
    r"\bB\d{5,}\b",
    r"\bPREVBID\b",
    r"\bLRN_?\d{4,}\b",
]
SKIP_PARTS = {"baseline_snapshots"}  # snapshots may intentionally preserve prior aggregate reports; review separately.


def test_no_raw_ids_in_exportable_text_artifacts():
    for root_name in SAFE_DIRS_TO_SCAN:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            if path.suffix.lower() not in {".md", ".json", ".csv", ".txt", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pat in FORBIDDEN_ID_PATTERNS:
                assert not re.search(pat, text), f"Possible raw ID in {path}: pattern={pat}"
