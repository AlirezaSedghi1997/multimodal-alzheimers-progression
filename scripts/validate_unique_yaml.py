from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival.strict_yaml import DuplicateYamlKeyError, load_unique_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject duplicate YAML mapping keys")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        value = load_unique_yaml(args.path)
    except (OSError, UnicodeDecodeError, DuplicateYamlKeyError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": "PASS", "top_level_type": type(value).__name__}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
