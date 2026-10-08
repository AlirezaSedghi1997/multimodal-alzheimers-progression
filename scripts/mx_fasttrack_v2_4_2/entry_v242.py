"""Run the v2.4.1 locked evaluator with proof-bound checkpoint projection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mx_fasttrack_v2_4_1 import entry_v241
from scripts.mx_fasttrack_v2_4_2.checkpoint_projection import evaluator_checkpoint_projection


def main() -> int:
    # Parse only the project/output boundary here; v2.4.1 validates all mode
    # options and preserves the original evaluator's command-line contract.
    arguments = sys.argv[1:]
    if ("--project-root" not in arguments or "--output-root" not in arguments
        or arguments.count("--project-root") != 1 or arguments.count("--output-root") != 1):
        raise RuntimeError("V242_PROJECT_AND_OUTPUT_ROOT_REQUIRED")
    if (arguments.index("--project-root") + 1 >= len(arguments)
        or arguments.index("--output-root") + 1 >= len(arguments)):
        raise RuntimeError("V242_PROJECT_AND_OUTPUT_ROOT_VALUE_MISSING")
    root = Path(arguments[arguments.index("--project-root") + 1]).resolve()
    output_relative = Path(arguments[arguments.index("--output-root") + 1])
    out = (root / output_relative).resolve()
    if root != ROOT.resolve() or not out.is_relative_to(root) or out == root:
        raise RuntimeError("V242_PROJECT_OR_OUTPUT_ROOT_INVALID")
    from scripts.mx_fasttrack_v2_4_2.supervise_v242 import ensure_v242_amendment
    ensure_v242_amendment(root, out, create=False)
    with evaluator_checkpoint_projection(root, out):
        return entry_v241.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
