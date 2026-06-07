from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_GOVERNANCE = Path("results/governance")
PLACEHOLDER = "[INSERT PRESPECIFIED CUTOFF BEFORE MODELING]"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Block modeling until Phase 0 governance is ready.")
    parser.add_argument("--governance_dir", default=str(DEFAULT_GOVERNANCE))
    parser.add_argument(
        "--require_label_freeze",
        action="store_true",
        help="Require results/governance/label_freeze.json with a real cutoff.",
    )
    parser.add_argument(
        "--require_feature_matrix_check",
        action="store_true",
        help="Require PASS method_x_feature_matrix_pet_leakage_check.json.",
    )
    args = parser.parse_args()

    gov = Path(args.governance_dir)
    required = {
        "phase0_governance_manifest": gov / "phase0_governance_manifest.json",
        "cohort_probe_protocol": gov / "cohort_probe_protocol.json",
        "deconfounding_scope": gov / "deconfounding_scope.json",
        "feature_allowlist_summary": gov / "feature_allowlist_summary.json",
        "label_freeze_template": gov / "label_freeze_template.json",
    }
    status = {"schema_version": 1, "status": "PASS", "missing": [], "failures": []}
    for name, path in required.items():
        if not path.exists():
            status["missing"].append({"artifact": name, "path": str(path)})
    if args.require_label_freeze:
        freeze_path = gov / "label_freeze.json"
        if not freeze_path.exists():
            status["missing"].append({"artifact": "label_freeze", "path": str(freeze_path)})
        else:
            freeze = load_json(freeze_path)
            if freeze.get("status") != "FROZEN_BEFORE_MODELING":
                status["failures"].append("label_freeze status is not FROZEN_BEFORE_MODELING")
            if freeze.get("centiloid_threshold") in {None, "", PLACEHOLDER}:
                status["failures"].append("centiloid cutoff is not frozen")
    if args.require_feature_matrix_check:
        check_path = gov / "method_x_feature_matrix_pet_leakage_check.json"
        if not check_path.exists():
            status["missing"].append({"artifact": "method_x_feature_matrix_pet_leakage_check", "path": str(check_path)})
        else:
            check = load_json(check_path)
            if check.get("status") != "PASS":
                status["failures"].append("Method X feature matrix leakage check did not PASS")
    if status["missing"] or status["failures"]:
        status["status"] = "BLOCKED"
    print(json.dumps(status, indent=2))
    return 0 if status["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
