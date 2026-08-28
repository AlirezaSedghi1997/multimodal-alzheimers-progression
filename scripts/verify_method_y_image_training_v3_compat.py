#!/usr/bin/env python3
"""Verify a finalized Method Y image run after Git-clean EOL normalization.

The v3.2 run contract predates canonical text hashing and therefore binds raw
working-tree bytes.  On Windows, a later clean checkout can rewrite LF/CRLF
bytes without changing the executable Git content.  This verifier permits
only that narrow case: the saved source tree must have been Git-clean, the
current Git-canonical source must exactly match the saved commit, and every
runtime/math field must remain identical.  It then runs the original final
verifier with the saved software contract injected into its config.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.method_y_revival.atomic_io import atomic_write_json, read_json  # noqa: E402
from src.method_y_revival.contracts import sha256_mapping  # noqa: E402
from src.method_y_revival.software_contract_v3 import (  # noqa: E402
    SOURCE_PATHS,
    bind_software_runtime_contract,
)


class CanonicalContractCompatibilityError(RuntimeError):
    """Raised when raw-byte drift cannot be proven semantically harmless."""


def _require_self_hash(contract: dict[str, Any], label: str) -> None:
    body = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if contract.get("contract_sha256") != sha256_mapping(body):
        raise CanonicalContractCompatibilityError(f"{label}_CONTRACT_SELF_HASH_MISMATCH")


def _runtime_and_math_payload(contract: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "contract_sha256",
        "git_commit",
        "scoped_dirty_digest",
        "training_source",
    }
    return {key: copy.deepcopy(value) for key, value in contract.items() if key not in excluded}


def _run_git(project_root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
    )


def verify_canonical_source_compatibility(
    *,
    project_root: str | Path,
    saved_contract: dict[str, Any],
    current_contract: dict[str, Any],
    source_paths: Iterable[str] = SOURCE_PATHS,
) -> dict[str, Any]:
    """Prove that raw source-hash drift is only Git-canonical representation.

    This deliberately refuses compatibility when the saved source tree was
    dirty: without the original bytes, a dirty working tree cannot be proven
    equivalent to the saved commit.
    """

    root = Path(project_root).resolve()
    paths = tuple(sorted(set(str(path).replace("\\", "/") for path in source_paths)))
    _require_self_hash(saved_contract, "SAVED_SOFTWARE")
    _require_self_hash(current_contract, "CURRENT_SOFTWARE")

    empty_dirty_digest = hashlib.sha256(b"").hexdigest()
    if saved_contract.get("scoped_dirty_digest") != empty_dirty_digest:
        raise CanonicalContractCompatibilityError("SAVED_SOURCE_WAS_NOT_GIT_CLEAN")

    if _runtime_and_math_payload(saved_contract) != _runtime_and_math_payload(current_contract):
        raise CanonicalContractCompatibilityError("RUNTIME_OR_MATH_POLICY_DRIFT")

    saved_source = saved_contract.get("training_source")
    current_source = current_contract.get("training_source")
    if not isinstance(saved_source, dict) or not isinstance(current_source, dict):
        raise CanonicalContractCompatibilityError("TRAINING_SOURCE_CONTRACT_MISSING")
    saved_files = saved_source.get("files")
    current_files = current_source.get("files")
    if not isinstance(saved_files, dict) or not isinstance(current_files, dict):
        raise CanonicalContractCompatibilityError("TRAINING_SOURCE_FILE_HASHES_MISSING")
    if set(saved_files) != set(paths) or set(current_files) != set(paths):
        raise CanonicalContractCompatibilityError("TRAINING_SOURCE_PATH_SET_MISMATCH")
    if any(not (root / path).is_file() for path in paths):
        raise CanonicalContractCompatibilityError("CURRENT_TRAINING_SOURCE_MISSING")

    saved_commit = str(saved_contract.get("git_commit", ""))
    live_commit = str(current_contract.get("git_commit", ""))
    if len(saved_commit) != 40 or any(ch not in "0123456789abcdef" for ch in saved_commit.lower()):
        raise CanonicalContractCompatibilityError("SAVED_GIT_COMMIT_INVALID")
    commit_check = _run_git(root, "cat-file", "-e", f"{saved_commit}^{{commit}}")
    if commit_check.returncode != 0:
        raise CanonicalContractCompatibilityError("SAVED_GIT_COMMIT_UNAVAILABLE")

    canonical_diff = _run_git(root, "diff", "--quiet", saved_commit, "--", *paths)
    if canonical_diff.returncode == 1:
        raise CanonicalContractCompatibilityError("CANONICAL_SOURCE_DRIFT")
    if canonical_diff.returncode != 0:
        raise CanonicalContractCompatibilityError("CANONICAL_SOURCE_COMPARISON_FAILED")

    raw_hash_mismatch_paths = sorted(
        path for path in paths if saved_files.get(path) != current_files.get(path)
    )
    if not raw_hash_mismatch_paths:
        mode = "EXACT_SOURCE_BYTES"
    else:
        mode = "GIT_CANONICAL_SOURCE_EQUIVALENCE"
    return {
        "status": "PASS",
        "mode": mode,
        "saved_commit": saved_commit,
        "live_commit": live_commit,
        "saved_source_tree_was_clean": True,
        "git_canonical_diff_exit_code": 0,
        "source_path_count": len(paths),
        "raw_hash_mismatch_paths": raw_hash_mismatch_paths,
        "runtime_and_math_policy_match": True,
    }


def _saved_contract_binder(saved_contract: dict[str, Any]):
    def bind(config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
        del project_root
        bound = copy.deepcopy(config)
        bound["training_source_sha256"] = saved_contract["training_source"][
            "bundle_sha256"
        ]
        bound["software_runtime_contract"] = copy.deepcopy(saved_contract)
        return bound

    return bind


def verify_final_with_canonical_compatibility(
    *, config_path: Path, output_root: Path
) -> dict[str, Any]:
    import scripts.verify_method_y_image_training_v3 as official_verifier

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    current_bound = bind_software_runtime_contract(raw_config, ROOT)
    run_contract = read_json(output_root / "run_contract.json")
    payload = run_contract.get("payload")
    if not isinstance(payload, dict) or run_contract.get("digest") != sha256_mapping(payload):
        raise CanonicalContractCompatibilityError("RUN_CONTRACT_SELF_HASH_MISMATCH")
    saved_contract = payload.get("software_runtime_contract")
    if not isinstance(saved_contract, dict):
        raise CanonicalContractCompatibilityError("SAVED_SOFTWARE_CONTRACT_MISSING")

    compatibility = verify_canonical_source_compatibility(
        project_root=ROOT,
        saved_contract=saved_contract,
        current_contract=current_bound["software_runtime_contract"],
    )
    original_binder = official_verifier.bind_software_runtime_contract
    try:
        official_verifier.bind_software_runtime_contract = _saved_contract_binder(
            saved_contract
        )
        verification = official_verifier.verify_final(config_path, output_root)
    finally:
        official_verifier.bind_software_runtime_contract = original_binder
    if verification.get("status") != "PASS":
        raise CanonicalContractCompatibilityError("OFFICIAL_FINAL_VERIFICATION_DID_NOT_PASS")
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "verification_scope": "FINAL_METHOD_Y_IMAGE_V3_2",
        "compatibility": compatibility,
        "official_verifier_result": verification,
        "run_contract_sha256": run_contract["digest"],
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed Method Y final verifier for Git-clean EOL drift"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "method_y_image_training_v3_2.yaml"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results" / "method_y_revival_v3_image_v2"),
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    try:
        result = verify_final_with_canonical_compatibility(
            config_path=config_path,
            output_root=output_root,
        )
        receipt = output_root / "provenance" / "FINAL_VERIFICATION_CANONICAL_COMPAT.json"
        atomic_write_json(receipt, result)
        print(json.dumps({**result, "receipt": str(receipt)}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "verification_scope": "FINAL_METHOD_Y_IMAGE_V3_2",
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
