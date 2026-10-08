"""Project only the imported baseline's contract digest in evaluator memory.

The historical checkpoint keeps its old digest by design.  The five-fold
source-only migration receipt and byte hashes prove equivalence to the new
contract.  Calls from migration/lock verifiers always see original bytes.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

from scripts.mx_fasttrack_v2.run import read, sha
from src.training.method_x_resume import _migration_validate_completed_receipt

ALLOWED_CALLERS = {
    "evaluate_hybrid_pair.py", "evaluate_locked.py", "xai_runner.py",
}


def baseline_binding(root: Path, out: Path) -> dict:
    directory = out / "inner_tasks/hybrid_baseline"
    import_receipt = read(out / "reports/mx05_resume_import.v1.json")
    lock = read(out / "reports/fasttrack_inner_lock.json")
    old = import_receipt.get("baseline_historical_run_contract_digest")
    new = (lock.get("tasks") or {}).get("hybrid_baseline", {}).get("run_contract_digest")
    if (import_receipt.get("status") != "IMPORTED_UNDER_HASH_PROOF"
        or not isinstance(old, str) or not isinstance(new, str)
        or len(old) != 64 or len(new) != 64 or old == new):
        raise RuntimeError("V242_SOURCE_ONLY_DIGEST_PAIR_INVALID")
    expected_checkpoints = lock["tasks"]["hybrid_baseline"].get("checkpoint_sha256")
    if not isinstance(expected_checkpoints, list) or len(expected_checkpoints) != 5:
        raise RuntimeError("V242_BASELINE_LOCK_CHECKPOINTS_INVALID")
    migration_paths = set()
    paths = {}
    for fold in range(5):
        state = read(directory / f"method_x_3d_fold_{fold}_state.json")
        details = state.get("details") or {}
        reference = details.get("source_only_migration_receipt") or {}
        relative = Path(str(reference.get("path") or ""))
        receipt = (directory / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts
            or not receipt.is_relative_to((directory / "source_only_contract_migrations").resolve())
            or not receipt.is_file() or reference.get("sha256") != sha(receipt)
            or state.get("stage") != "FINALIZED" or state.get("contract_digest") != new
            or details.get("outer_test_access_count") != 0):
            raise RuntimeError(f"V242_BASELINE_MIGRATION_STATE_INVALID:{fold}")
        migration_paths.add(receipt)
        checkpoint = (directory / f"method_x_3d_fold_{fold}.pt").resolve()
        if (sha(checkpoint) != expected_checkpoints[fold]
            or details.get("checkpoint_sha256") != expected_checkpoints[fold]
            or import_receipt["baseline_checkpoint_sha256"].get(str(fold)) != expected_checkpoints[fold]):
            raise RuntimeError(f"V242_BASELINE_CHECKPOINT_HASH_INVALID:{fold}")
        paths[checkpoint] = fold
    if len(migration_paths) != 1:
        raise RuntimeError("V242_BASELINE_MIGRATION_RECEIPT_NOT_SHARED")
    migration = _migration_validate_completed_receipt(
        next(iter(migration_paths)), old_digest=old, new_digest=new)
    if (migration.get("outer_test_touched") is not False
        or migration.get("outer_loader_invocation_count") != 0
        or {row.get("fold") for row in migration.get("folds") or []} != set(range(5))):
        raise RuntimeError("V242_BASELINE_MIGRATION_SCOPE_INVALID")
    for row in migration["folds"]:
        fold = row["fold"]
        if (row.get("artifact_contract_digest") != old
            or row.get("post_checkpoint_sha256") != expected_checkpoints[fold]
            or row.get("split_membership_digest") != lock["tasks"]["hybrid_baseline"]["fold_membership_digest"][fold]):
            raise RuntimeError(f"V242_BASELINE_MIGRATION_FOLD_INVALID:{fold}")
    return {"old_digest": old, "new_digest": new, "paths": paths,
            "migration_receipt_sha256": sha(next(iter(migration_paths)))}


@contextmanager
def evaluator_checkpoint_projection(root: Path, out: Path):
    import torch

    binding = baseline_binding(root, out)
    original = torch.load

    def projected(path, *args, **kwargs):
        resolved = Path(path).resolve() if isinstance(path, (str, Path)) else None
        fold = binding["paths"].get(resolved)
        caller = Path(sys._getframe(1).f_code.co_filename).name
        payload = original(path, *args, **kwargs)
        if fold is None or caller not in ALLOWED_CALLERS:
            return payload
        if (not isinstance(payload, dict)
            or payload.get("fold") != fold
            or payload.get("run_contract_digest") != binding["old_digest"]
            or payload.get("hpo_selection_only") is not True
            or payload.get("outer_test_access_count") != 0
            or payload.get("outer_test_evaluation_count") != 0):
            raise RuntimeError(f"V242_HISTORICAL_CHECKPOINT_IDENTITY_INVALID:{fold}")
        viewed = dict(payload)
        viewed["run_contract_digest"] = binding["new_digest"]
        return viewed

    torch.load = projected
    try:
        yield binding
    finally:
        torch.load = original
