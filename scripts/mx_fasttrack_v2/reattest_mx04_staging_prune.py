"""Re-attest the completed MX-04 staging-prune plan inside the re-preregistered tree.

Why a separate step is required
-------------------------------
``prune_completed_task_staging`` is single-use.  When no plan exists it *requires*
the C-drive checkpoint payloads to still be present, because its job is to record
them into the plan and then delete them.  The 20260923 tree already consumed those
payloads, so the runner cannot regenerate the plan in a second tree; it aborts with
``STAGING_PRUNE_C_FOLD_MISSING``.

What this tool does
-------------------
It transfers the plan, not the payloads.  For every fold it rewrites the three
project-root-relative paths from the old output tree to the new one and then
*re-verifies each recorded digest against the imported bytes on disk*.  A verbatim
string copy would be worthless; the point is that the H-mirror and checkpoint
digests are re-proved against this tree, so the plan states a fact about these
files rather than about the old ones.

The receipt is deliberately **not** written here.  The runner regenerates it from
the plan with ``write_once``, so the accounting (``removed_file_count``,
``removed_bytes``, ``c_payloads_absent_after_verified_h_commit``) is recomputed by
the runner rather than asserted by this tool.

Refusals
--------
* a source receipt that is not COMPLETED
* a source plan whose recorded digest does not match its own receipt
* any C payload still present (that would mean the prune never completed)
* any imported H artifact whose bytes do not match the recorded digest
* any contract/metrics digest disagreement with the imported tree
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_OUT = Path("results/final_results/method_x_fasttrack_v2_20260923")
TARGET_OUT = Path("results/final_results/method_x_fasttrack_v2_20260926_mx04_import")
TASKS = ("mx04_scratch", "mx04_head_last")
STAGING_ROOT = Path(r"C:\Users\USER\AppData\Local\AMUSE-X\checkpoint_staging")
PATH_KEYS = ("final_checkpoint", "mirror", "mirror_state")
HASH_KEYS = ("final_checkpoint_sha256", "mirror_sha256", "mirror_state_sha256")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def refuse(reason: str) -> None:
    raise SystemExit(f"STAGING_PRUNE_REATTEST_REFUSED:{reason}")


def reattest(task_id: str, source_out: Path, target_out: Path) -> dict:
    plan_name, receipt_name = f"staging_prune_{task_id}_plan.json", f"staging_prune_{task_id}.json"
    source_plan_path = source_out / "reports" / plan_name
    source_receipt_path = source_out / "reports" / receipt_name
    for path in (source_plan_path, source_receipt_path):
        if not path.is_file():
            refuse(f"SOURCE_EVIDENCE_MISSING:{path.name}")

    source_receipt = read(source_receipt_path)
    if (source_receipt.get("status") != "COMPLETED"
        or source_receipt.get("task_id") != task_id
        or source_receipt.get("c_payloads_absent_after_verified_h_commit") is not True
        or source_receipt.get("preexisting_staging_outside_task_preserved") is not True):
        refuse(f"SOURCE_PRUNE_NOT_COMPLETED:{task_id}")
    if source_receipt.get("plan_sha256") != sha(source_plan_path):
        refuse(f"SOURCE_RECEIPT_PLAN_DIGEST_MISMATCH:{task_id}")

    plan = read(source_plan_path)
    c_files = plan.get("c_files") or []
    if not isinstance(c_files, list) or not c_files:
        refuse(f"SOURCE_PLAN_C_FILES_INVALID:{task_id}")
    # The prune must genuinely have happened: none of the recorded C payloads may
    # still exist, in either tree.
    for row in c_files:
        relative = Path(str(row.get("relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            refuse(f"SOURCE_PLAN_C_PATH_UNSAFE:{task_id}")
        if (STAGING_ROOT / relative).exists():
            refuse(f"C_PAYLOAD_STILL_PRESENT:{task_id}:{relative.as_posix()}")

    task_dir = target_out / "inner_tasks" / task_id
    contract = read(task_dir / "run_contract.json")
    digest = str(contract.get("run_contract_digest") or "")
    if len(digest) != 64 or (contract.get("run_contract") or {}).get("digest") != digest:
        refuse(f"IMPORTED_RUN_CONTRACT_DIGEST_INVALID:{task_id}")
    if plan.get("run_contract_digest") != digest:
        refuse(f"PLAN_CONTRACT_DIGEST_DISAGREES_WITH_IMPORT:{task_id}")
    if plan.get("metrics_sha256") != sha(task_dir / "three_d_metrics.json"):
        refuse(f"PLAN_METRICS_DIGEST_DISAGREES_WITH_IMPORT:{task_id}")
    if plan.get("task_id") != task_id or plan.get("schema_version") != 1:
        refuse(f"SOURCE_PLAN_HEADER_INVALID:{task_id}")

    rewritten, reverified = [], []
    # Plan paths are project-root-relative POSIX strings (Path.as_posix()), so the
    # prefixes being rewritten must be relative too, not the resolved absolutes.
    source_prefix = (SOURCE_OUT / "inner_tasks" / task_id).as_posix()
    target_prefix = (TARGET_OUT / "inner_tasks" / task_id).as_posix()
    for row in plan.get("h_folds") or []:
        if set(PATH_KEYS + HASH_KEYS) - set(row):
            refuse(f"SOURCE_PLAN_H_FOLD_KEYS_INVALID:{task_id}:{row.get('fold')}")
        new_row = dict(row)
        for key in PATH_KEYS:
            original = str(row[key]).replace("\\", "/")
            if source_prefix not in original:
                refuse(f"SOURCE_PLAN_PATH_NOT_IN_SOURCE_TREE:{task_id}:{key}:{original}")
            new_row[key] = original.replace(source_prefix, target_prefix)
            resolved = (ROOT / new_row[key]).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.is_file():
                refuse(f"REWRITTEN_PATH_MISSING:{task_id}:{key}:{new_row[key]}")
        for key in HASH_KEYS:
            path_key = key[: -len("_sha256")]
            resolved = (ROOT / Path(new_row[path_key])).resolve()
            actual = sha(resolved)
            if actual != row[key]:
                refuse(f"IMPORTED_ARTIFACT_DIGEST_MISMATCH:{task_id}:{row.get('fold')}:{path_key}")
            reverified.append({"fold": row.get("fold"), "artifact": path_key,
                               "sha256": actual, "rewritten_path": new_row[path_key]})
        rewritten.append(new_row)
    if len(rewritten) != int(plan.get("expected_folds", 0)):
        refuse(f"SOURCE_PLAN_H_FOLD_COUNT_MISMATCH:{task_id}")

    new_plan = dict(plan)
    new_plan["h_folds"] = rewritten
    new_plan["scope"] = str(plan.get("scope") or "")
    # The C payloads recorded in this plan were deleted in the earlier tree, not here.
    # Recording that in the immutable plan lets the runner's receipt state the truth
    # while staying byte-stable across re-invocations.
    new_plan["c_payload_removal_location"] = (
        f"source_tree:{source_out.relative_to(ROOT).as_posix()}")
    new_plan["reattested_from_earlier_tree"] = True
    target_plan_path = target_out / "reports" / plan_name
    if target_plan_path.exists():
        refuse(f"TARGET_PLAN_ALREADY_EXISTS:{plan_name}")
    target_plan_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_plan_path.with_suffix(target_plan_path.suffix + ".new")
    tmp.write_text(json.dumps(new_plan, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target_plan_path)
    return {
        "task_id": task_id,
        "source_plan_sha256": sha(source_plan_path),
        "source_receipt_sha256": sha(source_receipt_path),
        "source_removed_file_count": source_receipt.get("removed_file_count"),
        "source_removed_bytes": source_receipt.get("removed_bytes"),
        "target_plan_sha256": sha(target_plan_path),
        "h_folds_reverified_against_imported_bytes": reverified,
        "c_payloads_confirmed_absent": len(c_files),
        "receipt_deliberately_not_written": "run.py regenerates it from this plan via write_once",
    }


def main() -> int:
    source_out = (ROOT / SOURCE_OUT).resolve()
    target_out = (ROOT / TARGET_OUT).resolve()
    print("[1/2] re-attesting staging-prune plans with digest re-verification", flush=True)
    evidence = [reattest(task_id, source_out, target_out) for task_id in TASKS]
    for row in evidence:
        print(f"      {row['task_id']}: {len(row['h_folds_reverified_against_imported_bytes'])} "
              f"H artifacts re-verified, {row['c_payloads_confirmed_absent']} C payloads confirmed absent", flush=True)

    print("[2/2] writing the re-attestation receipt", flush=True)
    receipt = {
        "schema_version": 1,
        "receipt_id": "method_x_fasttrack_v2_20260926_mx04_staging_prune_reattest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "STAGING_PRUNE_REATTESTED",
        "append_only": True,
        "outer_test_access_count": 0,
        "reason": (
            "prune_completed_task_staging is single-use: with no plan it requires the C "
            "checkpoint payloads to still exist so it can record and delete them. The 20260923 "
            "tree already consumed them, so the plan had to be transferred rather than "
            "regenerated. Every recorded H digest was re-verified against the imported bytes; "
            "no path string was trusted."
        ),
        "correction_to_import_manifest": (
            "import_mx04.py originally stated the runner would regenerate these plans. That was "
            "wrong: the runner cannot, for the single-use reason above. This receipt supersedes "
            "that claim and the manifest's deliberately_not_copied note."
        ),
        "scientific_changes": [],
        "tasks": evidence,
    }
    path = target_out / "reports" / "mx04_staging_prune_reattest.v1.json"
    if path.exists():
        refuse("REATTEST_RECEIPT_ALREADY_EXISTS")
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "receipt": path.relative_to(ROOT).as_posix(),
                      "receipt_sha256": sha(path)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(3)
