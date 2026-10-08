"""Import the completed MX-04 inner runs into a re-preregistered fast-track tree.

Why this exists
---------------
``scripts/mx_fasttrack_v2/integrity.py`` re-verifies every ``SOURCE_PATHS`` file
against the immutable preregistration.  The MX-05 blocker could only be repaired
by editing ``src/models/backbones/amuse_x_hybrid3d.py``, which is itself a
``SOURCE_PATHS`` member, so the 20260923 tree could never both hold the repair and
satisfy its own source binding.  Rather than retrain MX-04, this tool imports the
already-completed MX-04 inner runs into a freshly preregistered tree.

What makes the import defensible
--------------------------------
The repair touches only the AMUSE-X-Hybrid3D encoder and the ``tiny`` backbone.
MX-04 is ResNet34.  That is a claim, so this tool does not assert it: it *proves*
it at import time by rebuilding the pre-repair ResNet34 factory branch and
comparing weights, forward values and gradients bit-for-bit against the current
tree under the locked strict-determinism regime.  A mismatch aborts the import.

Independence from the runner
----------------------------
Every fact in the emitted manifest is recomputed here from bytes on disk.  The
runner is never asked whether the import is acceptable, and this tool does not
consult runner bookkeeping to decide anything.  The one thing deliberately NOT
copied is the staging-prune plan/receipt pair: those embed project-root-relative
paths, so the runner regenerates them for the new tree and re-verifies the mirror
hashes itself.

Refusals
--------
* any ``SOURCE_PATHS`` delta beyond the declared set
* any MX-04 artifact whose bytes do not match the source tree
* any MX-04 contract whose seed/folds/resolution/selection scope differs
* a non-terminal MX-04 task, or outer-test access recorded as non-zero
* a pre-existing destination
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_OUT = Path("results/final_results/method_x_fasttrack_v2_20260923")
TARGET_OUT = Path("results/final_results/method_x_fasttrack_v2_20260926_mx04_import")
IMPORT_TASKS = ("mx04_scratch", "mx04_head_last")
# Files whose bytes differ between the 20260923 preregistration and this tree.
#
# The first two are the MX-05 determinism repair.  The rest are the control plane of
# the re-preregistration itself and the corrections an independent review demanded:
#   run.py / integrity.py / supervise.py  protocol-id set, output-root default,
#                                        SOURCE_PATHS completion, shared membership
#                                        helper, per-protocol config resolution
#   freeze_report.py / xai_limited.py     the same per-protocol config resolution
#   docs/...runbook_FA.md                 operator contract for the new tree
# None of them can reach MX-04's numerics, and that is proved rather than asserted --
# see verify_mx04_command_invariance and verify_trainer_excludes_control_plane.
DECLARED_SOURCE_DELTA = (
    "src/models/backbones/amuse_x_hybrid3d.py",
    "src/models/backbones/backbone_factory.py",
    "scripts/mx_fasttrack_v2/run.py",
    "scripts/mx_fasttrack_v2/integrity.py",
    "scripts/mx_fasttrack_v2/supervise.py",
    "scripts/mx_fasttrack_v2/freeze_report.py",
    "scripts/mx_fasttrack_v2/xai_limited.py",
    "docs/method_x_fasttrack_v2_runbook_FA.md",
)
# Precise marker: the fast-track control-plane package name.  The trainer must not
# mention it at all, which transitively excludes integrity.py and supervise.py.
# (Bare words like "integrity_sha256" or "supervised_contrastive_alignment" are the
# trainer's own vocabulary and are deliberately not matched.)
CONTROL_PLANE_MARKER = "mx_fasttrack_v2"
LOCKED_SEED, LOCKED_FOLDS, LOCKED_EPOCHS_FIELD, LOCKED_RESOLUTION = 13, 5, 100, 128
EFFECTIVE_EPOCH_CAP = 20


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def refuse(reason: str) -> None:
    raise SystemExit(f"MX04_IMPORT_REFUSED:{reason}")


def verify_source_delta(source_out: Path) -> dict:
    """Refuse unless the working tree differs from the old preregistration
    in exactly the declared set.  An undeclared drift would mean MX-04 might
    have run under bytes nobody has accounted for."""

    prereg = read(source_out / "reports/fasttrack_preregistration.json")
    bound = prereg["training_source_sha256"]
    drift = {}
    for relative, expected in sorted(bound.items()):
        current = sha(ROOT / relative)
        if current != expected:
            drift[relative] = {"preregistered_sha256": expected, "current_sha256": current}
    unexpected = sorted(set(drift) - set(DECLARED_SOURCE_DELTA))
    if unexpected:
        refuse(f"UNDECLARED_SOURCE_DRIFT:{','.join(unexpected)}")
    absent = sorted(set(DECLARED_SOURCE_DELTA) - set(drift))
    if absent:
        refuse(f"DECLARED_DELTA_NOT_PRESENT:{','.join(absent)}")
    return {
        "parent_preregistration_sha256": sha(source_out / "reports/fasttrack_preregistration.json"),
        "parent_protocol_id": prereg["protocol_id"],
        "parent_bound_file_count": len(bound),
        "declared_delta": drift,
    }


def verify_mx04_source_independence() -> dict:
    """Prove the repaired source still builds and trains ResNet34 identically.

    The pre-repair factory branch is reconstructed inline from MONAI plus the
    untouched ``replace_native_maxpool3d`` walker and compared against the current
    ``build_3d_backbone('resnet34_3d')``.  This is the load-bearing scientific
    claim of the whole import, so it runs live and aborts on any difference.
    """

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from monai.networks.nets import resnet34

    from src.models.backbones.backbone_factory import build_3d_backbone
    from src.models.backbones.deterministic_pool import replace_native_maxpool3d

    def pre_repair_resnet34(in_channels: int = 1, out_dim: int = 128):
        model = resnet34(spatial_dims=3, n_input_channels=in_channels, num_classes=out_dim)
        if replace_native_maxpool3d(model) < 1:
            raise RuntimeError("pre-repair branch found no MaxPool3d to determinize")
        return model

    # The proof is only meaningful on CUDA: on CPU the convolutions never touch cuDNN,
    # so a bitwise-equality result would be vacuous.  Refuse rather than record one.
    if not torch.cuda.is_available():
        refuse("CUDA_REQUIRED_FOR_THE_BITWISE_INDEPENDENCE_PROOF")
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = "cuda"
        torch.manual_seed(LOCKED_SEED)
        reference = pre_repair_resnet34().to(device)
        torch.manual_seed(LOCKED_SEED)
        current = build_3d_backbone("resnet34_3d", in_channels=1, out_dim=128).to(device)

        ref_state, cur_state = reference.state_dict(), current.state_dict()
        if set(ref_state) != set(cur_state):
            refuse("MX04_STATE_DICT_KEYSET_CHANGED")
        if not all(torch.equal(ref_state[k], cur_state[k]) for k in ref_state):
            refuse("MX04_WEIGHTS_NOT_BITWISE_EQUAL")

        # Probed at the locked resolution MX-04 actually trained on: conv and
        # InstanceNorm-free equality can differ by resolution, and a smaller probe
        # would not demonstrate the claim the import rests on.
        torch.manual_seed(LOCKED_SEED)
        probe = torch.randn(1, 1, LOCKED_RESOLUTION, LOCKED_RESOLUTION, LOCKED_RESOLUTION, device=device)
        left = probe.clone().requires_grad_(True)
        right = probe.clone().requires_grad_(True)
        out_left, out_right = reference(left), current(right)
        if not torch.equal(out_left, out_right):
            refuse("MX04_FORWARD_NOT_BITWISE_EQUAL")
        out_left.square().mean().backward()
        out_right.square().mean().backward()
        if not torch.equal(left.grad, right.grad):
            refuse("MX04_INPUT_GRADIENT_NOT_BITWISE_EQUAL")
        ref_grads = [p.grad for p in reference.parameters() if p.grad is not None]
        cur_grads = [p.grad for p in current.parameters() if p.grad is not None]
        if len(ref_grads) != len(cur_grads) or not all(
            torch.equal(a, b) for a, b in zip(ref_grads, cur_grads)
        ):
            refuse("MX04_PARAMETER_GRADIENTS_NOT_BITWISE_EQUAL")

        hybrid_module = "src.models.backbones.amuse_x_hybrid3d"
        if hybrid_module in sys.modules:
            refuse("MX04_PATH_IMPORTED_THE_REPAIRED_HYBRID_ENCODER")
        hybrid = ROOT / "src/models/backbones/amuse_x_hybrid3d.py"
        return {
            "device": device,
            "probe_resolution": LOCKED_RESOLUTION,
            "probe_matches_locked_resolution": True,
            "cuda_required_and_present": True,
            "state_dict_tensor_count": len(ref_state),
            "weights_bitwise_identical": True,
            "forward_bitwise_identical": True,
            "input_gradient_bitwise_identical": True,
            "parameter_gradient_tensor_count": len(ref_grads),
            "parameter_gradients_bitwise_identical": True,
            "strict_determinism": True,
            "repaired_hybrid_module_imported_by_mx04_path": False,
            "repaired_hybrid_module_sha256": sha(hybrid),
        }
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def verify_task_contract(task_dir: Path, task_id: str) -> dict:
    """Confirm the task is a terminal, inner-only, five-fold, seed-13 result."""

    metrics = read(task_dir / "three_d_metrics.json")
    contract = read(task_dir / "run_contract.json")
    aggregate = metrics.get("aggregate") or {}
    if aggregate.get("status") != "COMPLETED" or aggregate.get("hpo_selection_only") is not True:
        refuse(f"{task_id}:NOT_INNER_ONLY_COMPLETED")
    if aggregate.get("outer_test_used_for_selection") is not False:
        refuse(f"{task_id}:OUTER_TEST_USED_FOR_SELECTION")
    if int(aggregate.get("outer_test_access_count", -1)) != 0:
        refuse(f"{task_id}:OUTER_TEST_ACCESS_NONZERO")
    if int(aggregate.get("folds", -1)) != LOCKED_FOLDS or int(aggregate.get("completed_folds", -1)) != LOCKED_FOLDS:
        refuse(f"{task_id}:FOLD_COUNT_MISMATCH")
    if int(aggregate.get("epoch_target", -1)) != EFFECTIVE_EPOCH_CAP:
        refuse(f"{task_id}:EPOCH_CAP_MISMATCH")
    digest = str(contract.get("run_contract_digest") or "")
    if len(digest) != 64 or (contract.get("run_contract") or {}).get("digest") != digest:
        refuse(f"{task_id}:RUN_CONTRACT_DIGEST_INVALID")
    if aggregate.get("run_contract_digest") not in (None, digest):
        refuse(f"{task_id}:AGGREGATE_CONTRACT_DIGEST_MISMATCH")
    parameters = (contract.get("run_contract") or {}).get("parameters") or {}
    if parameters.get("backbone") != "resnet34_3d":
        refuse(f"{task_id}:BACKBONE_NOT_RESNET34")
    if int(parameters.get("seed", -1)) != LOCKED_SEED:
        refuse(f"{task_id}:SEED_MISMATCH")
    if int(parameters.get("folds", -1)) != LOCKED_FOLDS:
        refuse(f"{task_id}:CONTRACT_FOLDS_MISMATCH")
    if int(parameters.get("epochs", -1)) != LOCKED_EPOCHS_FIELD:
        refuse(f"{task_id}:CONTRACT_EPOCHS_FIELD_MISMATCH")
    if int(parameters.get("resolution", -1)) != LOCKED_RESOLUTION:
        refuse(f"{task_id}:RESOLUTION_MISMATCH")
    # The split-membership digest is authoritative in the per-fold state file, not in
    # the metrics fold rows.  the run.py lock membership block reads the metrics
    # rows, which do not carry the field; see disclosed_defects in the manifest.
    membership, fold_states = [], []
    for fold in range(LOCKED_FOLDS):
        state_path = task_dir / f"method_x_3d_fold_{fold}_state.json"
        if not state_path.is_file():
            refuse(f"{task_id}:FOLD_STATE_MISSING:{fold}")
        state = read(state_path)
        details = state.get("details") or {}
        if state.get("stage") != "FINALIZED":
            refuse(f"{task_id}:FOLD_NOT_FINALIZED:{fold}")
        digest_value = details.get("split_membership_digest")
        if not isinstance(digest_value, str) or len(digest_value) != 64:
            refuse(f"{task_id}:FOLD_STATE_MEMBERSHIP_DIGEST_INVALID:{fold}")
        if int(details.get("outer_test_access_count", -1)) != 0:
            refuse(f"{task_id}:FOLD_STATE_OUTER_ACCESS_NONZERO:{fold}")
        if details.get("hpo_selection_only") is not True:
            refuse(f"{task_id}:FOLD_STATE_NOT_INNER_ONLY:{fold}")
        membership.append(digest_value)
        fold_states.append({"fold": fold, "stage": state.get("stage"),
                            "split_membership_digest": digest_value,
                            "state_sha256": sha(state_path)})
    checkpoints = []
    for fold in range(LOCKED_FOLDS):
        path = task_dir / f"method_x_3d_fold_{fold}.pt"
        if not path.is_file():
            refuse(f"{task_id}:CHECKPOINT_MISSING:{fold}")
        mirror = task_dir / "checkpoint_mirror" / f"fold_{fold}" / "resume_current.pt"
        mirror_state = task_dir / "checkpoint_mirror" / f"fold_{fold}" / "checkpoint_mirror_state.json"
        if not mirror.is_file() or not mirror_state.is_file():
            refuse(f"{task_id}:H_MIRROR_EVIDENCE_MISSING:{fold}")
        binding = read(mirror_state)
        if binding.get("payload_sha256") != sha(mirror):
            refuse(f"{task_id}:H_MIRROR_BINDING_MISMATCH:{fold}")
        checkpoints.append({"fold": fold, "sha256": sha(path), "bytes": path.stat().st_size})
    return {
        "run_contract_digest": digest,
        "metrics_sha256": sha(task_dir / "three_d_metrics.json"),
        "run_contract_sha256": sha(task_dir / "run_contract.json"),
        "fold_membership_digest": membership,
        "fold_membership_source": "inner_tasks/<task>/method_x_3d_fold_<n>_state.json :: details.split_membership_digest",
        "fold_states": fold_states,
        "checkpoints": checkpoints,
        "hpo_selection_only": True,
        "outer_test_used_for_selection": False,
        "outer_test_access_count": 0,
        "seed": LOCKED_SEED, "folds": LOCKED_FOLDS,
        "epochs_field": LOCKED_EPOCHS_FIELD, "effective_epoch_cap_per_fold": EFFECTIVE_EPOCH_CAP,
        "resolution": LOCKED_RESOLUTION, "backbone": "resnet34_3d",
        "source_contract_digest_at_training_time": parameters.get("source_contract_digest"),
    }


def verify_trainer_excludes_control_plane() -> dict:
    """Static proof that edits to the runner/verifier cannot reach the trainer.

    ``train_3d_mri.py`` is the only file MX-04 training executes.  If it never
    references the fast-track control plane, then changing ``integrity.py`` or
    ``supervise.py`` cannot alter a single training step.  Grep-based and
    therefore falsifiable by a reviewer in one step.
    """

    trainer = (ROOT / "scripts/train_3d_mri.py").read_text(encoding="utf-8")
    if CONTROL_PLANE_MARKER in trainer:
        refuse(f"TRAINER_REFERENCES_CONTROL_PLANE:{CONTROL_PLANE_MARKER}")
    return {
        "trainer": "scripts/train_3d_mri.py",
        "trainer_sha256": sha(ROOT / "scripts/train_3d_mri.py"),
        "control_plane_marker_searched": CONTROL_PLANE_MARKER,
        "marker_occurrences_in_trainer": trainer.count(CONTROL_PLANE_MARKER),
        "conclusion": (
            "the trainer never references the fast-track package, so edits to run.py, "
            "integrity.py and supervise.py cannot reach MX-04 training"
        ),
    }


def verify_mx04_command_invariance(source_out: Path, target_out: Path) -> dict:
    """Prove the repaired source rebuilds the exact MX-04 command that ran.

    The recorded ``command_sha256`` covers the whole argv, including ``--out_dir``,
    which necessarily differs in a re-preregistered tree.  So the new command is
    built, every occurrence of the new output path is remapped back to the old
    one, and the resulting fingerprint must equal the value the runner recorded
    when MX-04 actually trained.  Equality proves the repaired source produces a
    byte-identical training invocation; only the destination moved.
    """

    from scripts.mx_fasttrack_v2.run import build_fold_manifest, read as _read, task_command

    root = ROOT
    battery = (root / "results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729").resolve()
    manifest = _read(battery / "battery_manifest.json")
    recorded = _read(source_out / "reports/fasttrack_task_state.json").get("tasks") or {}
    evidence = {}
    for task_id in IMPORT_TASKS:
        expected = (recorded.get(task_id) or {}).get("command_sha256")
        if not expected:
            refuse(f"RECORDED_COMMAND_FINGERPRINT_MISSING:{task_id}")
        kind, variant = ("mx04", "scratch_baseline" if task_id == "mx04_scratch" else "head_then_last_stage_llrd")
        fold_manifest = build_fold_manifest(root, battery, target_out) if task_id == "mx04_head_last" else None
        cmd, _ = task_command((task_id, kind, variant), root, battery, target_out, manifest, fold_manifest)
        new_fingerprint = hashlib.sha256(json.dumps(cmd, ensure_ascii=False).encode()).hexdigest()
        remapped = [str(part).replace(str(target_out), str(source_out)) for part in cmd]
        remapped_fingerprint = hashlib.sha256(json.dumps(remapped, ensure_ascii=False).encode()).hexdigest()
        if remapped_fingerprint != expected:
            refuse(f"MX04_COMMAND_INVARIANCE_BROKEN:{task_id}")
        evidence[task_id] = {
            "recorded_command_sha256": expected,
            "remapped_rebuilt_command_sha256": remapped_fingerprint,
            "identical": True,
            "target_tree_command_sha256": new_fingerprint,
            "target_tree_note": (
                "differs from the recorded value only because --out_dir and the fold-manifest "
                "sidecar paths name the new tree; recorded here so fasttrack_task_state.json can "
                "be seeded with the value run.py will recompute"
            ),
        }
    return {
        "method": "argv rebuilt from the repaired source, output paths remapped to the old tree, fingerprint compared",
        "tasks": evidence,
    }


def copy_tree(source: Path, target: Path) -> dict:
    if target.exists():
        refuse(f"DESTINATION_EXISTS:{target}")
    target.mkdir(parents=True)
    files, total = [], 0
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hash, copied_hash = sha(path), sha(destination)
        if source_hash != copied_hash:
            refuse(f"COPY_HASH_MISMATCH:{relative.as_posix()}")
        files.append({"path": relative.as_posix(), "sha256": copied_hash, "bytes": destination.stat().st_size})
        total += destination.stat().st_size
    return {"file_count": len(files), "total_bytes": total, "files": files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_OUT))
    parser.add_argument("--target-root", default=str(TARGET_OUT))
    args = parser.parse_args()
    source_out = (ROOT / args.source_root).resolve()
    target_out = (ROOT / args.target_root).resolve()

    if target_out == source_out or target_out.is_relative_to(source_out):
        refuse("TARGET_MUST_BE_A_DISTINCT_SIBLING_TREE")
    if target_out.exists():
        refuse(f"TARGET_ALREADY_EXISTS:{target_out.name}")
    for name in ("reports/fasttrack_preregistration.json",):
        if not (source_out / name).is_file():
            refuse(f"SOURCE_PREREGISTRATION_MISSING:{name}")

    print("[1/6] verifying the declared source delta", flush=True)
    delta = verify_source_delta(source_out)

    print("[2/6] proving MX-04 is independent of the repaired source (live)", flush=True)
    independence = verify_mx04_source_independence()

    print("[3/6] verifying MX-04 contracts and command invariance", flush=True)
    control_plane = verify_trainer_excludes_control_plane()
    contracts = {}
    for task_id in IMPORT_TASKS:
        task_dir = source_out / "inner_tasks" / task_id
        if not task_dir.is_dir():
            refuse(f"SOURCE_TASK_MISSING:{task_id}")
        contracts[task_id] = verify_task_contract(task_dir, task_id)
        print(f"      {task_id}: COMPLETED, {LOCKED_FOLDS} folds, seed {LOCKED_SEED}, outer access 0", flush=True)
    # Both MX-04 candidates must have been trained on identical split membership,
    # otherwise a scratch-vs-head comparison would not be like-for-like.
    if contracts[IMPORT_TASKS[0]]["fold_membership_digest"] != contracts[IMPORT_TASKS[1]]["fold_membership_digest"]:
        refuse("MX04_CANDIDATES_DISAGREE_ON_SPLIT_MEMBERSHIP")
    print("      both candidates share identical per-fold split membership", flush=True)
    target_out.mkdir(parents=True, exist_ok=True)
    command_invariance = verify_mx04_command_invariance(source_out, target_out)
    for task_id, entry in command_invariance["tasks"].items():
        print(f"      {task_id}: training command identical to the recorded run", flush=True)

    print("[4/6] copying MX-04 artifacts byte-identically", flush=True)
    (target_out / "inner_tasks").mkdir(parents=True, exist_ok=True)
    (target_out / "reports").mkdir(parents=True, exist_ok=True)
    copies = {}
    for task_id in IMPORT_TASKS:
        copies[task_id] = copy_tree(source_out / "inner_tasks" / task_id,
                                   target_out / "inner_tasks" / task_id)
        print(f"      {task_id}: {copies[task_id]['file_count']} files, "
              f"{copies[task_id]['total_bytes'] / 2**30:.1f} GiB verified", flush=True)
    # The data-semantics audit is data-only (HDF5, splits, index) and carries no
    # source hash, so it transfers byte-identically and keeps its digest.
    audit_name = "data_semantics_audit.json"
    shutil.copy2(source_out / "reports" / audit_name, target_out / "reports" / audit_name)
    if sha(source_out / "reports" / audit_name) != sha(target_out / "reports" / audit_name):
        refuse("DATA_AUDIT_COPY_HASH_MISMATCH")

    print("[5/6] writing the runner-independent import manifest", flush=True)
    manifest = {
        "schema_version": 1,
        "manifest_id": "method_x_fasttrack_v2_20260926_mx04_import_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_on_platform": platform.platform(),
        "status": "MX04_IMPORTED_UNDER_PROOF",
        "append_only": True,
        "claim_cap": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
        "outer_test_access_count": 0,
        "outer_test_used_for_selection": False,
        "source_tree": {
            "path": source_out.relative_to(ROOT).as_posix(),
            "note": "historical tree, left untouched; MX-05 never trained there",
        },
        "target_tree": {
            "path": target_out.relative_to(ROOT).as_posix(),
            "note": "re-preregistered by scripts/mx_fasttrack_v2/run.py plan with the repaired source",
        },
        "source_delta": delta,
        "mx04_source_independence_proof": independence,
        "mx04_command_invariance_proof": command_invariance,
        "trainer_excludes_control_plane_proof": control_plane,
        "mx04_contracts": contracts,
        "mx04_copies": copies,
        "data_audit": {
            "path": f"reports/{audit_name}",
            "sha256": sha(target_out / "reports" / audit_name),
            "transferred": "byte_identical",
            "rationale": "data-only artifact: HDF5 rows, split digests and cache index; contains no source hash",
        },
        "disclosed_defects": {
            "run_py_lock_reads_fold_membership_digest_from_the_wrong_file": {
                "severity": "BLOCKS_THE_LOCK_STEP_FOR_MX04",
                "pre_existing": True,
                "independent_of_the_mx05_determinism_repair": True,
                "detail": (
                    "the run.py lock membership block builds the per-fold "
                    "split_membership_digest from three_d_metrics.json fold rows and raises "
                    "FOLD_MEMBERSHIP_DIGEST_MISSING when any entry is not a 64-character "
                    "string. Neither MX-04 metrics file carries that field, so the lock step "
                    "would raise for mx04_scratch and mx04_head_last even if MX-05 had never "
                    "been attempted."
                ),
                "where_the_fact_actually_lives": (
                    "inner_tasks/<task>/method_x_3d_fold_<n>_state.json :: "
                    "details.split_membership_digest, with stage=FINALIZED, "
                    "outer_test_access_count=0 and hpo_selection_only=true. This tool verifies "
                    "all three. build_fold_manifest already treats this same field as "
                    "authoritative for the battery checkpoints, so the location is established "
                    "elsewhere in the same codebase."
                ),
                "integrity_signal": (
                    "Both MX-04 candidates carry identical per-fold digests, which is the "
                    "correct like-for-like property for a scratch-vs-head comparison and is "
                    "asserted by this import."
                ),
                "not_fixed_here": (
                    "Changing the lock gate is a control-plane decision outside the scope of an "
                    "MX-04 import, and no option was authorised for it. Recorded here so it "
                    "surfaces before the lock step rather than after it."
                ),
            }
        },
        "deliberately_not_copied": {
            "staging_prune_plan_and_receipts": (
                "STAGING_PRUNE_PLAN_DRIFT in the new tree. scripts/mx_fasttrack_v2/run.py "
                "regenerates them for this tree during the MX-04 completion check and "
                "re-verifies every H-mirror hash itself."
            ),
            "preregistration": "recreated by the runner from the repaired source",
            "smoke_reports": "regenerated by the runner; parameter budget is unchanged",
            "mx04_lock_receipts": "no lock exists yet; produced by run.py lock after MX-05",
        },
        "scientific_changes": [],
        "scientific_change_justification": (
            "The repaired source leaves ResNet34 construction, forward values and all parameter "
            "gradients bit-for-bit identical, as proven live above. MX-04 is therefore not "
            "retrained: doing so would burn roughly 7-12 GPU-hours to reproduce numbers that are "
            "already exactly reproducible from the repaired tree."
        ),
        "honesty_statement": (
            "This manifest exists because the new preregistration records the repaired source "
            "hashes under status LOCKED_BEFORE_GPU, while MX-04 physically executed under the "
            "parent tree's bytes for the two declared files. That gap is real and is disclosed "
            "here rather than papered over. Without this manifest the new preregistration would "
            "misstate MX-04's execution source."
        ),
        "runner_independence": (
            "Every digest in this manifest was recomputed from bytes on disk by this tool. The "
            "runner was not consulted about whether the import is acceptable, and no runner "
            "bookkeeping was used to decide anything."
        ),
        "mx05_status_at_import": "NOT_TRAINED_TRAINABILITY_PROVEN_IN_SEPARATE_TEMP_REDIRECT",
        "original_mx04_mx05_status_unchanged": True,
    }
    manifest_path = target_out / "reports" / "mx04_import_manifest.v1.json"
    if manifest_path.exists():
        refuse("MANIFEST_ALREADY_EXISTS")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[6/6] done", flush=True)
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha(manifest_path),
        "imported_tasks": list(IMPORT_TASKS),
        "imported_bytes": sum(c["total_bytes"] for c in copies.values()),
        "next": "capture preflight, then run.py plan, then seed fasttrack_task_state.json",
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(3)
