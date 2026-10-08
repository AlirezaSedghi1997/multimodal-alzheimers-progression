"""Read-only command plan for the six resumable GPU arms of the MX-05 add-on.

Deliberately no run mode.  A separate execution preregistration must bind this
source and all downstream analysis/evaluation code after parent MX-07 is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from scripts.mx_fasttrack_v2.run import execution_contract, read
from scripts.mx_science_extension_v1.audit_and_lock import CONFIG, OUTPUT
from scripts.mx_science_extension_v1.terminal import require_parent_mx7_terminal
from scripts.run_method_x_hybrid_orchestrator import _protocol, build_hybrid_command

BATTERY = Path("results/final_results/method_x_model_battery_v3_depth_and_diversity_20260729/battery_manifest.json")
HYBRID_CONFIG = Path("configs/experiments/method_x_hybrid_v1.yaml")


def _arg(command: list[str], key: str) -> str:
    if command.count(key) != 1:
        raise RuntimeError(f"EXTENSION_COMMAND_FLAG_COUNT_INVALID:{key}")
    position = command.index(key)
    if position + 1 == len(command):
        raise RuntimeError(f"EXTENSION_COMMAND_FLAG_VALUE_MISSING:{key}")
    return command[position + 1]


def make_arm_commands(root: Path) -> dict[str, list[str]]:
    config = read(root / CONFIG)
    battery = read(root / BATTERY)
    full = yaml.safe_load((root / HYBRID_CONFIG).read_text(encoding="utf-8"))
    protocol = _protocol(battery)
    if (protocol["folds"] != 5 or protocol["epochs"] != 100
        or protocol["resolution"] != 128 or protocol["seed"] != 13
        or config["replication_seeds"] != [29, 47]
        or config["hybrid_variants"] != ["baseline", "plus_dense_skip"]
        or config["mri_only_arm"]["input_mode"] != "mri_only"
        or config["controls"]["random_labels"]["permutation_scope"] != "fit_partition_only"):
        raise RuntimeError("EXTENSION_ARM_PROTOCOL_DRIFT")
    contract = execution_contract(root, "amuse_x_hybrid3d", full["exact_resume"])
    arms: dict[str, list[str]] = {}
    specs = [(f"hybrid_{variant}_seed{seed}", variant, seed, "mri_plus_tabular", False)
             for seed in config["replication_seeds"] for variant in config["hybrid_variants"]]
    specs += [
        ("mri_only_baseline_seed13", "baseline", 13, "mri_only", False),
        ("random_label_baseline_seed13", "baseline", 13, "mri_plus_tabular", True),
    ]
    for name, variant, seed, input_mode, random_labels in specs:
        output = root / OUTPUT / "inner_tasks" / name
        command = build_hybrid_command(
            root=root, output_dir=output, variant=variant, seed=seed,
            selection_only=True, protocol=protocol, config=full,
            execution_contract=contract,
        )
        command += ["--input_mode", input_mode, "--val_fraction", "0.15"]
        if random_labels:
            # The trainer correctly forbids random-label controls in final
            # reporting mode.  Keep the same architecture and budget but mark
            # this negative control as dev/selection-only, never a final model.
            command[command.index("--phase") + 1] = "dev"
            command.append("--sanity_random_labels")
            command[2] = str(root / "scripts/mx_science_extension_v1/random_label_train.py")
        if (_arg(command, "--folds") != "5" or _arg(command, "--epochs") != "100"
            or _arg(command, "--seed") != str(seed)
            or _arg(command, "--phase") != ("dev" if random_labels else "final")
            or _arg(command, "--input_mode") != input_mode
            or _arg(command, "--hybrid_variant") != variant
            or _arg(command, "--backbone") != "amuse_x_hybrid3d"
            or _arg(command, "--fusion") != "cross_attention"
            or _arg(command, "--device") != "cuda"
            or _arg(command, "--batch_size") != "1"
            or _arg(command, "--val_fraction") != "0.15"
            or "--require_gpu" not in command
            or "--hpo_selection_only" not in command
            or "--require_exact_resume_contract" not in command
            or "--cache_sha256" not in command
            or "--spm_snapshot_sha256" not in command
            or "--no_resume" in command or "--q2_channel_gate" in command
            or "--latent_alignment_lambda" in command
            or ("--sanity_random_labels" in command) != random_labels):
            raise RuntimeError(f"EXTENSION_ARM_COMMAND_INVALID:{name}")
        expected_entry = ("scripts/mx_science_extension_v1/random_label_train.py"
                          if random_labels else "scripts/train_3d_mri.py")
        if Path(command[2]).resolve() != (root / expected_entry).resolve():
            raise RuntimeError(f"EXTENSION_ARM_ENTRY_INVALID:{name}")
        arms[name] = command
    if len(arms) != 6 or len({_arg(cmd, "--out_dir") for cmd in arms.values()}) != 6:
        raise RuntimeError("EXTENSION_ARM_OUTPUT_COLLISION")
    return arms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if root != ROOT.resolve():
        raise RuntimeError("EXTENSION_PROJECT_ROOT_INVALID")
    lock = read(root / OUTPUT / "reports/design_lock.json")
    if lock.get("status") != "DESIGN_LOCKED_IMPLEMENTATION_PENDING" or lock.get("gpu_authorized") is not False:
        raise RuntimeError("EXTENSION_DAY1_DESIGN_LOCK_MISSING")
    commands = make_arm_commands(root)
    try:
        require_parent_mx7_terminal(root)
        parent_done = True
    except RuntimeError:
        parent_done = False
    print(json.dumps({
        "status": "PLANNED_NOT_GPU_AUTHORIZED",
        "parent_mx7_terminal": parent_done,
        "arm_command_sha256": {
            name: hashlib.sha256(json.dumps(cmd, separators=(",", ":")).encode()).hexdigest()
            for name, cmd in sorted(commands.items())
        },
        "arm_count": len(commands),
        "resumable_trainer_flag_present": all("--require_exact_resume_contract" in cmd
                                               for cmd in commands.values()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
