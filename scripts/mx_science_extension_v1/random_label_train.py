"""Fit-only prevalence-preserving label permutation for the negative control.

The locked trainer's native sanity flag draws new Bernoulli bits.  This
versioned adapter instead shuffles the existing fit labels, preserving their
class count.  Validation and outer labels are untouched.  The adapter source
is bound by the extension execution preregistration before GPU optimization.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    arguments = sys.argv[1:]
    required = ("--sanity_random_labels", "--hpo_selection_only",
                "--require_exact_resume_contract", "--require_gpu")
    if any(flag not in arguments for flag in required):
        raise RuntimeError("EXTENSION_RANDOM_CONTROL_REQUIRED_FLAGS_MISSING")
    for flag, value in (("--phase", "dev"), ("--input_mode", "mri_plus_tabular"),
                        ("--hybrid_variant", "baseline")):
        if (arguments.count(flag) != 1 or arguments.index(flag) + 1 == len(arguments)
            or arguments[arguments.index(flag) + 1] != value):
            raise RuntimeError(f"EXTENSION_RANDOM_CONTROL_ARGUMENT_INVALID:{flag}")
    from scripts import train_3d_mri as trainer

    original = trainer.H5MethodXDataset.__init__

    def init(self, *args, **kwargs):
        shuffle_fit = kwargs.get("sanity_random_labels") is True
        forwarded = dict(kwargs)
        forwarded["sanity_random_labels"] = False
        original(self, *args, **forwarded)
        if shuffle_fit:
            before = list(self.labels)
            after = before.copy()
            random.Random(int(kwargs.get("seed", 13)) + len(after)).shuffle(after)
            if sorted(after) != sorted(before):
                raise RuntimeError("EXTENSION_RANDOM_CONTROL_PREVALENCE_CHANGED")
            self.labels = after

    trainer.H5MethodXDataset.__init__ = init
    try:
        return trainer.main()
    finally:
        trainer.H5MethodXDataset.__init__ = original


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(3)

