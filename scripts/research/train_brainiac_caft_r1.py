"""Protected BrainIAC-CAFT R1 training library CLI (Plan 006 / B4).

--mode status    : read-only readiness of the training library
--mode fixture   : run the tiny synthetic self-test (non-scientific)

Real fitting on study data, outer snapshots and promotion are Plan 007
orchestrator territory; this CLI structurally cannot start them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import brainiac_caft_training as ct
from src.research import brainiac_caft_evaluation as ce
from src.research.brainiac_caft import CaftConfig


def status() -> dict[str, Any]:
    config = CaftConfig()
    config.validate()
    return {
        "status": "PASS",
        "library": "brainiac_caft_training/evaluation",
        "local_embedding_dim": config.local_embedding_dim,
        "arms": [1, 2, 3, 4, 5, 6],
        "neural_seeds": list(ct.NEURAL_SEEDS),
        "logistic_c_grid": list(ct.LOGISTIC_C_GRID),
        "outer_loader_calls_allowed_before_authorization": 0,
        "real_training": "FORBIDDEN_UNTIL_PLAN_007_AND_BATTERY_TERMINAL",
        "claim_ceiling": "POST_HOC_INTERNAL_TECHNICAL_EVIDENCE",
    }


def fixture() -> dict[str, Any]:
    """Tiny synthetic self-test; explicitly non-scientific."""

    import numpy as np

    def view(n: int, seed: int):
        rng = np.random.default_rng(seed)
        labels = (rng.random(n) < 0.4).astype("int64")
        return {
            "subject_hashes": [f"f_{seed}_{i:04d}" for i in range(n)],
            "locked_logit": rng.normal(size=(n, 1)).astype("float32"),
            "local_embedding": rng.normal(size=(n, 128)).astype("float32"),
            "brainiac_embedding": rng.normal(size=(n, 768)).astype("float32") + labels[:, None] * 0.8,
            "tabular": rng.normal(size=(n, 5)).astype("float32"),
            "class_values": labels,
        }

    roles = {role: view(40, index + 1) for index, role in enumerate(ct.ROLE_ORDER)}
    logistic = ct.fit_logistic_arm(roles, use_tabular=False, seed=13)
    neural = ct.train_neural_arm(
        roles, arm_id=6, seed=13, max_epochs=2, batch_size=8, progress=False
    )
    return {
        "status": "PASS",
        "note": "SYNTHETIC_FIXTURE_NON_SCIENTIFIC",
        "logistic_selected_c": logistic["selected_c"],
        "neural_best_auc": neural["best_selection"]["roc_auc"],
        "outer_access_count": neural["state"]["outer_access_count"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("status", "fixture"), default="status")
    args = parser.parse_args(argv)
    result = status() if args.mode == "status" else fixture()
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)
    return 0 if result.get("status") == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
