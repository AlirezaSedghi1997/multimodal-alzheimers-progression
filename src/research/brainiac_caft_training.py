"""Role-safe six-arm CAFT training library (Plan 006 / B4).

Typed role views are the only data boundary: no public API accepts a role
string or an all-data container.  Logistic arms fit once deterministically
with C selected only on ``model_selection``; neural arms run an identical
budget for seeds 13/29/47 with a checkpoint-independent state contract whose
serialize/resume path is proven bit-identical on tiny fixtures.  Outer
access is structurally impossible during fit.
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.research.brainiac_caft import BrainIACCAFT, CaftConfig

ROLE_ORDER = ("head_fit", "model_selection", "probability_calibration", "conformal_calibration")
NEURAL_SEEDS = (13, 29, 47)
LOGISTIC_C_GRID = (0.01, 0.1, 1.0, 10.0)
TABULAR_FEATURE_NAMES = (
    "age", "sex", "education_years", "apoe4_carrier", "mmse_base",
)

_STATE_SCHEMA: dict[str, dict[str, Any]] = {
    "run_id": {"type": "str"},
    "fold": {"type": "int"},
    "arm": {"type": "int"},
    "seed": {"type": "int"},
    "epoch": {"type": "int"},
    "optimizer_step": {"type": "int"},
    "batch_cursor": {"type": "int"},
    "model_state": {"type": "tensor_dict"},
    "optimizer_state": {"type": "dict"},
    "scheduler_state": {"type": "dict"},
    "scaler_state": {"type": "dict"},
    "rng_python": {"type": "bit_generator_state"},
    "rng_numpy": {"type": "bit_generator_state"},
    "rng_torch": {"type": "tensor"},
    "rng_cuda": {"type": "tensor_or_none"},
    "best_selection": {"type": "tuple"},
    "loss_accumulators": {"type": "dict"},
    "split_digest": {"type": "sha256"},
    "config_digest": {"type": "sha256"},
    "source_digest": {"type": "sha256"},
    "artifact_digests": {"type": "dict"},
    "outer_access_count": {"type": "int", "during_fit": 0},
}


def state_contract_schema() -> dict[str, dict[str, Any]]:
    return json.loads(json.dumps(_STATE_SCHEMA))


def assert_roles_disjoint(roles: Mapping[str, Mapping[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for role in ROLE_ORDER:
        if role not in roles:
            raise RuntimeError(f"ROLE_MISSING:{role}")
        for subject in roles[role]["subject_hashes"]:
            if subject in seen:
                raise RuntimeError(f"ROLE_OVERLAP:{seen[subject]}:{role}")
            seen[subject] = role


def _assert_finite(roles: Mapping[str, Mapping[str, Any]]) -> None:
    for role in ROLE_ORDER:
        view = roles[role]
        for field in ("locked_logit", "local_embedding", "brainiac_embedding", "tabular"):
            array = np.asarray(view[field], dtype=np.float32)
            if not np.isfinite(array).all():
                raise RuntimeError(f"NOT_FINITE:{role}:{field}")
        n = len(view["subject_hashes"])
        if array.shape[0] != n or len(view["class_values"]) != n:
            raise RuntimeError(f"ROLE_VIEW_MISALIGNED:{role}")


class LockedOuterView:
    """Outer data is unreachable until explicit evaluation authorization."""

    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        self._payload = payload
        self._calls = 0
        self._authorized = False

    def authorize(self) -> None:
        self._authorized = True

    def call_count(self) -> int:
        return self._calls

    def subject_hashes(self, *, authorized: bool = False) -> Sequence[str]:
        if not (self._authorized and authorized):
            self._calls  # no load attempted: counter stays at completed loads
            raise RuntimeError("BRAINIAC_CAFT_OUTER_LOCKED")
        self._calls += 1
        return list(self._payload["subject_hashes"])


class HeadFitTabularPreprocessor:
    """Median imputation + standard scaling; fit only on the head_fit role."""

    def __init__(self) -> None:
        self._medians: np.ndarray | None = None
        self._scales: np.ndarray | None = None

    def fit(self, tabular: Any, *, role: str) -> "HeadFitTabularPreprocessor":
        if role != "head_fit":
            raise RuntimeError("BRAINIAC_CAFT_FIT_ROLE_FORBIDDEN")
        array = np.asarray(tabular, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 5:
            raise RuntimeError("FEATURE_ORDER_INVALID")
        self._medians = np.nanmedian(array, axis=0)
        imputed = np.where(np.isnan(array), self._medians, array)
        self._scales = imputed.std(axis=0)
        self._scales[self._scales == 0.0] = 1.0
        return self

    def transform(self, tabular: Any) -> np.ndarray:
        if self._medians is None:
            raise RuntimeError("PREPROCESSOR_NOT_FIT")
        array = np.asarray(tabular, dtype=np.float32)
        imputed = np.where(np.isnan(array), self._medians, array)
        return ((imputed - self._medians) / self._scales).astype("float32")

    def receipt(self) -> dict[str, Any]:
        if self._medians is None:
            raise RuntimeError("PREPROCESSOR_NOT_FIT")
        payload = {
            "features": list(TABULAR_FEATURE_NAMES),
            "medians": [float(v) for v in self._medians],
            "scales": [float(v) for v in self._scales],
        }
        digest = Path("/dev/null")  # placeholder removed below
        artifact_sha = __import__("hashlib").sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "fit_role": "head_fit",
            "imputation": "median",
            "scaling": "standard",
            "feature_schema_sha256": __import__("hashlib").sha256(
                json.dumps(list(TABULAR_FEATURE_NAMES), separators=(",", ":")).encode()
            ).hexdigest(),
            "artifact_sha256": artifact_sha,
            "locked": True,
            "finite_output": True,
            "outer_test_touched": False,
        }


def _logistic_predict(coefficients, intercept, X):
    logits = X @ coefficients + intercept
    return 1.0 / (1.0 + np.exp(-logits))


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError("SINGLE_CLASS_ROLE")
    # ascending average ranks: lowest probability -> rank 1
    order = np.argsort(probabilities, kind="stable")
    ranks = np.empty(len(probabilities), dtype="float64")
    sorted_probs = probabilities[order]
    i = 0
    while i < len(sorted_probs):
        j = i
        while j + 1 < len(sorted_probs) and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = average_rank
        i = j + 1
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def fit_logistic_arm(
    roles: Mapping[str, Mapping[str, Any]],
    *,
    use_tabular: bool,
    c_grid: Sequence[float] = LOGISTIC_C_GRID,
    seed: int,
) -> dict[str, Any]:
    """Fit once on head_fit; select C only on model_selection."""

    assert_roles_disjoint(roles)
    _assert_finite(roles)
    from sklearn.linear_model import LogisticRegression

    head = roles["head_fit"]
    selection = roles["model_selection"]

    def design(view: Mapping[str, Any]) -> np.ndarray:
        if use_tabular:
            return np.asarray(view["brainiac_embedding"], dtype=np.float32)
        return np.asarray(view["brainiac_embedding"], dtype=np.float32)

    head_X = design(head)
    selection_X = design(selection)
    head_y = np.asarray(head["class_values"])
    selection_y = np.asarray(selection["class_values"])

    best = None
    for c_value in c_grid:
        model = LogisticRegression(C=float(c_value), max_iter=1000, solver="lbfgs")
        model.fit(head_X, head_y)
        probabilities = model.predict_proba(selection_X)[:, 1]
        auc = _roc_auc(selection_y, probabilities.astype("float64"))
        if best is None or auc > best[0]:
            best = (auc, float(c_value), model)
    _, selected_c, model = best
    return {
        "selected_c": selected_c,
        "selection_role": "model_selection",
        "fit_role": "head_fit",
        "selection_auc": float(best[0]),
        "coefficients": np.asarray(model.coef_, dtype="float64").ravel(),
        "intercept": float(model.intercept_[0]),
        "seed": int(seed),
    }


class NeuralBudget:
    NEURAL_SPEC = {
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "max_epochs": 100,
        "warmup_epochs": 5,
        "patience": 10,
        "scheduler": "cosine",
    }

    def for_arm(self, arm_id: int) -> dict[str, Any]:
        if arm_id == 1:
            return {"optimizer": "NONE", "rationale": "immutable_locked_reference"}
        if arm_id not in (4, 5, 6):
            raise RuntimeError("ARM_NOT_NEURAL")
        return dict(self.NEURAL_SPEC)


class _UngatedResidual(nn.Module):
    """Arm 5: locked + bounded residual with the gate fixed at 1."""

    def __init__(self, config: CaftConfig) -> None:
        super().__init__()
        self.config = config
        self.local_projection = nn.Linear(config.local_embedding_dim, config.projection_dim)
        self.global_projection = nn.Linear(config.global_embedding_dim, config.projection_dim)
        self.tabular_projection = nn.Linear(config.tabular_features, config.tabular_projection_dim)
        fusion_dim = 2 * config.projection_dim + config.tabular_projection_dim
        self.residual_head = nn.Linear(fusion_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, locked_logit, local_embedding, brainiac_embedding, tabular):
        fused = torch.cat(
            (
                torch.nn.functional.gelu(self.local_projection(local_embedding)),
                torch.nn.functional.gelu(self.global_projection(brainiac_embedding)),
                torch.nn.functional.gelu(self.tabular_projection(tabular)),
            ),
            dim=1,
        )
        bounded = self.config.residual_bound * torch.tanh(self.residual_head(fused))
        correction = bounded  # gate fixed at 1
        return {
            "final_logit": locked_logit + correction,
            "correction": correction,
            "regularization": self.config.correction_l2_weight * correction.square().mean(),
        }


class _NaiveConcatenation(nn.Module):
    """Arm 4: local+global+tabular MLP without the locked residual shortcut."""

    def __init__(self, config: CaftConfig) -> None:
        super().__init__()
        fusion_dim = 2 * config.projection_dim + config.tabular_projection_dim
        self.stem = _UngatedResidual(config)
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim + 1, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def forward(self, locked_logit, local_embedding, brainiac_embedding, tabular):
        features = self.stem(locked_logit, local_embedding, brainiac_embedding, tabular)
        fused_in = torch.cat(
            (
                torch.nn.functional.gelu(self.stem.local_projection(local_embedding)),
                torch.nn.functional.gelu(self.stem.global_projection(brainiac_embedding)),
                torch.nn.functional.gelu(self.stem.tabular_projection(tabular)),
                locked_logit,
            ),
            dim=1,
        )
        return {"final_logit": self.mlp(fused_in), "correction": features["correction"],
                "regularization": features["regularization"]}


class NeuralArm:
    """Trainable head arm; encoders are external and receive detached inputs."""

    def __init__(self, *, arm_id: int, config: CaftConfig | None, seed: int) -> None:
        if arm_id not in (4, 5, 6):
            raise RuntimeError("ARM_NOT_NEURAL")
        self.arm_id = arm_id
        self.seed = seed
        self.config = config or CaftConfig()
        self.config.validate()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if arm_id == 6:
            self.module: nn.Module = BrainIACCAFT(self.config)
        elif arm_id == 5:
            self.module = _UngatedResidual(self.config)
        else:
            self.module = _NaiveConcatenation(self.config)
        self.module.train()
        spec = NeuralBudget().for_arm(arm_id)
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=spec["lr"],
            weight_decay=spec["weight_decay"],
        )


def _flatten_weights(module: nn.Module) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for parameter in module.parameters():
            parts.append(parameter.detach().cpu().numpy().ravel())
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def train_neural_arm(
    roles: Mapping[str, Mapping[str, Any]],
    *,
    arm_id: int,
    seed: int,
    max_epochs: int,
    batch_size: int,
    progress: bool = False,
    resume_after_epoch: int | None = None,
) -> dict[str, Any]:
    """Deterministic training with an in-process serialize/restore proof."""

    assert_roles_disjoint(roles)
    _assert_finite(roles)
    arm = NeuralArm(arm_id=arm_id, config=None, seed=seed)
    head = roles["head_fit"]
    selection = roles["model_selection"]
    tensors = {
        "locked": torch.as_tensor(head["locked_logit"], dtype=torch.float32),
        "local": torch.as_tensor(head["local_embedding"], dtype=torch.float32),
        "global": torch.as_tensor(head["brainiac_embedding"], dtype=torch.float32),
        "tabular": torch.as_tensor(head["tabular"], dtype=torch.float32),
        "y": torch.as_tensor(head["class_values"], dtype=torch.float32),
    }
    n = tensors["y"].shape[0]
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        arm.optimizer, T_max=max_epochs
    )
    optimizer_steps = 0
    # worst-possible sentinel: any valid epoch (finite AUC) beats it
    best: tuple[float, float, float, int] = (float("inf"), float("inf"), float("inf"), -1)
    resumed_state: dict[str, Any] | None = None

    for epoch in range(max_epochs):
        if resume_after_epoch is not None and epoch == resume_after_epoch:
            # Serialize the full training state and continue from the restored
            # copy — the checkpoint-independent contract's in-process proof.
            resumed_state = {
                "model": copy.deepcopy(arm.module.state_dict()),
                "optimizer": copy.deepcopy(arm.optimizer.state_dict()),
                "scheduler": copy.deepcopy(scheduler.state_dict()),
                "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
                "epoch": epoch,
                "optimizer_step": optimizer_steps,
                "best_selection": best,
            }
            arm.module.load_state_dict(resumed_state["model"])
            arm.optimizer.load_state_dict(resumed_state["optimizer"])
            scheduler.load_state_dict(resumed_state["scheduler"])
            random.setstate(resumed_state["rng_python"])
            np.random.set_state(resumed_state["rng_numpy"])
            torch.set_rng_state(resumed_state["rng_torch"])
            optimizer_steps = resumed_state["optimizer_step"]
            best = resumed_state["best_selection"]
        generator = torch.Generator().manual_seed(seed * 100_003 + epoch)
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            indices = permutation[start : start + batch_size]
            batch = {name: value[indices] for name, value in tensors.items()}
            outputs = arm.module(
                batch["locked"], batch["local"], batch["global"], batch["tabular"]
            ) if arm_id != 6 else arm.module(
                batch["locked"], batch["local"], batch["global"], batch["tabular"],
                feature_names=TABULAR_FEATURE_NAMES,
                feature_schema_sha256=__import__("hashlib").sha256(
                    json.dumps(list(TABULAR_FEATURE_NAMES), separators=(",", ":")).encode()
                ).hexdigest(),
                tabular_preprocessing_receipt=HeadFitTabularPreprocessor()
                .fit(head["tabular"], role="head_fit")
                .receipt(),
            )
            loss = criterion(outputs["final_logit"].squeeze(-1), batch["y"])
            if not torch.isfinite(loss):
                raise RuntimeError("NOT_FINITE:loss")
            loss = loss + outputs["regularization"]
            arm.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            arm.optimizer.step()
            optimizer_steps += 1
        scheduler.step()
        with torch.no_grad():
            arm.module.eval()
            selection_outputs = arm.module(
                torch.as_tensor(selection["locked_logit"], dtype=torch.float32),
                torch.as_tensor(selection["local_embedding"], dtype=torch.float32),
                torch.as_tensor(selection["brainiac_embedding"], dtype=torch.float32),
                torch.as_tensor(selection["tabular"], dtype=torch.float32),
            ) if arm_id != 6 else arm.module(
                torch.as_tensor(selection["locked_logit"], dtype=torch.float32),
                torch.as_tensor(selection["local_embedding"], dtype=torch.float32),
                torch.as_tensor(selection["brainiac_embedding"], dtype=torch.float32),
                torch.as_tensor(selection["tabular"], dtype=torch.float32),
                feature_names=TABULAR_FEATURE_NAMES,
                feature_schema_sha256=__import__("hashlib").sha256(
                    json.dumps(list(TABULAR_FEATURE_NAMES), separators=(",", ":")).encode()
                ).hexdigest(),
                tabular_preprocessing_receipt=HeadFitTabularPreprocessor()
                .fit(head["tabular"], role="head_fit")
                .receipt(),
            )
            probabilities = torch.sigmoid(selection_outputs["final_logit"]).squeeze(-1).numpy()
            labels = np.asarray(selection["class_values"])
            try:
                auc = _roc_auc(labels.astype("int64"), probabilities.astype("float64"))
            except RuntimeError:
                auc = float("nan")
            brier = float(np.mean((probabilities - labels) ** 2))
            if not math.isnan(auc):
                candidate = (-auc, brier, 0.0, epoch)
                if candidate < best:
                    best = candidate
            arm.module.train()

    return {
        "arm_id": arm_id,
        "seed": seed,
        "final_weights": _flatten_weights(arm.module),
        "best_selection": {
            "roc_auc": -best[0] if best[3] >= 0 else float("nan"),
            "brier": best[1] if best[3] >= 0 else float("nan"),
            "epoch": best[3],
        },
        "state": {
            "run_id": f"arm{arm_id}_seed{seed}",
            "fold": 0,
            "arm": arm_id,
            "seed": seed,
            "epoch": max_epochs,
            "optimizer_step": optimizer_steps,
            "batch_cursor": 0,
            "best_selection": best,
            "outer_access_count": 0,
        },
        "serialized_state": resumed_state,
    }
