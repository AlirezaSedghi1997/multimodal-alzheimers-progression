#!/usr/bin/env python3
"""SynIB tabular+MRI fusion trainer (surrogate MLP, arXiv:2606.09853)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_baseline import average_precision, brier, guarded_bootstrap_ci, roc_auc, safe_float
from src.data.mri_feature_constants import MRI_FEATURES
from src.models.losses.synib_synergy_loss import SynIBConfig, mask_modalities, synib_multimodal_loss
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.run_manifest import write_run_manifest

TABULAR_FEATURES = ["age", "education_years", "apoe4_carrier", "mmse_base"]
TABULAR_DIM = len(TABULAR_FEATURES) + 1  # + sex


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def feature_index(path: Path) -> dict[str, dict[str, str]]:
    return {row["sample_hash"]: row for row in read_rows(path) if row.get("mri_feature_status") == "OK"}


def vector(row: dict[str, str], frow: dict[str, str]) -> list[float]:
    sex = row.get("sex", "")
    sex_val = 1.0 if str(sex).strip() in {"1", "M", "Male", "male"} else 0.0
    tab = [safe_float(row.get(f)) for f in TABULAR_FEATURES] + [sex_val]
    mri = [safe_float(frow.get(f)) for f in MRI_FEATURES]
    return tab + mri


def build_matrix(rows: list[dict[str, str]], features: dict[str, dict[str, str]]) -> tuple[list[list[float]], list[int]]:
    x, y = [], []
    for row in rows:
        frow = features.get(row["sample_hash"])
        if frow is None:
            continue
        x.append(vector(row, frow))
        y.append(int(float(row["ab_label"])))
    return x, y


def train_fold(
    x_train: list[list[float]],
    y_train: list[int],
    x_test: list[list[float]],
    *,
    epochs: int,
    synib_cfg: SynIBConfig,
    seed: int,
) -> list[float]:
    import torch
    import torch.nn as nn
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    xt = scaler.fit_transform(imp.fit_transform(x_train))
    xe = scaler.transform(imp.transform(x_test))
    x_t = torch.tensor(xt, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    x_e = torch.tensor(xe, dtype=torch.float32)

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x_t.shape[1], 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits_full = model(x_t).squeeze(-1)
        x_tab = mask_modalities(x_t, tabular_dim=synib_cfg.tabular_dim, mask_tab=True, mask_mri=False)
        x_mri = mask_modalities(x_t, tabular_dim=synib_cfg.tabular_dim, mask_tab=False, mask_mri=True)
        logits_tab = model(x_tab).squeeze(-1)
        logits_mri = model(x_mri).squeeze(-1)
        loss, _ = synib_multimodal_loss(logits_full, logits_tab, logits_mri, y_t, config=synib_cfg)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_e).squeeze(-1))
    return [float(p) for p in probs.tolist()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--features", default="data/processed/mri_features.csv")
    ap.add_argument("--out_dir", default="results/final_results/synib_fusion")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lambda_syn", type=float, default=0.5)
    ap.add_argument("--mask_prob", type=float, default=0.5)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except Exception:
        report = {"status": "BLOCKED", "blocker": "PYTORCH_MISSING", "literature_anchor": "LIT019"}
        out_dir = PROJECT_ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "synib_fusion_metrics.json", report)
        print(json.dumps(report, indent=2))
        return 1

    split_dir = PROJECT_ROOT / args.splits
    features = feature_index(PROJECT_ROOT / args.features)
    synib_cfg = SynIBConfig(lambda_syn=args.lambda_syn, mask_prob=args.mask_prob, tabular_dim=TABULAR_DIM, mri_dim=len(MRI_FEATURES))

    fold_metrics = []
    all_y: list[int] = []
    all_p: list[float] = []
    for test_path in sorted(split_dir.glob("fold_*_test.csv")):
        fold = int(test_path.name.split("_")[1])
        x_train, y_train = build_matrix(read_rows(split_dir / f"fold_{fold}_train.csv"), features)
        x_test, y_test = build_matrix(read_rows(test_path), features)
        preds = train_fold(x_train, y_train, x_test, epochs=args.epochs, synib_cfg=synib_cfg, seed=13 + fold)
        fold_metrics.append(
            {
                "fold": fold,
                "n_test": len(y_test),
                "roc_auc": roc_auc(y_test, preds),
                "pr_auc": average_precision(y_test, preds),
                "brier": brier(y_test, preds),
            }
        )
        all_y.extend(y_test)
        all_p.extend(preds)

    aggregate = {
        "literature_anchor": "LIT019 arXiv:2606.09853",
        "method": "SynIB_tabular_mri_surrogate",
        "n": len(all_y),
        "epochs": args.epochs,
        "synib_config": {"lambda_syn": args.lambda_syn, "mask_prob": args.mask_prob},
        "roc_auc": roc_auc(all_y, all_p),
        "pr_auc": average_precision(all_y, all_p),
        "brier": brier(all_y, all_p),
        "roc_auc_ci": guarded_bootstrap_ci(all_y, all_p, roc_auc, n_boot=500),
    }
    out = {"status": "COMPLETED", "folds": fold_metrics, "aggregate": aggregate}
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "synib_fusion_metrics.json", out)
    md = [
        "# SynIB Fusion (tabular+MRI surrogate)\n\n",
        f"- literature: `{aggregate['literature_anchor']}`\n",
        f"- roc_auc: `{aggregate['roc_auc']:.4f}`\n",
        f"- pr_auc: `{aggregate['pr_auc']:.4f}`\n",
        f"- brier: `{aggregate['brier']:.4f}`\n",
        f"- lambda_syn: `{args.lambda_syn}`\n",
    ]
    atomic_write_text(out_dir / "synib_fusion_metrics.md", "".join(md))
    write_run_manifest("results/logs/synib_fusion_manifest.json", stage="synib_fusion", data_root=None, extra={"aggregate": aggregate})
    print(json.dumps({"roc_auc": aggregate["roc_auc"], "status": "COMPLETED"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
