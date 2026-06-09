#!/usr/bin/env python3
"""Aggregate MRI/PET preprocess QC JSON artifacts into audit tables and gate reports."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.a4_schema import resolve_data_root
from src.utils.io_helpers import atomic_write_json, atomic_write_text
from src.utils.preprocess_quality import (
    classify_mri_numeric_qc,
    flag_icv_mad_outliers,
    load_icv_ml_by_sample_hash,
    load_json,
    load_official_centiloid_by_bid,
    load_official_suvr_by_bid,
    pearson_correlation,
)
from src.utils.visual_qc import build_visual_review_manifest


def _iter_qc_json_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*.json"))


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "p05": round(_percentile(values, 5) or 0.0, 6),
        "p95": round(_percentile(values, 95) or 0.0, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def audit_mri_qc(mri_qc_dir: Path, icv_by_sample: dict[str, float] | None = None) -> dict:
    status_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    brain_volumes: list[float] = []
    icv_values: list[float] = []
    gm_wm_cnr: list[float] = []
    rows: list[dict] = []

    for path in _iter_qc_json_files(mri_qc_dir):
        raw = load_json(path)
        if not raw:
            continue
        sample_hash = str(raw.get("sample_hash") or path.stem)
        if icv_by_sample and sample_hash in icv_by_sample and "icv_ml_official" not in raw:
            raw = {**raw, "icv_ml_official": icv_by_sample[sample_hash]}
        qc = classify_mri_numeric_qc(raw)
        status = str(qc.get("qc_status") or "UNKNOWN")
        status_counts[status] += 1
        for f in qc.get("failures") or []:
            failure_counts[str(f)] += 1
        for w in qc.get("warnings") or []:
            warning_counts[str(w)] += 1
        if qc.get("brain_volume_ml") is not None:
            brain_volumes.append(float(qc["brain_volume_ml"]))
        if qc.get("icv_ml_official") is not None:
            icv_values.append(float(qc["icv_ml_official"]))
        if qc.get("gm_wm_cnr") is not None:
            gm_wm_cnr.append(float(qc["gm_wm_cnr"]))
        rows.append(
            {
                "sample_hash": sample_hash,
                "qc_status": status,
                "brain_volume_ml": qc.get("brain_volume_ml"),
                "icv_ml_official": qc.get("icv_ml_official"),
                "gm_wm_cnr": qc.get("gm_wm_cnr"),
                "failures": qc.get("failures") or [],
                "warnings": qc.get("warnings") or [],
            }
        )

    icv_brain_pairs = [
        (float(r["brain_volume_ml"]), float(r["icv_ml_official"]))
        for r in rows
        if r.get("brain_volume_ml") is not None and r.get("icv_ml_official") is not None
    ]
    icv_corr = pearson_correlation([p[0] for p in icv_brain_pairs], [p[1] for p in icv_brain_pairs])
    icv_mad_outliers = flag_icv_mad_outliers(rows)

    total = sum(status_counts.values())
    return {
        "modality": "MRI",
        "qc_dir": str(mri_qc_dir),
        "total_qc_files": total,
        "status_counts": dict(status_counts),
        "fail_rate": round(status_counts.get("FAIL", 0) / total, 4) if total else None,
        "hold_rate": round(status_counts.get("HOLD", 0) / total, 4) if total else None,
        "failure_reasons": dict(failure_counts),
        "warning_reasons": dict(warning_counts),
        "brain_volume_ml": _distribution(brain_volumes),
        "icv_ml_official": _distribution(icv_values),
        "brain_volume_icv_pearson_r": icv_corr,
        "icv_mad_outliers": icv_mad_outliers[:50],
        "gm_wm_cnr": _distribution(gm_wm_cnr),
        "fail_samples": [r for r in rows if r["qc_status"] == "FAIL"][:50],
        "hold_samples": [r for r in rows if r["qc_status"] == "HOLD"][:50],
    }


def audit_pet_qc(pet_qc_dir: Path, project_root: Path) -> dict:
    status_counts: Counter[str] = Counter()
    gate_failures: Counter[str] = Counter()
    ncc_values: list[float] = []
    official_centiloids: list[float] = []
    derived_centiloids: list[float] = []
    centiloid_delta_rows: list[dict] = []
    overlay_review_rows: list[dict] = []
    visual_qc_pending = 0
    rows: list[dict] = []

    for path in _iter_qc_json_files(pet_qc_dir):
        payload = load_json(path)
        if not payload:
            continue
        status = str(payload.get("status") or payload.get("gate", {}).get("status") or "UNKNOWN")
        status_counts[status] += 1
        gate = payload.get("gate") or {}
        for f in gate.get("failures") or []:
            gate_failures[str(f)] += 1
        qc_after = payload.get("qc_after") or {}
        ncc = qc_after.get("ncc_pet_t1")
        if ncc is not None:
            try:
                ncc_values.append(float(ncc))
            except (TypeError, ValueError):
                pass
        official = qc_after.get("official_centiloid")
        derived = qc_after.get("derived_centiloid")
        if official is not None:
            try:
                official_centiloids.append(float(official))
            except (TypeError, ValueError):
                pass
        if derived is not None:
            try:
                derived_centiloids.append(float(derived))
            except (TypeError, ValueError):
                pass
        if official is not None and derived is not None:
            try:
                centiloid_delta_rows.append(
                    {
                        "sample_hash": payload.get("sample_hash") or path.stem,
                        "official_centiloid": float(official),
                        "derived_centiloid": float(derived),
                        "official_suvr": qc_after.get("official_suvr"),
                        "abs_delta": abs(float(derived) - float(official)),
                    }
                )
            except (TypeError, ValueError):
                pass
        if payload.get("visual_qc_required_for_training") and payload.get("visual_qc_decision") is None:
            visual_qc_pending += 1
        sample_hash = payload.get("sample_hash") or path.stem
        tracer = str(payload.get("tracer") or "fbp").lower()
        overlay_rel = Path("results/qc/method_y_pet/overlays") / tracer / f"{sample_hash}_pet_on_t1.png"
        rows.append(
            {
                "sample_hash": sample_hash,
                "status": status,
                "ncc_pet_t1": ncc,
                "official_centiloid": official,
                "derived_centiloid": derived,
                "gate_failures": gate.get("failures") or [],
            }
        )
        if ncc is not None:
            overlay_review_rows.append(
                {
                    "sample_hash": sample_hash,
                    "ncc_pet_t1": float(ncc),
                    "status": status,
                    "overlay_exists": (project_root / overlay_rel).exists(),
                }
            )

    overlay_review_rows.sort(key=lambda r: (r["ncc_pet_t1"] if r["ncc_pet_t1"] is not None else 999.0))
    centiloid_pairs = [
        (float(r["official_centiloid"]), float(r["derived_centiloid"]))
        for r in centiloid_delta_rows
        if r.get("official_centiloid") is not None and r.get("derived_centiloid") is not None
    ]
    centiloid_corr = pearson_correlation([p[0] for p in centiloid_pairs], [p[1] for p in centiloid_pairs])

    total = sum(status_counts.values())
    return {
        "modality": "PET",
        "qc_dir": str(pet_qc_dir),
        "total_qc_files": total,
        "status_counts": dict(status_counts),
        "qc_fail_rate": round(status_counts.get("QC_FAIL", 0) / total, 4) if total else None,
        "pass_rate": round(status_counts.get("PASS", 0) / total, 4) if total else None,
        "gate_failure_reasons": dict(gate_failures),
        "ncc_pet_t1": _distribution(ncc_values),
        "official_centiloid": _distribution(official_centiloids),
        "derived_centiloid": _distribution(derived_centiloids),
        "official_vs_derived_centiloid_pearson_r": centiloid_corr,
        "centiloid_delta_review": sorted(centiloid_delta_rows, key=lambda r: r.get("abs_delta", 0), reverse=True)[:50],
        "worst_ncc_overlays": overlay_review_rows[:10],
        "visual_qc_pending_count": visual_qc_pending,
        "fail_samples": [r for r in rows if r["status"] in {"QC_FAIL", "FAIL"}][:50],
    }


def build_gate_verdict(mri: dict, pet: dict, visual_manifest: dict | None) -> dict:
    blockers: list[str] = []
    if mri.get("fail_rate") and mri["fail_rate"] > 0.05:
        blockers.append("MRI_FAIL_RATE_ABOVE_5PCT")
    if pet.get("total_qc_files", 0) > 0:
        if pet.get("pass_rate") is not None and pet["pass_rate"] < 0.85:
            blockers.append("PET_PASS_RATE_BELOW_85PCT")
        if pet.get("visual_qc_pending_count", 0) > 0:
            blockers.append("PET_VISUAL_QC_SIGNOFF_INCOMPLETE")
    if visual_manifest and visual_manifest.get("pending_signoff_count", 0) > 0:
        blockers.append("VISUAL_REVIEW_MANIFEST_PENDING")
    return {
        "training_ready_after_audit": len(blockers) == 0,
        "blockers": blockers,
        "recommended_actions": [
            "Review FAIL/HOLD MRI samples and exclude from cache build.",
            "Review PET QC_FAIL overlays under results/qc/method_y_pet/overlays/.",
            "Run scripts/build_visual_qc_manifest.py then scripts/sign_visual_qc.py for each reviewed item.",
            "Re-run preprocess batches without --overwrite on PASS items.",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Preprocess Quality Audit",
        "",
        f"- Generated from: `{report.get('script')}`",
        f"- MRI QC dir: `{report['mri']['qc_dir']}`",
        f"- PET QC dir: `{report['pet']['qc_dir']}`",
        "",
        "## MRI summary",
        "",
        f"- Total QC files: **{report['mri']['total_qc_files']}**",
        f"- Status counts: `{report['mri']['status_counts']}`",
        f"- Fail rate: **{report['mri'].get('fail_rate')}**",
        f"- Hold rate: **{report['mri'].get('hold_rate')}**",
        f"- Brain volume vs ICV Pearson r: **{report['mri'].get('brain_volume_icv_pearson_r')}**",
        "",
        "## PET summary",
        "",
        f"- Total QC files: **{report['pet']['total_qc_files']}**",
        f"- Status counts: `{report['pet']['status_counts']}`",
        f"- Pass rate: **{report['pet'].get('pass_rate')}**",
        f"- Official vs derived centiloid Pearson r: **{report['pet'].get('official_vs_derived_centiloid_pearson_r')}**",
        f"- Visual QC pending: **{report['pet'].get('visual_qc_pending_count')}**",
        "",
        "## Gate verdict",
        "",
        f"- Training ready after audit: **{report['gate_verdict']['training_ready_after_audit']}**",
        f"- Blockers: `{report['gate_verdict']['blockers']}`",
        "",
    ]
    worst = report["pet"].get("worst_ncc_overlays") or []
    if worst:
        lines.extend(["## Worst NCC overlays (manual review priority)", ""])
        for row in worst[:10]:
            lines.append(f"- `{row['sample_hash']}` ncc={row['ncc_pet_t1']} status={row['status']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate MRI/PET preprocess QC into audit reports.")
    ap.add_argument("--mri_qc_dir", default="data/preprocessed_mri_96/qc")
    ap.add_argument("--pet_qc_dir", default="data/preprocessed_pet/fbp/qc")
    ap.add_argument("--catalog", default="data/processed/master_catalog.csv")
    ap.add_argument("--data_root", default="", help="Optional DATA_ROOT for official lookups.")
    ap.add_argument("--build_visual_manifest", action="store_true")
    ap.add_argument("--out_json", default="results/tables/preprocess_quality_audit.json")
    ap.add_argument("--out_md", default="results/tables/preprocess_quality_audit.md")
    args = ap.parse_args()

    icv_by_sample: dict[str, float] = {}
    official_lookup_size = 0
    suvr_lookup_size = 0
    if args.data_root:
        root = resolve_data_root(args.data_root)
        icv_by_sample = load_icv_ml_by_sample_hash(root, PROJECT_ROOT / args.catalog)
        official_lookup_size = len(load_official_centiloid_by_bid(root))
        suvr_lookup_size = len(load_official_suvr_by_bid(root))

    mri_qc_dir = PROJECT_ROOT / args.mri_qc_dir
    pet_qc_dir = PROJECT_ROOT / args.pet_qc_dir
    mri = audit_mri_qc(mri_qc_dir, icv_by_sample=icv_by_sample)
    pet = audit_pet_qc(pet_qc_dir, PROJECT_ROOT)

    visual_manifest = None
    if args.build_visual_manifest or pet_qc_dir.exists():
        visual_manifest = build_visual_review_manifest(pet_qc_dir, project_root=PROJECT_ROOT)
        atomic_write_json(PROJECT_ROOT / "results/qc/visual_review_manifest.json", visual_manifest)

    gate = build_gate_verdict(mri, pet, visual_manifest)

    report = {
        "script": "scripts/run_preprocess_quality_audit.py",
        "mri": mri,
        "pet": pet,
        "gate_verdict": gate,
        "official_centiloid_lookup_rows": official_lookup_size,
        "official_suvr_lookup_rows": suvr_lookup_size,
        "visual_review_manifest": {
            "pending_signoff_count": (visual_manifest or {}).get("pending_signoff_count"),
            "all_signed_off": (visual_manifest or {}).get("all_signed_off"),
        },
    }
    out_json = PROJECT_ROOT / args.out_json
    out_md = PROJECT_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_json, report)
    atomic_write_text(out_md, render_markdown(report))
    print(json.dumps({"status": "OK", "out_json": str(out_json), "gate_verdict": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
