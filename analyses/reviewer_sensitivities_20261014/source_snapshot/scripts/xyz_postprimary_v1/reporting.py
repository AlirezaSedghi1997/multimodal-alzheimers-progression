"""Aggregate-only reporting; diagnostic completion never implies a scientific PASS."""
from __future__ import annotations

from collections import Counter
from scripts.xyz_fusion_v1.store import atomic_json, atomic_text
from src.method_y_revival.privacy import scan_public_bundle
from .context import output_hashes


def public_summary(statistics, pet, tau, cases, parent):
    pet = {k: v for k, v in pet.items() if k != "private_rows"}
    tau = {k: v for k, v in tau.items() if k != "private_rows"}
    return {"status": "POSTPRIMARY_DIAGNOSTICS_COMPLETED_NOT_ALL_EXTENSIONS",
            "claim_status": "POST_HOC_INTERNAL_EXPLORATORY", "n": parent["n_after_image_qc"],
            "events": parent["events_after_new_landmark"], "statistics": statistics,
            "pet_image_feasibility": pet, "tau_feasibility": tau,
            "spatial": {"cases": len(cases), "statuses": dict(Counter(c["status"] for c in cases)),
                        "population_localization_proof": False, "patient_images_public": False,
                        "atlas_roi_naming": "BLOCKED_DIFFERENT_TEMPLATE_SPACE"},
            "primary_training_repeated": False, "clinical_claim_allowed": False,
            "external_claim_allowed": False, "positive_quantitative_equivalence_claim_allowed": False,
            "deep_mri_comparison": "SEPARATE_VERSIONED_MATCHED_EXPERIMENT_NOT_AUTOMATICALLY_SUPERIOR"}


def render_bundle(store, summary):
    root = store.root / "public"
    atomic_json(root / "summary.json", summary)
    text = ["# Post-primary fusion diagnostics", "", summary["status"], "",
            "The original 50 MRI fits and 45 Cox fits are preserved. No training is repeated here.",
            "Claims remain POST_HOC_INTERNAL_EXPLORATORY. No diagnostic, causal, external or quantitative-equivalence claim.",
            "", "## Multiplicity-aware incremental results", "",
            "| Base | Added model | Simultaneous delta IBS interval | Incremental support |",
            "|---|---|---|---|"]
    for c in summary["statistics"]["comparisons"]:
        text.append(f"| {c['base']} | {c['augmented']} | {c['simultaneous_ibs_ci']} | {c['incremental_support_after_multiplicity']} |")
    text += ["", "## Spatial interpretation", "",
             "Signed Integrated Gradients explains MRI progression risk and its affine contribution to frozen Cox log hazard.",
             "Native-space images are private actual-subject overlays, not disease-lesion maps. Failed diagnostics remain failed.",
             "Selection is outcome-blind; ten cases are a bounded diagnostic sample, not whole-population XAI validation.",
             "The historical atlas uses a different template space; regional names are not assigned by shape-only resampling.",
             "", "## Still separate", "",
             "PET-image training needs new ancestry-safe cross-fitting. Tau needs the new landmark and documented regional mapping.",
             "A deeper MRI comparator is a new post-hoc experiment, not a replacement of unfavorable original results.",
             "External data are unavailable. Conditional bootstrap does not account for model-development uncertainty."]
    atomic_text(root / "REPORT.md", "\n".join(text)+"\n")
    alignment = """# Reporting alignment: core items, not an exhaustive checklist certification

| Domain | Evidence / unresolved issue |
|---|---|
| TRIPOD+AI population and endpoint | A4-only common-landmark sustained future CDR; inclusion/exclusion and counts preserved |
| TRIPOD+AI model development | Nested MRI cross-fitting, fixed low-dimensional Cox, heldout ancestry bound to original artifacts |
| Performance and uncertainty | IPCW-IBS, Uno-C, calibration, paired conditional bootstrap and multiplicity intervals |
| PROBAST+AI participants | Selected baseline-MRI cohort; QC/blood selection imbalance reported, MNAR cannot be excluded |
| PROBAST+AI predictors | Chronology audited; original biomarker provenance limits remain; no cohort/site input |
| PROBAST+AI analysis | Post-hoc internal results; external validity, formal penalized PH test and training uncertainty unresolved |
| CLAIM image provenance | Raw MRI, registered input, locked transforms and cache retained; all subject overlays private |
| CLAIM interpretability | Numerical, randomization and equal-volume perturbation checks reported separately; no lesion truth |
| Missing extensions | PET joint cross-fitting, tau mapping, template-compatible ROI atlas, deeper MRI comparison |

No human sign-off is imposed. Automated verification is not a substitute for missing scientific evidence.
"""
    atomic_text(root / "REPORTING_ALIGNMENT.md", alignment)
    violations = scan_public_bundle(root)
    if violations: raise ValueError("POSTPRIMARY_PUBLIC_PRIVACY_FAILED:"+str(violations))
    return {"status": "AGGREGATE_REPORTS_VERIFIED", "outputs": output_hashes(store.root,
            [root/"summary.json", root/"REPORT.md", root/"REPORTING_ALIGNMENT.md"])}
