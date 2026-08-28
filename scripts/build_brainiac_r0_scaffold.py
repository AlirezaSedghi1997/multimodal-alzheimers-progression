"""Materialize the BrainIAC R0 scaffold on H. Scaffold only: no fetch, no weights.

Seeds the evidence registry with the sources already vetted in the corrective
program. It deliberately does *not* invent additional studies to reach the 24-30
target: the registry reports itself incomplete instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.brainiac_provenance import (  # noqa: E402
    NOVELTY_CEILING,
    R0_MIN_PEER_REVIEWED,
    R0_TARGET_SOURCE_RANGE,
    UNRESOLVED,
    UPSTREAM_PINNED_COMMIT,
    build_upstream_manifest,
    evidence_registry_status,
    validate_evidence_registry,
    validate_upstream_manifest,
    assert_current_r0_write_target,
)

ROOT = PROJECT_ROOT / "research" / "brainiac_caft_20260808"
LITERATURE_FREEZE = "2026-08-08"

# Sources carried over from the corrective program's pre-review. Citation status
# reflects what citeprobe actually returned: 12 identifier checks passed, the
# AnatCL DOI had no multi-source candidate, and Crossref/OpenAlex errored during
# the run, so corroboration overall was INCONCLUSIVE.
PRIMARY_SOURCES = [
    ("s01", "A generalizable foundation model for analysis of human brain MRI", 2026,
     "peer_reviewed", "https://www.nature.com/articles/s41593-026-02202-6",
     "multiple_downstream_tasks", "multi_cohort", "T1_MRI",
     "rigid_mni_skull_stripped_96", "reported", "partial", "not_reported",
     "not_reported", "public", "IDENTIFIER_VERIFIED",
     "Foundation encoder candidate; endpoint was not PET-defined amyloid."),
    ("s02", "MRI-based prediction of PET-defined amyloid status", 2025,
     "peer_reviewed", "https://pubmed.ncbi.nlm.nih.gov/40579043/",
     "pet_defined_amyloid", "clinical_cohort", "T1_and_FLAIR",
     "not_fully_reported", "reported", "none", "not_reported", "not_reported",
     "unavailable", "IDENTIFIER_VERIFIED",
     "T1-only AUC around 0.61; T1+FLAIR around 0.67. Endpoint is hard."),
    ("s03", "Transfer from brain age to amyloid prediction", 2025,
     "peer_reviewed", "https://pubmed.ncbi.nlm.nih.gov/40450379/",
     "amyloid_status", "clinical_cohort", "T1_MRI", "not_fully_reported",
     "reported", "none", "not_reported", "not_reported", "unavailable",
     "IDENTIFIER_VERIFIED",
     "Related pretraining can help; performance varies with disease stage."),
    ("s04", "AnatCL: anatomy-guided contrastive representations", 2025,
     "peer_reviewed",
     "https://www.sciencedirect.com/science/article/pii/S0167865525003848",
     "representation_learning", "multi_cohort", "T1_MRI", "not_fully_reported",
     "reported", "none", "not_reported", "not_reported", "partial", "UNKNOWN",
     "DOI unresolved by citeprobe: no multi-source candidate. Not upgraded."),
    ("s05", "Building a General SimCLR Self-Supervised Foundation Model Across "
     "Neurological Diseases", 2025, "peer_reviewed",
     "https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/html/Kaczmarek_Building_a_General_SimCLR_Self-Supervised_Foundation_Model_Across_Neurological_Diseases_ICCVW_2025_paper.html",
     "representation_learning", "multi_cohort", "volumetric_MRI",
     "not_fully_reported", "reported", "none", "not_reported", "not_reported",
     "partial", "IDENTIFIER_VERIFIED",
     "Workshop tier. Volumetric SSL transfer is feasible; helps in low data."),
    ("s06", "FOMO25 challenge report", 2026, "preprint",
     "https://arxiv.org/abs/2604.11679", "multiple_downstream_tasks",
     "multi_cohort", "MRI", "not_fully_reported", "reported", "none",
     "not_reported", "not_reported", "partial", "IDENTIFIER_VERIFIED",
     "No single objective or model size wins everywhere; scaling is not a given."),
    ("s07", "Cross-scanner reliability of brain MRI foundation embeddings", 2026,
     "preprint",
     "https://www.medrxiv.org/content/10.64898/2026.03.23.26348808v2.full",
     "embedding_stability", "multi_scanner", "T1_MRI", "not_fully_reported",
     "reported", "none", "scanner_probe", "not_reported", "unavailable",
     "IDENTIFIER_VERIFIED",
     "BrainIAC embeddings may be unstable across real scanners. Drives the audit."),
    ("s08", "When Brains Disagree: structural MRI versus amyloid PET", 2026,
     "preprint", "https://arxiv.org/abs/2605.11867", "mri_pet_concordance",
     "clinical_cohort", "MRI_and_PET", "not_fully_reported", "reported", "none",
     "not_reported", "not_reported", "unavailable", "IDENTIFIER_VERIFIED",
     "MRI measures neurodegeneration, PET measures amyloid; mapping is ambiguous."),
    ("s09", "PET-guided MRI distillation", 2026, "preprint",
     "https://arxiv.org/abs/2604.12574", "amyloid_status", "clinical_cohort",
     "MRI_trained_with_PET", "not_fully_reported", "reported", "none",
     "not_reported", "not_reported", "partial", "IDENTIFIER_VERIFIED",
     "Uses PET at training time, so it is outside strict Method X."),
    ("s10", "BrainDINO: frozen encoder with lightweight heads", 2026, "preprint",
     "https://arxiv.org/abs/2604.27277", "representation_learning",
     "multi_cohort", "T1_MRI", "not_fully_reported", "reported", "none",
     "not_reported", "not_reported", "partial", "IDENTIFIER_VERIFIED",
     "Frozen encoder plus light head is hardware-feasible here."),
    ("s11", "RegionFM: region-additive attribution", 2026, "preprint",
     "https://arxiv.org/abs/2607.16325", "interpretability", "multi_cohort",
     "MRI", "not_fully_reported", "reported", "none", "not_reported",
     "not_reported", "partial", "IDENTIFIER_VERIFIED",
     "Additive regional attribution is a fit for the XAI contract."),
    ("s12", "Generalization of brain foundation models to African cohorts", 2026,
     "preprint", "https://arxiv.org/abs/2607.28771", "classification",
     "external_cohort", "MRI", "not_fully_reported", "reported", "reported",
     "cohort_probe", "not_reported", "unavailable", "IDENTIFIER_VERIFIED",
     "Foundation models do not always beat a scratch baseline on classification."),
    ("s13", "ADLIP: multimodal representation for AD/MCI/CN", 2026,
     "peer_reviewed", "https://pubmed.ncbi.nlm.nih.gov/41457461/",
     "ad_mci_cn_classification", "clinical_cohort", "multimodal",
     "not_fully_reported", "reported", "none", "not_reported", "not_reported",
     "unavailable", "IDENTIFIER_VERIFIED",
     "Endpoint is AD/MCI/CN, not amyloid status."),
    ("s14", "Longitudinal MRI progression modelling", 2025, "peer_reviewed",
     "https://link.springer.com/article/10.1186/s13195-025-01827-2",
     "progression", "clinical_cohort", "T1_MRI", "not_fully_reported",
     "reported", "none", "not_reported", "not_reported", "unavailable",
     "INCONCLUSIVE",
     "Relevant to a future conversion/survival endpoint, not the current one."),
    ("s15", "Skull-stripping induced shortcut learning", 2026, "peer_reviewed",
     "https://pubmed.ncbi.nlm.nih.gov/41428317/", "shortcut_audit",
     "multi_cohort", "T1_MRI", "skull_stripped", "reported", "none",
     "mask_boundary_probe", "not_reported", "unavailable", "IDENTIFIER_VERIFIED",
     "Mask boundary can create a shortcut. Drives the mask-edge gate."),
    ("s16", "Plasma p-tau217 as a standalone biomarker", 2025, "peer_reviewed",
     "https://jamanetwork.com/journals/jama/fullarticle/10.1001/jamaneurol.2025.3217",
     "amyloid_status", "clinical_cohort", "plasma", "not_applicable",
     "reported", "reported", "not_reported", "not_reported", "unavailable",
     "IDENTIFIER_VERIFIED",
     "A future orthogonal branch that could reduce MRI-amyloid ambiguity."),
    ("s17", "MRI-to-PET image synthesis", 2026, "peer_reviewed",
     "https://pubmed.ncbi.nlm.nih.gov/41495456/", "pet_synthesis",
     "clinical_cohort", "MRI_to_PET", "not_fully_reported", "reported", "none",
     "not_reported", "not_reported", "unavailable", "INCONCLUSIVE",
     "Promising but ill-posed; cohort dependence remains a risk."),
    ("s18", "BrainAnytime: cross-modal pretraining", 2026, "preprint",
     "https://arxiv.org/abs/2605.13059", "representation_learning",
     "multi_cohort", "multimodal", "not_fully_reported", "reported", "none",
     "not_reported", "not_reported", "partial", "IDENTIFIER_VERIFIED",
     "A future direction, not a change to the strict Method X battery."),
]

# Outside the prespecified 2025-01-01 review window. Retained as background so
# the primary count stays honest rather than padded.
BACKGROUND_SOURCES = [
    {
        "id": "b01",
        "title": "MRI enhancement foundation model",
        "year": 2024,
        "tier": "peer_reviewed",
        "url": "https://www.nature.com/articles/s41551-024-01283-7",
        "citation_status": "IDENTIFIER_VERIFIED",
        "excluded_reason": "PUBLISHED_BEFORE_REVIEW_WINDOW_START_2025_01_01",
        "note": "Harmonization is potentially useful but would change locked preprocessing.",
    }
]

FIELDS = (
    "id", "title", "year", "tier", "url", "endpoint", "cohort", "modality",
    "preprocessing", "split", "external_validation", "confound_audit",
    "uncertainty", "code_weights", "citation_status", "design_impact",
)


def build_registry() -> dict:
    sources = [dict(zip(FIELDS, row)) for row in PRIMARY_SOURCES]
    registry = {
        "schema_version": 1,
        "review_id": "brainiac_caft_r0_20260808",
        "review_window": {"start": "2025-01-01", "freeze": LITERATURE_FREEZE},
        "target_source_range": list(R0_TARGET_SOURCE_RANGE),
        "minimum_peer_reviewed": R0_MIN_PEER_REVIEWED,
        "sources": sources,
        "background_sources_outside_window": BACKGROUND_SOURCES,
        "citation_tool": "citeprobe",
        "citation_corroboration": "INCONCLUSIVE",
        "citation_note": (
            "Crossref and OpenAlex returned transient errors during the run. No "
            "source is marked fully verified on the strength of a partial result."
        ),
    }
    validate_evidence_registry(registry)
    registry["status"] = evidence_registry_status(registry)["status"]
    registry["coverage"] = evidence_registry_status(registry)
    return registry


def build_novelty_matrix() -> dict:
    return {
        "schema_version": 1,
        "claim_ceiling": NOVELTY_CEILING,
        "gap_statement": (
            "Whether the global representation of a 3D foundation model, combined "
            "with the local 128-cubed representation of a locked CNN, adds "
            "information for PET-defined amyloid status while cohort/scanner "
            "leakage, MRI-amyloid ambiguity, calibration, and the option to "
            "abstain are explicitly controlled."
        ),
        "required_components": [
            {"id": "c1", "component": "BrainIAC global embedding",
             "prior_art_covers": True},
            {"id": "c2", "component": "CNN local 128-cubed with identical split",
             "prior_art_covers": True},
            {"id": "c3", "component": "PET-free inference for PET-defined amyloid",
             "prior_art_covers": True},
            {"id": "c4", "component": "Confound/scanner representation audit",
             "prior_art_covers": True},
            {"id": "c5", "component": "Ambiguity-aware calibrated abstention",
             "prior_art_covers": True},
        ],
        "prior_art_exact_match_found": False,
        "prior_art_note": (
            "No reviewed source implements all five components together in "
            "A4/LEARN. This is a candidate gap, not a proven novelty claim: the "
            "review is still incomplete."
        ),
        "downgrade_rule": (
            "If prior art matching all five components is found, the novelty "
            "claim converts to an honest replication/extension."
        ),
    }


def build_environment_spec() -> dict:
    return {
        "schema_version": 1,
        "created": False,
        "target_path": "H:/New project/.venvs/brainiac",
        "isolation_reason": (
            "BrainIAC dependencies must not perturb the environment that produced "
            "the locked v3 battery results."
        ),
        "python": "3.11",
        "declared_dependencies": [
            "torch", "monai", "numpy", "scipy", "scikit-learn", "nibabel",
        ],
        "pinned_versions": UNRESOLVED,
        "creation_gate": "R1a_BRAINIAC_AFTER_FIVE_MODEL_BATTERY_COMPLETES",
    }


def main() -> int:
    assert_current_r0_write_target(ROOT / "evidence_registry.json")
    ROOT.mkdir(parents=True, exist_ok=True)

    manifest = build_upstream_manifest()
    validate_upstream_manifest(manifest)
    (ROOT / "upstream_boundary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    registry = build_registry()
    (ROOT / "evidence_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    (ROOT / "novelty_matrix.json").write_text(
        json.dumps(build_novelty_matrix(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ROOT / "environment_spec.json").write_text(
        json.dumps(build_environment_spec(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    coverage = registry["coverage"]
    (ROOT / "review_protocol.md").write_text(
        f"""# BrainIAC-CAFT R0 review protocol

Systematic scoping review with a PRISMA-style log.

- Review window: 2025-01-01 to {LITERATURE_FREEZE} (literature freeze).
- Target: {R0_TARGET_SOURCE_RANGE[0]}-{R0_TARGET_SOURCE_RANGE[1]} primary studies,
  at least {R0_MIN_PEER_REVIEWED} peer-reviewed.
- Preprints are held in a separate evidence tier and never merged into the
  peer-reviewed count.
- Per source, record endpoint, cohort, modality, preprocessing, split, external
  validation, confound audit, uncertainty handling, and code/weight availability.

## Current state

| Measure | Value |
|---|---|
| Primary sources | {coverage['total_count']} |
| Peer-reviewed | {coverage['peer_reviewed_count']} |
| Preprints | {coverage['preprint_count']} |
| Unverified citations | {coverage['inconclusive_count']} |
| Status | `{coverage['status']}` |

Blockers: {', '.join(coverage['blockers']) or 'none'}

The registry is seeded with the sources already vetted in the corrective
program. Studies are **not** invented to reach the target range; the review
reports itself incomplete until real additional sources are screened.

## Claim ceiling

`{NOVELTY_CEILING}` until the matrix is complete. All BrainIAC results carry
`POST_HOC_INTERNAL_TECHNICAL_EVIDENCE`.
""",
        encoding="utf-8",
    )

    (ROOT / "search_log.md").write_text(
        f"""# R0 search log

Freeze date: {LITERATURE_FREEZE}

## Search axes

```text
brain MRI foundation model + Alzheimer/amyloid
T1 MRI + PET-defined amyloid + deep learning
BrainIAC/AnatCL/3D-Neuro-SimCLR + transfer
scanner/site/cohort robustness + brain MRI embeddings
biological ambiguity + MRI-to-amyloid
uncertainty/conformal/selective prediction + neuroimaging
```

## Screening ledger

| Stage | Count |
|---|---|
| Carried from program pre-review | {len(PRIMARY_SOURCES) + len(BACKGROUND_SOURCES)} |
| Excluded, outside review window | {len(BACKGROUND_SOURCES)} |
| Included as primary | {len(PRIMARY_SOURCES)} |
| Remaining to reach target | {max(0, R0_TARGET_SOURCE_RANGE[0] - len(PRIMARY_SOURCES))} |

## Citation validation

Tool: `citeprobe`. Identifier checks passed for most key sources. The AnatCL DOI
returned no multi-source candidate and is recorded `UNKNOWN`. Crossref and
OpenAlex returned transient errors, so overall corroboration is `INCONCLUSIVE`.
No source is promoted to verified on a partial tool result.
""",
        encoding="utf-8",
    )

    (ROOT / "claim_language_policy.md").write_text(
        f"""# Claim language policy

Every BrainIAC-CAFT output is labelled `POST_HOC_INTERNAL_TECHNICAL_EVIDENCE`.
The novelty ceiling is `{NOVELTY_CEILING}` until the review matrix is complete.

## Permitted

```text
Regions with higher attribution in the model decision
PET-free technical estimate of biomarker status
Internal, single-programme technical evidence
Indeterminate research output
```

## Forbidden

Enforced mechanically by `assert_claim_language` in
`src/research/brainiac_provenance.py`. The rejected set covers diagnosis
language, claims of external validation, causal or disease-region attribution,
clinical triage, and regional quantitative biomarker claims.

A confirmatory claim would require a genuinely untouched cohort, which this
programme does not have.
""",
        encoding="utf-8",
    )

    print(json.dumps({
        "root": str(ROOT),
        "artifacts": sorted(p.name for p in ROOT.iterdir()),
        "registry_status": coverage["status"],
        "primary_sources": coverage["total_count"],
        "peer_reviewed": coverage["peer_reviewed_count"],
        "preprints": coverage["preprint_count"],
        "upstream": validate_upstream_manifest(manifest)["status"],
        "pinned_commit": UPSTREAM_PINNED_COMMIT,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
