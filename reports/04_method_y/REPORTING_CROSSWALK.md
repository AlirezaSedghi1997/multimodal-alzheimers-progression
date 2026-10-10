> Internal/exploratory evidence snapshot; not external/clinical validation. A completed computation is not a positive science gate. This narrative predates the FP32 review; current sensitivity findings are summarized in docs/FP32_SENSITIVITY.md.

# Method Y item-addressed reporting assessment

Official item IDs map to project evidence hashes. Item text is not reproduced. This is not a completed manuscript or per-model certification. UNKNOWN and limitations remain visible; agent review does not replace original clinical annotation.

Sources: [TRIPOD_AI](https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf), [PROBAST_AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11931409/#tbl2), [CLAIM_2024](https://pure.eur.nl/ws/files/165782184/tejani-et-al-2024-checklist-for-artificial-intelligence-in-medical-imaging-_claim_-2024-update.pdf)

| Standard/phase | Item | Status | Evidence keys | Project disposition |
|---|---|---|---|---|
| TRIPOD_AI/reporting | 1 | PENDING_MANUSCRIPT | INTERPRETATION | Title must state internal preclinical prognosis, not diagnosis. |
| TRIPOD_AI/reporting | 2 | PENDING_MANUSCRIPT | RESULTS | Abstract not finalized; disclose negative and non-estimable findings. |
| TRIPOD_AI/reporting | 3a | EVIDENCE_LINKED | INTERPRETATION | PET incremental prognostic question; prior multimodal work acknowledged. |
| TRIPOD_AI/reporting | 3b | EVIDENCE_LINKED | POLICY | Research use only in selected local A4 population. |
| TRIPOD_AI/reporting | 3c | DOCUMENTED_LIMITATION | REVIEW | Demographic and selection disparities are not eliminated. |
| TRIPOD_AI/reporting | 4 | EVIDENCE_LINKED | REPAIR | Post-hoc internal evaluation; no retrospective preregistration claim. |
| TRIPOD_AI/reporting | 5a | EVIDENCE_LINKED | AMYLOID_ENDPOINT,TAU_ENDPOINT | A4 development; LEARN is not independent external validation. |
| TRIPOD_AI/reporting | 5b | PARTIAL | AMYLOID_ENDPOINT,TAU_ENDPOINT | Date definitions exist; publication accrual/follow-up table still required. |
| TRIPOD_AI/reporting | 6a | DOCUMENTED_LIMITATION | SITE | Exact inventory sidecars do not provide usable site; no fabricated centres. |
| TRIPOD_AI/reporting | 6b | EVIDENCE_LINKED | AMYLOID_ENDPOINT,TAU_ENDPOINT | Baseline and sustained progression eligibility are contracted. |
| TRIPOD_AI/reporting | 6c | EVIDENCE_LINKED | RESULTS,V4_CONTRACT | A4 treatment structure retained; not a treatment-effect study. |
| TRIPOD_AI/reporting | 7 | PARTIAL | V4_CONTRACT,AMYLOID_QUANT,TAU_QUANT | Pattern preprocessing recorded; quantitative equivalence unproven. |
| TRIPOD_AI/reporting | 8a | EVIDENCE_LINKED | AMYLOID_ENDPOINT,TAU_ENDPOINT | Future sustained CDR and censoring, fixed horizons. |
| TRIPOD_AI/reporting | 8b | UNKNOWN_SOURCE_ASSESSOR_DETAILS | AMYLOID_ENDPOINT,TAU_ENDPOINT | Original clinical rater qualifications are not verified here. |
| TRIPOD_AI/reporting | 8c | UNKNOWN_SOURCE_BLINDING | AMYLOID_ENDPOINT,TAU_ENDPOINT | Computational separation does not prove source clinical blinding. |
| TRIPOD_AI/reporting | 9a | EVIDENCE_LINKED | V4_CONTRACT | Feature sets and factorial variants fixed in contract. |
| TRIPOD_AI/reporting | 9b | PARTIAL | V4_CONTRACT,AMYLOID_QUANT,TAU_QUANT | Measured values available; image-scale equivalence is not established. |
| TRIPOD_AI/reporting | 9c | UNKNOWN_SOURCE_ASSESSOR_DETAILS | POLICY | Agent QC is not expert-rater adjudication. |
| TRIPOD_AI/reporting | 10 | DOCUMENTED_LIMITATION | RESULTS | Available cohort; sparse tau/manufacturer support; no retrospective power guarantee. |
| TRIPOD_AI/reporting | 11 | PARTIAL | RESULTS,REVIEW | Imputation and PACC missingness described; MAR not proven. |
| TRIPOD_AI/reporting | 12a | EVIDENCE_LINKED | V4_CONTRACT,V4_STATE | Subject-grouped nested design; LMO is a separate three-manufacturer experiment. |
| TRIPOD_AI/reporting | 12b | EVIDENCE_LINKED | V4_CONTRACT | Fold-local tabular processing and within-brain image pattern normalization. |
| TRIPOD_AI/reporting | 12c | EVIDENCE_LINKED | V4_CONTRACT | Model, HPO metric and variant contracts; A4 LMO is post-hoc. |
| TRIPOD_AI/reporting | 12d | DOCUMENTED_LIMITATION | RESULTS,SITE | Manufacturer analysis exists; site and tau multi-manufacturer transport unsupported. |
| TRIPOD_AI/reporting | 12e | EVIDENCE_LINKED | RESULTS,REPAIR | IPCW metrics and support limits; invalid risk-score-delta inference removed. |
| TRIPOD_AI/reporting | 12f | NOT_PERFORMED | REVIEW | No holdout-driven recalibration credited to original model. |
| TRIPOD_AI/reporting | 12g | EVIDENCE_LINKED | V4_CONTRACT,V5_CONTRACT | Locked model predictions reused; v5 reanalysis does not retrain CNN. |
| TRIPOD_AI/reporting | 13 | EVIDENCE_LINKED | V4_CONTRACT | Survival likelihood; augmentation is not independent participants/events. |
| TRIPOD_AI/reporting | 14 | DOCUMENTED_LIMITATION | REVIEW,SUBGROUP | Cognition conditional Brier CIs added; QC/cohort lack development variation. No fairness claim. |
| TRIPOD_AI/reporting | 15 | EVIDENCE_LINKED | V4_CONTRACT | Survival/event-risk outputs; no clinical decision threshold validated. |
| TRIPOD_AI/reporting | 16 | EVIDENCE_LINKED | RESULTS | Held-manufacturer and image-selected populations distinguished. |
| TRIPOD_AI/reporting | 17 | UNKNOWN_ADMINISTRATIVE | POLICY | Do not invent IRB approval, consent waiver or DUA authority. |
| TRIPOD_AI/reporting | 18a | UNKNOWN_ADMINISTRATIVE | INTERPRETATION | Author funding declaration not verified in this code audit. |
| TRIPOD_AI/reporting | 18b | UNKNOWN_ADMINISTRATIVE | INTERPRETATION | Author conflicts declaration not inferred by an agent. |
| TRIPOD_AI/reporting | 18c | EVIDENCE_LINKED | REPAIR,V4_CONTRACT,V5_CONTRACT | Versioned local protocols and supersessions available. |
| TRIPOD_AI/reporting | 18d | NOT_DEMONSTRATED | REPAIR | Local lock is not public prospective registration. |
| TRIPOD_AI/reporting | 18e | EVIDENCE_LINKED | PRIVACY | Restricted subject data; release aggregate only. |
| TRIPOD_AI/reporting | 18f | PARTIAL | V5_CONTRACT | Code exists locally; public release licence/dependency archive not certified. |
| TRIPOD_AI/reporting | 19 | NOT_DOCUMENTED | INTERPRETATION | Agent review is not patient/public involvement. |
| TRIPOD_AI/reporting | 20a | PARTIAL | RESULTS,REVIEW | Use exact image subset counts and PACC flow; not largest regional cohort. |
| TRIPOD_AI/reporting | 20b | PARTIAL | RESULTS,REVIEW | Branch-specific counts exist; manuscript characteristics table not finalized. |
| TRIPOD_AI/reporting | 20c | PARTIAL | RESULTS,REVIEW | Selection/missingness diagnostics retained; no claim of equivalent distributions. |
| TRIPOD_AI/reporting | 21 | EVIDENCE_LINKED | V4_CONTRACT,V4_STATE,RESULTS | Count patients, not seeds/scans; five-fold factorial differs from LMO. |
| TRIPOD_AI/reporting | 22 | PARTIAL | V4_CONTRACT | Architecture/checkpoints local; no approved deployed clinical model. |
| TRIPOD_AI/reporting | 23a | EVIDENCE_LINKED | RESULTS,REVIEW | Conditional paired CIs; support failures kept; calibration not PASS. |
| TRIPOD_AI/reporting | 23b | EVIDENCE_LINKED | RESULTS,SITE | Manufacturer heterogeneity; site cannot be estimated. |
| TRIPOD_AI/reporting | 24 | NOT_PERFORMED | REVIEW | No external or holdout model update. |
| TRIPOD_AI/reporting | 25 | EVIDENCE_LINKED | INTERPRETATION | A4 LMO superiority unproven; paired tau negative remains. |
| TRIPOD_AI/reporting | 26 | EVIDENCE_LINKED | INTERPRETATION | Internal exposure, censoring, selection, scale and XAI limitations explicit. |
| TRIPOD_AI/reporting | 27a | RESEARCH_ONLY | POLICY | Invalid inputs fail; missing PET is not zero-filled synthetic imaging. |
| TRIPOD_AI/reporting | 27b | RESEARCH_ONLY | POLICY | No clinical usability claim; automated/agent review honestly labelled. |
| TRIPOD_AI/reporting | 27c | EVIDENCE_LINKED | INTERPRETATION | External work deferred; fusion is future, not an accomplished result. |
| CLAIM_2024/reporting | 1 | PENDING_MANUSCRIPT | INTERPRETATION | Image survival CNN; no AD diagnostic claim. |
| CLAIM_2024/reporting | 2 | PENDING_MANUSCRIPT | RESULTS | Abstract requires exact experiment counts and limitations. |
| CLAIM_2024/reporting | 3 | EVIDENCE_LINKED | INTERPRETATION | Local incremental prognosis rationale. |
| CLAIM_2024/reporting | 4 | EVIDENCE_LINKED | REPAIR | Historical exposure and post-hoc analyses disclosed. |
| CLAIM_2024/reporting | 5 | EVIDENCE_LINKED | REPAIR | Secondary retrospective modelling of existing longitudinal data. |
| CLAIM_2024/reporting | 6 | EVIDENCE_LINKED | POLICY | Exploratory internal progression prediction. |
| CLAIM_2024/reporting | 7 | EVIDENCE_LINKED | AMYLOID_ENDPOINT,TAU_ENDPOINT | Local A4/LEARN; no independent external data. |
| CLAIM_2024/reporting | 8 | EVIDENCE_LINKED | AMYLOID_ENDPOINT,TAU_ENDPOINT | Contracted baseline/risk-set eligibility. |
| CLAIM_2024/reporting | 9 | PARTIAL | V4_CONTRACT,AMYLOID_QUANT,TAU_QUANT | Pattern scale and transform lineage; no official quantitative equivalence. |
| CLAIM_2024/reporting | 10 | EVIDENCE_LINKED | REVIEW | Image-selected cohort and outcome-blind diagnostic subset distinguished. |
| CLAIM_2024/reporting | 11 | PARTIAL | PRIVACY | Aggregate privacy scan passed; not legal de-identification certification. |
| CLAIM_2024/reporting | 12 | PARTIAL | RESULTS | Observed missingness described; informative dropout remains possible. |
| CLAIM_2024/reporting | 13 | PARTIAL | TRACER,SITE | Tracer evidence retained; incomplete source UID/site metadata. |
| CLAIM_2024/reporting | 14 | EVIDENCE_LINKED | AMYLOID_ENDPOINT,TAU_ENDPOINT | CDR progression definition, not image-derived positivity target. |
| CLAIM_2024/reporting | 15 | EVIDENCE_LINKED | INTERPRETATION | Future progression instead of circular PET target. |
| CLAIM_2024/reporting | 16 | UNKNOWN_SOURCE_ASSESSOR_DETAILS | AMYLOID_ENDPOINT,TAU_ENDPOINT | Clinical source rater qualifications not reconstructed. |
| CLAIM_2024/reporting | 17 | PARTIAL | AMYLOID_ENDPOINT,TAU_ENDPOINT | Automated endpoint rule consistent; source rater blinding unknown. |
| CLAIM_2024/reporting | 18 | NOT_AVAILABLE | AMYLOID_ENDPOINT,TAU_ENDPOINT | No source rater-repeat data; cannot invent agreement. |
| CLAIM_2024/reporting | 19 | EVIDENCE_LINKED | V4_CONTRACT | Locked subject nested partitions. |
| CLAIM_2024/reporting | 20 | EVIDENCE_LINKED | V4_CONTRACT | Subject grouping, not independent scan/seed observations. |
| CLAIM_2024/reporting | 21 | DOCUMENTED_LIMITATION | RESULTS | Sparse supported groups; no retrospective adequacy proof. |
| CLAIM_2024/reporting | 22 | EVIDENCE_LINKED | V4_CONTRACT | Locked variant architectures, strict reconstruction for XAI pilot. |
| CLAIM_2024/reporting | 23 | EVIDENCE_LINKED | V4_CONTRACT,V5_CONTRACT | Software/runtime contracts; no whole-repository certification. |
| CLAIM_2024/reporting | 24 | EVIDENCE_LINKED | V4_CONTRACT | Initialization and seeds under run contract. |
| CLAIM_2024/reporting | 25 | EVIDENCE_LINKED | V4_CONTRACT | Training/augmentation and resume policies distinct from inference. |
| CLAIM_2024/reporting | 26 | EVIDENCE_LINKED | V4_CONTRACT,REPAIR | Inner-only selection; global A4 diagnostic cannot become unbiased winner. |
| CLAIM_2024/reporting | 27 | NOT_IMPLEMENTED_FOR_XYZ | INTERPRETATION | Proposed fusion not a completed ensemble result. |
| CLAIM_2024/reporting | 28 | EVIDENCE_LINKED | RESULTS | IPCW IBS/Uno; support and horizons explicit. |
| CLAIM_2024/reporting | 29 | EVIDENCE_LINKED | RESULTS | Subject-paired bootstrap conditional on trained models. |
| CLAIM_2024/reporting | 30 | DOCUMENTED_LIMITATION | RESULTS,SITE | Amyloid manufacturer analysis; tau/site limits retained. |
| CLAIM_2024/reporting | 31 | DOCUMENTED_LIMITATION | XAI,XAI_CONTRACT,SPATIAL,MATRIX | Signed-CAM/LayerCAM diagnosis followed by a fixed-panel individual-input IG pilot with numerical, randomization and equal-volume voxel controls; full-panel review and spatial validity remain open. |
| CLAIM_2024/reporting | 32 | EVIDENCE_LINKED | V4_STATE,RESULTS | Internal results only; technical completion is not scientific PASS. |
| CLAIM_2024/reporting | 33 | NOT_PERFORMED | POLICY | No external dataset available. |
| CLAIM_2024/reporting | 34 | NOT_APPLICABLE_TO_NEW_TRIAL | REPAIR | No new interventional trial; source A4 registration remains source study fact. |
| CLAIM_2024/reporting | 35 | PARTIAL | REVIEW,RESULTS | Exact experiment counts required; scans not subjects. |
| CLAIM_2024/reporting | 36 | PARTIAL | REVIEW | Manuscript partition-characteristics table still to render. |
| CLAIM_2024/reporting | 37 | EVIDENCE_LINKED | RESULTS,REVIEW | CIs and negative incremental findings recorded. |
| CLAIM_2024/reporting | 38 | NOT_APPLICABLE_DIAGNOSTIC | POLICY | Censoring-aware prognosis, not diagnostic sensitivity at an invented cutoff. |
| CLAIM_2024/reporting | 39 | PARTIAL | REVIEW,XAI | Calibration and saliency failure analysis; no case-level clinical adjudication. |
| CLAIM_2024/reporting | 40 | EVIDENCE_LINKED | INTERPRETATION | No external, scale equivalence or anatomic claim. |
| CLAIM_2024/reporting | 41 | RESEARCH_ONLY | POLICY | Clinical deployment and usability not validated. |
| CLAIM_2024/reporting | 42 | EVIDENCE_LINKED | V4_CONTRACT,V5_CONTRACT,REPAIR | Versioned protocols; old artifacts preserved. |
| CLAIM_2024/reporting | 43 | PARTIAL | PRIVACY,V5_CONTRACT | Local code/checkpoints; subject data not released. |
| CLAIM_2024/reporting | 44 | UNKNOWN_ADMINISTRATIVE | INTERPRETATION | Funding/support cannot be inferred from repo execution. |
| PROBAST_AI/development | 1.1 | PY | AMYLOID_ENDPOINT,TAU_ENDPOINT | Appropriate local longitudinal sources for narrow internal question. |
| PROBAST_AI/development | 1.2 | PY | AMYLOID_ENDPOINT,TAU_ENDPOINT | Future progression avoids same-PET circular target. |
| PROBAST_AI/development | 1.3 | PN | REVIEW | Image availability and A4 enrichment limit representativeness. |
| PROBAST_AI/development | 2.1 | NI | AMYLOID_QUANT,TAU_QUANT | Cross-scanner quantitative equivalence is unproven. |
| PROBAST_AI/development | 2.2 | PY | V4_CONTRACT | Pattern pipeline contracted within branch, not cross-tracer equivalence. |
| PROBAST_AI/development | 2.3 | NI | REPAIR | Computational leakage protection does not prove acquisition/measurement blinding. |
| PROBAST_AI/development | 2.4 | PY | AMYLOID_ENDPOINT,TAU_ENDPOINT | Y baseline is before future endpoint; this does not approve legacy Z. |
| PROBAST_AI/development | 3.1 | PY | AMYLOID_ENDPOINT,TAU_ENDPOINT | Sustained CDR progression algorithm is explicit. |
| PROBAST_AI/development | 3.2 | NI | AMYLOID_ENDPOINT,TAU_ENDPOINT | Original clinical inter-site measurement equivalence unverified. |
| PROBAST_AI/development | 3.3 | NI | AMYLOID_ENDPOINT,TAU_ENDPOINT | Original outcome-rater blinding not documented here. |
| PROBAST_AI/development | 3.4 | PY | AMYLOID_ENDPOINT,TAU_ENDPOINT | Post-origin endpoint and follow-up support rules explicit. |
| PROBAST_AI/development | 4.1 | PN | RESULTS | Event support is sparse for tau complexity/manufacturer strata. |
| PROBAST_AI/development | 4.2 | PY | V4_CONTRACT | Fold-local scaling and fixed features. |
| PROBAST_AI/development | 4.3 | NI | RESULTS | IPCW models censoring; conditional independence/MAR cannot be verified. |
| PROBAST_AI/development | 4.4 | NA | V4_CONTRACT | No class balancing interpreted as extra events. |
| PROBAST_AI/development | 4.5 | PY | V4_CONTRACT | Nested HPO/regularization; post-hoc variant choice still restricted. |
| PROBAST_AI/evaluation | 1.1 | PY | AMYLOID_ENDPOINT,TAU_ENDPOINT | Appropriate local longitudinal sources for narrow internal question. |
| PROBAST_AI/evaluation | 1.2 | PY | AMYLOID_ENDPOINT,TAU_ENDPOINT | Future progression avoids same-PET circular target. |
| PROBAST_AI/evaluation | 1.3 | PN | REVIEW | Image availability and A4 enrichment limit representativeness. |
| PROBAST_AI/evaluation | 2.1 | NI | AMYLOID_QUANT,TAU_QUANT | Cross-scanner quantitative equivalence is unproven. |
| PROBAST_AI/evaluation | 2.2 | PY | V4_CONTRACT | Pattern pipeline contracted within branch, not cross-tracer equivalence. |
| PROBAST_AI/evaluation | 2.3 | NI | REPAIR | Computational leakage protection does not prove acquisition/measurement blinding. |
| PROBAST_AI/evaluation | 2.4 | PY | AMYLOID_ENDPOINT,TAU_ENDPOINT | Y baseline is before future endpoint; this does not approve legacy Z. |
| PROBAST_AI/evaluation | 3.1 | PY | AMYLOID_ENDPOINT,TAU_ENDPOINT | Sustained CDR progression algorithm is explicit. |
| PROBAST_AI/evaluation | 3.2 | NI | AMYLOID_ENDPOINT,TAU_ENDPOINT | Original clinical inter-site measurement equivalence unverified. |
| PROBAST_AI/evaluation | 3.3 | NI | AMYLOID_ENDPOINT,TAU_ENDPOINT | Original outcome-rater blinding not documented here. |
| PROBAST_AI/evaluation | 3.4 | PY | AMYLOID_ENDPOINT,TAU_ENDPOINT | Post-origin endpoint and follow-up support rules explicit. |
| PROBAST_AI/evaluation | 4.1 | Y | V4_STATE,RESULTS | Held-out internal predictions exist, not training accuracy only. |
| PROBAST_AI/evaluation | 4.2 | PN | RESULTS | Wide intervals and sparse manufacturers; no power guarantee. |
| PROBAST_AI/evaluation | 4.3 | NI | RESULTS | Censoring-aware metrics do not remove unmeasured informative censoring. |
| PROBAST_AI/evaluation | 4.4 | PY | V4_CONTRACT | Held-out patients are not augmented as independent observations. |
| PROBAST_AI/evaluation | 4.5 | PN | REPAIR | Historical test exposure remains; corrective rerun is not untouched confirmation. |
| PROBAST_AI/evaluation | 4.6 | PN | RESULTS | Fixed-model bootstrap excludes model-selection/retraining variance. |
| PROBAST_AI/evaluation | 4.7 | PY | RESULTS,REVIEW | Multiple valid survival metrics; clinical net benefit not claimed. |

Hash/path lookup: `evidence_registry.json`. Counts: 130.
PROBAST answers: Y=yes, PY=probably yes, PN=probably no, NI=insufficient information, NA=not applicable. No per-model low-risk conclusion is issued. Clinical deployment applicability is high concern; source-assessor details remain unknown.
