"""Item-addressed reporting evidence, not manuscript certification or refitting."""
from __future__ import annotations
import csv
import hashlib
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.method_y_revival.atomic_io import atomic_write_json
from src.utils.io_helpers import atomic_write_text

OUT = ROOT / 'results/method_y_reporting_crosswalk_v1'
SOURCES = {
    'TRIPOD_AI': 'https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf',
    'PROBAST_AI': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11931409/#tbl2',
    'CLAIM_2024': 'https://pure.eur.nl/ws/files/165782184/tejani-et-al-2024-checklist-for-artificial-intelligence-in-medical-imaging-_claim_-2024-update.pdf',
}
EVIDENCE = {
    'RESULTS': 'results/method_y_local_validation_v5/public_bundle/facts.json',
    'REVIEW': 'results/method_y_post_v5_review_20260929/public_bundle/review.json',
    'SITE': 'results/method_y_exact_site_screen_v1/public_bundle/site_screen.json',
    'XAI': 'results/method_y_xai_signed_diagnostic_v1/public_bundle/summary.json',
    'XAI_CONTRACT': 'results/method_y_xai_signed_diagnostic_v1/private/contract.json',
    'SPATIAL': 'results/xy_spatial_validation_v3/public_bundle/summary.json',
    'SUBGROUP': 'results/method_y_remaining_subgroups_v2/public_bundle/subgroups.json',
    'MATRIX': 'docs/METHOD_Y_REMAINING_WORK_MATRIX_20260930_FA.md',
    'V4_CONTRACT': 'results/method_y_revival_v4_image_factorial_v1/run_contract.json',
    'V5_CONTRACT': 'results/method_y_local_validation_v5/run_contract.json',
    'V4_STATE': 'results/method_y_revival_v4_image_factorial_v1/orchestration_state.json',
    'AMYLOID_ENDPOINT': 'results/method_y_revival_v2/endpoints/amyloid_endpoint_contract.json',
    'TAU_ENDPOINT': 'results/method_y_revival_v2/endpoints/tau_endpoint_contract.json',
    'AMYLOID_QUANT': 'results/method_y_revival_v2/provenance/scan_level_amyloid_image_regional_validation.json',
    'TAU_QUANT': 'results/method_y_revival_v2/provenance/scan_level_tau_quantitative_validation.json',
    'TRACER': 'results/method_y_revival_v2/provenance/tau_tracer_consistency_audit_v2.json',
    'POLICY': 'docs/decisions/ADR-2026-08-24-method-y-image-dual-gate.md',
    'REPAIR': 'docs/METHOD_Y_REPAIR_AND_FUSION_20260929_FA.md',
    'INTERPRETATION': 'docs/METHOD_Y_V5_RESULTS_AND_XYZ_FUSION_20260929_FA.md',
    'PRIVACY': 'results/method_y_post_v5_review_20260929/public_bundle/privacy_scan.json',
}

# The official item text is not copied. Each row records this project's answer.
T = [
 ('1','PENDING_MANUSCRIPT','INTERPRETATION','Title must state internal preclinical prognosis, not diagnosis.'),
 ('2','PENDING_MANUSCRIPT','RESULTS','Abstract not finalized; disclose negative and non-estimable findings.'),
 ('3a','EVIDENCE_LINKED','INTERPRETATION','PET incremental prognostic question; prior multimodal work acknowledged.'),
 ('3b','EVIDENCE_LINKED','POLICY','Research use only in selected local A4 population.'),
 ('3c','DOCUMENTED_LIMITATION','REVIEW','Demographic and selection disparities are not eliminated.'),
 ('4','EVIDENCE_LINKED','REPAIR','Post-hoc internal evaluation; no retrospective preregistration claim.'),
 ('5a','EVIDENCE_LINKED','AMYLOID_ENDPOINT,TAU_ENDPOINT','A4 development; LEARN is not independent external validation.'),
 ('5b','PARTIAL','AMYLOID_ENDPOINT,TAU_ENDPOINT','Date definitions exist; publication accrual/follow-up table still required.'),
 ('6a','DOCUMENTED_LIMITATION','SITE','Exact inventory sidecars do not provide usable site; no fabricated centres.'),
 ('6b','EVIDENCE_LINKED','AMYLOID_ENDPOINT,TAU_ENDPOINT','Baseline and sustained progression eligibility are contracted.'),
 ('6c','EVIDENCE_LINKED','RESULTS,V4_CONTRACT','A4 treatment structure retained; not a treatment-effect study.'),
 ('7','PARTIAL','V4_CONTRACT,AMYLOID_QUANT,TAU_QUANT','Pattern preprocessing recorded; quantitative equivalence unproven.'),
 ('8a','EVIDENCE_LINKED','AMYLOID_ENDPOINT,TAU_ENDPOINT','Future sustained CDR and censoring, fixed horizons.'),
 ('8b','UNKNOWN_SOURCE_ASSESSOR_DETAILS','AMYLOID_ENDPOINT,TAU_ENDPOINT','Original clinical rater qualifications are not verified here.'),
 ('8c','UNKNOWN_SOURCE_BLINDING','AMYLOID_ENDPOINT,TAU_ENDPOINT','Computational separation does not prove source clinical blinding.'),
 ('9a','EVIDENCE_LINKED','V4_CONTRACT','Feature sets and factorial variants fixed in contract.'),
 ('9b','PARTIAL','V4_CONTRACT,AMYLOID_QUANT,TAU_QUANT','Measured values available; image-scale equivalence is not established.'),
 ('9c','UNKNOWN_SOURCE_ASSESSOR_DETAILS','POLICY','Agent QC is not expert-rater adjudication.'),
 ('10','DOCUMENTED_LIMITATION','RESULTS','Available cohort; sparse tau/manufacturer support; no retrospective power guarantee.'),
 ('11','PARTIAL','RESULTS,REVIEW','Imputation and PACC missingness described; MAR not proven.'),
 ('12a','EVIDENCE_LINKED','V4_CONTRACT,V4_STATE','Subject-grouped nested design; LMO is a separate three-manufacturer experiment.'),
 ('12b','EVIDENCE_LINKED','V4_CONTRACT','Fold-local tabular processing and within-brain image pattern normalization.'),
 ('12c','EVIDENCE_LINKED','V4_CONTRACT','Model, HPO metric and variant contracts; A4 LMO is post-hoc.'),
 ('12d','DOCUMENTED_LIMITATION','RESULTS,SITE','Manufacturer analysis exists; site and tau multi-manufacturer transport unsupported.'),
 ('12e','EVIDENCE_LINKED','RESULTS,REPAIR','IPCW metrics and support limits; invalid risk-score-delta inference removed.'),
 ('12f','NOT_PERFORMED','REVIEW','No holdout-driven recalibration credited to original model.'),
 ('12g','EVIDENCE_LINKED','V4_CONTRACT,V5_CONTRACT','Locked model predictions reused; v5 reanalysis does not retrain CNN.'),
 ('13','EVIDENCE_LINKED','V4_CONTRACT','Survival likelihood; augmentation is not independent participants/events.'),
 ('14','DOCUMENTED_LIMITATION','REVIEW,SUBGROUP','Cognition conditional Brier CIs added; QC/cohort lack development variation. No fairness claim.'),
 ('15','EVIDENCE_LINKED','V4_CONTRACT','Survival/event-risk outputs; no clinical decision threshold validated.'),
 ('16','EVIDENCE_LINKED','RESULTS','Held-manufacturer and image-selected populations distinguished.'),
 ('17','UNKNOWN_ADMINISTRATIVE','POLICY','Do not invent IRB approval, consent waiver or DUA authority.'),
 ('18a','UNKNOWN_ADMINISTRATIVE','INTERPRETATION','Author funding declaration not verified in this code audit.'),
 ('18b','UNKNOWN_ADMINISTRATIVE','INTERPRETATION','Author conflicts declaration not inferred by an agent.'),
 ('18c','EVIDENCE_LINKED','REPAIR,V4_CONTRACT,V5_CONTRACT','Versioned local protocols and supersessions available.'),
 ('18d','NOT_DEMONSTRATED','REPAIR','Local lock is not public prospective registration.'),
 ('18e','EVIDENCE_LINKED','PRIVACY','Restricted subject data; release aggregate only.'),
 ('18f','PARTIAL','V5_CONTRACT','Code exists locally; public release licence/dependency archive not certified.'),
 ('19','NOT_DOCUMENTED','INTERPRETATION','Agent review is not patient/public involvement.'),
 ('20a','PARTIAL','RESULTS,REVIEW','Use exact image subset counts and PACC flow; not largest regional cohort.'),
 ('20b','PARTIAL','RESULTS,REVIEW','Branch-specific counts exist; manuscript characteristics table not finalized.'),
 ('20c','PARTIAL','RESULTS,REVIEW','Selection/missingness diagnostics retained; no claim of equivalent distributions.'),
 ('21','EVIDENCE_LINKED','V4_CONTRACT,V4_STATE,RESULTS','Count patients, not seeds/scans; five-fold factorial differs from LMO.'),
 ('22','PARTIAL','V4_CONTRACT','Architecture/checkpoints local; no approved deployed clinical model.'),
 ('23a','EVIDENCE_LINKED','RESULTS,REVIEW','Conditional paired CIs; support failures kept; calibration not PASS.'),
 ('23b','EVIDENCE_LINKED','RESULTS,SITE','Manufacturer heterogeneity; site cannot be estimated.'),
 ('24','NOT_PERFORMED','REVIEW','No external or holdout model update.'),
 ('25','EVIDENCE_LINKED','INTERPRETATION','A4 LMO superiority unproven; paired tau negative remains.'),
 ('26','EVIDENCE_LINKED','INTERPRETATION','Internal exposure, censoring, selection, scale and XAI limitations explicit.'),
 ('27a','RESEARCH_ONLY','POLICY','Invalid inputs fail; missing PET is not zero-filled synthetic imaging.'),
 ('27b','RESEARCH_ONLY','POLICY','No clinical usability claim; automated/agent review honestly labelled.'),
 ('27c','EVIDENCE_LINKED','INTERPRETATION','External work deferred; fusion is future, not an accomplished result.'),
]
C = [
 ('1','PENDING_MANUSCRIPT','INTERPRETATION','Image survival CNN; no AD diagnostic claim.'),
 ('2','PENDING_MANUSCRIPT','RESULTS','Abstract requires exact experiment counts and limitations.'),
 ('3','EVIDENCE_LINKED','INTERPRETATION','Local incremental prognosis rationale.'),
 ('4','EVIDENCE_LINKED','REPAIR','Historical exposure and post-hoc analyses disclosed.'),
 ('5','EVIDENCE_LINKED','REPAIR','Secondary retrospective modelling of existing longitudinal data.'),
 ('6','EVIDENCE_LINKED','POLICY','Exploratory internal progression prediction.'),
 ('7','EVIDENCE_LINKED','AMYLOID_ENDPOINT,TAU_ENDPOINT','Local A4/LEARN; no independent external data.'),
 ('8','EVIDENCE_LINKED','AMYLOID_ENDPOINT,TAU_ENDPOINT','Contracted baseline/risk-set eligibility.'),
 ('9','PARTIAL','V4_CONTRACT,AMYLOID_QUANT,TAU_QUANT','Pattern scale and transform lineage; no official quantitative equivalence.'),
 ('10','EVIDENCE_LINKED','REVIEW','Image-selected cohort and outcome-blind diagnostic subset distinguished.'),
 ('11','PARTIAL','PRIVACY','Aggregate privacy scan passed; not legal de-identification certification.'),
 ('12','PARTIAL','RESULTS','Observed missingness described; informative dropout remains possible.'),
 ('13','PARTIAL','TRACER,SITE','Tracer evidence retained; incomplete source UID/site metadata.'),
 ('14','EVIDENCE_LINKED','AMYLOID_ENDPOINT,TAU_ENDPOINT','CDR progression definition, not image-derived positivity target.'),
 ('15','EVIDENCE_LINKED','INTERPRETATION','Future progression instead of circular PET target.'),
 ('16','UNKNOWN_SOURCE_ASSESSOR_DETAILS','AMYLOID_ENDPOINT,TAU_ENDPOINT','Clinical source rater qualifications not reconstructed.'),
 ('17','PARTIAL','AMYLOID_ENDPOINT,TAU_ENDPOINT','Automated endpoint rule consistent; source rater blinding unknown.'),
 ('18','NOT_AVAILABLE','AMYLOID_ENDPOINT,TAU_ENDPOINT','No source rater-repeat data; cannot invent agreement.'),
 ('19','EVIDENCE_LINKED','V4_CONTRACT','Locked subject nested partitions.'),
 ('20','EVIDENCE_LINKED','V4_CONTRACT','Subject grouping, not independent scan/seed observations.'),
 ('21','DOCUMENTED_LIMITATION','RESULTS','Sparse supported groups; no retrospective adequacy proof.'),
 ('22','EVIDENCE_LINKED','V4_CONTRACT','Locked variant architectures, strict reconstruction for XAI pilot.'),
 ('23','EVIDENCE_LINKED','V4_CONTRACT,V5_CONTRACT','Software/runtime contracts; no whole-repository certification.'),
 ('24','EVIDENCE_LINKED','V4_CONTRACT','Initialization and seeds under run contract.'),
 ('25','EVIDENCE_LINKED','V4_CONTRACT','Training/augmentation and resume policies distinct from inference.'),
 ('26','EVIDENCE_LINKED','V4_CONTRACT,REPAIR','Inner-only selection; global A4 diagnostic cannot become unbiased winner.'),
 ('27','NOT_IMPLEMENTED_FOR_XYZ','INTERPRETATION','Proposed fusion not a completed ensemble result.'),
 ('28','EVIDENCE_LINKED','RESULTS','IPCW IBS/Uno; support and horizons explicit.'),
 ('29','EVIDENCE_LINKED','RESULTS','Subject-paired bootstrap conditional on trained models.'),
 ('30','DOCUMENTED_LIMITATION','RESULTS,SITE','Amyloid manufacturer analysis; tau/site limits retained.'),
 ('31','DOCUMENTED_LIMITATION','XAI,XAI_CONTRACT,SPATIAL,MATRIX','Signed-CAM/LayerCAM diagnosis followed by a fixed-panel individual-input IG pilot with numerical, randomization and equal-volume voxel controls; full-panel review and spatial validity remain open.'),
 ('32','EVIDENCE_LINKED','V4_STATE,RESULTS','Internal results only; technical completion is not scientific PASS.'),
 ('33','NOT_PERFORMED','POLICY','No external dataset available.'),
 ('34','NOT_APPLICABLE_TO_NEW_TRIAL','REPAIR','No new interventional trial; source A4 registration remains source study fact.'),
 ('35','PARTIAL','REVIEW,RESULTS','Exact experiment counts required; scans not subjects.'),
 ('36','PARTIAL','REVIEW','Manuscript partition-characteristics table still to render.'),
 ('37','EVIDENCE_LINKED','RESULTS,REVIEW','CIs and negative incremental findings recorded.'),
 ('38','NOT_APPLICABLE_DIAGNOSTIC','POLICY','Censoring-aware prognosis, not diagnostic sensitivity at an invented cutoff.'),
 ('39','PARTIAL','REVIEW,XAI','Calibration and saliency failure analysis; no case-level clinical adjudication.'),
 ('40','EVIDENCE_LINKED','INTERPRETATION','No external, scale equivalence or anatomic claim.'),
 ('41','RESEARCH_ONLY','POLICY','Clinical deployment and usability not validated.'),
 ('42','EVIDENCE_LINKED','V4_CONTRACT,V5_CONTRACT,REPAIR','Versioned protocols; old artifacts preserved.'),
 ('43','PARTIAL','PRIVACY,V5_CONTRACT','Local code/checkpoints; subject data not released.'),
 ('44','UNKNOWN_ADMINISTRATIVE','INTERPRETATION','Funding/support cannot be inferred from repo execution.'),
]
P_SHARED = [
 ('1.1','PY','AMYLOID_ENDPOINT,TAU_ENDPOINT','Appropriate local longitudinal sources for narrow internal question.'),
 ('1.2','PY','AMYLOID_ENDPOINT,TAU_ENDPOINT','Future progression avoids same-PET circular target.'),
 ('1.3','PN','REVIEW','Image availability and A4 enrichment limit representativeness.'),
 ('2.1','NI','AMYLOID_QUANT,TAU_QUANT','Cross-scanner quantitative equivalence is unproven.'),
 ('2.2','PY','V4_CONTRACT','Pattern pipeline contracted within branch, not cross-tracer equivalence.'),
 ('2.3','NI','REPAIR','Computational leakage protection does not prove acquisition/measurement blinding.'),
 ('2.4','PY','AMYLOID_ENDPOINT,TAU_ENDPOINT','Y baseline is before future endpoint; this does not approve legacy Z.'),
 ('3.1','PY','AMYLOID_ENDPOINT,TAU_ENDPOINT','Sustained CDR progression algorithm is explicit.'),
 ('3.2','NI','AMYLOID_ENDPOINT,TAU_ENDPOINT','Original clinical inter-site measurement equivalence unverified.'),
 ('3.3','NI','AMYLOID_ENDPOINT,TAU_ENDPOINT','Original outcome-rater blinding not documented here.'),
 ('3.4','PY','AMYLOID_ENDPOINT,TAU_ENDPOINT','Post-origin endpoint and follow-up support rules explicit.'),
]
P_DEV = [
 ('4.1','PN','RESULTS','Event support is sparse for tau complexity/manufacturer strata.'),
 ('4.2','PY','V4_CONTRACT','Fold-local scaling and fixed features.'),
 ('4.3','NI','RESULTS','IPCW models censoring; conditional independence/MAR cannot be verified.'),
 ('4.4','NA','V4_CONTRACT','No class balancing interpreted as extra events.'),
 ('4.5','PY','V4_CONTRACT','Nested HPO/regularization; post-hoc variant choice still restricted.'),
]
P_EVAL = [
 ('4.1','Y','V4_STATE,RESULTS','Held-out internal predictions exist, not training accuracy only.'),
 ('4.2','PN','RESULTS','Wide intervals and sparse manufacturers; no power guarantee.'),
 ('4.3','NI','RESULTS','Censoring-aware metrics do not remove unmeasured informative censoring.'),
 ('4.4','PY','V4_CONTRACT','Held-out patients are not augmented as independent observations.'),
 ('4.5','PN','REPAIR','Historical test exposure remains; corrective rerun is not untouched confirmation.'),
 ('4.6','PN','RESULTS','Fixed-model bootstrap excludes model-selection/retraining variance.'),
 ('4.7','PY','RESULTS,REVIEW','Multiple valid survival metrics; clinical net benefit not claimed.'),
]


def main():
    evidence = {}
    for key, rel in EVIDENCE.items():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f'MISSING_EVIDENCE:{key}:{rel}')
        evidence[key] = {'path': rel, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    rows = []
    for standard, section, entries in [('TRIPOD_AI','reporting',T), ('CLAIM_2024','reporting',C),
        ('PROBAST_AI','development',P_SHARED+P_DEV), ('PROBAST_AI','evaluation',P_SHARED+P_EVAL)]:
        for item, status, refs, rationale in entries:
            keys = refs.split(',')
            assert all(k in evidence for k in keys)
            rows.append({'standard':standard, 'section':section, 'item':item,
                         'status':status, 'evidence_keys':refs, 'project_disposition':rationale})
    assert len({(r['standard'],r['section'],r['item']) for r in rows}) == len(rows)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    atomic_write_text(OUT / 'item_crosswalk.csv', stream.getvalue())
    atomic_write_json(OUT / 'evidence_registry.json', {'sources':SOURCES, 'evidence':evidence,
        'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    atomic_write_json(OUT / 'assessment_scope.json', {
        'status':'ITEM_CROSSWALK_COMPLETED_WITH_EXPLICIT_GAPS',
        'tripod_item_rows':len(T), 'claim_item_rows':len(C), 'probast_dev_rows':len(P_SHARED+P_DEV),
        'probast_eval_rows':len(P_SHARED+P_EVAL),
        'scope':'BRANCH_SHARED_EVIDENCE_SCREEN_NOT_PER_MODEL_CERTIFICATION_OR_FINAL_MANUSCRIPT',
        'full_compliance_claim':False, 'overall_evaluation_bias_concern':'HIGH_FOR_CONFIRMATORY_OR_EXTERNAL_CLAIMS',
        'reason':'Historical exposure, selection, sparse support and unverified source measurement; local exploratory reporting allowed.',
        'new_training':False, 'human_signoff_required':False,
        'administrative_facts_may_not_be_invented':True,
    })
    lines = ['# Method Y item-addressed reporting assessment', '',
        'Official item IDs map to project evidence hashes. Item text is not reproduced. '
        'This is not a completed manuscript or per-model certification. '
        'UNKNOWN and limitations remain visible; agent review does not replace original clinical annotation.', '',
        'Sources: '+', '.join(f'[{k}]({v})' for k,v in SOURCES.items()), '',
        '| Standard/phase | Item | Status | Evidence keys | Project disposition |',
        '|---|---|---|---|---|']
    lines += [f"| {r['standard']}/{r['section']} | {r['item']} | {r['status']} | {r['evidence_keys']} | {r['project_disposition']} |" for r in rows]
    lines += ['', 'Hash/path lookup: `evidence_registry.json`. Counts: '+str(len(rows))+'.',
        'PROBAST answers: Y=yes, PY=probably yes, PN=probably no, NI=insufficient information, NA=not applicable. '
        'No per-model low-risk conclusion is issued. Clinical deployment applicability is high concern; source-assessor details remain unknown.']
    atomic_write_text(OUT / 'REPORTING_CROSSWALK.md', '\n'.join(lines)+'\n')
    print(f'ITEM_CROSSWALK_WRITTEN:{len(rows)};NO_COMPLIANCE_CERTIFICATION')


if __name__ == '__main__':
    main()
