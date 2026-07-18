# مسیرهای ردشده یا deferred

| idea_id | verdict | hard_fail_gates | practical_continuation_score | penalties |
|---|---|---|---|---|
| conflict_aware_fusion | ELIGIBLE | [] | 67.0 | [{'reason': 'direct_multimodal_conflict_precedent', 'points': 10}] |
| selective_prediction | REJECTED_NOVELTY_GATE | [] | 66.0 | [{'reason': 'direct_selective_prediction_precedent', 'points': 8}] |
| value_of_pet | REJECTED_HARD_GATE | ['G1'] | 47.0 | [{'reason': 'external_decision_dataset_missing', 'points': 15}, {'reason': 'direct_precedent', 'points': 8}] |
| two_visit_delta | ELIGIBLE | [] | 61.0 | [{'reason': 'paired_availability_uncertain', 'points': 10}] |
| faq_informant | REJECTED_HARD_GATE | ['G0'] | 42.0 | [{'reason': 'endpoint_not_locked', 'points': 20}] |
| underused_biomarkers | REJECTED_HARD_GATE | ['G1'] | 49.0 | [{'reason': 'roi_volumetry_not_ready', 'points': 12}] |

Rejection is due to gates/data/precedent, not aesthetic preference.
