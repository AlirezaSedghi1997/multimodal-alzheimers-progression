# Release gates

This is the selected code-and-aggregate repository package, now publicly visible at the owner's request. Technical package verification and
the scientific scope of each analysis are recorded separately.

| Gate | Status |
|---|---|
| Copy/syntax/integrity and bounded secret/privacy checks | See package_verification.json |
| Live study source unchanged by packaging | Recorded separately in local audit |
| FP32 execution, statistics and image review | Completed; aggregate sensitivity and failed controls included; attribution not fully validated |
| Final post-fusion promotion | Pending actual post-run review |
| Fresh-machine end-to-end reconstruction | Not executed |
| Dynamic import sites | 2 inventoried; no full runtime proof |
| Code license and author/citation metadata | Rights-holder decision required |
| Restricted data/weights transfer to supervisors | Applicable agreement and recipient access required |
| Dependency advisory audit | Executed; known findings remain; not security-cleared |
| External/clinical/causal claims | Not supported |

Begin with the offline package check and the reproduction guide. Run study scripts
only after reconstructing their protected inputs and versioned contracts. Future
additions produce a new release version.
Packaging does not erase provenance or override required disclosures.

Source-binding comparison counts: {"MATCH": 150, "MATCH_VERIFIED_PH_DATATYPE_ONLY_AMENDMENT": 2}. The two PH datatype-adapter differences were verified against their recorded amendment without modifying the parent; source copies correspond to repaired execution, not the obsolete pre-repair bytes.


## 14 October research update

The [new research update](RESEARCH_UPDATE_20261014.md) adds completed statistical analyses, OOF sensitivity and the gated CNN/Jacobian pilot decision. The [publication checks](RESEARCH_UPDATE_CHECKS_20261014.json) verify this update's bounded payload and execution tests. Full-cohort CNN/Jacobian work did not proceed; external validation and fresh-machine end-to-end reproduction remain unperformed.
