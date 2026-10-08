# Release gates

This package is a local candidate, not an upload approval or completed research claim.

| Gate | Status |
|---|---|
| Copy/syntax/integrity and bounded secret/privacy checks | See package_verification.json |
| Live study source unchanged by packaging | Recorded separately in local audit |
| Active FP32 results and image review | Pending; no partial result exported |
| Final post-fusion promotion | Pending actual post-run review |
| Fresh-machine end-to-end reconstruction | Not executed |
| Dynamic import sites | 2 inventoried; no full runtime proof |
| Code license and author/citation metadata | Rights-holder decision required |
| Restricted data/weights transfer to supervisors | Applicable agreement and recipient access required |
| Dependency advisory audit | Executed; known findings remain; not security-cleared |
| External/clinical/causal claims | Not supported |

The safe action is to inspect and verify the local package, not run a historical
all-stages script in the live tree. Future additions produce a new release version.
Packaging does not erase provenance or override required disclosures.

Source-binding comparison counts: {"MATCH": 150, "MATCH_VERIFIED_PH_DATATYPE_ONLY_AMENDMENT": 2}. The two PH datatype-adapter differences were verified against their recorded amendment without modifying the parent; source copies correspond to repaired execution, not the obsolete pre-repair bytes.
