# Bio-Cog / Method Z — Phase 3 Blood Branch

- status: `COMPLETED`
- claim boundary: Minimally invasive biomarker-assisted branch; never supports the non-invasive Method X claim.

## enrolled_a4_learn

- rows: `1637`
- cohort-label Cramér's V: `1.0`

- **core_blood_only**: status=`OK` roc_auc=`0.8085818422796553` n=`1613`
- **full_blood_panel**: status=`OK` roc_auc=`0.8691200342031262` n=`1637`

## sf_inclusive

- rows: `2623`
- cohort-label Cramér's V: `0.9286464476261044`

- **core_blood_only**: status=`OK` roc_auc=`0.706716966034491` n=`2594`
- **full_blood_panel**: status=`OK` roc_auc=`0.8363914283812551` n=`2623`

## train-SF → test-enrolled

`{'status': 'OK', 'n_train': 986, 'n_test': 1637, 'variant': 'full_blood_panel', 'roc_auc': 0.7300438314453254, 'pr_auc': 0.8402550393097035, 'brier': 0.21183600540167533, 'decoupling_disclaimer': 'Internal decoupling / stress-test only. SF is not external validation and must never be described as independent or quasi-external validation.'}`
