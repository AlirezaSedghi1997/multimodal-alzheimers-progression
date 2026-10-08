> Internal/exploratory evidence snapshot; not external/clinical validation. A completed computation is not a positive science gate. The FP32 companion is excluded pending review.

# Corrected internal post-fusion analysis

POST_HOC_INTERNAL_EXPLORATORY

This report does not rerun fusion training or reselect a model using outer outcomes.

| Analysis | Arm | N | Events | IPCW IBS | Conditional IBS CI | Weighted fold Uno C |
|---|---|---:|---:|---:|---|---:|
| mri | blood_clinical_amyloid | 696 | 166 | 0.040528 | [0.03273340954347326, 0.048908663331144804] | 0.6977490855607165 |
| mri | blood_clinical_amyloid_blood | 696 | 166 | 0.040155 | [0.03230689642914566, 0.04832029833209613] | 0.7074949133163857 |
| mri | blood_clinical_mri_amyloid | 696 | 166 | 0.040551 | [0.032715729821931695, 0.04891398509714158] | 0.7002774800837186 |
| mri | blood_clinical_mri_amyloid_blood | 696 | 166 | 0.040111 | [0.03224905613185417, 0.048357335216483235] | 0.7089774876473043 |
| mri | clinical | 1059 | 239 | 0.040687 | [0.03418703144662744, 0.04780611388752025] | 0.6686554483238794 |
| mri | clinical_amyloid | 1059 | 239 | 0.040300 | [0.03378463199383528, 0.04719528983006302] | 0.6855558941646316 |
| mri | clinical_mri | 1059 | 239 | 0.040762 | [0.03422697770063003, 0.047890559742520995] | 0.6698828415237823 |
| mri | clinical_mri_amyloid | 1059 | 239 | 0.040402 | [0.03395591973659065, 0.04727993991290668] | 0.6854348286450931 |
| mri | clinical_volumes | 1059 | 239 | 0.040777 | [0.03422990481394312, 0.0479455950311995] | 0.6752003052212774 |
| mri | clinical_volumes_amyloid | 1059 | 239 | 0.040628 | [0.03405045388694637, 0.047686385953199704] | 0.6963344755190763 |
| mri | clinical_volumes_mri_amyloid | 1059 | 239 | 0.040750 | [0.034231039896081325, 0.04788753665135921] | 0.692481662093371 |
| mri | mri_only | 1059 | 239 | 0.042698 | [0.03577169474776692, 0.050188140783060345] | 0.5712684794949295 |
| pet | clinical | 544 | 112 | 0.040641 | [0.031243164761422682, 0.050751101462346154] | 0.7000680130973196 |
| pet | clinical_amyloid | 544 | 112 | 0.040547 | [0.03142681878740153, 0.05033779259503741] | 0.7063647669483905 |
| pet | clinical_mri | 544 | 112 | 0.040770 | [0.03134518555217814, 0.05078333485620243] | 0.7080337646864817 |
| pet | clinical_mri_amyloid | 544 | 112 | 0.040637 | [0.03146595057832925, 0.05051927515409161] | 0.7187130345101003 |
| pet | clinical_mri_amyloid_pet | 544 | 112 | 0.040592 | [0.03146826520661478, 0.05032042667826079] | 0.7059805889605117 |
| pet | clinical_volumes | 544 | 112 | 0.039982 | [0.030796868801232873, 0.049867634684754227] | 0.7241801043194591 |
| pet | clinical_volumes_amyloid | 544 | 112 | 0.040395 | [0.031233466306838636, 0.05040105058586524] | 0.7432537347891983 |
| pet | clinical_volumes_mri_amyloid | 544 | 112 | 0.040722 | [0.031408369475895155, 0.05070987937962939] | 0.7476641903263781 |
| pet | mri_only | 544 | 112 | 0.042708 | [0.03227250189564934, 0.05330863744591624] | 0.5637132645997212 |
| scalar_confirmation_time_cox_ridge | blood_clinical | 696 | 166 | 0.040252 | [0.03249950089833512, 0.04874327585853926] | 0.6850381479165583 |
| scalar_confirmation_time_cox_ridge | blood_clinical_amyloid | 696 | 166 | 0.040490 | [0.03266029470832077, 0.048975195608085074] | 0.6941725022414396 |
| scalar_confirmation_time_cox_ridge | blood_clinical_amyloid_blood | 696 | 166 | 0.040117 | [0.032256981050386574, 0.04830140917244307] | 0.7064284876244303 |
| scalar_confirmation_time_cox_ridge | blood_clinical_blood | 696 | 166 | 0.039942 | [0.032197334690185354, 0.048031676332297865] | 0.7101112369560293 |
| scalar_confirmation_time_cox_ridge | clinical | 1059 | 239 | 0.040601 | [0.034144893544338834, 0.04767128868844745] | 0.6654066401679701 |
| scalar_confirmation_time_cox_ridge | clinical_amyloid | 1059 | 239 | 0.040296 | [0.033753667442362745, 0.047284566043057266] | 0.6836196530828519 |
| scalar_confirmation_time_cox_ridge | clinical_volumes | 1059 | 239 | 0.040797 | [0.03429122784680886, 0.047955034170936765] | 0.6677122125727948 |
| scalar_confirmation_time_cox_ridge | clinical_volumes_amyloid | 1059 | 239 | 0.040717 | [0.03409201958675803, 0.047911043434150064] | 0.687408332811167 |
| scalar_confirmation_time_rsf | blood_clinical | 696 | 166 | 0.040645 | [0.03285793280172099, 0.04898873069465913] | 0.6795386101943081 |
| scalar_confirmation_time_rsf | blood_clinical_amyloid | 696 | 166 | 0.040259 | [0.03255238575886647, 0.04862682422704622] | 0.6916760758240273 |
| scalar_confirmation_time_rsf | blood_clinical_amyloid_blood | 696 | 166 | 0.040243 | [0.03252929564929254, 0.04846940276510224] | 0.6998957699026543 |
| scalar_confirmation_time_rsf | blood_clinical_blood | 696 | 166 | 0.040210 | [0.03246474445613585, 0.04842638063207838] | 0.6958959868139168 |
| scalar_confirmation_time_rsf | clinical | 1059 | 239 | 0.041097 | [0.034549053525084736, 0.048332685127040725] | 0.6406101292484887 |
| scalar_confirmation_time_rsf | clinical_amyloid | 1059 | 239 | 0.040448 | [0.033930735455200235, 0.04751523160313486] | 0.6699533081113977 |
| scalar_confirmation_time_rsf | clinical_volumes | 1059 | 239 | 0.040952 | [0.03445703974964015, 0.04807211572594508] | 0.6632119792944575 |
| scalar_confirmation_time_rsf | clinical_volumes_amyloid | 1059 | 239 | 0.040433 | [0.03392262926054947, 0.047396511835523236] | 0.6793856460238137 |
| scalar_last_negative_censor_cox_ridge | blood_clinical | 692 | 166 | 0.113893 | [0.09844389665254186, 0.13022631219937386] | 0.6693303240710348 |
| scalar_last_negative_censor_cox_ridge | blood_clinical_amyloid | 692 | 166 | 0.111579 | [0.09616630067570518, 0.12739972705744806] | 0.6897055036522829 |
| scalar_last_negative_censor_cox_ridge | blood_clinical_amyloid_blood | 692 | 166 | 0.109922 | [0.09436179579086544, 0.1255983521962905] | 0.707226853751862 |
| scalar_last_negative_censor_cox_ridge | blood_clinical_blood | 692 | 166 | 0.110271 | [0.09506992685470063, 0.1259585378149616] | 0.7071615698416002 |
| scalar_last_negative_censor_cox_ridge | clinical | 1052 | 239 | 0.110553 | [0.09781806904193056, 0.12375962202599378] | 0.651079882089476 |
| scalar_last_negative_censor_cox_ridge | clinical_amyloid | 1052 | 239 | 0.107486 | [0.09488897326800441, 0.12066968410707238] | 0.6806090532209194 |
| scalar_last_negative_censor_cox_ridge | clinical_volumes | 1052 | 239 | 0.110223 | [0.09722329826125022, 0.12382396344233049] | 0.6678968121575701 |
| scalar_last_negative_censor_cox_ridge | clinical_volumes_amyloid | 1052 | 239 | 0.108131 | [0.09538054790581554, 0.12170513712593882] | 0.6939435079674444 |
| scalar_last_visit_positive_cox_ridge | blood_clinical | 696 | 241 | 0.119902 | [0.1047665627923929, 0.13539028376243345] | 0.668790819995104 |
| scalar_last_visit_positive_cox_ridge | blood_clinical_amyloid | 696 | 241 | 0.118330 | [0.1030984596742721, 0.13333181030773408] | 0.6832681901559878 |
| scalar_last_visit_positive_cox_ridge | blood_clinical_amyloid_blood | 696 | 241 | 0.117139 | [0.10212040749839069, 0.13250270091371844] | 0.6986163402413259 |
| scalar_last_visit_positive_cox_ridge | blood_clinical_blood | 696 | 241 | 0.117131 | [0.10215805865294735, 0.13193705779006312] | 0.698956257689679 |
| scalar_last_visit_positive_cox_ridge | clinical | 1059 | 346 | 0.116689 | [0.1035489612167042, 0.13064281132333036] | 0.6534365850292199 |
| scalar_last_visit_positive_cox_ridge | clinical_amyloid | 1059 | 346 | 0.113896 | [0.10072720024141682, 0.127642515247277] | 0.6827951273491524 |
| scalar_last_visit_positive_cox_ridge | clinical_volumes | 1059 | 346 | 0.116244 | [0.10297081955047412, 0.13041926826207742] | 0.6679540632237909 |
| scalar_last_visit_positive_cox_ridge | clinical_volumes_amyloid | 1059 | 346 | 0.114267 | [0.10083384129690781, 0.12814023604748648] | 0.6911417906067852 |
| scalar_legacy_onset_cox_ridge | blood_clinical | 696 | 166 | 0.112857 | [0.09778036815110812, 0.12842892541070633] | 0.66872630587622 |
| scalar_legacy_onset_cox_ridge | blood_clinical_amyloid | 696 | 166 | 0.110610 | [0.09509660559204776, 0.12619480780947956] | 0.6900341249266451 |
| scalar_legacy_onset_cox_ridge | blood_clinical_amyloid_blood | 696 | 166 | 0.109355 | [0.0942705023719278, 0.12452203413531196] | 0.7048884419477498 |
| scalar_legacy_onset_cox_ridge | blood_clinical_blood | 696 | 166 | 0.109747 | [0.09482713961969848, 0.1247448349383221] | 0.7057047895061448 |
| scalar_legacy_onset_cox_ridge | clinical | 1059 | 239 | 0.109733 | [0.09675025258932392, 0.12391185143384052] | 0.6497673881041082 |
| scalar_legacy_onset_cox_ridge | clinical_amyloid | 1059 | 239 | 0.106665 | [0.09386930693181215, 0.12018185666933241] | 0.6799480420619801 |
| scalar_legacy_onset_cox_ridge | clinical_volumes | 1059 | 239 | 0.109624 | [0.09663852719539257, 0.1235031787185405] | 0.6659655521647623 |
| scalar_legacy_onset_cox_ridge | clinical_volumes_amyloid | 1059 | 239 | 0.107415 | [0.09470370793388198, 0.12134169854277876] | 0.6925031291289168 |
| scalar_single_crossing_cox_ridge | blood_clinical | 696 | 291 | 0.159728 | [0.14422969580992487, 0.17587138860207] | 0.6577264398600199 |
| scalar_single_crossing_cox_ridge | blood_clinical_amyloid | 696 | 291 | 0.158505 | [0.14306832569724223, 0.1747641916038813] | 0.6690759151963401 |
| scalar_single_crossing_cox_ridge | blood_clinical_amyloid_blood | 696 | 291 | 0.157324 | [0.14165838453490617, 0.17359475518144274] | 0.6717717852126961 |
| scalar_single_crossing_cox_ridge | blood_clinical_blood | 696 | 291 | 0.157086 | [0.14154824860593448, 0.17327990730365614] | 0.6714107557263679 |
| scalar_single_crossing_cox_ridge | clinical | 1059 | 426 | 0.156837 | [0.14439615033697786, 0.16999381901751176] | 0.6486737562941876 |
| scalar_single_crossing_cox_ridge | clinical_amyloid | 1059 | 426 | 0.154832 | [0.14241350629500665, 0.16786616846627916] | 0.665732951006893 |
| scalar_single_crossing_cox_ridge | clinical_volumes | 1059 | 426 | 0.156670 | [0.1439273412814645, 0.16998953587503424] | 0.6550715010027233 |
| scalar_single_crossing_cox_ridge | clinical_volumes_amyloid | 1059 | 426 | 0.155178 | [0.14257873961615372, 0.16813163500918474] | 0.6688969481219745 |
| tau_cox_ridge | clinical_amyloid | 358 | 92 | 0.049564 | [0.038369631955839645, 0.06076726557821838] | 0.704159479208027 |
| tau_cox_ridge | clinical_amyloid_tau | 358 | 92 | 0.046799 | [0.03618709116026135, 0.057679547540868986] | 0.7771182446145738 |
| tau_rsf | clinical_amyloid | 358 | 92 | 0.051133 | [0.03972096336835957, 0.06275516413534145] | 0.6548380595317449 |
| tau_rsf | clinical_amyloid_tau | 358 | 92 | 0.049715 | [0.03867051758207216, 0.06121194657027723] | 0.7097628786785716 |

## Additional matched analyses

Tau: {'status': 'COMPLETED_MATCHED_INTERNAL_EXTENSION', 'subjects': 358, 'events': 92, 'comparisons': [{'augmented': 'plus_tau', 'base': 'base', 'delta_ibs_bonferroni_ci': [-0.005479288862910477, -0.0005089601418510036], 'delta_ibs_ci95': [-0.005479288862910477, -0.0005089601418510036], 'delta_ibs_mean': -0.002979315991996438, 'delta_uno_bonferroni_ci': [0.037153151281008025, 0.1294904475481983], 'delta_uno_ci95': [0.037153151281008025, 0.1294904475481983], 'delta_uno_mean': 0.0824645063137252, 'family_size': 1, 'multiplicity': 'EXPLORATORY_NO_CONFIRMATORY_SIGNIFICANCE_CLAIM', 'resamples': 2000, 'resampling': 'PAIRED_SUBJECT_WITHIN_OUTER_FOLD', 'uncertainty': 'CONDITIONAL_ON_FITTED_MODELS_AND_TRAIN_CENSORING_NOT_SELECTION_UNCERTAINTY', 'valid_ibs_resamples': 2000, 'valid_uno_resamples': 1967, 'verdict': 'INTERNAL_EXPLORATORY_SUPPORT'}]}

Four-way: {'status': 'NOT_ESTIMABLE_COMMON_META_SUPPORT', 'subjects': 109, 'events': 24, 'action': 'DO_NOT_FIT_UNSUPPORTED_FOUR_WAY_HEADS'}

PACC status: COMPLETED_SECONDARY_ASSOCIATION

All confidence intervals condition on the already fitted predictors and training censoring. No clinical, external, prospective, or causal anatomical claim is licensed by these results.
