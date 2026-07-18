# برنامه شش‌هفته‌ای

| هفته | کارها | ورودی/خروجی | پذیرش | fallback |
|---|---|---|---|---|
| ۱ | freeze، endpoint/leakage audit، provenance و split check | v1 artifacts → audit tables | no leakage; hashes locked | stop and report blocker |
| ۲ | matched clinical/PET baselines، calibration | locked folds → baseline metrics | all outer folds complete | negative-only report |
| ۳ | tau-beyond-amyloid locked comparison، paired bootstrap | OOF predictions → deltas | exact-once outer eval | preserve negative result |
| ۴ | optional cross-fitted discordance + null/shuffle | fold-local residual → ablation | null control and stability pass | omit extension |
| ۵ | scanner/site/subgroup/censoring sensitivity | committed predictions → audits | no shortcut gate failure | exploratory only |
| ۶ | final reruns, CI, reports, privacy bundle | all artifacts → manuscript-ready package | consistency guard pass | no claim/public release |

هر هفته CPU-first است؛ image branch و external validation به‌دلیل گیت‌ها مسیر اصلی نیستند.
