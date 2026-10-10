> Internal/exploratory evidence snapshot; not external/clinical validation. A completed computation is not a positive science gate. This narrative predates the FP32 review; current sensitivity findings are summarized in docs/FP32_SENSITIVITY.md.

# فصل نتایج ــ Method X fast-track v2.2

تمام چهار بازو، حتی اگر نتیجهٔ نامطلوب داشته باشند، گزارش می‌شوند. این مقادیر از fold testهای داخلیِ تاریخاً دیده‌شده‌اند، نه مجموعهٔ آزمون مستقل.

| بازو | ROC-AUC | AP | Brier |
|---|---:|---:|---:|
| MX4 scratch | 0.6842 | 0.8138 | 0.2283 |
| MX4 head/last | 0.6771 | 0.8109 | 0.2343 |
| Hybrid baseline | 0.6785 | 0.8116 | 0.2275 |
| Hybrid dense skip | 0.7185 | 0.8349 | 0.2208 |

بازوی برگزیده با inner loss: `mx04_scratch`. تفاوت ROC-AUC dense منهای baseline: 0.0400. CI 95٪ bootstrap بیمار: [0.0217, 0.0587].

میانگین بهترین loss در inner validation برای همهٔ بازوها:

| بازو | mean best validation loss |
|---|---:|
| hybrid_baseline | 0.3835 |
| hybrid_dense | 0.3828 |
| mx04_head_last | 0.3819 |
| mx04_scratch | 0.3821 |

| فولد | n | AUC پایه | AUC dense | AP پایه | AP dense | Brier پایه | Brier dense |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 268 | 0.7461 | 0.7238 | 0.8602 | 0.8470 | 0.2217 | 0.2095 |
| 1 | 269 | 0.7153 | 0.7150 | 0.8284 | 0.8292 | 0.2119 | 0.2118 |
| 2 | 267 | 0.6448 | 0.6687 | 0.7921 | 0.8101 | 0.2376 | 0.2335 |
| 3 | 266 | 0.7428 | 0.7211 | 0.8600 | 0.8549 | 0.2241 | 0.2348 |
| 4 | 266 | 0.4332 | 0.7732 | 0.6870 | 0.8645 | 0.2423 | 0.2143 |

برای MX4، ارزیابی outer نمرهٔ تجمیعی دارد؛ معیارهای per-fold آن در این بستهٔ گزارش نهایی موجود نیست و از aggregate بازسازی نمی‌شود.

شمارش حساسیت پنجرهٔ PET–MRI، صرفاً توصیفی:

| فاصلهٔ حداکثر | n | مثبت | منفی |
|---:|---:|---:|---:|
| 90 روز | 259 | 225 | 34 |
| 180 روز | 448 | 394 | 54 |
| 365 روز | 978 | 842 | 136 |

XAI ۲.۲: BLOCKED_EXPLORATORY_GATE؛ ۴۰ تصویر در پنج فولد؛ میانگین فولدی افت احتمال هدف برای ROI منتخب: -0.0003؛ میانگین فولدی اختلاف افت هدف نسبت به کنترل: -0.0002. این شاخص صرفاً وفاداریِ اکتشافی به مدل است.

| فولد | ROI منتخب جهت اول/دوم (atlas ID) | افت هدف | اختلاف جفتی با کنترل | gate |
|---:|---|---:|---:|---|
| 0 | 216/213 | 0.0030 | 0.0030 | PASS_EXPLORATORY |
| 1 | 9/9 | -0.0015 | -0.0015 | BLOCKED_EXPLORATORY_GATE |
| 2 | 28/36 | -0.0023 | -0.0015 | BLOCKED_EXPLORATORY_GATE |
| 3 | 214/214 | -0.0007 | -0.0007 | BLOCKED_EXPLORATORY_GATE |
| 4 | 35/35 | -0.0001 | -0.0001 | BLOCKED_EXPLORATORY_GATE |

CI 95٪ bootstrap در سطح پنج فولد برای اختلاف جفتی: [-0.0013, 0.0015].

وضعیت pseudo-holdout در freeze والد: `UNKNOWN_OR_NOT_DONE`.

اعتبارسنجی مستقل: `NOT_DONE_NO_INDEPENDENT_COHORT_OR_REGIONAL_REFERENCE`.

## ردگیری اعداد

SHA256 بستهٔ منجمد والد: `f1e99125f390188ef42a933f435d63da43dc95b6e5c9c8124766d9b5978f2df4`؛ SHA256 مقایسهٔ Hybrid: `6cff6ab9a1bbd3b6f1df36ee85cd1df99f71e03b180ea880ef0fc78521ee633c`؛ SHA256 تحلیل XAI نسخهٔ ۲.۲: `48513afa69701abe418b3b4c28da4aa95b2b3c3ba5fdd79b9df7294b1412306f`. نگاشت کامل فصل به artifact در manifest همین نسخه ثبت شده است.
