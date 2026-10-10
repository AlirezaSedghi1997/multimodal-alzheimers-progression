> Internal/exploratory evidence snapshot; not external/clinical validation. A completed computation is not a positive science gate. This narrative predates the FP32 review; current sensitivity findings are summarized in docs/FP32_SENSITIVITY.md.

# پروندهٔ یکپارچهٔ محاسبات اصلاح‌شده پس از فیوژن

این پوشه نتیجهٔ اجرای محاسبات است؛ پیش از استفادهٔ نهایی در پایان‌نامه، بازبینی واقعی اعداد و تصاویر توسط computational review باقی می‌ماند.
تأیید انسانی مستقل شرط نیست. هیچ فایل Muse پاک نشده و مرجع رسمی هنوز خودکار جایگزین نشده است.

| گام | وضعیت ثبت‌شده |
|---|---|
| F00 | PARENT_STRUCTURAL_AND_ARTIFACT_VERIFICATION_PASSED |
| F01 | COMPLETED |
| F02 | EXACT_INPUT_AND_ALL_COMPONENT_ANCESTRY_VERIFIED |
| F03 | COMPLETED_MATCHED_INTERNAL_EXTENSION |
| F04 | COMPLETED_SPATIAL_DIAGNOSTICS_LABEL_CONTROL_PENDING |
| F05 | PET_MANUFACTURER_AUDIT_COMPLETED; PROBES_FROM_CORRECTED_POSTALL |
| F06 | COMPLETED_SECONDARY_ASSOCIATION |
| F07 | NOT_ESTIMABLE_COMMON_META_SUPPORT |
| F08 | COMBINED_DOSSIER_COMPUTED_PENDING_ACTUAL_POST_RUN_REVIEW |

## محل نتایج

- postall: مقایسه‌های matched، calibration، CI، Tau، PACC، probe و XAI.
- manufacturer: حساسیت محدود واقعی PET-manufacturer با MRI/PET expertهای تازه و بدون نشت سازندهٔ کنارگذاشته‌شده.
- regional: ارتباط حجم نواحی پایه با CDR آینده و PACC، تعدیل مخدوش‌گرها و اصلاح چندگانگی.

## مواردی که اجرای کامل آن‌ها را ثابت نمی‌کند

- داده خارجی نداریم؛ تمام تحلیل‌ها داخلی و post-hoc هستند.
- چهاروجهی در support ناکافی fit نمی‌شود؛ NOT_ESTIMABLE نتیجهٔ امکان‌سنجی است.
- XAI نامعتبر، صفر، ناپایدار یا دارای خطای عددی با عنوان تشخیص ضایعه منتشر نمی‌شود.
- آزمون permutation برچسب کل ensemble اجرا نشده؛ random-weight کنترل مستقل دیگری است، نه معادل آن.
- در F05 یک seed و سه جزء استفاده شده است؛ نتیجهٔ حساسیت با ensemble اصلیِ نه‌جزئی یکسان نیست.
- native-space lesion localization و تشخیص اختصاصی آلزایمر اثبات نشده‌اند.
- داده‌های فردی، checkpoint و تصاویر خصوصی در این bundle کپی نشده‌اند.
