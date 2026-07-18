# تصمیم اجرایی نهایی ممیزی پنج‌ایجنت

## تصمیم در یک نگاه

**توصیه اصلی:** `continuation_prespecified_method_y` با Practical Continuation Score برابر **84.0**. مسیر اصلی، تکمیل و بازاجرای کنترل‌شده‌ی Method Y موجود با endpoint آینده‌نگر، ارزیابی nested و گزارش صادقانه‌ی نتیجه‌ی منفی tau-beyond-amyloid است. این انتخاب به‌معنای اثبات نوآوری مثبت یا external validation نیست.

**چرا:** داده و کد محلی آماده‌تر از هر pivot بیرونی است؛ event scarcity و نبود DUA مدل complex یا external-dependent را از مسیر اصلی خارج می‌کند؛ و وضعیت علمی فعلی نیازمند قفل‌کردن provenance، leakage و claim policy است.

**نوآوری هسته‌ای شماره ۱:** پروتکل matched و leakage-closed برای سنجش ارزش افزوده‌ی PET در برابر clinical+cognition، با outer-test دقیقاً یک‌باره.

**نوآوری هسته‌ای شماره ۲:** لازم نیست برای MVP. discordance cross-fitted فقط extension مشروط است، نه novelty مفروض.

**افزونه اختیاری:** `biology_function_discordance`، فقط پس از عبور null/shuffle، fold-stability و scanner gates.

**داده و endpoint:** A4 برای development، LEARN برای internal transport؛ endpoint محلی future sustained CDR progression و PACC فقط secondary. DUA خارجی موجود نیست.

**Claim مجاز:** «در محیط داخلی A4، با پروتکل از پیش‌تعیین‌شده، PET در صورت عبور gate می‌تواند اطلاعات پیش‌آگهی افزوده ارائه کند.» اگر gate شکست بخورد، claim فقط exploratory/technical است.

**Claim ممنوع:** تشخیص آلزایمر، MCI→AD محلی، causal effect، external generalization، یا novelty صرفاً به‌دلیل conflict/discordance.

**زمان:** MVP چهار هفته؛ full internal evaluation حداکثر شش هفته.

**منابع محاسباتی:** CPU/ROI-first؛ RTX 4060 فقط برای تحلیل تصویری ثانویه پس از قرارداد مدل.

**بزرگ‌ترین ریسک:** scanner/site shortcut و کمبود event در tau.

**شرط توقف:** هر leakage، transform/provenance failure، outer-fold ناقص، null-control failure، یا نبود incremental value.

**Fallback:** `biology_function_discordance`؛ در صورت شکست هر gate، فقط نتیجه‌ی منفی/اکتشافی فعلی تکمیل و گزارش شود.

## وضعیت ممیزی زنده

- Method Y amyloid endpoint (locked v1 evidence): 1,613 eligible subjects / 278 events; A4 development 1,106/240; LEARN transport 507/38.
- Tau and paired A/T endpoint (locked v1 evidence): 369 subjects / 91 events.
- Tau beyond amyloid: locked negative result; this is retained, not overwritten.
- External data: DUA/raw manifest unavailable; external claim is not permitted.
- FTP transform writer active at audit start: True; its count is intentionally read from the live snapshot and is not frozen here.

## نکته‌ی ایمنی

writer فعال FTP→T1→MNI متوقف یا دست‌کاری نشد. T0 و Tfinal باید اختلاف‌های مسیر transform را با process writer نسبت دهند، نه به این ممیزی.
