# Plan 011: nested-CV image survival trainer و outer evaluation دقیقاً یک‌باره را بساز

> **دستور ایجنت مجری:** Planهای 009 و 010 باید DONE باشند و cache summary هر
> دو شاخه COMPLETED باشد. این Plan ابتدا trainer را با fixture مصنوعی می‌سازد و
> تست می‌کند؛ training واقعی production را شروع نکن. هیچ outer member نباید
> پیش از checkpoint lock حتی dataset شود.
>
> **Drift check:**
>
> ```powershell
> Set-Location -LiteralPath 'H:\New project'
> git diff --stat 5405716..HEAD -- src/method_y_revival scripts configs tests/method_y_revival
> Get-FileHash -Algorithm SHA256 `
>   src\method_y_revival\image_survival_v3.py, `
>   src\method_y_revival\image_training_runtime_v3.py, `
>   src\method_y_revival\outer_evaluation_control.py, `
>   src\method_y_revival\science_gates.py
> ```
>
> اگر APIهای زیر drift کرده‌اند، STOP و plan را به bytes زنده rebase کن.

## وضعیت

- **اولویت:** P0
- **اندازه:** XL
- **ریسک:** HIGH؛ خطا می‌تواند leakage یا تکرار outer-test ایجاد کند
- **وابسته به:** 009, 010
- **دسته:** correctness / statistics / architecture / tests
- **Planned at:** commit `5405716`, 2026-08-24

## چرا مهم است

کد موجود فقط اجزای پایین‌سطح را دارد: شبکه/loss، dataset، epoch checkpoint و
cache. هیچ consumer برای config v3، inner HPO، انتخاب winner، fit روی outer
train، قفل checkpoint، ساخت outer loader پس از lock و aggregation OOF وجود
ندارد. وجود `READY_FOR_FINAL_INTERNAL_IMAGE_TRAINING` به معنی انجام این مراحل
نیست.

مقایسهٔ علمی باید matched باشد:

```text
Amyloid base: clinical+cognition+treatment+global amyloid scalar
Amyloid augmented: همان base + pattern-normalized FBP image

Tau base: clinical+cognition+treatment+amyloid scalar+global tau PERSI scalar
Tau augmented: همان base + pattern-normalized FTP image
```

این طراحی «ارزش افزودهٔ spatial pattern فراتر از scalar burden» را می‌سنجد؛
تصویر کمی رسمی فرض نمی‌شود.

## APIهای موجود برای reuse

- `PatternSurvivalNet` و `TabularSurvivalNet`:
  `src/method_y_revival/image_survival_v3.py:165-254`.
- `FoldTabularTransformer` و `fit_survival_model`:
  `image_training_runtime_v3.py:178-215` و `286-536`.
- one-standard-error selector:
  `survival_models.py:10-26`.
- state machine قوی outer:
  `outer_evaluation_control.py:45-210`. از این class استفاده کن؛
  `evaluation.OuterEvaluationStore` ساده‌تر است و مسیر ambiguity را fail-closed
  نمی‌کند.
- paired OOF bootstrap و incremental gate:
  `science_gates.py:161-225`.
- الگوی censoring-aware metric:
  `regional_training.py:78-142`.

## فایل‌های مجاز

```text
src/method_y_revival/image_modeling_v3.py                    (new)
src/method_y_revival/image_training_orchestrator_v3.py       (new)
scripts/run_method_y_image_training_v3.py                    (new)
configs/method_y_image_training_v3_1.yaml                    (فقط افزودن contract fields لازم)
tests/method_y_revival/test_image_modeling_v3.py              (new)
tests/method_y_revival/test_image_training_orchestrator_v3.py (new)
tests/method_y_revival/test_image_outer_exactly_once_v3.py     (new)
advisor-plans/README.md                                       (status only)
```

خارج scope: PS1 نهایی، robustness reports، external data، v1/v2، تغییر
endpoint/split/features، XAI و training production.

## قرارداد مدل‌سازی

نام variantها دقیقاً:

```text
scalar_tabular
scalar_tabular_plus_pattern_image
```

هیچ variant image-only به تحلیل اصلی اضافه نشود. image-only اگر بعداً ساخته
شد فقط negative/diagnostic benchmark و output root جدا است.

برای candidateها در config `complexity_rank` صریح اضافه کن. چون معماری یکسان
است، rank فقط tie-break prespecified است. پیشنهاد:

```yaml
hpo_candidates:
  - candidate_id: lr1e4_wd1e3
    learning_rate: 0.0001
    weight_decay: 0.001
    complexity_rank: 0
  - candidate_id: lr1e4_wd1e4
    learning_rate: 0.0001
    weight_decay: 0.0001
    complexity_rank: 1
  - candidate_id: lr3e4_wd1e4
    learning_rate: 0.0003
    weight_decay: 0.0001
    complexity_rank: 2
```

ترتیب را بعد از دیدن metrics تغییر نده.

## مراحل

### مرحله 1: RED contract tests

ابتدا تست‌های زیر را بنویس:

```text
test_outer_dataset_factory_is_not_called_during_inner_hpo
test_winner_uses_mean_inner_ibs_not_validation_nll_or_outer_metric
test_one_se_tie_break_uses_locked_complexity_rank
test_tabular_transformer_fits_only_inner_or_outer_train
test_matched_variants_have_identical_features_budget_splits_and_hpo_grid
test_outer_checkpoint_is_locked_before_test_dataset_creation
test_committed_outer_result_resume_does_not_construct_test_dataset
test_interrupt_after_outer_result_commit_recovers_without_second_access
test_incomplete_outer_cv_cannot_be_reported_complete
test_run_contract_change_fails_closed
test_private_oof_rows_are_not_written_to_public_reports
```

### مرحله 2: مدل factory و contract

در `image_modeling_v3.py`:

```python
def build_model(variant, model_config, tabular_dim, hazard_bins):
    if variant == "scalar_tabular":
        return TabularSurvivalNet(...)
    if variant == "scalar_tabular_plus_pattern_image":
        return PatternSurvivalNet(...)
    raise ImageModelingV3Error("UNKNOWN_LOCKED_VARIANT")
```

run contract هر واحد training باید شامل موارد زیر باشد:

```python
{
  "branch", "variant", "outer_fold", "inner_fold_or_final",
  "candidate_id", "model_config", "optimizer_config",
  "hazard_bin_edges", "feature_columns", "tabular_transformer",
  "membership_file_sha256", "cache_file_sha256", "split_file_sha256",
  "readiness_run_contract_sha256", "seed", "device_math_policy"
}
```

هر path یک output directory immutable دارد:

```text
development/<branch>/<variant>/outer_<k>/hpo/<candidate>/inner_<j>/
development/<branch>/<variant>/outer_<k>/final/
locked/<branch>/<variant>/outer_<k>/
private/outer_evaluation/<branch>/<variant>/fold_<k>/
```

### مرحله 3: survival interpolation و metrics

شبکه probability را در binها می‌دهد. برای IBS/horizonهای بین binها، یک قرارداد
piecewise-constant hazard بساز و قبل از outcome analysis تست کن:

```python
def survival_at_times(logits, bin_edges, evaluation_times):
    hazards = sigmoid(logits)
    # در هر interval hazard rate ثابت فرض شود.
    # در boundary دقیق با cumulative discrete survival برابر باشد.
    # خروجی در time صعودی monotone non-increasing و [0,1] باشد.
```

تست‌ها:

```text
boundary equality
monotonicity
zero/one hazard stability
grid outside locked support rejected
same fold grid for matched variants
```

evaluation همان الگوی `regional_training.py:86-142` را رعایت کند:

1. cutoff اداری فقط از train/test support و نه prediction؛
2. censoring distribution فقط outer-train؛
3. `integrated_brier_score`؛
4. `concordance_index_ipcw` با risk=`1-S(tau)`؛
5. dynamic AUC در horizonهای estimable؛
6. calibration input و OOF row خصوصی؛
7. event/at-risk counts.

grid باید پیش از مدل و یکسان برای matched variants از support همان fold ساخته
شود. requested horizons هرگز بر اساس نتیجه test تغییر نکنند.

### مرحله 4: inner HPO

شبه‌کد دقیق:

```python
for branch in ["amyloid", "tau"]:
  membership = load_hash_verified_private_membership(branch)
  development = membership[role == "development"]
  cache = verify_completed_cache(branch)

  for variant in LOCKED_VARIANTS:
    for outer in locked_splits:
      outer_train = rows(outer.train ∩ development)
      # outer test object هنوز ساخته نشود

      candidate_records = []
      for candidate in locked_hpo_candidates:
        inner_scores = []
        inner_best_epochs = []
        for inner in outer.inner_folds:
          inner_train = rows(inner.train ∩ outer_train)
          inner_val = rows(inner.validation ∩ outer_train)
          transformer = FoldTabularTransformer.fit(inner_train, features)
          train_ds = dataset(inner_train, transformer, include_image=variant_has_image)
          val_ds = dataset(inner_val, transformer, include_image=variant_has_image)
          fit_survival_model(... early stopping on INNER validation NLL ...)
          locked = strict_load_locked_checkpoint()
          ibs = evaluate_inner_ibs(locked, inner_train, inner_val)
          inner_scores.append(ibs)
          inner_best_epochs.append(best_epoch)

        candidate_records.append({
          mean_ibs, standard_error, complexity_rank,
          inner_fold_ibs, inner_best_epochs, status="COMPLETED"
        })

      winner = select_inner_winner(candidate_records)
      atomic_write winner + all candidates + hashes
```

validation NLL فقط برای early stopping داخل همان inner fold است؛ candidate
selection فقط IBS است.

### مرحله 5: final outer-train fit بدون test

```python
fixed_epochs = int(max(1, round(median(winner.inner_best_epochs))))
transformer = FoldTabularTransformer.fit(outer_train, features)
final_train_ds = dataset(outer_train, transformer, include_image=...)
fit_survival_model(
    validation_dataset=None,
    fixed_epochs=fixed_epochs,
    ...
)
checkpoint_sha = sha256(locked_checkpoint)
controller.lock_checkpoint(checkpoint_sha)
```

در fixed fit، outer test یا validation ساخته نشود. `validation_loss=train_loss`
فقط logging است و نباید best selection نامیده شود.

### مرحله 6: outer test دقیقاً یک‌باره

فقط پس از `CHECKPOINT_LOCKED`:

```python
def evaluator():
    outer_test = outer_test_factory()  # counter increment اینجاست
    return evaluate_locked_model(...)

metrics = controller.evaluate_once(evaluator)
controller.finalize()
```

result خصوصی می‌تواند patient hash و predictions داشته باشد؛ aggregate report
نباید داشته باشد. اگر state `OUTER_EVAL_IN_PROGRESS` بدون result معتبر یافت شد،
همان رفتار fail-closed موجود حفظ شود؛ outer دوباره خوانده نشود.

### مرحله 7: matched OOF comparison

بعد از پنج fold کامل هر دو variant:

```python
require_complete_outer_cv(..., expected=5)
bootstrap = cached_paired_oof_bootstrap(
  scalar_tabular,
  scalar_tabular_plus_pattern_image,
  iterations=2000,
  seed=locked_seed,
)
```

در این Plan `scanner_sensitivity_passed=False` باقی بماند؛ بنابراین positive
claim باز نشود. Plan 012 آن را ارزیابی می‌کند.

### مرحله 8: orchestration state رزوم‌پذیر

state machine:

```text
INIT
→ INPUTS_VERIFIED
→ INNER_HPO_RUNNING
→ INNER_SELECTION_DONE
→ FINAL_TRAINING
→ CHECKPOINT_LOCKED
→ OUTER_EVAL_IN_PROGRESS
→ OUTER_EVAL_COMMITTED
→ BRANCH_OUTER_CV_COMPLETE
```

state artifact بعد از artifact write و hash verification commit شود. اگر artifact
وجود دارد ولی state عقب مانده، hash recovery انجام شود؛ HPO/outer تکرار نشود.

## آزمون‌ها: integration اجباری

یک fixture 24–40 subject، 16³، پنج outer/سه inner و 2 candidate بساز:

1. اجرای بدون interruption؛
2. interruption پس از epoch 1 و resume؛
3. interruption بعد از outer result commit؛
4. مقایسهٔ byte/hash/weights/history/metrics؛
5. counter outer access برای هر variant/fold دقیقاً 1؛
6. تغییر seed/cache/split/config با همان root = contract mismatch.

فرمان:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -p no:cacheprovider `
  tests\method_y_revival\test_image_modeling_v3.py `
  tests\method_y_revival\test_image_training_orchestrator_v3.py `
  tests\method_y_revival\test_image_outer_exactly_once_v3.py -q
```

## معیار پایان

- [ ] high-level trainer وجود دارد و config v3 را مصرف می‌کند.
- [ ] دو variant matched هستند.
- [ ] winner فقط mean inner IBS است.
- [ ] final epoch count فقط از inner evidence است.
- [ ] outer dataset قبل از lock ساخته نمی‌شود.
- [ ] committed outer result بدون reaccess بازیابی می‌شود.
- [ ] OOF خصوصی و reports aggregate جدا هستند.
- [ ] synthetic integration interruption test پاس است.
- [ ] هیچ training production اجرا نشده است.
- [ ] کل Method Y zero failure است.

## STOP conditions

- cache summary/hash ناقص یا mismatch؛
- API drift در controller/runtime؛
- کمتر از دو evaluation time در common support؛
- هر fold بدون event؛
- outer access پیش از lock یا access counter >1؛
- نیاز به feature/endpoint/split جدید؛
- OOM در synthetic test که با 16³ رخ دهد؛
- تغییر فایل خارج scope.

## نگهداری

هیچ family یا candidate جدید بدون config/output version جدید اضافه نشود.
نتیجهٔ image v3 ثانویه است و negative incremental result باید همان‌قدر قابل
انتشار باشد که نتیجهٔ مثبت.
