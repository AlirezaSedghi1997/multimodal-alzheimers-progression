"""Promote only independently reviewed post-fusion results; preserve history."""
from .core import atomic_json, atomic_text, digest, file_sha, read_json
from .release import verify_final


def candidate(ctx):
    verification = verify_final(ctx)
    if not verification.get('verifier', '').startswith('PASS'):
        raise ValueError('FINAL_VERIFICATION_REQUIRED_FOR_CANDIDATE')
    release = ctx.units.read('f08')
    return {'status': 'VERIFIED_CANDIDATE_PENDING_POST_RUN_REVIEW', 'contract': ctx.units.binding,
            'f08_digest': digest(release), 'verification_digest': digest(verification),
            'result_root': ctx.out.relative_to(ctx.root).as_posix(),
            'parent_root_unchanged': ctx.parent.root.relative_to(ctx.root).as_posix(),
            'supersedes': ['results/xyz_fusion_postall_v1', 'results/xyz_fusion_postall_v1_maps'],
            'old_files_deleted_or_rewritten': False,
            'promote_only_after_independent_post_run_result_and_image_review': True}


def promote(ctx):
    expected = candidate(ctx)
    path = ctx.out / 'private/post_run_review.json'
    if not path.is_file():
        raise ValueError('POST_RUN_SUPERVISORY_REVIEW_NOT_YET_RECORDED')
    review = read_json(path)
    if (review.get('contract') != ctx.units.binding or review.get('f08_digest') != expected['f08_digest']
            or review.get('status') != 'PASS' or review.get('unresolved_technical_defects') != []
            or review.get('actual_subject_overlays_reviewed') is not True
            or review.get('numerical_and_negative_results_reconciled') is not True
            or review.get('no_hidden_training_or_parent_change') is not True):
        raise ValueError('POST_RUN_REVIEW_INCOMPLETE_OR_STALE')
    final = dict(expected, status='FINAL_INTERNAL_POSTFUSION_REFERENCE', review_sha256=file_sha(path),
                 claim_status='POST_HOC_INTERNAL_EXPLORATORY')
    registry = ctx.root / 'governance/xyz_postfusion_current.json'
    atomic_json(registry, final)
    atomic_text(ctx.root / 'docs/XYZ_POSTFUSION_CURRENT_FA.md',
                '# مرجع رسمی پس از فیوژن\n\n'
                'پس از ممیزی و بازبینی نهایی، `results/xyz_fusion_postall_v2` مرجع محاسبات و اعتبارسنجی داخلی است.\n\n'
                'خروجی‌های postall v1 و v1_maps جایگزین شده‌اند و فقط سابقه ممیزی هستند؛ برای نتیجه نهایی پایان‌نامه استفاده نشوند. '
                'فیوژن والد، checkpointها و داده‌های آن تغییر نکرده‌اند. محدودیت‌ها و نتایج منفی نسخه جدید باید همراه اعداد گزارش شوند.\n\n'
                'سقف ادعا: `POST_HOC_INTERNAL_EXPLORATORY`؛ نه تشخیص بالینی، نه تعمیم خارجی، نه تعیین ضایعه آلزایمر.\n')
    return final
