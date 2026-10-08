from pathlib import Path
import pytest
from scripts.xyz_fusion_postall_v2.core import Units
from scripts.xyz_postfusion_final_v1 import closure


def test_combined_dossier_uses_whitelist_and_requires_every_stage(tmp_path,monkeypatch):
    from scripts.xyz_fusion_postall_v2.release import PUBLIC_FILES
    from scripts.xyz_pet_manufacturer_audit_v1.reporting import PUBLIC_FILES as LMO_FILES
    approved={'postall':PUBLIC_FILES,'manufacturer':LMO_FILES,
        'regional':{'REGIONAL_EVIDENCE.json','REGIONAL_EVIDENCE_FA.md','regional_hazard_ratios.svg'}}
    for key,names in approved.items():
        root=tmp_path/closure.ROOTS[key]/'public';root.mkdir(parents=True)
        for name in names:
            (root/name).write_text('{}' if name.endswith('.json') else '<svg/>' if name.endswith('.svg') else 'Aggregate fixture',encoding='utf-8')
    result={'status':'COMPUTED_PENDING_REVIEW','stage_matrix':{f'F{i:02d}':'COMPUTED' for i in range(9)},
        'claim_status':'POST_HOC_INTERNAL_EXPLORATORY'}
    deps={'f08':'fixed','manufacturer':'fixed','regional':'fixed'}
    monkeypatch.setattr(closure,'evidence',lambda _: (result,deps))
    units=Units(tmp_path/'results/combined',{'fixture':True})
    receipt=closure.build(tmp_path,units)
    assert receipt['privacy']=='PASS' and len(receipt['outputs'])==17
    assert closure.verify(tmp_path,units)['status'].startswith('PASS_ALL')
    (units.root/'public/private_extra.csv').write_text('patient_hash\nfixture',encoding='utf-8')
    with pytest.raises(ValueError,match='WHITELIST'):
        closure.verify(tmp_path,units)


def test_combined_release_does_not_promote_without_review(tmp_path,monkeypatch):
    monkeypatch.setattr(closure,'evidence',lambda _: (_ for _ in ()).throw(ValueError('FINAL_POSTALL_INCOMPLETE')))
    units=Units(tmp_path,{'fixture':True})
    with pytest.raises(ValueError,match='INCOMPLETE'):closure.build(tmp_path,units)
    assert units.read('combined_release') is None
    assert not (tmp_path/'public').exists()
