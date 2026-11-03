from pathlib import Path
from phone_texture.automatic import implementation_manifest


def test_runtime_helper_and_atlas_changes_are_visible_without_git(tmp_path):
    files={'phone_texture/antenna_material.py':'source detector',
           'phone_texture/side_material.py':'hardware detector',
           'phone_texture/recognition_data/atlas.npz':'frozen resource',
           'phone_texture/lighting_review.py':'acceptance gate',
           'phone_normalizer/api.py':'normalizer',
           'scripts/segmentation.py':'segmentation',
           'skills/phone-aluminum-frame-restore/scripts/harmonize_aluminum.py':'lighting'}
    for rel,value in files.items():
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    before=implementation_manifest(tmp_path)
    assert set(before)==set(files)
    changed='phone_texture/antenna_material.py';(tmp_path/changed).write_text('revised source detector')
    after=implementation_manifest(tmp_path)
    assert after[changed]!=before[changed]
    assert all(after[k]==before[k] for k in files if k!=changed)


def test_complete_source_checkout_records_new_material_helpers():
    manifest=implementation_manifest(Path(__file__).resolve().parents[1])
    assert {'phone_texture/antenna_material.py','phone_texture/side_material.py',
            'phone_texture/view_recognition.py','phone_texture/lighting_review.py',
            'phone_texture/recognition_data/atlas.npz'}<=manifest.keys()
