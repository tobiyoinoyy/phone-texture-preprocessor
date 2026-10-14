import hashlib
import numpy as np
from PIL import Image
import pytest
from phone_texture.protection_policy import check_sticker_only
from phone_texture import lighting_review as r
from test_lighting_review import fixture,perimeter_delivery
from test_six_view_workflow import w


def test_policy_rejects_nonpaper_lines_even_if_unchanged(tmp_path):
    a=np.zeros((20,20,4),dtype=np.uint8);a[...,:3]=80;a[...,3]=255
    src=tmp_path/'source.png';Image.fromarray(a).save(src)
    paper=np.zeros((20,20),bool);paper[10:18,8:12]=True
    Image.fromarray((paper*255).astype('uint8')).save(tmp_path/'paper.png')
    c=dict(protection_policy='sticker_only',source_sha256=r.digest(src),sticker_mask=str(tmp_path/'paper.png'))
    assert check_sticker_only(c,src,paper,~paper)['extra_protected_pixels']==0
    stale=paper.copy();stale[2,1:5]=True
    with pytest.raises(ValueError,match='extra protected'):check_sticker_only(c,src,stale,~stale)
    with pytest.raises(ValueError,match='editable'):check_sticker_only(c,src,paper,np.ones_like(paper))


def test_local_dark_spot_is_not_hidden_by_good_edge_median(tmp_path):
    work,spec=fixture(tmp_path)
    m=np.zeros((32,16),np.uint8);m[14:16,0:2]=255;Image.fromarray(m).save(work/'spot.png')
    spec['local_checks']={'button-root':dict(mask=str(work/'spot.png'),min_L=50,max_L=80,percentile=10)}
    a=r.pixels(work/'good.png');a[14:16,0:2,:3]=20;Image.fromarray(a).save(work/'good.png')
    allowed=np.ones((32,16),bool)
    with pytest.raises(ValueError,match='Local feature effect failed'):
        r.measure_perimeter(spec,work/'source.png',work/'good.png',allowed,~allowed)
    # The original eight median witnesses alone would pass.
    spec.pop('local_checks');assert len(r.measure_perimeter(spec,work/'source.png',work/'good.png',allowed,~allowed))==8


def test_publisher_enforces_sticker_only_from_frozen_scope(tmp_path):
    work,plan=perimeter_delivery(tmp_path)
    # Create a new reviewed scope for the explicitly changed user policy.
    frozen=r.read(work/'lighting-scope.json')
    f=frozen['specs']['SKU_01']['front'];f.update(protection_policy='sticker_only',sticker_mask=str(work/'protected.png'))
    r.write(work/'lighting-scope.json',frozen);r.write(work/'lighting-scope-proof.json',dict(sha256=r.digest(work/'lighting-scope.json')))
    a=r.pixels(work/'front.png');src=r.pixels(work/'geometry/SKU_01/front.png');a[15,8]=src[15,8];Image.fromarray(a).save(work/'front.png')
    m=np.zeros((32,16),np.uint8);m[15,8]=255;Image.fromarray(m).save(work/'stale-protect.png')
    Image.fromarray(255-m).save(work/'allowed.png');plan['SKU_01']['front']['protected_mask']=str(work/'stale-protect.png')
    w.save(work/'plan.json',plan)
    with pytest.raises(ValueError,match='extra protected'):w.publish(work,work/'plan.json',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_missing_policy_cannot_silently_inherit_stale_masks(tmp_path):
    work,spec=fixture(tmp_path);spec.pop('protection_policy')
    with pytest.raises(ValueError,match='Explicit aluminum protection policy'):
        r.validate_spec(spec,work/'source.png')
