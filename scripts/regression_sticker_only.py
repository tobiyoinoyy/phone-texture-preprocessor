"""Replay the source-bound Pro Max local shadow/protection regression."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from phone_texture import lighting_review as r
from phone_texture.protection_policy import check_sticker_only


def run(process, repair, report_path):
    process,repair=Path(process),Path(repair)
    spec=r.read(repair/'back-scope.json');src=process/'source/back.png'
    old=process/'edge-v3/candidate/iphone17promax/back.png';new=repair/'candidate/iphone17promax/back.png'
    paper=np.array(Image.open(repair/'sticker-mask.png'))>0;allowed=(r.pixels(src)[...,3]>0)&~paper
    old_protection=np.array(Image.open(process/'edge-v3/back-protected.png'))>0
    errors=[]
    for check in (lambda:check_sticker_only(spec,src,old_protection),
                  lambda:r.measure_perimeter(spec,src,old,allowed,paper)):
        try:check()
        except ValueError as e:errors.append(str(e))
        else:raise AssertionError('Known faulty version passed')
    assert 'extra protected' in errors[0] and 'Local feature effect failed' in errors[1]
    policy=check_sticker_only(spec,src,paper,allowed)
    metrics=r.measure_perimeter(spec,src,new,allowed,paper)
    a,b=r.pixels(src),r.pixels(new)
    assert np.array_equal(a[...,3],b[...,3]) and np.array_equal(a[paper],b[paper])
    other={}
    for face in ('front','left','right','top','bottom'):
        other[face]=r.digest(process/'edge-v3/candidate/iphone17promax'/f'{face}.png')==r.digest(repair/'candidate/iphone17promax'/f'{face}.png')
        assert other[face]
    result={'source_sha256':r.digest(src),'output_sha256':r.digest(new),'old_rejected':errors,
            'policy':policy,'new_effect_passed':metrics,'other_faces_unchanged':other}
    r.write(report_path,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--process',required=True);p.add_argument('--repair',required=True);p.add_argument('--report',required=True)
    a=p.parse_args();run(a.process,a.repair,a.report);print('Old protection/dark spots rejected; revised back passes')
