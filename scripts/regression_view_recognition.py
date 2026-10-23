"""Validate frozen view atlas with leave-one-SKU-out and unseen captures.

Reference SKU_01–06 are excluded when testing that SKU. SKU_07–10 and Pro Max
are held out of the atlas. Filenames supply ground truth only in this harness.
No references are updated during validation.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path
from PIL import Image, ImageOps
from phone_texture.view_recognition import FACES, PROFILE, recognize
from phone_texture.web_app import App


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--promax',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    rows=[];negative=[];start=time.monotonic()
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp)
        for folder in sorted(a.root.glob('SKU_*')):
            for face in FACES:
                source=folder/(face+'.JPG')
                result=recognize(source,exclude_group=folder.name)
                rows.append(dict(sku=folder.name,expected=face,correct=result['face']==face,**result))
                if face in ('left','right'):
                    with Image.open(source) as im:ImageOps.mirror(im).save(tmp/'mirror.png')
                    proof=recognize(tmp/'mirror.png',exclude_group=folder.name)
                    negative.append(dict(case=folder.name+'-'+face+'-mirror',rejected=proof['face'] is None,**proof))
        if a.promax:
            for face in FACES:
                result=recognize(a.promax/(face+'.jpg'))
                rows.append(dict(sku='ProMax-unseen',expected=face,correct=result['face']==face,**result))
        for face in FACES:
            with Image.open(a.root/'SKU_09'/(face+'.JPG')) as im:im.transpose(Image.Transpose.ROTATE_90).save(tmp/'rotated.png')
            proof=recognize(tmp/'rotated.png')
            negative.append(dict(case=face+'-rotated',rejected=proof['face'] is None,**proof))
        Image.new('RGB',(800,800),'white').save(tmp/'blank.png')
        proof=recognize(tmp/'blank.png');negative.append(dict(case='blank',rejected=proof['face'] is None,**proof))
        # Arbitrary filenames carry no view hints. Copies retain original bytes.
        raw=tmp/'unnamed';raw.mkdir();expected={}
        for i,face in enumerate(reversed(FACES)):
            dest=raw/f'IMG_{791+i}.jpg';dest.write_bytes((a.root/'SKU_09'/(face+'.JPG')).read_bytes());expected[face]=dest.read_bytes()
        app=App(tmp/'app');scan=app.scan(raw)
        assert scan['valid']==0 and scan['renameable']==1
        output=app.apply_names(scan['id']);assert output['valid']==1 and output['renameable']==0
        assert app.apply_names(scan['id'])['id']==output['id']
        for face in FACES:assert (Path(output['source'])/'unnamed'/(face+'.jpg')).read_bytes()==expected[face]
        assert sorted(p.name for p in raw.iterdir())==[f'IMG_{791+i}.jpg' for i in range(6)]
        # Canonical names swapped must not pass silently.
        swap=tmp/'swapped';swap.mkdir()
        for face in FACES:(swap/(face+'.jpg')).write_bytes(expected[dict(front='back',back='front').get(face,face)])
        swapped=app.scan(swap);assert swapped['renameable']==1 and len(swapped['rows'][0]['changes'])==2
        # Duplicate view must not be filled in by elimination.
        (swap/'back.jpg').write_bytes(expected['back']);assert app.scan(swap)['renameable']==0
        # Confirmation must be bound to inspected bytes.
        (raw/'IMG_791.jpg').write_bytes(expected['front'])
        # Use a fresh, unconfirmed scan, then alter its bytes.
        changed=app.scan(Path(output['source'])/'unnamed')
        (Path(output['source'])/'unnamed'/'front.jpg').write_bytes(expected['back'])
        try:app.start(dict(scan_id=changed['id'],skus=['unnamed'],concurrency=1,model='iphone17pro'))
        except ValueError as exc:assert '图片已改变' in str(exc)
        else:raise AssertionError('Changed images accepted')
    report=dict(profile=PROFILE,reference_groups='SKU_01–06, self excluded',
                held_out='SKU_07–10 plus Pro Max',correct=sum(r['correct'] for r in rows),total=len(rows),
                rejected=sum(r['rejected'] for r in negative),negative_total=len(negative),
                integration='unnamed, swapped, duplicate, copy byte integrity, idempotency, changed source passed',
                seconds=time.monotonic()-start,rows=rows,negative=negative)
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','negative')},ensure_ascii=False,indent=2))
    assert all(r['correct'] for r in rows),[r for r in rows if not r['correct']]
    assert all(r['rejected'] for r in negative),[r for r in negative if not r['rejected']]


if __name__=='__main__':main()
