from phone_texture.geometry_review import CHECKS, digest, write, make_evidence
from phone_texture.pipeline import batch_files
from PIL import Image


def approved_fixture(geometry, work):
    """Tiny known-solid fixture, with real evidence; not used for photo approval."""
    entries=[]; items={}
    for sku,faces in batch_files(geometry).items():
        for face,p in faces.items():
            im=Image.open(p).convert('RGBA')
            artifacts=make_evidence(p,im,{'after_support':im,'after_rotation':im},work/'evidence'/sku/face)
            entries.append(dict(sku=sku,face=face,source=str(p),source_sha256=digest(p),output_sha256=digest(p),geometry_evidence=artifacts))
            items[sku+'/'+face]=dict.fromkeys(CHECKS,True)
            items[sku+'/'+face]['notes']='Synthetic solid rectangle; exact known full coverage, no stand.'
    path=work/'geometry-report.json'
    write(path,dict(geometry_status='approved',candidates_complete=True,images=entries,geometry_review=dict(reviewer='synthetic-fixture',items=items)))
    return path


def review_fixture(process):
    import json
    path=process/'report.json';report=json.loads(path.read_text())
    items={}
    for e in report['images']:
        items[e['sku']+'/'+e['face']]={**dict.fromkeys(CHECKS,True),'notes':'Synthetic rounded rectangle, all source boundary pixels reviewed.'}
    out=process/'review.json';write(out,dict(report_sha256=digest(path),reviewer='synthetic-test',items=items))
    return out
