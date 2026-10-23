"""Recompute rear lighting from a geometry baseline and reject the old dark rim.

The baseline is used only for isolated lighting regression. Production auto
continues to segment and measure every six-face input from original photos.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from phone_texture import lighting, lighting_review
from phone_texture.auto_material import propose_job, lighting_scope
from phone_texture.automatic import measure_masks
from phone_texture.geometry_review import digest
from phone_texture.pipeline import FACES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True,help='One SKU geometry directory with six PNGs')
    p.add_argument('--old',type=Path,required=True,help='Previously processed back.png')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=args.geometry.resolve()/'back.png';old=args.old.resolve()
    job,proof,allowed,paper=propose_job(source,'back',out/'materials',
        {f:args.geometry.resolve()/(f+'.png') for f in FACES})
    # Freeze source-only witnesses and targets before computing the new image.
    scope=lighting_scope(source,'back',job,out/'scope-assets')
    job.update(output_dir=str(out/'images'),records_dir=str(out/'records'))
    for name,value in [('job',job),('scope',scope),('analysis',proof)]:
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    before_hashes={str(path):digest(path) for path in [source,old]}
    old_error=None
    try:lighting_review.measure_perimeter(scope,source,old,allowed,paper)
    except ValueError as exc:old_error=str(exc)
    if old_error is None:raise AssertionError('Old dark border passed the new effect gate')
    with contextlib.redirect_stdout(io.StringIO()):lighting.run('aluminum',out/'job.json')
    output=out/'images/back.png'
    checks=lighting_review.measure_perimeter(scope,source,output,allowed,paper)
    invariants=measure_masks(source,output,allowed,paper)
    rim=np.array(Image.open(out/'materials/rear_perimeter.png'))>0
    previous=np.array(Image.open(old).convert('RGBA'));new=np.array(Image.open(output).convert('RGBA'))
    if previous.shape!=new.shape:raise AssertionError('Old comparison uses a different geometry')
    repair= rim.copy()
    from scipy.ndimage import binary_dilation,distance_transform_edt
    masks={d['name']:np.array(Image.open(d['mask']))>0 for d in job['views'][0]['domains']}
    for blend in job['views'][0].get('correction_blends',[]):
        union=np.logical_or.reduce([masks[n] for n in blend['domains']])&allowed&~paper
        primary=masks[blend['boundary_domain']]&union;other=union&~primary
        seam=(primary&binary_dilation(other))|(other&binary_dilation(primary))
        repair |= union&(distance_transform_edt(~seam)<3*blend['sigma'])
    changed=np.any(previous!=new,axis=2)
    nonrim=int(changed[~rim].sum());outside=int(changed[~repair].sum())
    if outside:raise AssertionError(f'Rear repair changed {outside} pixels outside metal perimeter and internal transition')
    if any(digest(path)!=sha for path,sha in before_hashes.items()):raise AssertionError('Input was overwritten')
    lighting_review.evidence_images(scope,source,output,out/'visual-review')
    report=dict(source=str(source),old=str(old),output=str(output),source_hashes=before_hashes,
        old_rejected=old_error,checks=checks,invariants=invariants,
        nonperimeter_pixels_changed_from_old=nonrim,outside_repair_pixels_changed=outside,visual_approval=False)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(dict(output=str(output),old_rejected=True,nonperimeter_changed=nonrim,outside_repair_changed=outside)))


if __name__=='__main__':main()
