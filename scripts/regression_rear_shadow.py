"""Replay saved rear geometry and verify the incremental ring-shadow operation.

Use --verify-existing to check saved before/after-native.png and job.json only.
No original image, historical batch, UI job or model UV is changed.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from phone_texture import auto_material, lighting, shadow_lift_profile
from phone_texture.back_lens import lens_centres
from phone_texture.geometry_review import digest, write


def verify(folder):
    job = json.loads((folder/'job.json').read_text())
    view = job['views'][0]; plan = view['rear_shadow_correction']
    assert plan['status'] == 'ready'
    assert digest(view['source']) == plan['source_sha256']
    assert digest(plan['material_mask']) == plan['material_sha256']
    before = np.array(Image.open(folder/'before-native.png').convert('RGBA'))
    after = np.array(Image.open(folder/'after-native.png').convert('RGBA'))
    assert before.shape == after.shape
    assert np.array_equal(before[..., 3], after[..., 3])
    material = np.array(Image.open(plan['material_mask']).convert('L')) > 0
    changed = np.any(before != after, axis=2)
    assert not changed[~material].any()
    h,w = material.shape; yy,xx = np.indices((h,w))
    ring = np.zeros((h,w),bool); footprint = ring.copy()
    shadows = []
    render = lighting.engine('aluminum')
    L0 = render.rgb_to_lab(before[...,:3])[...,0]
    L1 = render.rgb_to_lab(after[...,:3])[...,0]
    for lens in plan['lenses']:
        cx,cy,aa,bb,angle = lens['ellipse']; phi = np.deg2rad(angle)
        U = (np.cos(phi)*(xx-cx)+np.sin(phi)*(yy-cy))/(aa/2)
        V = (-np.sin(phi)*(xx-cx)+np.cos(phi)*(yy-cy))/(bb/2)
        R = np.sqrt(U*U+V*V); theta = np.mod(np.arctan2(V,U),2*np.pi)
        starts = lens['start_radii']
        edge = np.interp(theta.ravel(),np.linspace(0,2*np.pi,361),starts+[starts[0]]).reshape(h,w)
        hardware = lens.get('hardware_radii', starts)
        hardware_edge = np.interp(theta.ravel(),np.linspace(0,2*np.pi,361),hardware+[hardware[0]]).reshape(h,w)
        ring |= R <= hardware_edge
        footprint |= (R>hardware_edge)&(R<edge+.42)&material
        # Independent user-visible cast-shadow witness: source-right of each
        # lens, beyond the measured joint. No assertion on a synthetic target.
        witness = (R>edge+.025)&(R<edge+.14)&(xx>cx+.8*(aa+bb)/4)&material
        shadows.append(dict(samples=int(witness.sum()),median_added_L=float(np.median((L1-L0)[witness]))))
    assert not changed[ring].any(), 'Glass/ring pixels changed'
    assert not changed[~footprint].any(), 'Correction escaped the measured footprint'
    assert all(s['median_added_L']>2 for s in shadows), shadows
    centres0,_ = lens_centres(before); centres1,_ = lens_centres(after)
    assert centres0 is not None and centres1 is not None
    drift = float(np.max(np.linalg.norm(centres1-centres0,axis=1)))
    assert drift < .25, drift
    proof = dict(passed=True,source_sha256=digest(view['source']),
        before_sha256=digest(folder/'before-native.png'),after_sha256=digest(folder/'after-native.png'),
        alpha_unchanged=True,glass_and_bright_rim_unchanged=True,outside_footprint_unchanged=True,
        changed_pixels=int(changed.sum()),shadow_witnesses=shadows,lens_measurement_drift_px=drift)
    write(folder/'regression.json',proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--skus',nargs='+',default=['sku1','sku2','sku3','sku4'])
    parser.add_argument('--verify-existing',action='store_true')
    args=parser.parse_args(); output=args.output.resolve(); results={}
    for sku in args.skus:
        folder=output/sku
        if not args.verify_existing:
            if args.cases is None:parser.error('--cases required for replay')
            folder.mkdir(parents=True,exist_ok=False)
            old=json.loads((args.cases/sku/'work/material-inputs/back/job.json').read_text())
            source=old['views'][0]['source']
            assert digest(source)==old['views'][0]['source_sha256']
            job,analysis,_,_=auto_material.propose_job(source,'back',folder/'materials',protection_policy='none')
            job,profile=shadow_lift_profile.apply(job,'back',folder/'materials')
            write(folder/'job.json',job);write(folder/'analysis.json',dict(analysis,lighting_profile=profile))
            render=lighting.engine('aluminum');target,_=render.target_color({'views':job['views']})
            baseline=copy.deepcopy(job);baseline['views'][0].pop('rear_shadow_correction')
            for name,item in [('before',baseline),('after',job)]:
                pixels,_,_,proof=render.process_view(item['views'][0],target,dict(item['defaults'],color_mode=item['color_mode']))
                Image.fromarray(pixels).save(folder/(name+'-native.png'))
                write(folder/(name+'-validation.json'),proof)
        results[sku]=verify(folder)
    write(output/'regression.json',results)
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
