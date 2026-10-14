"""Replay the reviewed Pro Max perimeter case. Images stay outside the repository."""
import argparse
import copy
from pathlib import Path

import numpy as np
from PIL import Image
from phone_texture.lighting_review import read, write, digest, measure_perimeter, lightness, witness, evidence_images


def run(process, output):
    process, output = Path(process).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixtures = Path(__file__).resolve().parents[1]/'tests/fixtures/aluminum-perimeter'
    cases = read(fixtures/'case.json')
    report = {'case': 'iphone17promax-20261013', 'faces': {}, 'other_faces_unchanged': {}}
    for face, original in cases.items():
        spec = copy.deepcopy(original)
        spec['required_metal_mask'] = str(fixtures/spec['required_metal_mask'])
        for z in spec['zones'].values():z['mask'] = str(fixtures/z['mask'])
        src = process/'source'/f'{face}.png'
        old = process/'candidate/iphone17promax'/f'{face}.png'
        good = process/'edge-v3/candidate/iphone17promax'/f'{face}.png'
        allowed = np.array(Image.open(process/'edge-v3'/f'{face}-editable.png'))>0
        protected = np.array(Image.open(process/'edge-v3'/f'{face}-protected.png'))>0
        # Use the GOOD allowed mask for BOTH versions: proves actual effect is checked,
        # rather than merely rejecting an old mask or relying on the action label.
        try:
            measure_perimeter(spec,src,old,allowed,protected)
        except ValueError as e:
            if 'Perimeter effect failed' not in str(e):raise
            rejection = str(e)
        else:raise AssertionError('Old black-rim image incorrectly passed')
        metrics = measure_perimeter(spec,src,good,allowed,protected)
        L = lightness(np.array(Image.open(old))[...,:3])
        failed = []
        for name,z in spec['zones'].items():
            value=float(np.median(L[witness(z['mask'],L.shape)]))
            metrics[name]['old_L']=value
            if not z['min_L'] <= value <= z['max_L']:failed.append(name)
        evidence_images(spec,src,good,output/face)
        report['faces'][face]={'old_rejected':True,'reason':rejection,'failed_old_zones':failed,
                              'corrected_effect_passed':True,'source_sha256':digest(src),
                              'old_sha256':digest(old),'corrected_sha256':digest(good),'metrics':metrics}
    for f in ('left','right','top','bottom'):
        equal=digest(process/'candidate/iphone17promax'/f'{f}.png') == digest(process/'edge-v3/candidate/iphone17promax'/f'{f}.png')
        if not equal:raise AssertionError(f'Unexpected change to {f}')
        report['other_faces_unchanged'][f]=equal
    report['visual_review']='pending; inspect generated 1x/4x light/dark crops; metrics alone cannot approve material appearance'
    write(output/'regression.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--process',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=run(a.process,a.output)
    print({f:len(v['failed_old_zones']) for f,v in result['faces'].items()})
