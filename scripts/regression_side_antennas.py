"""Recheck source-bound side antenna complaints without bundling photos."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from phone_texture import lighting_review as lr


def run(process, source, previous):
    process, source, previous = map(Path, (process, source, previous))
    results = {}
    for face in ('left', 'right'):
        spec = lr.read(process / f'{face}-scope.json')
        src, old = source / f'{face}.png', previous / f'{face}.png'
        new = process / 'candidate/iphone17promax' / f'{face}.png'
        allowed = np.array(Image.open(process / f'{face}-repair-mask.png')) > 0
        protected = np.zeros(allowed.shape, bool)
        try:
            lr.measure_local(spec, src, old, allowed, protected)
        except ValueError as exc:
            if 'Local feature effect failed' not in str(exc):
                raise
        else:
            raise AssertionError('Previous endpoint defect was not rejected')
        results[face] = lr.measure_local(spec, src, new, allowed, protected)
        assert np.array_equal(lr.pixels(old)[~allowed], lr.pixels(new)[~allowed])
        lr.check_review(spec, src, new, process / f'{face}-evidence/review.json')
    for face in ('front', 'back', 'top', 'bottom'):
        assert lr.digest(previous / f'{face}.png') == lr.digest(process / 'candidate/iphone17promax' / f'{face}.png')
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--process', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--previous', required=True)
    a = p.parse_args()
    run(a.process, a.source, a.previous)
    print('Side antenna regression passed: old rejected, new accepted, unrelated pixels locked')
