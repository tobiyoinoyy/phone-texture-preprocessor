"""Default planar perspective rectification of front/back (R10)."""
import copy
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from phone_texture import face_perspective as fp
from phone_texture.geometry_review import digest


def _phone(face, keystone=0.):
    """Rectangle phone with an in-plane reference, then a synthetic keystone.

    keystone > 0 makes the bottom wider than the top (camera nearer the bottom).
    """
    h, w = 1000, 480
    a = np.zeros((h, w, 4), np.uint8)
    body = np.zeros((h, w), np.uint8)
    cv2.rectangle(body, (20, 20), (w-21, h-21), 255, -1)
    a[..., 3] = body
    if face == 'front':
        a[body > 0, :3] = 170                               # bright rim
        cv2.rectangle(a, (34, 34), (w-35, h-35), (18, 18, 20, 255), -1)   # dark glass
    else:
        a[body > 0, :3] = 200
        cv2.rectangle(a, (40, 300), (w-41, h-31), (120, 120, 120, 255), 2)  # panel seam
    if keystone:
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = np.float32([[keystone, 0], [w-keystone, 0], [0, h], [w, h]])
        m = cv2.getPerspectiveTransform(src, dst)
        a = cv2.warpPerspective(a, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=0)
    return a


@pytest.mark.parametrize('face', ['front', 'back'])
def test_keystoned_face_is_rectified_to_a_rectangle(face):
    a = _phone(face, keystone=6.)
    before, _ = fp.reference_quad(a, face)
    assert fp.tapers(before)[0] > 2*fp.MIN_TAPER_PX
    out, proof = fp.rectify(a, face)
    assert out is not None and proof['action'] == 'rectified'
    after, _ = fp.reference_quad(out, face)
    assert max(abs(v) for v in fp.tapers(after)) < 1.
    assert proof['reference'] == ('front_screen_glass_edge' if face == 'front' else 'rear_glass_panel_seam')
    assert .97 <= proof['opaque_area_ratio'] <= 1.03


@pytest.mark.parametrize('face', ['front', 'back'])
def test_square_face_is_kept_byte_for_byte(face):
    out, proof = fp.rectify(_phone(face), face)
    assert out is None and proof['action'] == 'unchanged_taper_within_noise'


def test_prepare_rebinds_geometry_and_copies_other_faces(tmp_path):
    files, rows = {}, []
    for face in ('front', 'back', 'left', 'right', 'top', 'bottom'):
        img = _phone(face if face in ('front', 'back') else 'back', keystone=6. if face == 'back' else 0.)
        path = tmp_path/(face+'.png')
        Image.fromarray(img).save(path)
        files[face] = path
        rows.append(dict(sku='s', face=face, output_sha256=digest(path), geometry=dict(output_size=[img.shape[1], img.shape[0]])))
    geometry = dict(candidates_complete=True, images=rows)
    original = copy.deepcopy(geometry)
    selected, effective, report = fp.prepare(files, geometry, tmp_path/'process')
    assert geometry == original
    assert report['applied_faces'] == ['back'] and report['faces']['front']['applied'] is False
    for face in ('front', 'left', 'right', 'top', 'bottom'):
        assert digest(selected[face]) == digest(files[face])
    row = next(r for r in effective['images'] if r['face'] == 'back')
    assert row['output_sha256'] == digest(selected['back']) != row['pre_perspective_sha256']
    with Image.open(selected['back']) as image:
        assert list(image.size) == row['geometry']['output_size']
    assert json.loads((tmp_path/'process/report.json').read_text())['default_enabled'] is True


def test_unfittable_face_keeps_source_with_notice(tmp_path):
    files, rows = {}, []
    blank = np.zeros((400, 200, 4), np.uint8)
    blank[20:380, 20:180] = 255                 # no reference edge at all
    for face in ('front', 'back', 'left', 'right', 'top', 'bottom'):
        path = tmp_path/(face+'.png')
        Image.fromarray(blank).save(path)
        files[face] = path
        rows.append(dict(sku='s', face=face, output_sha256=digest(path)))
    selected, _, report = fp.prepare(files, dict(candidates_complete=True, images=rows), tmp_path/'p')
    assert report['applied_faces'] == []
    assert {n['face'] for n in report['notices'] if n['code'] == 'perspective_not_applied'} == {'front', 'back'}
    assert all(digest(selected[f]) == digest(files[f]) for f in files)
