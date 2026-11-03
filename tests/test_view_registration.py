"""Bounded framing recovery, independent witnesses and fail-closed routing.

The frozen atlas supplies descriptors, never newly captured references. The
simple RGB body is a separate source-closure fixture, not classifier training.
Real raw-image leave-one-SKU-out/held-out validation remains a separate check.
"""
import hashlib

import cv2
import numpy as np
from PIL import Image
import pytest

from phone_texture import view_recognition as recognition


@pytest.fixture
def frozen():
    meta, arrays = recognition.atlas()
    return meta['references'], arrays


def body(face='back'):
    rgb = np.full((192, 192, 3), 230, np.uint8)
    rgb[24:176, 60:132] = 30 if face == 'front' else 130
    return rgb


def shifted(values, dy, background=0):
    size = values.shape[0]
    return cv2.warpAffine(values, np.float32([[1, 0, 0], [0, 1, dy]]),
                          (size, size), borderValue=background)


@pytest.mark.parametrize('face', ['front', 'back'])
@pytest.mark.parametrize('group', ['SKU_01', 'SKU_03', 'SKU_06'])
def test_small_framing_shift_recovers_without_new_templates(frozen, face, group):
    refs, arrays = frozen
    feature = arrays[f'{group}_{face}_edge'].reshape(48, 48)
    moved = shifted(feature, -1.5).ravel()
    rgb = shifted(body(face), -6, (230, 230, 230))
    proof = recognition.register_plate(rgb, moved, refs, arrays)
    assert proof['accepted'] and proof['face'] == face
    assert proof['score'] >= .90 and proof['margin'] >= .025
    assert proof['interior_scores'][face] >= .90
    assert abs(proof['translation_cells'][1] - 1.5) <= .5
    assert .90 < proof['scale'] < 1.10


def test_pose_estimator_cannot_use_internal_identity_pixels(frozen):
    refs, arrays = frozen
    references = [arrays[r['id'] + '_edge'] for r in refs if r['face'] == 'front']
    feature = arrays['SKU_03_front_edge'].reshape(48, 48).copy()
    _, interior = recognition._plate_regions()
    before, _ = recognition._outline_registration(feature, references)
    # Values much stronger than any real feature must still not alter pose.
    feature[interior] = np.random.default_rng(17).uniform(10, 100, interior.sum())
    after, _ = recognition._outline_registration(feature, references)
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize('face', ['front', 'back'])
def test_outline_alone_cannot_pass_by_high_full_image_score(frozen, face):
    refs, arrays = frozen
    feature = arrays[f'SKU_03_{face}_edge'].reshape(48, 48).copy()
    feature[8:20, 17:31] = 0
    proof = recognition.register_plate(body(face), feature.ravel(), refs, arrays)
    # The full-image gate itself passes. A genuinely independent test must fail.
    assert proof['score'] >= .90
    assert not proof['accepted']
    assert 'interior' in proof['reason']


@pytest.mark.parametrize('kind', ['mirror', 'rotate90', 'rotate180', 'large_shift', 'large_scale'])
def test_back_pose_and_orientation_negatives(frozen, kind):
    refs, arrays = frozen
    feature = arrays['SKU_03_back_edge'].reshape(48, 48)
    if kind == 'mirror':
        feature = feature[:, ::-1]
    elif kind.startswith('rotate'):
        feature = np.rot90(feature, 1 if kind == 'rotate90' else 2)
    elif kind == 'large_shift':
        feature = shifted(feature, -6)
    else:
        matrix = cv2.getRotationMatrix2D((23.5, 23.5), 0, 1.3)
        feature = cv2.warpAffine(feature, matrix, (48, 48))
    proof = recognition.register_plate(body(), feature.ravel(), refs, arrays)
    assert not proof['accepted']


@pytest.mark.parametrize('face', ['front', 'back'])
def test_missing_lower_body_rejected_even_when_descriptor_is_unchanged(frozen, face):
    refs, arrays = frozen
    feature = arrays[f'SKU_03_{face}_edge']
    rgb = body(face)
    rgb[154:] = 230
    proof = recognition.register_plate(rgb, feature, refs, arrays)
    assert proof['score'] >= .90
    assert not proof['accepted'] and 'bottom closure' in proof['reason']


def test_front_requires_dark_glass_in_source_not_only_atlas_edges(frozen):
    refs, arrays = frozen
    rgb = body('front')
    rgb[32:160, 65:127] = 230  # Outline remains, screen interior is background.
    proof = recognition.register_plate(rgb, arrays['SKU_03_front_edge'], refs, arrays)
    assert proof['score'] >= .90
    assert not proof['accepted'] and 'dark screen' in proof['reason']


def test_blank_image_rejected_and_bytes_preserved(tmp_path):
    path = tmp_path / 'arbitrary-name.png'
    Image.new('RGB', (192, 192), 'white').save(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert recognition.recognize(path)['face'] is None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_duplicate_registered_faces_are_not_filled_by_elimination(tmp_path, monkeypatch, frozen):
    refs, arrays = frozen
    feature = shifted(arrays['SKU_03_back_edge'].reshape(48, 48), -1.5).ravel()
    rgb = shifted(body(), -6, (230, 230, 230))
    monkeypatch.setattr(recognition, 'edge_feature', lambda _: feature)
    paths = []
    for i in range(6):
        path = tmp_path / f'capture-{i}.png'
        Image.fromarray(rgb).save(path)
        paths.append(path)
    report = recognition.inspect_images(paths)
    assert not report['complete']
    assert all(row['face'] == 'back' for row in report['images'])
    assert any('同一方向' in error for error in report['errors'])


def test_side_rejection_never_invokes_plate_registration(tmp_path, monkeypatch, frozen):
    _, arrays = frozen
    feature = arrays['SKU_03_left_edge'] * .8
    monkeypatch.setattr(recognition, 'edge_feature', lambda _: feature)
    def forbidden(*args):
        pytest.fail('Plate registration must not rescue a rejected side')
    monkeypatch.setattr(recognition, 'register_plate', forbidden)
    path = tmp_path / 'source.png'
    Image.fromarray(body()).save(path)
    result = recognition.recognize(path)
    assert result['face'] is None and result['suggested_face'] == 'left'
