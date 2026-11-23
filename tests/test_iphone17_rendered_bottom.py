import numpy as np
import pytest
from PIL import Image

from phone_texture import bottom_rectify as br, iphone17_port as port


def test_rendered_port_requires_convergence_between_seed_windows(monkeypatch):
    candidates = [dict(threshold=t, port=[x, 60, x - 55, 40, w, h])
                  for t, x, w, h in [(50, 515, 121, 36), (60, 515, 121, 37), (70, 516, 123, 40),
                                     (80, 516, 139, 56), (90, 516, 140, 56), (100, 516, 141, 56)]]
    gray = np.zeros((120, 1030), dtype='uint8')
    centers = iter((515.5, 516.0))
    def verify(image, samples, scale):
        x = next(centers)
        return np.array([x, 60, 446, 32, 140, 56]), 0, [], {'selected': {'score': .8}}
    monkeypatch.setattr(br, 'port_rounded_rect', verify)
    assert len(br._rendered_port_consensus(gray, candidates, 10, 1)[3]['agreeing_rendered_windows']) == 2
    centers = iter((515.5, 521.0))
    with pytest.raises(br.Rejected, match='competing_rendered_usb_c_contours'):
        br._rendered_port_consensus(gray, candidates, 10, 1)


def test_rendered_validation_can_use_local_contrast_but_not_move_the_transform(monkeypatch):
    image = Image.new('RGBA', (1030, 140))
    nominal = port.DRAWING_HOLES
    scale = (image.width - 1) / port.WIDTH_MM
    holes = np.c_[nominal * scale, np.full(8, 60)]
    measured = dict(nominal=nominal, holes=holes, mirrored=False, port=np.array([514.5, 60]),
                    screws=np.array([[42.65 * scale, 61]]), selected_thresholds=[70, 75, 80],
                    port_contour_evidence={'selected': {'support': dict(left_end=.7, right_end=.7, top=.7, bottom=.7)}})
    def detect(a, model, contrast, rendered=False):
        if contrast != 'local':
            raise br.Rejected('independent_hole_consensus_failed')
        assert rendered
        return measured
    monkeypatch.setattr(br, 'measure', detect)
    proof = dict(measurement_method='independent_whole_port', hole_contrast='raw', hole_nominal_mm=nominal.tolist())
    assert port.final_port(image, proof)[0] == 514.5
    assert proof['rendered_hole_contrast'] == 'local'
    assert [s['status'] for s in proof['rendered_screw_checks']] == ['not_observed', 'confirmed']
    measured['holes'][0, 0] += 2
    with pytest.raises(ValueError, match='rendered_holes_outside_tolerance'):
        port.final_port(image, proof)
