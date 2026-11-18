"""Per-band rear evidence must expose actual minimum support to consumers."""
import numpy as np
import pytest
from PIL import Image
from phone_texture.auto_structure import _rear_reference_per_band


def candidates():
    # No top-ranked column sees both bands, but all three independent windows
    # contain the same supported horizontal alternatives.
    rows = [[40, 50, 400, 410], [40, 50, 420, 430], [60, 70, 420, 430]]
    return [(r, [1, -1, 1, -1], [10+i*8, 17+i*8],
             dict(scores=[10, 10], supported_central_candidates=[
                 [[40, 50], [60, 70]], [[400, 410], [420, 430]]]))
            for i, r in enumerate(rows)]


def test_per_band_result_reports_minimum_actual_support():
    image = Image.fromarray(np.full((500, 100, 4), 255, np.uint8))
    rows, _, _, proof = _rear_reference_per_band(image, candidates(), 3)
    assert len(rows) == 4
    assert proof['agreeing_columns'] == 3
    assert proof['agreeing_columns_by_band'] == [3, 3]
    assert len(proof['per_band_selection']) == 2


def test_two_windows_are_not_promoted_to_three():
    image = Image.fromarray(np.full((500, 100, 4), 255, np.uint8))
    with pytest.raises(ValueError, match='neighbouring-column consensus'):
        _rear_reference_per_band(image, candidates()[:2], 3)


def test_nearby_seed_support_does_not_certify_a_shifted_returned_pair():
    image = Image.fromarray(np.full((500, 100, 4), 255, np.uint8))
    rows = [[37, 47, 400, 410], [40, 50, 400, 410], [43, 53, 400, 410]]
    observations = [(r, [1, -1, 1, -1], [10+i*8, 17+i*8],
                     dict(scores=[10, 10], supported_central_candidates=[
                         [r[:2]], [r[2:]]])) for i, r in enumerate(rows)]
    # The seed at 40 has three neighbours, but the first column's returned
    # pair at 37 has only two. It must not inherit the seed's count.
    with pytest.raises(ValueError, match='neighbouring-column consensus'):
        _rear_reference_per_band(image, observations, 3)
