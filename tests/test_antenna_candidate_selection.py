"""A central button edge must not prevent testing a real full-width strip."""
import numpy as np
import pytest
from PIL import Image
from phone_texture.auto_structure import antenna_rows


def source():
    a = np.full((700, 100, 4), 255, np.uint8)
    a[..., :3] = 180
    a[58:70, :, :3] = 140
    a[630:642, :, :3] = 140
    # A stronger, narrower central feature lies in the lower search window.
    a[585:590, 30:70, :3] = 15
    return a


def test_full_span_pair_disambiguates_stronger_button_cap():
    a = source()
    before = a.copy()
    raw, _, _ = antenna_rows(Image.fromarray(a), [40, 60])
    assert raw[2] < 600
    rows, _, proof = antenna_rows(Image.fromarray(a), [40, 60], require_full_span=True)
    assert abs(rows[2]-630) <= 1 and abs(rows[3]-642) <= 1
    assert proof['full_span_candidate_selection'][1]['rejected_candidates']
    np.testing.assert_array_equal(a, before)


def test_no_supported_alternative_still_rejects_missing_endpoint():
    a = source()
    a[:, :8, :3] = 80
    with pytest.raises(ValueError, match='endpoint.*evidence'):
        antenna_rows(Image.fromarray(a), [40, 60], require_full_span=True)


def test_rear_narrow_sampling_keeps_its_independent_consensus_path():
    a = source()
    # Rear callers have a separate neighbouring-column consensus. They do not
    # claim full-frame side evidence from one narrow sample window.
    raw, _, proof = antenna_rows(Image.fromarray(a), [40, 60])
    assert raw[2] < 600
    assert 'full_span_candidate_selection' not in proof
