import numpy as np
from phone_texture.frame_end_color import correct


def test_local_cast_on_both_orientations_preserves_metal_detail_and_exclusions():
    frame = np.full((60, 600, 4), [238, 158, 95, 255], dtype='uint8')
    frame[:, :55, :3] = [160, 129, 119]
    frame[::2, :55, :3] += 3
    mask = np.ones(frame.shape[:2], bool)
    mask[20:36, 12:24] = False
    frame[:3, :, 3] = 0
    result, evidence = correct(frame, mask)
    assert [r['status'] for r in evidence] == ['corrected', 'within_tolerance']
    assert result[30, 35, 0] > frame[30, 35, 0] + 15
    assert np.all(result[30, 35, :3] > result[31, 35, :3])
    assert np.array_equal(result[..., 3], frame[..., 3])
    assert np.array_equal(result[~mask], frame[~mask])
    assert np.array_equal(result[:3], frame[:3])
    assert np.array_equal(result[:, 100:], frame[:, 100:])
    vertical, _ = correct(frame.swapaxes(0, 1), mask.T)
    assert np.array_equal(vertical, result.swapaxes(0, 1))


def test_silver_and_unmeasurable_reference_preserve_source():
    frame = np.full((70, 610, 4), [233, 238, 240, 255], dtype='uint8')
    frame[:, :35, :3] -= 5
    for mask in (np.ones(frame.shape[:2],bool),np.zeros(frame.shape[:2],bool)):
        result, evidence = correct(frame, mask)
        assert np.array_equal(result, frame)
        assert all(item['status'] != 'corrected' for item in evidence)


def test_a15_saturates_only_coloured_terminal_paint():
    from phone_texture.frame_end_saturation import enhance
    from phone_texture.lighting import engine
    frame = np.full((60, 600, 4), [210, 140, 100, 255], dtype='uint8')
    frame[12:18, :50, 3] = 0
    mask = np.ones(frame.shape[:2], dtype=bool)
    mask[20:24] = False
    ends = [dict(end='start', status='corrected', reference=[70., 22., 35.]),
            dict(end='end', status='within_tolerance', reference=[70., 1., 2.])]
    result = enhance(frame, mask, ends, .12)
    lab = engine('aluminum').rgb_to_lab
    current, original = (lab(image[36, 15, :3]) for image in (result, frame))
    assert np.linalg.norm(current[1:]) > np.linalg.norm(original[1:]) + 2.
    assert abs(current[0] - original[0]) < .4
    assert np.array_equal(result[..., 3], frame[..., 3])
    assert np.array_equal(result[~mask], frame[~mask])
    assert np.array_equal(result[frame[..., 3] == 0], frame[frame[..., 3] == 0])
    assert np.array_equal(result[:, 100:], frame[:, 100:])
    assert np.array_equal(enhance(frame.swapaxes(0, 1), mask.T, ends, .12), result.swapaxes(0, 1))
