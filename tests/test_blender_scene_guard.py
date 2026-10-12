import copy
import unittest
from texture_alignment.blender_scene_guard import fingerprint, require_match
from texture_alignment.seam_targets import transfer_row


class SceneGuardTest(unittest.TestCase):
    def record(self, payload):
        return {'payload': payload, 'sha256': fingerprint(payload)}

    def test_same_model_name_does_not_hide_uv_change(self):
        original = {'name': 'phone', 'uv': [[0.1, 0.2]]}
        changed = copy.deepcopy(original)
        changed['uv'][0][1] = 0.201
        with self.assertRaisesRegex(ValueError, 'recalibrate'):
            require_match(self.record(original), self.record(changed))

    def test_changed_record_without_new_hash_is_rejected(self):
        expected = self.record({'uv': [0.1]})
        changed = copy.deepcopy(expected)
        changed['payload']['uv'][0] = 0.2
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            require_match(expected, changed)

    def test_unchanged_record_can_be_reused(self):
        record = self.record({'uv': [0.1], 'mapping': [0, 0, 0]})
        require_match(record, copy.deepcopy(record))

    def test_back_uv_scale_can_preserve_top_but_shift_bottom(self):
        # Same model and side UV, back UV scaled around its upper end.
        # A passing upper seam cannot validate the lower seam.
        side = [[-2, .91], [2, .10]]
        original_back = [[-2, .90], [2, .10]]
        edited_back = [[-2, .906], [2, .10]]
        upper = .10 * 2000 - .5
        lower = .89 * 2000 - .5
        self.assertAlmostEqual(transfer_row(upper, 2000, 1979, original_back, side),
                               transfer_row(upper, 2000, 1979, edited_back, side))
        before = transfer_row(lower, 2000, 1979, original_back, side)
        after = transfer_row(lower, 2000, 1979, edited_back, side)
        self.assertGreater(before - after, 10)
        self.assertLess(after, before)  # Bottom band must move upward in side image.
