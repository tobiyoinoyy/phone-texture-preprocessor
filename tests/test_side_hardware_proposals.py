import unittest
import numpy as np
from PIL import Image
from phone_texture.side_material import _independent_hardware_proposals as propose_outer, _distinct_hardware_proposals as deduplicate, hardware_materials, _hardware_material_step as independent_material_step

def shape(h=1800, w=125, x=35, y=1100, bw=55, bh=250):
    (yy, xx) = np.indices((h, w))
    r = (bw - 1) / 2
    return (xx - x - r) ** 2 + (yy - np.clip(yy, y + r, y + bh - 1 - r)) ** 2 <= r * r

def canvas():
    a = np.zeros((1800, 125, 4), np.uint8)
    a[:] = [145, 165, 195, 255]
    return a

class OuterCapsuleTests(unittest.TestCase):

    def test_true_outer_rail_with_internal_groove(self):
        a = canvas()
        outer = shape()
        a[outer, :3] = [42, 52, 67]
        inner = shape(x=43, y=1110, bw=39, bh=230)
        inner2 = shape(x=45, y=1112, bw=35, bh=226)
        a[inner & ~inner2, :3] = [70, 80, 95]
        (proposals, entries) = propose_outer(Image.fromarray(a), 'right')
        p = deduplicate(proposals, 1.875)
        self.assertEqual(len(p), 1)
        self.assertGreater((p[0][0] & outer).sum() / (p[0][0] | outer).sum(), 0.95)
        self.assertTrue(p[0][1]['independent_material']['accepted'])

    def test_external_noise_branch_does_not_expand_material(self):
        a = canvas()
        outer = shape()
        a[outer, :3] = [42, 52, 67]
        a[1097:1102, 57:62, :3] = [40, 50, 65]
        a[1094:1098, 53:59, :3] = [40, 50, 65]
        p = deduplicate(propose_outer(Image.fromarray(a), 'right')[0], 1.875)
        self.assertLessEqual(len(p), 1)
        if p:
            self.assertGreater((p[0][0] & outer).sum() / (p[0][0] | outer).sum(), 0.95)

    def test_off_center_cap_noise_retains_true_outer_boundary(self):
        for (x, y, x1, y1) in [(44, 1096, 49, 1109), (46, 1097, 50, 1107), (48, 1097, 51, 1105)]:
            with self.subTest(branch=[x, y, x1, y1]):
                a = canvas()
                outer = shape()
                a[outer, :3] = [42, 52, 67]
                a[y:y1, x:x1, :3] = [40, 50, 65]
                p = deduplicate(propose_outer(Image.fromarray(a), 'right')[0], 1.875)
                self.assertEqual(len(p), 1)
                self.assertGreater((p[0][0] & outer).sum() / (p[0][0] | outer).sum(), 0.95)

    def test_rectangle_noise_is_rejected(self):
        a = canvas()
        a[1100:1350, 35:90, :3] = [42, 52, 67]
        self.assertEqual(len(propose_outer(Image.fromarray(a), 'right')[0]), 0)

    def test_missing_cap_is_rejected(self):
        a = canvas()
        a[shape(), :3] = [42, 52, 67]
        a[1090:1125, 45:80, :3] = [145, 165, 195]
        self.assertEqual(len(propose_outer(Image.fromarray(a), 'right')[0]), 0)

    def test_subopaque_domain_is_rejected(self):
        a = canvas()
        a[shape(), :3] = [42, 52, 67]
        a[1200, 60, 3] = 239
        self.assertEqual(len(propose_outer(Image.fromarray(a), 'right')[0]), 0)

    def test_duplicate_localization_is_not_competing(self):
        m = shape()
        p = deduplicate([(m, {}), (m.copy(), {})], 1.875)
        self.assertEqual(len(p), 1)

    def test_two_distinct_valid_nested_materials_are_ambiguous(self):
        a = canvas()
        outer = shape()
        inner = shape(x=43, y=1110, bw=39, bh=230)
        a[outer, :3] = [85, 100, 120]
        a[inner, :3] = [30, 40, 50]
        (proposals, entries) = propose_outer(Image.fromarray(a), 'right')
        self.assertGreaterEqual(len(proposals), 2)
        with self.assertRaisesRegex(ValueError, 'competing'):
            deduplicate(proposals, 1.875)

    def test_expected_hardware_count_not_relaxed(self):
        a = canvas()
        a[shape(), :3] = [42, 52, 67]
        with self.assertRaisesRegex(ValueError, 'expected 2'):
            hardware_materials(Image.fromarray(a), 'right')

    def test_line_groove_inside_same_material_is_not_material_edge(self):
        a = canvas()
        a[shape(), :3] = [42, 52, 67]
        inner = shape(x=43, y=1110, bw=39, bh=230)
        inner2 = shape(x=45, y=1112, bw=35, bh=226)
        a[inner & ~inner2, :3] = [70, 80, 95]
        m = independent_material_step(a, [43, 1110, 82, 1340], 1.875)
        self.assertFalse(m['accepted'])

    def test_wide_dark_groove_with_same_material_core_is_rejected(self):
        for thickness in (8, 10):
            with self.subTest(thickness=thickness):
                a = canvas()
                outer = shape()
                inner = shape(x=35 + thickness, y=1100 + thickness, bw=55 - 2 * thickness, bh=250 - 2 * thickness)
                a[outer & ~inner, :3] = [42, 52, 67]
                self.assertEqual(propose_outer(Image.fromarray(a), 'right')[0], [])

    def test_three_candidate_bridge_is_rejected_for_every_permutation(self):
        import itertools
        proposals = [(shape(x=x), {}) for x in (35, 34, 36)]
        for order in itertools.permutations(proposals):
            with self.assertRaisesRegex(ValueError, 'competing'):
                deduplicate(order, 1.875)

    def test_valid_existing_results_do_not_invoke_fallback(self):
        from unittest.mock import patch
        a = canvas()
        a[shape(), :3] = [42, 52, 67]
        a[shape(x=42, y=450, bw=41, bh=220), :3] = [42, 52, 67]
        image = Image.fromarray(a)
        with patch('phone_texture.side_material._independent_hardware_proposals', side_effect=AssertionError('unexpected fallback')):
            result = hardware_materials(image, 'right')
        self.assertEqual(len(result), 2)
        self.assertTrue(all(('independent_proposal_review' not in proof for (_, proof) in result)))

    def test_proposal_detection_keeps_source_pixels_unchanged(self):
        a = canvas()
        a[shape(), :3] = [42, 52, 67]
        image = Image.fromarray(a)
        before = image.tobytes()
        propose_outer(image, 'right')
        self.assertEqual(image.tobytes(), before)

    def test_filling_missing_component_preserves_existing_mask_and_proof(self):
        import copy
        from phone_texture.side_material import _resolve_missing_hardware

        a = canvas()
        a[shape(), :3] = [42, 52, 67]
        a[shape(x=42, y=450, bw=41, bh=220), :3] = [42, 52, 67]
        image = Image.fromarray(a)
        original = hardware_materials(image, 'right')[0]
        mask_before = original[0].copy()
        proof_before = copy.deepcopy(original[1])
        completed, review = _resolve_missing_hardware(image, 'right', [original])

        self.assertEqual(len(completed), 2)
        self.assertEqual(review['added'], 1)
        self.assertIs(completed[0][0], original[0])
        self.assertIs(completed[0][1], original[1])
        np.testing.assert_array_equal(completed[0][0], mask_before)
        self.assertEqual(completed[0][1], proof_before)
        self.assertNotIn('proposal_method', completed[0][1])
if __name__ == '__main__':
    unittest.main()
