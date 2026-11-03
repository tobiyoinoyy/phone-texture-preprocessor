import hashlib
import unittest
import numpy as np
from phone_texture.bottom_paper_audit import audit_bottom_paper_end, guard_bottom_paper_end

class BottomAuditGuardTests(unittest.TestCase):
    def fixture(self,end=95):
        a=np.zeros((120,700,4),np.uint8);a[:]=[25,35,58,255]
        a[:end,290:330,:3]=220
        for y in range(12,min(76,end),10): a[y:y+2,298:322,:3]=40
        mask=np.zeros((120,700),bool);mask[:,290:330]=True
        proof={'components':[{'box':[290,0,330,end]}]}
        return a,mask,proof
    def test_known_metal_inside_oversized_mask_rejected(self):
        a,m,p=self.fixture();before=a.tobytes(),m.tobytes()
        r=audit_bottom_paper_end(a,m,p)
        self.assertEqual(r['status'],'rejected')
        self.assertGreater(r['protected_metal_pixels'],0)
        self.assertGreaterEqual(r['protected_metal_source_bbox'][1],98)
        self.assertFalse(r['certifies_complete_paper'])
        with self.assertRaisesRegex(ValueError,'source_bbox'): guard_bottom_paper_end(a,m,p)
        self.assertEqual(before,(a.tobytes(),m.tobytes()))
    def test_correct_mask_not_rejected(self):
        a,m,p=self.fixture();m[95:]=False
        r=audit_bottom_paper_end(a,m,p)
        self.assertNotEqual(r['status'],'rejected');self.assertFalse(r['certifies_complete_paper'])
    def test_canvas_continuing_paper_not_rejected(self):
        a,m,p=self.fixture(120)
        r=audit_bottom_paper_end(a,m,p)
        self.assertEqual(r['status'],'unresolved');self.assertFalse(r['certifies_complete_paper'])
    def test_common_metal_lighting_shoulder_not_paper_edge(self):
        a,m,p=self.fixture(120);a[95:,:,:3]=np.maximum(a[95:,:,:3].astype(int)-20,0)
        r=audit_bottom_paper_end(a,m,{'components':[{'box':[290,0,330,95]}]})
        self.assertNotEqual(r['status'],'rejected')
    def test_no_lateral_opaque_metal_is_unresolved(self):
        a,m,p=self.fixture();a[:,:290,3]=0;a[:,330:,3]=0
        r=audit_bottom_paper_end(a,m,p)
        self.assertEqual(r['status'],'unresolved')
    def test_distinct_material_after_edge_not_rejected(self):
        a,m,p=self.fixture();a[95:,290:330,:3]=[100,110,130]
        r=audit_bottom_paper_end(a,m,p)
        self.assertNotEqual(r['status'],'rejected')
    def test_no_paper_color_contrast_not_certified(self):
        a,m,p=self.fixture();a[:,:,:3]=[25,35,58]
        r=audit_bottom_paper_end(a,m,p)
        self.assertEqual(r['status'],'unresolved');self.assertFalse(r['certifies_complete_paper'])
    def test_printed_horizontal_strokes_are_not_a_physical_paper_end(self):
        a,m,p=self.fixture(120)
        # Text crosses much of the printed interior, while both source paper
        # margins continue through the stripe and to the face canvas.
        a[108:114,297:324,:3]=[25,35,58]
        before=a.copy(),m.copy()
        returned,proof=guard_bottom_paper_end(a,m,p)
        self.assertNotEqual(proof['bottom_source_audit']['status'],'rejected')
        self.assertFalse(proof['bottom_source_audit']['certifies_complete_paper'])
        np.testing.assert_array_equal(returned,before[1])
        np.testing.assert_array_equal(a,before[0])

    def test_unresolved_extrapolated_strip_is_not_released_after_inconclusive_audit(self):
        from phone_texture.auto_material import trim_strip_to_source_end
        a,m,p=self.fixture(120)
        p['source_strip_edges']=[[0,290],[0,329]]
        # Even genuinely continuing paper cannot be certified by side fits
        # alone. A safe rejection is preferable to silently freezing metal.
        before=a.copy(),m.copy()
        self.assertEqual(audit_bottom_paper_end(a,m,p)['status'],'unresolved')
        with self.assertRaisesRegex(ValueError,'Bottom paper end unresolved'):
            trim_strip_to_source_end(a,m,p)
        np.testing.assert_array_equal(a,before[0])
        np.testing.assert_array_equal(m,before[1])

    def test_closed_non_extrapolated_mask_keeps_existing_audit(self):
        from phone_texture.auto_material import trim_strip_to_source_end
        a,m,p=self.fixture();m[95:]=False
        returned,proof=trim_strip_to_source_end(a,m,p)
        np.testing.assert_array_equal(returned,m)
        self.assertFalse(proof['bottom_source_audit']['certifies_complete_paper'])

if __name__=='__main__':unittest.main()
