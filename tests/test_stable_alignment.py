"""Behavioral regressions for phase-one registration and review routing."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phone_normalizer.body_geometry import measure_body
from phone_normalizer.uv_profile import align_to_profile
from phone_normalizer.uv import process_six_views
from phone_normalizer.uv_cli import main
from scripts.precise_geometry import normalize_candidate
from test_uv_profile import make_profile
from test_uv_pipeline import FakeBackend, make_sources, settings


def body(bump=None):
    a = np.zeros((240, 100, 4), np.uint8)
    a[20:220, 30:50] = [70, 110, 160, 255]
    if bump:
        x0,y0,x1,y1 = bump
        a[y0:y1,x0:x1] = [70,110,160,255]
    return Image.fromarray(a)


class StableAlignmentTests(unittest.TestCase):
    def test_connected_narrow_tip_does_not_move_internal_points(self):
        profile = make_profile(size=(100,240), bbox=(30,20,50,220))
        source = body((38,10,42,20))
        self.assertEqual(measure_body(source)['bbox'], [30.,20.,50.,220.])
        output, info = align_to_profile(source,'left',profile)
        self.assertEqual(info['geometry_gate'],'passed')
        np.testing.assert_array_equal(np.asarray(output), np.asarray(source))

    def test_long_protrusion_does_not_compress_body_and_is_flagged(self):
        source = body((50,30,60,160))
        evidence = measure_body(source)
        self.assertEqual(evidence['bbox'], [30.,20.,50.,220.])
        self.assertIn('ambiguous_body_edges', evidence['review_reasons'])
        profile = make_profile(size=(100,240), bbox=(30,20,50,220))
        output, info = align_to_profile(source,'left',profile)
        self.assertEqual(info['geometry_gate'],'review_required')
        np.testing.assert_array_equal(np.asarray(output), np.asarray(source))

    def test_excess_width_scaling_outputs_only_similarity_preview(self):
        source = body()
        profile = make_profile(size=(100,240), bbox=(20,20,60,220))
        _, info = align_to_profile(source,'left',profile)
        self.assertIn('anisotropic_scale_exceeds_limit', info['geometry_gate_reasons'])
        self.assertEqual(info['requested_scale_xy'],[2.,1.])
        matrix = np.asarray(info['candidate_to_final_matrix'])
        self.assertEqual(matrix[0,0], matrix[1,1])
        self.assertEqual(info['output_role'],'review_preview')

    def test_small_verified_anisotropy_is_allowed(self):
        profile = make_profile(size=(100,240), bbox=(30,20,50.8,220))
        _, info = align_to_profile(body(),'left',profile)
        self.assertEqual(info['geometry_gate'],'passed')
        self.assertAlmostEqual(info['anisotropy'],.04)
        self.assertAlmostEqual(info['candidate_to_final_matrix'][0][0],1.04)

    def test_alpha_instability_is_reported(self):
        a = np.asarray(body()).copy()
        a[10:20,30:50] = [70,110,160,100]
        evidence = measure_body(Image.fromarray(a))
        self.assertIn('alpha_boundary_unstable',evidence['review_reasons'])
        self.assertEqual(evidence['threshold_edge_spread_px'][1],10.)

    def test_horizontal_and_rotated_body_have_consistent_geometry(self):
        image=body((38,10,42,20))
        evidence=measure_body(image.transpose(Image.Transpose.TRANSPOSE))
        self.assertEqual(evidence['bbox'],[20.,30.,220.,50.])

    def test_registration_canvas_keeps_soft_pixels_before_final_crop(self):
        a=np.asarray(body()).copy()
        a[19,30:50]=[70,110,160,20]
        full, info=normalize_candidate(a,remove_support=False,deskew=False,preserve_canvas=True)
        np.testing.assert_array_equal(full,a)
        self.assertEqual(info['nonzero_pixels_outside_crop'],0)
        self.assertTrue(info['canvas_preserved_for_registration'])

    def test_batch_routes_rejected_images_and_emits_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=make_sources(root)
            profile=make_profile(size=(100,240),bbox=(30,20,50,220))
            path=root/'profile.json'; path.write_text(json.dumps(profile))
            backend=FakeBackend()
            report=process_six_views(source,root/'out',uv_profile=path,config=settings(),backend_instance=backend)
            self.assertTrue(report['success'])
            self.assertEqual(report['geometry_gate'],'review_required')
            self.assertEqual(len(report['review_required_faces']),6)
            self.assertEqual(len(list((root/'out').glob('*.png'))),0)
            self.assertEqual(len(list((root/'out/review').glob('*.png'))),6)
            self.assertEqual(len(list((root/'out/diagnostics').glob('*.png'))),6)
            with self.assertRaises(FileExistsError):
                process_six_views(source,root/'out',uv_profile=path,config=settings(),backend_instance=backend)
            with patch('phone_normalizer.uv_cli.process_six_views',return_value=report):
                self.assertEqual(main([str(source),'--output-dir',str(root/'unused'),'--uv-profile',str(path)]),2)

    def test_invalid_scale_limit_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=make_sources(root); backend=FakeBackend()
            for limit in (float('nan'),-1,2,True):
                with self.assertRaises(ValueError):
                    process_six_views(source,root/'out',max_anisotropy=limit,backend_instance=backend)
            self.assertEqual(backend.calls,0)
            self.assertFalse((root/'out').exists())

    def test_diagnostic_collision_is_detected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=make_sources(root); backend=FakeBackend()
            path=root/'out/diagnostics/front.png'; path.parent.mkdir(parents=True); path.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                process_six_views(source,root/'out',config=settings(),backend_instance=backend)
            self.assertEqual(path.read_bytes(),b'keep')
            self.assertEqual(backend.calls,0)


if __name__=='__main__':
    unittest.main()
