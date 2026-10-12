"""Frame-only alpha changes must preserve visible source RGB and round corners."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
import numpy as np
from PIL import Image,ImageDraw
from phone_normalizer.frame_strip import extract_frame_strip, normalize_frame_strip, deskew_cropped_frame
from phone_normalizer.frame_cli import process_frames


def sample():
    mask=Image.new('L',(400,300));d=ImageDraw.Draw(mask)
    d.rounded_rectangle((25,100,374,153),radius=10,fill=255)
    d.rectangle((100,70,280,100),fill=255)
    d.rectangle((115,153,280,299),fill=255)
    yy,xx=np.mgrid[:300,:400]
    a=np.stack((xx%256,yy%256,(xx+yy)%256,np.asarray(mask)),axis=-1).astype(np.uint8)
    return Image.fromarray(a)


class Backend:
    name='birefnet';model_name='ZhengPeng7/BiRefNet_dynamic';device='cpu'
    def segment(self,image):return sample()


class FrameStripTests(unittest.TestCase):
    def test_camera_and_stand_removed_without_resampling_content(self):
        original=sample();result,info=extract_frame_strip(original,margin_px=0)
        self.assertEqual(info['crop_xyxy'],[25,100,375,154])
        self.assertEqual(result.size,(350,54))
        a=np.asarray(result);visible=a[:,:,3]>0
        np.testing.assert_array_equal(a[:,:,:3][visible],np.asarray(original)[100:154,25:375,:3][visible])
        self.assertEqual(int(a[0,0,3]),0)
        self.assertEqual(int(a[20,10,3]),255)

    def test_input_is_unchanged(self):
        im=sample();before=np.asarray(im).copy();extract_frame_strip(im)
        np.testing.assert_array_equal(np.asarray(im),before)

    def test_non_frame_and_unsupported_face_are_rejected(self):
        with self.assertRaises(ValueError):extract_frame_strip(sample(),face='back')
        with self.assertRaises(ValueError):extract_frame_strip(Image.new('RGBA',(400,300),(1,2,3,255)))
        with self.assertRaises(ValueError):extract_frame_strip(sample(),margin_px=-1)

    def test_side_orientation_is_exact_and_not_mirrored(self):
        image=sample();a=np.asarray(image).copy()
        a[:100,:,3]=0
        a[75:101,40:110,3]=255  # Camera over one end, clear central rail.
        image=Image.fromarray(a)
        horizontal,_=extract_frame_strip(image,face='right')
        right,ri=normalize_frame_strip(image,face='right',deskew=False)
        left,li=normalize_frame_strip(image,face='left',deskew=False)
        np.testing.assert_array_equal(np.asarray(right),np.rot90(np.asarray(horizontal),-1))
        np.testing.assert_array_equal(np.asarray(left),np.rot90(np.asarray(horizontal),1))
        self.assertEqual(ri['orientation_quarter_turns_ccw'],-1)
        self.assertEqual(li['orientation_quarter_turns_ccw'],1)
        self.assertFalse(ri['quarter_turn_resampled'])

    def test_side_deskew_precedes_orientation(self):
        from scripts.geometry_postprocess import estimate_long_angle
        a=np.asarray(sample()).copy();a[:100,:,3]=0;a[75:101,40:110,3]=255
        source=Image.fromarray(a).rotate(2.,resample=Image.Resampling.BICUBIC,expand=True)
        result,info=normalize_frame_strip(source,face='right')
        horizontal=result.transpose(Image.Transpose.ROTATE_90)
        angle=estimate_long_angle(np.asarray(horizontal)[...,3],128)
        self.assertLess(abs(angle),.2)
        self.assertAlmostEqual(info['deskew_rotation_ccw_degrees'],-2.,delta=.2)
        self.assertEqual(info['resampling_passes'],1)
        self.assertGreater(result.height,result.width)

    def test_formal_runner_preserves_sources_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'SKU_01';folder.mkdir();p=folder/'bottom.png'
            sample().convert('RGB').save(p);before=p.read_bytes()
            report=process_frames([p],root/'out',backend_instance=Backend())
            self.assertTrue(report['success'])
            self.assertTrue((root/'out/SKU_01/bottom.png').exists())
            self.assertEqual(before,p.read_bytes())
            with self.assertRaises(FileExistsError):process_frames([p],root/'out',backend_instance=Backend())

    def test_top_bottom_are_deskewed_by_default(self):
        from scripts.geometry_postprocess import estimate_long_angle
        source=sample().rotate(1.5,resample=Image.Resampling.BICUBIC,expand=True)
        for face in ('top','bottom'):
            result,info=normalize_frame_strip(source,face=face)
            self.assertLess(abs(estimate_long_angle(np.asarray(result)[...,3],128)),.2)
            self.assertEqual(info['resampling_passes'],1)
            self.assertEqual(info['orientation_quarter_turns_ccw'],0)

    def test_cropped_frame_preserves_markings_and_uses_one_rigid_rotation(self):
        from scripts.geometry_postprocess import rotate_alpha_safe
        cropped,_=extract_frame_strip(sample())
        source=Image.fromarray(rotate_alpha_safe(np.asarray(cropped),1.2))
        before=np.asarray(source).copy()
        result,info=deskew_cropped_frame(source,face='bottom')
        self.assertEqual(info['resampling_passes'],1)
        self.assertLess(abs(np.mean(info['after_rail_angles_degrees'])),.06)
        np.testing.assert_array_equal(before,np.asarray(source))
        # Entire output equals a rigid transform of the original including labels.
        rotated=rotate_alpha_safe(before,info['rotation_ccw_degrees'])
        x0,y0,x1,y1=info['crop_xyxy']
        np.testing.assert_array_equal(np.asarray(result),rotated[y0:y1,x0:x1])


if __name__=='__main__':unittest.main()
