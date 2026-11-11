import json
import cv2
import numpy as np
import pytest
from PIL import Image
from phone_texture.automatic import no_zone_effects


def image(payload):
 code=cv2.QRCodeEncoder_create().encode(payload)
 code=cv2.resize(code,(240,240),interpolation=cv2.INTER_NEAREST)
 a=np.full((300,300,4),255,np.uint8);a[30:270,30:270,:3]=code[...,None]
 return a


def context(tmp_path,before,after):
 source=tmp_path/'source.png';output=tmp_path/'output.png';mask=tmp_path/'label-fit.png'
 Image.fromarray(before).save(source);Image.fromarray(after).save(output);Image.fromarray(np.ones(before.shape[:2],np.uint8)*255).save(mask)
 return source,output,dict(source_label_fit_exclusion={'path':str(mask)})


def test_lighting_changes_label_rgb_without_changing_its_readable_payload(tmp_path):
 a=image('SKU-READABILITY-TEST');b=a.copy();b[...,:3]=np.rint(a[...,:3]*.55+70).astype('uint8')
 report=no_zone_effects(*context(tmp_path,a,b))
 assert report['label_pixels_changed']>0 and report['alpha_pixels_changed']==0
 assert report['source_QR_decoded'] and report['output_QR_decoded'] and report['QR_payload_unchanged']


def test_unconfirmed_readability_is_reported_not_faked_as_passed(tmp_path):
 a=image('SKU-READABILITY-TEST');b=np.full_like(a,255)
 report=no_zone_effects(*context(tmp_path,a,b))
 assert report['source_QR_decoded'] and not report['output_QR_decoded']
 assert not report['QR_payload_unchanged'] and report['review_warnings']


def test_an_actually_different_payload_is_not_accepted(tmp_path):
 a=image('SKU-READABILITY-TEST');b=image('DIFFERENT-SKU-CONTENT')
 with pytest.raises(ValueError,match='different source QR'):no_zone_effects(*context(tmp_path,a,b))
