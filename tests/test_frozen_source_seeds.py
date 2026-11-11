import numpy as np,pytest
from PIL import Image
from phone_texture.auto_structure import frozen_source_rows,antenna_rows,per_column_boundaries

def ambiguous_source():
 a=np.full((500,100,4),255,np.uint8);a[...,:3]=180
 a[42:51,:,:3]=145;a[65:74,:,:3]=40;a[450:459,:,:3]=100
 return a

def test_frozen_genuine_band_survives_a_stronger_complete_neighbor():
 image=Image.fromarray(ambiguous_source());rows,_,_=antenna_rows(image,[40,60],require_full_span=True)
 assert 63<=rows[0]<=66
 seeds=frozen_source_rows(image,[40,60],dict(source_seed_rows=[42,51,450,459]))
 curves,_=per_column_boundaries(image,seeds,return_proof=True)
 assert abs(np.median(curves[0])-42)<2
 assert abs(np.median(curves[1])-51)<2

def test_removed_frozen_band_cannot_borrow_its_neighbor_evidence():
 a=ambiguous_source();a[42:51,:,:3]=180
 with pytest.raises(ValueError,match='lack independent'):
  frozen_source_rows(Image.fromarray(a),[40,60],dict(source_seed_rows=[42,51,450,459]))

def test_invalid_frozen_coordinates_rejected():
 with pytest.raises(ValueError,match='Invalid frozen'):
  frozen_source_rows(Image.fromarray(ambiguous_source()),[40,60],dict(source_seed_rows=[42,51,float('nan'),459]))
