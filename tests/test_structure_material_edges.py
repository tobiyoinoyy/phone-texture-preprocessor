"""Final paired pixel evidence must survive polarity reversals, not bad maps."""
import numpy as np
import pytest
from PIL import Image

from phone_texture import structure_review as s
from test_structure_review import fixture


def paired_fixture(tmp_path):
    root,files,spec,path=fixture(tmp_path)
    for side in ('left','right'):
        spec['A'][side]['source_boundary_detection']=dict(
            method='paired_source_material_edges_v3',revision_bands=[0,1],
            bands=[dict(band=0),dict(band=1)])
    return root,files,spec,path


def test_final_pairs_keep_position_and_width_across_material_polarity(tmp_path):
    root,files,spec,path=paired_fixture(tmp_path)
    for side in ('left','right'):
        a=s.pixels(files['A'][side])
        metal=np.full(a.shape[1],150.)
        metal[:18]=np.linspace(30,150,18);metal[-18:]=np.linspace(150,30,18)
        a[...,:3]=metal[None,:,None]
        a[40:52,:,:3]=70;a[200:212,:,:3]=70
        a[110:145,25:35,:3]=90
        Image.fromarray(a).save(files['A'][side])
        spec['A'][side]['source_sha256']=s.digest(files['A'][side])
    s.write(path,spec);process=tmp_path/'structure';s.prepare(root,path,process)
    checks=s.read(process/'report.json')['checks']['A']
    for side in ('left','right'):
        assert checks[side]['pixel_measurement_method']=='final_paired_source_edges'
        for check in checks[side]['checks'].values():
            assert max(check['error_px'])<=2.5
            assert max(check['width_error_px'])<=2.5


@pytest.mark.parametrize('failure',['both_edges_shifted','width_changed'])
def test_consistent_warp_cannot_substitute_expected_rows_for_final_pixels(tmp_path,failure):
    root,files,spec,path=paired_fixture(tmp_path)
    lines=spec['A']['left']['source_lines']
    if failure=='both_edges_shifted':lines[1]+=4;lines[2]+=4
    else:lines[2]+=5
    s.write(path,spec)
    with pytest.raises(ValueError,match='position/width mismatch|paired antenna|Antenna'):
        s.prepare(root,path,tmp_path/'structure')


def test_unrelated_highlight_lines_cannot_replace_two_material_edges(tmp_path):
    root,files,spec,path=paired_fixture(tmp_path)
    a=s.pixels(files['A']['left']);a[...,:3]=150
    # Bright narrow lines at the claimed boundaries have no sustained
    # material band between them, even though individual gradients exist.
    for y in (40,52,200,212):a[y:y+1,:,:3]=230
    Image.fromarray(a).save(files['A']['left'])
    spec['A']['left']['source_sha256']=s.digest(files['A']['left']);s.write(path,spec)
    with pytest.raises(ValueError,match='antenna|Antenna'):
        s.prepare(root,path,tmp_path/'structure')


@pytest.mark.parametrize('mode',['manual','lower_only'])
def test_manual_and_lower_only_sources_retain_signed_window_detector(tmp_path,monkeypatch,mode):
    root,files,spec,path=fixture(tmp_path)
    if mode=='lower_only':
        for cfg in spec['A'].values():
            cfg['source_boundary_detection']=dict(method='paired_source_material_edges_v3',
                revision_bands=[1],bands=[dict(band=1)],upper_algorithm_retained=True)
        s.write(path,spec)
    def unexpected(*args,**kwargs):
        raise AssertionError('Legacy manual/lower-only specs must not use paired detector')
    monkeypatch.setattr(s,'_paired_final_measurements',unexpected)
    s.prepare(root,path,tmp_path/'structure')


def test_each_paired_witness_requires_actual_supported_columns(tmp_path):
    root,files,spec,path=paired_fixture(tmp_path)
    # Keep overall support >60%, but remove actual evidence at one end.
    a=s.pixels(files['A']['left']);a[:,:12,:3]=150
    Image.fromarray(a).save(files['A']['left'])
    spec['A']['left']['source_sha256']=s.digest(files['A']['left']);s.write(path,spec)
    with pytest.raises(ValueError,match='endpoint.*evidence|paired antenna'):
        s.prepare(root,path,tmp_path/'structure')
