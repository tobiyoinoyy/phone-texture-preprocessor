import numpy as np
from PIL import Image
import pytest
import importlib.util
import json
from pathlib import Path

from phone_texture.frame_seams import MODELS, profile, resized_chains


@pytest.fixture
def seam_engine(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'skills/phone-aluminum-frame-restore/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('frame_seam_field_test',scripts/'match_uv_seams.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_required_no_silent_shape_guess():
    for model in (None, '', 'iphone17', '../iphone17pro'):
        with pytest.raises(ValueError, match='请选择模型'):
            profile(model)


@pytest.mark.parametrize('model', MODELS)
def test_actual_inset_uv_survives_nonuniform_texture_resizing(tmp_path, model):
    data, _ = profile(model)
    files = {}
    for face, size in [('top',(1400,151)),('bottom',(1300,143)),
                       ('left',(138,2341)),('right',(139,2351))]:
        files[face] = tmp_path/(face+'.png')
        Image.new('RGBA',size,'orange').save(files[face])
    converted = resized_chains(data, files)
    for old, new in zip(data['chains'], converted):
        for a, b in zip(old['sides'], new['sides']):
            old_uv = (np.asarray(a['points'])+.5)/a['image_size']
            new_uv = (np.asarray(b['points'])+.5)/b['image_size']
            np.testing.assert_allclose(old_uv,new_uv,atol=1e-12)
            assert np.min(new_uv) > 0 and np.max(new_uv) < 1
            np.testing.assert_allclose(np.linalg.norm(b['normals'],axis=1),1)
            if b['name'] in ('top','bottom'):
                # All four seams are inside the image, not at PNG edges.
                assert min(np.min(new_uv[:,0]), 1-np.max(new_uv[:,0])) > .02
    assert len(converted)==4


@pytest.mark.parametrize('model', MODELS)
def test_curved_model_seams_do_not_create_medial_axis_color_steps(seam_engine, model):
    # A smooth boundary profile must remain smooth on the inside of a bend.
    # Nearest-segment extension jumps by 2.9--3.7 L on these actual UV curves,
    # despite there being no image texture or material boundary in this test.
    for chain in profile(model)[0]['chains']:
        arc=np.asarray(chain['arc']);ss=np.linspace(0,arc[-1],192)
        offset=np.zeros((len(ss),3))
        offset[:,0]=4+10*np.sin(np.pi*ss/arc[-1])
        slope=np.zeros_like(offset)
        for original in chain['sides']:
            side=dict(original,points=np.asarray(original['points']),
                      normals=np.asarray(original['normals']))
            w,h=side['image_size']
            field,_=seam_engine.apply_band((h,w),side,arc,ss,offset,slope,.2,18)
            metric=seam_engine.correction_continuity(field,np.ones((h,w),bool))
            assert metric['max_step_deltaE76'] < 1.25


def test_reversing_uv_chain_preserves_the_same_field(seam_engine):
    chain=profile('iphone17pro')[0]['chains'][0]
    arc=np.asarray(chain['arc']);ss=np.linspace(0,arc[-1],192)
    original=chain['sides'][0]
    side=dict(original,points=np.asarray(original['points']),normals=np.asarray(original['normals']))
    reverse=dict(side,points=side['points'][::-1],normals=side['normals'][::-1])
    offset=np.zeros((192,3));offset[:,0]=2+6*ss/arc[-1]
    slope=np.zeros_like(offset);slope[:,0]=np.sin(np.pi*ss/arc[-1])*8
    w,h=side['image_size']
    a,ma=seam_engine.apply_band((h,w),side,arc,ss,offset,slope,.2,18)
    b,mb=seam_engine.apply_band((h,w),reverse,arc[-1]-arc[::-1],ss,
                              offset[::-1],slope[::-1],.2,18)
    np.testing.assert_allclose(a,b,atol=1e-12)
    np.testing.assert_array_equal(ma,mb)


def test_curved_band_is_not_cut_at_a_normal_width_rectangle(seam_engine):
    chain=profile('iphone17pro')[0]['chains'][0]
    original=chain['sides'][0]
    side=dict(original,points=np.asarray(original['points']),normals=np.asarray(original['normals']))
    arc=np.asarray(chain['arc']);ss=np.linspace(0,arc[-1],192)
    offset=np.zeros((192,3));offset[:,0]=12.
    w,h=side['image_size']
    field,mask=seam_engine.apply_band((h,w),side,arc,ss,offset,np.zeros_like(offset),.2,18)
    x=int(np.ceil(side['points'][:,0].max())+np.ceil(.2*side['px_per_unit'])+3)
    y=int(round(side['points'][-1,1]))
    # Oblique end normals require a wider extent along the image x axis.
    # Cropping at the scalar normal width would introduce a rectangular step.
    assert mask[y,x]
    assert field[y,x,0] > .1
    assert abs(field[y,x,0]-field[y,x-1,0]) < .5


def test_turning_chain_is_rejected_instead_of_reusing_ambiguous_projection(seam_engine):
    side=dict(points=np.array([[4.,4.],[28.,4.],[20.,16.]]),
              normals=np.array([[0.,1.]]*3),px_per_unit=80)
    arc=np.array([0.,.5,1.]);ss=np.linspace(0,1,192)
    with pytest.raises(ValueError,match='monotone cross-section'):
        seam_engine.apply_band((40,40),side,arc,ss,np.ones((192,3)),
                               np.zeros((192,3)),.2,6)


def test_correction_gate_rejects_a_block_before_writing_images(seam_engine, monkeypatch, tmp_path):
    views={}
    originals={}
    for name,level in [('a',95),('b',145)]:
        source=tmp_path/(name+'.png')
        Image.new('RGBA',(64,64),(level,level,level,255)).save(source)
        originals[source]=source.read_bytes()
        editable=tmp_path/(name+'-metal.png');Image.new('L',(64,64),255).save(editable)
        protected=tmp_path/(name+'-paper.png');Image.new('L',(64,64),0).save(protected)
        views[name]=dict(source=str(source),editable_mask=str(editable),protected_mask=str(protected))
    scene=tmp_path/'scene.json';scene.write_text('{}')
    chains=tmp_path/'chains.json'
    chains.write_text(json.dumps(dict(chains=[dict(names=['a','b'],edges=[1],arc=[0.,1.],
        sides=[dict(name=n,points=[[16,8],[16,56]],normals=[[1,0],[1,0]],px_per_unit=80)
               for n in views])])))
    output=tmp_path/'candidate';job=tmp_path/'job.json'
    job.write_text(json.dumps(dict(scene=str(scene),seam_chains=str(chains),views=views,
        output_dir=str(output),records_dir=str(tmp_path/'records'),protection_feather=0,
        maximum_correction_step_deltaE76=3.0)))
    def block(shape,*args,**kwargs):
        field=np.zeros((*shape,3));field[:,32:,0]=8.
        return field,np.ones(shape,bool)
    monkeypatch.setattr(seam_engine,'apply_band',block)
    with pytest.raises(ValueError,match='correction field discontinuity'):
        seam_engine.run(job)
    assert not output.exists()
    for path,content in originals.items():
        assert path.read_bytes()==content
