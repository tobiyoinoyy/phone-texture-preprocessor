"""Replay a saved rear geometry through current material/lighting/canvas code.

Only the back is regenerated. The other five delivered faces are byte copies.
The old lighting job must reproduce its saved pixels before comparison is valid.
No segmentation, source edits, historical-result overwrite or UI publication.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy import ndimage
from PIL import Image,ImageDraw,ImageFont

from phone_texture import auto_material,lighting,lighting_review,shadow_lift_profile
from phone_texture.automatic import measure_masks,implementation_manifest
from phone_texture.back_canvas import standardize
from phone_texture.back_lens import lens_centres
from phone_texture.geometry_review import digest,write


def compare(images,labels,boxes,path,scale=1,background='#eeeeee'):
    widths=[(box[2]-box[0])*scale for box in boxes]
    heights=[(box[3]-box[1])*scale for box in boxes]
    column=max(widths)+24;canvas=Image.new('RGB',(column*len(images),sum(heights)+70*len(boxes)),background)
    draw=ImageDraw.Draw(canvas)
    font_path=next((p for p in (Path('/System/Library/Fonts/PingFang.ttc'),
        Path('/System/Library/Fonts/STHeiti Medium.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')) if p.exists()),None)
    font=ImageFont.truetype(str(font_path),24) if font_path else ImageFont.load_default(size=24)
    y=0
    for box,height in zip(boxes,heights):
        for k,(image,label) in enumerate(zip(images,labels)):
            draw.text((column*k+12,y+12),label,font=font,fill='#777777')
            crop=image.crop(box).resize(((box[2]-box[0])*scale,height),Image.Resampling.NEAREST)
            canvas.paste(crop,(column*k+12,y+52),crop.getchannel('A'))
        y+=height+70
    canvas.save(path)


def run(case,previous,output,model):
    case=Path(case).resolve();previous=Path(previous).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);work=case/'work'
    old_job_path=work/'material-inputs/back/job.json';old_job=json.loads(old_job_path.read_text())
    source=Path(old_job['views'][0]['source'])
    if digest(source)!=old_job['views'][0]['source_sha256']:raise ValueError('Saved lighting source changed')
    # Find the case name from the source path instead of assuming SKU numbering.
    saved=work/'lighting'/case.name/'back.png'
    case_result=json.loads((case/'result.json').read_text())
    inputs=[source,old_job_path,saved,case/'result.json',*previous.glob('*.png'),
            *Path(case_result['source']).glob('*.jpg')]
    hashes={str(p):digest(p) for p in inputs}
    render=lighting.engine('aluminum')
    target,_=render.target_color({'views':old_job['views']})
    old,_,_,_=render.process_view(old_job['views'][0],target,dict(old_job['defaults'],color_mode=old_job['color_mode']))
    if not np.array_equal(old,np.array(Image.open(saved).convert('RGBA'))):
        raise AssertionError('Old lighting replay differs from saved baseline')
    old_canvas=output/'previous-canvas-replay.png';standardize(saved,old_canvas,model)
    if not np.array_equal(np.array(Image.open(old_canvas)),np.array(Image.open(previous/'back.png'))):
        raise AssertionError('Saved rear has additional downstream edits; do not bypass them')
    job,analysis,_,_=auto_material.propose_job(source,'back',output/'materials',protection_policy=old_job['views'][0]['protection_policy'])
    job,profile=shadow_lift_profile.apply(job,'back',output/'materials',old_job['lighting_profile'])
    write(output/'analysis.json',dict(analysis,lighting_profile=profile))
    scope=auto_material.lighting_scope(source,'back',job,output/'scope')
    write(output/'scope.json',scope)
    job.update(output_dir=str(output/'lighting'),records_dir=str(output/'records'));write(output/'job.json',job)
    target,_=render.target_color({'views':job['views']})
    fixed,allowed,protected,validation=render.process_view(job['views'][0],target,dict(job['defaults'],color_mode=job['color_mode']))
    (output/'lighting').mkdir();new=output/'lighting/back.png';Image.fromarray(fixed).save(new)
    write(output/'lighting-validation.json',validation)
    invariants=measure_masks(source,new,allowed,protected)
    checks={}
    for name,function in [('perimeter',lighting_review.measure_perimeter),('local',lighting_review.measure_local)]:
        checks[name]=shadow_lift_profile.measure_effect(function,scope,source,new,allowed,protected,old_job['lighting_profile'])
    previous_warnings=next((v for v in case_result.get('lighting_effect_warnings',[]) if v.get('face')=='back'),{})
    warnings_unchanged=previous_warnings.get('effect_failures',[])==checks['perimeter'].get('effect_failures',[])
    if not warnings_unchanged:raise AssertionError('Rear effect warnings changed; inspect before delivery')
    write(output/'effect-checks.json',checks)
    delivery=output/'六面结果'/case.name;delivery.mkdir(parents=True)
    canvas=standardize(new,delivery/'back.png',model)
    final_pixels=np.array(Image.open(delivery/'back.png'))
    previous_pixels=np.array(Image.open(previous/'back.png'))
    if final_pixels.shape!=previous_pixels.shape or not np.array_equal(final_pixels[...,3],previous_pixels[...,3]):
        raise AssertionError('Final rear canvas/alpha changed')
    unchanged={}
    for face in ('front','left','right','top','bottom'):
        shutil.copy2(previous/(face+'.png'),delivery/(face+'.png'))
        unchanged[face]=digest(previous/(face+'.png'))==digest(delivery/(face+'.png'))
    source_pixels=np.array(Image.open(source).convert('RGBA'))
    raw_L=render.rgb_to_lab(source_pixels[...,:3])[...,0]
    old_L=render.rgb_to_lab(old[...,:3])[...,0];new_L=render.rgb_to_lab(fixed[...,:3])[...,0]
    mark=np.array(Image.open(output/'materials/rear-mark-fit-exclusion.png'))>0
    distance=ndimage.distance_transform_edt(~mark)
    metrics={}
    for name,region in [('logo',mark),('near_logo_background',(distance>3)&(distance<25)),('far_logo_background',(distance>90)&(distance<140))]:
        metrics[name]={stage:np.percentile((L-raw_L)[region],[5,50,95]).tolist() for stage,L in [('old_delta_L',old_L),('new_delta_L',new_L)]}
    # Detect the original halo as excess lift relative to nearby panel.
    halo_old=metrics['near_logo_background']['old_delta_L'][1]-metrics['far_logo_background']['old_delta_L'][1]
    halo_new=metrics['near_logo_background']['new_delta_L'][1]-metrics['far_logo_background']['new_delta_L'][1]
    if not (halo_old>3 and abs(halo_new)<1):raise AssertionError('Logo halo regression did not improve')
    glass,_=auto_material.rear_camera_glass(source_pixels,source_pixels[...,3]>0)
    if not np.array_equal(fixed[glass],source_pixels[glass]):raise AssertionError('Camera glass changed during lighting')
    def camera_mask(config):
        domain=next(d for d in config['views'][0]['domains'] if d['name']=='camera_surround')
        return np.array(Image.open(domain['mask']))>0
    released=camera_mask(job)&~camera_mask(old_job)
    release_change=float(np.mean(np.any(fixed!=source_pixels,axis=2)[released]))
    if release_change<.9:raise AssertionError('Released shadow remains frozen')
    before_centres,_=lens_centres(old);after_centres,_=lens_centres(fixed)
    if before_centres is None or after_centres is None:raise AssertionError('Lens centres unavailable')
    drift=float(abs(after_centres-before_centres).max())
    if drift>.5:raise AssertionError('Lighting changed measured lens geometry')
    old_images=[Image.open(previous/'back.png').convert('RGBA'),Image.open(delivery/'back.png').convert('RGBA')]
    w,h=old_images[0].size
    boxes=[(int(.05*w),int(.02*h),int(.57*w),int(.27*h)),(int(.32*w),int(.53*h),int(.68*w),int(.72*h))]
    compare(old_images,['优化前','优化后'],boxes,output/'局部对比.png')
    for background,name in [('#eeeeee','浅底'),('#222222','深底')]:
        compare(old_images,['优化前','优化后'],boxes,output/f'局部4倍-{name}.png',4,background)
    compare(old_images,['优化前','优化后'],[(0,0,w,h)],output/'整图对比.png')
    if not all(digest(p)==sha for p,sha in hashes.items()):raise AssertionError('Input modified')
    report=dict(old_lighting_replay_pixel_exact=True,old_canvas_replay_pixel_exact=True,input_hashes=hashes,
        all_input_hashes_unchanged=True,other_five_faces_byte_identical=unchanged,invariants=invariants,
        source_camera_glass_unchanged=True,lens_measurement_drift_px=drift,canvas=canvas,logo_metrics=metrics,
        released_shadow_pixels=int(released.sum()),released_shadow_changed_fraction=release_change,
        previous_effect_warnings_unchanged=warnings_unchanged,final_alpha_and_size_unchanged=True,
        halo_excess_lift_L=dict(old=halo_old,new=halo_new),source_geometry_reused=True,segmentation_rerun=False,
        model_uv_verified=False,user_visual_approval=False,web_history_overwritten=False,
        implementation=implementation_manifest(Path(__file__).resolve().parents[1]))
    write(output/'regression.json',report)
    print(json.dumps({k:report[k] for k in ('other_five_faces_byte_identical','lens_measurement_drift_px','halo_excess_lift_L')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('case','previous','output'):parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--model',choices=('iphone17pro','iphone17promax'),required=True)
    args=parser.parse_args();run(args.case,args.previous,args.output,args.model)
