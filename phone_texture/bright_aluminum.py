"""Opt-in bright aluminum finish on already aligned standard iPhone 17 PNGs.

The caller supplies the approved bright top/bottom as lightness references.
No segmentation, pose/UV changes, or generation; keep a pristine baseline.
"""
from pathlib import Path
import shutil
import numpy as np
import cv2
from PIL import Image
from . import lighting,workspace as ws
from .iphone17_material import job
from .iphone17_structure import antenna_rows,per_column_boundaries,hardware


def lab(a):
    return cv2.cvtColor(a[...,:3].astype('float32')/255,cv2.COLOR_RGB2LAB)


def refine_sides(source,work):
    source,work=Path(source),Path(work)
    work.mkdir(parents=True,exist_ok=False);dest=work/'candidates';dest.mkdir()
    files={f:source/(f+'.png') for f in ('front','back','left','right','top','bottom')}
    baseline={f:ws.sha(p) for f,p in files.items()};references=[]
    for face in ('top','bottom'):
        a=np.asarray(Image.open(files[face]).convert('RGBA'));h,w=a.shape[:2];L=lab(a)[...,0]
        for lo,hi in ((.01,.07),(.93,.99)):
            roi=np.zeros((h,w),bool);roi[int(h*.25):int(h*.75),int(w*lo):int(w*hi)]=True;roi &=a[...,3]>240
            references.append(dict(face=face,box=[int(w*lo),int(h*.25),int(w*hi),int(h*.75)],sha256=baseline[face],L=float(np.median(L[roi]))))
    target=float(np.median([r['L'] for r in references]));checks={}
    if not np.isfinite(target):raise ValueError('No finite approved metal reference')
    for face,path in files.items():
        if face not in ('left','right'):shutil.copy2(path,dest/path.name);continue
        im=Image.open(path).convert('RGBA');a=np.asarray(im);h,w=a.shape[:2]
        rows,_,_=antenna_rows(im,[int(w*.4),int(w*.6)])
        structure={face:dict(antenna_lines=per_column_boundaries(im,rows).tolist(),hardware_boxes=hardware(im,face))}
        spec,proof,_,paper=job(path,face,work/'materials'/face,structure)
        editable=np.zeros((h,w),bool);chosen=[]
        for domain in spec['views'][0]['domains']:
            name=domain['name']
            selected=name=='main_surface' or (name.endswith('_face') and (face=='left' or name=='hardware_0_face'))
            if not selected:continue
            chosen.append(name);editable |=np.asarray(Image.open(domain['mask']))>0
            domain.update(sigma=[max(6,h*.006),3.],strength=[.98,0,0],feather=0,correction_smooth=1.5)
            domain.pop('reference_boxes',None)
        spec['defaults'].update(max_lab_shift=[65,0,0],color_mode='source_lightness')
        ws.save(work/(face+'-job.json'),dict(spec,approved_reference_L=target,reference_evidence=references))
        rendered,_,_,_=lighting.engine('aluminum').process_view(spec['views'][0],np.array([target,0.,0.]),spec['defaults'])
        original_lab=lab(a);rendered_lab=lab(rendered)
        delta=np.maximum(0,rendered_lab[...,0]-original_lab[...,0])*editable
        result_lab=original_lab.copy();result_lab[...,0]+=delta
        rgb=np.clip(np.rint(cv2.cvtColor(result_lab,cv2.COLOR_LAB2RGB)*255),0,255).astype('uint8')
        out=a.copy();changed=(delta>0)&(a[...,3]>0);out[changed,:3]=rgb[changed]
        Image.fromarray(out).save(dest/path.name);np.save(work/(face+'-editable.npy'),editable)
        assert np.array_equal(a[...,3],out[...,3]) and np.array_equal(a[~editable],out[~editable])
        assert np.array_equal(a[paper],out[paper]) and np.array_equal(a[a[...,3]==0],out[a[...,3]==0])
        final_lab=lab(out);regions={}
        main=np.asarray(Image.open(next(d['mask'] for d in spec['views'][0]['domains'] if d['name']=='main_surface')))>0
        for label,lo,hi in [('upper_end',.01,.07),('middle',.45,.55),('lower_end',.93,.99)]:
            roi=np.zeros((h,w),bool);roi[int(h*lo):int(h*hi),int(w*.25):int(w*.75)]=True;roi &=main&(a[...,3]>240)
            regions[label]=dict(before_L=float(np.median(original_lab[...,0][roi])),after_L=float(np.median(final_lab[...,0][roi])),p10_L=float(np.percentile(final_lab[...,0][roi],10)))
            if abs(regions[label]['after_L']-target)>5:raise ValueError('Bright metal region failed approved reference comparison')
        checks[face]=dict(regions=regions,material_domains=chosen,alpha_changed=0,outside_material_changed=0,paper_changed=0,
                          max_lift_L=float(delta.max()),ab_error_p99=np.percentile(abs((final_lab-original_lab)[changed,1:]),99,axis=0).tolist())
    if any(ws.sha(files[f])!=v for f,v in baseline.items()):raise ValueError('Input changed during processing')
    report=dict(mode='bright_aluminum_opt_in_v1',source=str(source),baseline_sha256=baseline,target_L=target,references=references,
                checks=checks,candidate=str(dest),final_sha256={f:ws.sha(dest/(f+'.png')) for f in files},model_uv_checked=False,
                scope='left/right metal lightness only; antenna polymer, grooves and camera-control glass retain source pixels')
    ws.save(work/'report.json',report)
    return report
