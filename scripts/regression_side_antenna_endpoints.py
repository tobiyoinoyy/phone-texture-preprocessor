"""Source-derived lower-antenna repair of an existing validated side-lighting run.

The requested lower band is revised; upper mapping, hardware and all pixels
outside its finite illumination support remain exactly the accepted v1 pixels.
No segmentation, generated RGB, model UV changes or formal publication.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from phone_texture import auto_material, auto_structure, lighting, lighting_review as review, structure_review
from phone_texture.automatic import measure_masks


def run(bundle,v1,output,skus,faces=('left','right')):
    started=time.perf_counter();bundle=Path(bundle).resolve();v1=Path(v1).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);engine=lighting.engine('aluminum')
    report=dict(skus={},scope='lower antenna only',segmentation_rerun=False,model_uv_verified=False,published=False)
    for sku in skus:
        begin=time.perf_counter();folder=bundle/'images'/sku
        geometry={f:folder/'02_geometry_intermediates'/(f+'.png') for f in ('front','back','left','right','top','bottom')}
        marker_path=bundle/'logs'/sku/'process/cases'/sku/'work/geometry-structure-proposal.json'
        landmarks=review.read(marker_path);hashes={str(p):review.digest(p) for p in [*geometry.values(),marker_path]};case={}
        for face in faces:
            tick=time.perf_counter();old=v1/sku/face;work=output/sku/face;work.mkdir(parents=True)
            job=review.read(old/'job.json');hashes.update(lighting.dependencies(job))
            hashes[str(old/'job.json')]=review.digest(old/'job.json');hashes[str(old/'after-lighting.png')]=review.digest(old/'after-lighting.png')
            source=geometry[face];im=Image.open(source).convert('RGBA');a=np.asarray(im);h,w=a.shape[:2]
            if job['views'][0]['source_sha256']!=review.digest(source):raise ValueError('V1 job source hash mismatch')
            cfg=landmarks[face]
            if cfg['source_sha256']!=review.digest(source) or cfg['back']['sha256']!=review.digest(geometry['back']):
                raise ValueError('Frozen side/back landmarks belong to another geometry source')
            indices=cfg['antenna_indices'];source_lines=np.asarray(cfg['source_lines'])
            rows=np.median(source_lines[indices,int(w*.4):int(w*.6)],axis=1)
            fresh,boundary_proof=auto_structure.per_column_boundaries(im,rows,return_proof=True,revision_bands=(1,))
            for i in (2,3):cfg['source_lines'][indices[i]]=fresh[i].tolist()
            cfg['source_boundary_detection']=dict(boundary_proof,revision_bands=[1],upper_mapping_retained=True)
            view=job['views'][0];domains={d['name']:d for d in view['domains']}
            old_band=np.asarray(Image.open(domains['antenna_1']['mask']).convert('L'))>0
            old_metal=np.asarray(Image.open(domains['main_surface']['mask']).convert('L'))>0
            yy=np.arange(h)[:,None];band=(yy>=fresh[2])&(yy<fresh[3])&(a[...,3]>0)
            metal=(old_metal|old_band)&~band
            fit=np.asarray(Image.open(domains['main_surface']['fit_mask']).convert('L'))>0
            fit=(fit|old_band)&metal
            for name,mask in [('antenna_1',band),('main_surface',metal)]:
                domains[name]['mask']=auto_material.save_mask(work/(name+'.png'),mask)
                for x0,y0,x1,y1 in domains[name]['reference_boxes']:
                    if not mask[y0:y1,x0:x1].all():raise ValueError('Previous source reference crosses corrected material')
            domains['main_surface']['fit_mask']=auto_material.save_mask(work/'main_surface-fit.png',fit)
            view['correction_blends']=[dict(domains=['antenna_1','main_surface'],boundary_domain='antenna_1',sigma=1.25,kind='source_material_transition')]
            view['antenna_boundary_bands']=[1]
            view['source']=str(source);review.write(work/'job.json',job)
            target,_=engine.target_color(job);fixed,allowed,paper,validation=engine.process_view(view,target,dict(job['defaults'],color_mode=job['color_mode']))
            candidate=work/'after-lighting.png';Image.fromarray(fixed).save(candidate)
            old_pixels=np.asarray(Image.open(old/'after-lighting.png').convert('RGBA'))
            # Gaussian kernels have finite support; verify the whole accepted
            # upper body and hardware rather than restoring it after a failure.
            radius=int(np.ceil(4*domains['main_surface']['sigma'][0]+12))
            lo=max(0,int(min(source_lines[indices[2]].min(),fresh[2].min()))-radius)
            hi=min(h,int(max(source_lines[indices[3]].max(),fresh[3].max()))+radius+1)
            outside=np.ones((h,w),bool);outside[lo:hi]=False
            if not np.array_equal(fixed[outside],old_pixels[outside]):raise ValueError('Lower repair changed accepted pixels outside finite support')
            for name,d in domains.items():
                if name.startswith('hardware_') or name=='antenna_0':
                    m=np.asarray(Image.open(d['mask']))>0
                    if not np.array_equal(fixed[m],old_pixels[m]):raise ValueError('Lower repair changed upper antenna or hardware')
            invariants=measure_masks(source,candidate,allowed,paper)
            scope=auto_material.lighting_scope(source,face,job,work/'scope')
            review.write(work/'scope.json',scope)
            metrics=review.measure_local(scope,source,candidate,allowed,paper);rejected={}
            for name,check in scope['local_checks'].items():
                if not (name.startswith('antenna_1_') and name.endswith('_metal')):continue
                try:review.measure_local(dict(scope,local_checks={name:check}),source,old/'after-lighting.png',allowed,paper)
                except ValueError as e:rejected[name]=str(e)
            if sku=='SKU_02':
                complaint='antenna_1_'+('right' if face=='left' else 'left')+'_metal'
                if complaint not in rejected:raise AssertionError('User-marked v1 endpoint was not rejected')
            review.evidence_images(scope,old/'after-lighting.png',candidate,work/'v1-v2-evidence')
            for name,path in [('geometry.png',source),('v1-lighting.png',old/'after-lighting.png'),('original-old-lighting.png',old/'before-lighting.png')]:shutil.copy2(path,work/name)
            crop=(0,int(min(fresh[2].min(),source_lines[indices[2]].min()))-32,w,int(max(fresh[3].max(),source_lines[indices[3]].max()))+33)
            for scale in (1,4):
                for bg,color in [('dark','#222222'),('light','#eeeeee')]:
                    ims=[im,Image.fromarray(old_pixels),Image.fromarray(fixed)];cw,ch=crop[2],crop[3]-crop[1]
                    sheet=Image.new('RGB',(3*cw*scale+24,ch*scale+24),color);draw=ImageDraw.Draw(sheet)
                    for i,(pic,label) in enumerate(zip(ims,['SOURCE','V1','V2'])):
                        tile=pic.crop(crop).resize((cw*scale,ch*scale),Image.Resampling.NEAREST);x=i*(cw*scale+12);sheet.paste(tile,(x,24),tile);draw.text((x,4),label,fill='white' if bg=='dark' else '#111')
                    sheet.save(work/f'bottom-{bg}-{scale}x.png')
            result=dict(seconds=time.perf_counter()-tick,boundary_detection=boundary_proof,bottom_only=True,unchanged_upper_and_hardware=True,
                allowed_change_rows=[lo,hi],outside_scope_changed_pixels=0,changed_pixels_vs_v1=int(np.any(fixed!=old_pixels,axis=2).sum()),
                invariant_checks=invariants,local_checks=metrics,v1_rejected_checks=rejected,processing=validation,
                candidate_sha256=review.digest(candidate),visual_acceptance='pending',crop_xyxy=list(crop))
            review.write(work/'validation.json',result);case[face]=result
            print(sku,face,round(result['seconds'],3),'s',len(rejected),'v1 endpoint gates rejected',flush=True)
        if sku=='SKU_02' and set(faces)=={'left','right'}:
            stage=output/sku/'structure';light=stage/'lighting'/sku;light.mkdir(parents=True)
            before={}
            for face in geometry:
                p=output/sku/face/'after-lighting.png' if face in ('left','right') else folder/'03_lighting_intermediates'/(face+'.png')
                hashes[str(p)]=review.digest(p);dest=light/(face+'.png');shutil.copy2(p,dest);before[face]=dest
            spec={sku:auto_structure.bind_lighting_sources(landmarks,geometry,before)};review.write(stage/'spec.json',spec)
            case['structure']=structure_review.prepare(stage/'lighting',stage/'spec.json',stage/'records')['checks']
        if not all(review.digest(p)==v for p,v in hashes.items()):raise ValueError('Read-only source or v1 changed')
        case['input_hashes']=hashes;case['seconds']=time.perf_counter()-begin;report['skus'][sku]=case;review.write(output/'regression.json',report)
    report.update(seconds=time.perf_counter()-started,all_input_hashes_unchanged=True)
    review.write(output/'regression.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);p.add_argument('--v1',required=True);p.add_argument('--output',required=True)
    p.add_argument('--skus',nargs='+',default=['SKU_02']);p.add_argument('--faces',nargs='+',choices=['left','right'],default=['left','right'])
    a=p.parse_args();run(a.bundle,a.v1,a.output,a.skus,a.faces)
