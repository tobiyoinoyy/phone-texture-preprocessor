"""Serial side-lighting replay from saved geometry; never runs segmentation or publishes."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from phone_texture import auto_material, lighting, lighting_review as review
from phone_texture.automatic import measure_masks
from phone_texture import auto_structure, structure_review


def run(bundle, output, skus, baseline_ref='7659e67', structure=False):
    start=time.perf_counter();bundle=Path(bundle).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    code=subprocess.check_output(['git','show',baseline_ref+':phone_texture/auto_material.py'])
    baseline_path=output/'baseline_auto_material.py';baseline_path.write_bytes(code)
    spec=importlib.util.spec_from_file_location('phone_texture._baseline_material_replay',baseline_path)
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    engine=lighting.engine('aluminum');result={'skus':{},'segmentation_rerun':False,'published':False,
        'baseline_commit':subprocess.check_output(['git','rev-parse',baseline_ref]).decode().strip(),
        'structure_replayed':structure,'uv_color_seam_stage_rerun':False}
    for sku in skus:
        case_start=time.perf_counter();result['skus'][sku]={}
        folder=bundle/'images'/sku
        geometry={f:folder/'02_geometry_intermediates'/(f+'.png') for f in ('front','back','left','right','top','bottom')}
        landmark_path=bundle/'logs'/sku/'process/cases'/sku/'work/geometry-structure-proposal.json'
        landmarks=json.loads(landmark_path.read_text())
        fresh_landmarks=auto_structure.propose(geometry,coordinate_mode='canvas_pixel_centers')
        # Full reprocessing now checks both bands; the separate lower-only
        # replay preserves an existing accepted upper mapping explicitly.
        for side in ('left','right'):
            if landmarks[side]['source_sha256']!=review.digest(geometry[side]):
                raise ValueError('Stored side landmarks belong to another source')
        hashes={str(p):review.digest(p) for p in [*geometry.values(),landmark_path]}
        for face in ('right','left'):
            begin=time.perf_counter();source=geometry[face];work=output/sku/face;work.mkdir(parents=True)
            old_job,old_proof,_,_=baseline.propose_job(source,face,work/'old-materials',companions=geometry,landmarks=landmarks)
            target,_=engine.target_color(old_job)
            old,_,_,old_validation=engine.process_view(old_job['views'][0],target,dict(old_job['defaults'],color_mode=old_job['color_mode']))
            Image.fromarray(old).save(work/'before-lighting.png')
            cloud=folder/'03_lighting_intermediates'/(face+'.png')
            cloud_equal=None
            if cloud.exists():
                hashes[str(cloud)]=review.digest(cloud)
                cloud_equal=np.array_equal(old,np.asarray(Image.open(cloud).convert('RGBA')))
                if not cloud_equal:raise AssertionError('Baseline replay differs from cloud '+sku+'/'+face)
            job,proof,allowed,paper=auto_material.propose_job(source,face,work/'materials',companions=geometry,landmarks=fresh_landmarks)
            scope=auto_material.lighting_scope(source,face,job,work/'scope')
            review.write(work/'job.json',job);review.write(work/'analysis.json',proof);review.write(work/'scope.json',scope)
            target,_=engine.target_color(job)
            fixed,_,_,validation=engine.process_view(job['views'][0],target,dict(job['defaults'],color_mode=job['color_mode']))
            fixed_path=work/'after-lighting.png';Image.fromarray(fixed).save(fixed_path)
            shutil.copy2(source,work/'geometry.png')
            original=folder/'01_raw_originals'/(face+'.JPG')
            if original.exists():hashes[str(original)]=review.digest(original)
            final=folder/'04_candidates_needs_visual_review'/(face+'.png')
            if final.exists():
                hashes[str(final)]=review.digest(final);shutil.copy2(final,work/'cloud-structure.png')
            invariants=measure_masks(source,fixed_path,allowed,paper)
            metrics=review.measure_local(scope,source,fixed_path,allowed,paper)
            baseline_rejections={}
            for name,check in scope['local_checks'].items():
                if not name.startswith('side_metal_fragment_'):continue
                try:review.measure_local(dict(scope,local_checks={name:check}),source,work/'before-lighting.png',allowed,paper)
                except ValueError as error:baseline_rejections[name]=str(error)
            # Record the upper correction rather than treating a known old
            # material-label artifact as a pixel-invariance requirement.
            antenna=np.zeros(allowed.shape,bool)
            for domain in job['views'][0]['domains']:
                if domain['name']=='antenna_0':antenna |= np.asarray(Image.open(domain['mask']))>0
            upper_changed=int(np.any(fixed[antenna]!=old[antenna],axis=1).sum())
            review.evidence_images(scope,source,fixed_path,work/'source-vs-fixed')
            review.evidence_images(scope,work/'before-lighting.png',fixed_path,work/'before-vs-fixed')
            # Full-size source/old/new, and source-only material overlays.
            images=[Image.open(p).convert('RGBA') for p in (source,work/'before-lighting.png',fixed_path)]
            w,h=images[0].size;sheet=Image.new('RGB',(3*w+32,h+24),'#252525');draw=ImageDraw.Draw(sheet)
            for i,(label,img) in enumerate(zip(('SOURCE','OLD','FIXED'),images)):
                draw.text((i*(w+16),4),label,fill='white');sheet.paste(img,(i*(w+16),24),img)
            sheet.save(work/'full-size-comparison.png')
            row=dict(seconds=time.perf_counter()-begin,source_sha256=review.digest(source),
                cloud_baseline_pixel_equal=cloud_equal,old_hardware_box_count=len(old_proof['hardware_boxes']),
                new_hardware_component_count=len(proof['hardware_materials']),invariants=invariants,
                upper_antenna_changed_pixels=upper_changed,antenna_source_boundaries_refreshed=[0,1],local_checks=metrics,old_rejected_checks=baseline_rejections,
                validation=validation,visual_acceptance='pending',model_uv_verified=False)
            result['skus'][sku][face]=row;review.write(work/'validation.json',row)
            print(sku,face,round(row['seconds'],3),'s',len(baseline_rejections),'old false partitions rejected',flush=True)
        if structure:
            # Use frozen pre-lighting landmarks, never re-detect after color
            # changes. Keep all companion faces from the saved lighting stage.
            before={};stage=output/sku/'structure-replay';light=stage/'lighting'/sku
            light.mkdir(parents=True)
            for face in geometry:
                src=(output/sku/face/'after-lighting.png') if face in ('left','right') else folder/'03_lighting_intermediates'/(face+'.png')
                hashes[str(src)]=review.digest(src);dest=light/(face+'.png');shutil.copy2(src,dest);before[face]=dest
            spec={sku:auto_structure.bind_lighting_sources(fresh_landmarks,geometry,before)}
            spec_path=stage/'spec.json';review.write(spec_path,spec)
            structural=structure_review.prepare(stage/'lighting',spec_path,stage/'records')
            result['skus'][sku]['structure_checks']=structural['checks']
            # Alignment must keep the entire intervening hardware area intact.
            # This also preserves the pixels in the four reported SKU02 blocks.
            for face in ('left','right'):
                fixed=np.asarray(Image.open(before[face]).convert('RGBA'))
                after=np.asarray(Image.open(stage/'records/candidate'/sku/(face+'.png')).convert('RGBA'))
                cfg=spec[sku][face];lo=min(x[0] for x in cfg['button_ranges']);hi=max(x[1] for x in cfg['button_ranges'])
                assert np.array_equal(fixed[lo:hi],after[lo:hi])
        assert all(review.digest(path)==sha for path,sha in hashes.items())
        result['skus'][sku]['input_hashes']=hashes
        result['skus'][sku]['seconds']=time.perf_counter()-case_start
        review.write(output/'regression.json',result)
    result['seconds']=time.perf_counter()-start
    result['all_input_hashes_unchanged']=True
    review.write(output/'regression.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    p.add_argument('--skus',nargs='+',default=['SKU_02']);p.add_argument('--baseline-ref',default='7659e67')
    p.add_argument('--structure',action='store_true',help='Also replay frozen image-space mapping; excludes UV color seam stage')
    args=p.parse_args();run(args.bundle,args.output,args.skus,args.baseline_ref,args.structure)
