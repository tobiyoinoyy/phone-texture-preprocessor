"""Standard iPhone 17 studio photographs, separate from the metal-back Pro.

Measurements belong to the current source images. In particular, the glass
rear is never passed to the Pro rear-panel or fixed-model UV algorithms.
"""
from pathlib import Path
import time
import traceback
import shutil
import contextlib

import numpy as np
from PIL import Image, ImageOps

from . import pipeline, workspace as w, geometry_review as gr
from .auto_geometry import frame_geometry, plate_geometry, trim_visible_canvas
from .iphone17_material import paper as paper_mask

PROFILE = 'iphone17-glass-back-studio-v4-button-perspective'


def geometry(source, work, sku, device='auto', cpu_threads=2):
    import torch
    import cv2
    from phone_normalizer.api import Normalizer, NormalizerConfig
    torch.set_num_threads(cpu_threads)
    cv2.setNumThreads(1)
    manifest = w.read(work/'intake.json') if (work/'intake.json').exists() else w.intake(source, work, single_name=sku)
    if Path(manifest['source_root']).resolve()!=source.resolve() or set(manifest['skus'])!={sku}:
        raise ValueError('Standard profile workspace belongs to another source/SKU')
    current=w.face_files(source)
    if any(w.sha(current[f])!=manifest['skus'][sku][f]['sha256'] for f in pipeline.FACES):
        raise ValueError('Standard profile source differs from its frozen intake')
    w.check_sources(manifest)
    records = work/'geometry-records'
    folder = records/'staged'/sku
    folder.mkdir(parents=True, exist_ok=True)
    cache = records/'mattes'
    cache.mkdir(exist_ok=True)
    report = dict(mode=PROFILE, success=False, candidates_complete=False,
                  geometry_status='needs_review', images=[])
    with Normalizer(NormalizerConfig(remove_support=False, deskew=False, tight_crop=False, device=device)) as norm:
        for face in pipeline.FACES:
            print('geometry: '+face, flush=True)
            item = manifest['skus'][sku][face]
            raw = ImageOps.exif_transpose(Image.open(item['snapshot'])).convert('RGB')
            matte_path = cache/(face+'.png')
            identity = dict(source_sha256=item['sha256'], max_side=1536 if face in ('front','back') else 1024,
                            model='ZhengPeng7/BiRefNet_dynamic')
            cache_meta = cache/(face+'.json')
            if matte_path.exists() and cache_meta.exists() and w.read(cache_meta) == identity:
                matte = Image.open(matte_path).convert('RGBA')
            else:
                norm.backend.max_inference_side = identity['max_side']
                matte, _ = norm.normalize(raw)
                matte.save(matte_path)
                w.save(cache_meta, identity)
            stages = {}
            def capture(name, image):
                stages[name] = image.copy()
            # Pro-validated support-edge separation and printed-tail
            # reconstruction stay off: this glass-back sample was reviewed
            # on the earlier geometry, and its label layout differs.
            if face in pipeline.FRAMES:
                image, evidence = frame_geometry(matte, face, capture, source_image=raw, separate_support_edge=False)
            else:
                paper, info = paper_mask(matte, face)
                image, evidence = plate_geometry(matte, face, paper, capture, source_image=raw, reconstruct_paper_tail=False)
                evidence['paper'] = info
            paper, _ = paper_mask(image, face)
            image, trim = trim_visible_canvas(image, paper)
            evidence['visible_canvas_trim'] = trim
            path = folder/(face+'.png')
            image.save(path)
            artifacts = gr.make_evidence(item['snapshot'], image, stages, records/'geometry-review'/sku/face)
            report['images'].append(dict(sku=sku, face=face, source=item['snapshot'], source_sha256=item['sha256'],
                output_sha256=w.sha(path), geometry=evidence, review_reasons=evidence['review_reasons'], geometry_evidence=artifacts))
            w.save(records/'report.json', report)
    report['candidates_complete'] = True
    w.save(records/'report.json', report)
    return {face: folder/(face+'.png') for face in pipeline.FACES}


def run(source, work=None, output=None, *, single_name='SKU_01', device='auto', cpu_threads=2, **kwargs):
    if work is None or output is None:
        raise ValueError('iPhone 17 profile requires explicit --work and --output')
    source, work, output = pipeline.separate_paths(source, work, output)
    pipeline.fresh(output)
    if not 1<=cpu_threads<=8:raise ValueError('cpu_threads must be 1..8')
    if kwargs.get('skus') not in (None,[single_name]):raise ValueError('Standard iPhone 17 currently accepts one named six-face folder')
    work.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    report = dict(profile=PROFILE, processed=0, total=1, visual_approved=0,
                  status='failed', model_uv_verified=False)
    try:
        files = geometry(source, work/'work', single_name, device, cpu_threads)
        from .iphone17_structure import propose,apply as align
        from .iphone17_material import job
        from .iphone17_port import mapping as port_mapping,apply as port_apply,measure as port_measure
        from . import lighting
        from .automatic import measure_masks
        from .lighting_review import lightness
        structure=propose(files)
        w.save(work/'structure-proposal.json',structure)
        port_map,port_proof=port_mapping(Image.open(files['bottom']))
        w.save(work/'port-registration.json',port_proof)
        np.save(work/'bottom-source-x.npy',port_map)
        light=work/('lighting-'+str(time.time_ns()));light.mkdir()
        lighting_checks={}
        for face in pipeline.FACES:
            print('lighting: '+face,flush=True)
            assets=work/'material-inputs'/face
            spec,proof,allowed,sticker=job(files[face],face,assets,structure)
            records=work/'lighting-records'/face
            # Existing attempts are retained separately; no stale engine result
            # is silently accepted after material or parameter changes.
            if records.exists():
                records.rename(records.with_name(face+'-previous-'+str(time.time_ns())))
            spec.update(output_dir=str(light.resolve()),records_dir=str(records.resolve()))
            jp=assets/'job.json';w.save(jp,spec);w.save(assets/'analysis.json',proof)
            with (assets/'engine.log').open('w') as log,contextlib.redirect_stdout(log):
                lighting.run('aluminum',jp)
            dest=light/(face+'.png')
            check=measure_masks(files[face],dest,allowed,sticker)
            before=np.asarray(Image.open(files[face]).convert('RGBA'));after=np.asarray(Image.open(dest).convert('RGBA'))
            delta=lightness(after[...,:3])-lightness(before[...,:3])
            if abs(delta[allowed]).max()>35.5:raise ValueError('Standard illumination exceeded the frozen source correction limit')
            check.update(visual_approved=False,max_lightness_shift=float(abs(delta[allowed]).max()),
                material_domains=[d['name'] for d in spec['views'][0]['domains']])
            lighting_checks[face]=check
            w.save(work/'lighting-checks.json',lighting_checks)
        staged=work/'final-candidates'/single_name;staged.mkdir(parents=True,exist_ok=True)
        structural={}
        for face in pipeline.FACES:
            print('structure: '+face,flush=True)
            im=Image.open(light/(face+'.png')).convert('RGBA')
            if face in ('left','right'):
                im,structural[face]=align(im,structure[face])
                from .iphone17_buttons import propose as button_propose,apply as button_apply
                button_dir=work/'button-inputs';button_dir.mkdir(exist_ok=True)
                button_input=button_dir/(face+'.png');im.save(button_input)
                button_spec=button_propose(files['back'],button_input,face)
                w.save(work/(face+'-button-proposal.json'),button_spec)
                im,structural[face]['button_alignment']=button_apply(im,button_spec)
            elif face=='bottom':
                im=port_apply(im,port_map)
                measured=port_measure(im)['port']
                error=float(measured[0]-(im.width-1)/2)
                if abs(error)>2.5:raise ValueError('Final rendered USB-C centre failed independent measurement')
                port_proof.update(final_measured_port_x=float(measured[0]),final_rendered_center_error_px=error)
                structural[face]=port_proof
            else:
                shutil.copy2(light/(face+'.png'),staged/(face+'.png'));continue
            im.save(staged/(face+'.png'))
        w.save(work/'structure-checks.json',structural)
        final={face:staged/(face+'.png') for face in pipeline.FACES}
        pipeline.overview({single_name:files},{single_name:final},work/'overview.png',faces=pipeline.FACES)
        manifest=w.read(work/'work/intake.json');w.check_sources(manifest)
        report.update(status='processed_needs_visual_review',processed=1,six_faces_processed=True,
            lighting_checks=lighting_checks,structure_checks=structural,source_hashes_unchanged=True,
            candidate=str(staged),final_sha256={f:w.sha(p) for f,p in final.items()})
        report['lighting_directory']=str(light)
        report['implementation_sha256']={str(p.relative_to(Path(__file__).parent)):w.sha(p)
            for p in sorted(Path(__file__).parent.glob('iphone17*.py'))}
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    report['seconds'] = time.perf_counter()-start
    w.save(work/'report.json', report)
    print(report.get('error',''), flush=True)
    return report


def publish(work,review,output):
    """Release a source-bound, actually inspected standard-model candidate.

    Automatic processing never fills this review or invents a reviewer. This
    separate development acceptance is intentionally not an auto dependency.
    """
    work,output=Path(work).resolve(),Path(output).resolve()
    report=w.read(work/'report.json');record=w.read(review)
    if report['status']!='processed_needs_visual_review' or report.get('processed')!=1:
        raise ValueError('Complete standard-model processing required before image delivery')
    for side in ('left','right'):
        check=report.get('structure_checks',{}).get(side,{})
        if check.get('basis')!='same_sku_back_rim_measured_pixel_center_ratio' or not check.get('rear_reference_rows'):
            raise ValueError('Measured rear rim alignment required; native-side-only results cannot be published')
    if record.get('report_sha256')!=w.sha(work/'report.json') or not record.get('reviewer'):
        raise ValueError('Review is missing or belongs to a stale standard-model result')
    folder=Path(report['candidate']);files={f:folder/(f+'.png') for f in pipeline.FACES}
    from .iphone17_structure import validate_render_checks
    revision=report.get('partial_revision')
    if revision:
        prior_path=Path(revision['prior_report_path'])
        if w.sha(prior_path)!=revision['prior_report_sha256']:
            raise ValueError('Partial correction has a stale baseline report')
        prior=w.read(prior_path);prior_folder=Path(prior['candidate']);side=revision['side']
        if side not in ('left','right'):raise ValueError('Only a declared side can receive a lower correction')
        check=report['structure_checks'][side]
        validate_render_checks(check)
        if check['refined_indices']!=[2,3]:raise ValueError('Partial correction must affect only the lower antenna')
        for face in pipeline.FACES:
            old=prior_folder/(face+'.png')
            if w.sha(old)!=prior['final_sha256'][face]:raise ValueError('Partial correction baseline changed')
            if face!=side and w.sha(files[face])!=w.sha(old):
                raise ValueError('Partial correction changed another face')
        old=np.asarray(Image.open(prior_folder/(side+'.png')).convert('RGBA'))
        new=np.asarray(Image.open(files[side]).convert('RGBA'))
        if old.shape!=new.shape:raise ValueError('Partial correction changed image dimensions')
        # Range is derived from the frozen original geometry, never selected
        # from the candidate's difference mask.
        spec=w.read(prior_path.parent/'structure-proposal.json')[side]
        ranges=[[0,spec['locked_middle'][1]+1],spec['endcap_ranges'][1]]
        if check['unchanged_row_ranges']!=ranges:raise ValueError('Partial correction lock ranges differ from baseline')
        for lo,hi in ranges:
            if not np.array_equal(old[lo:hi],new[lo:hi]):raise ValueError('Partial correction changed upper/buttons/endcap')
    else:
        for side in ('left','right'):
            validate_render_checks(report['structure_checks'][side])
            from .iphone17_buttons import validate as validate_buttons
            validate_buttons(report['structure_checks'][side].get('button_alignment',{}))
    checks=('outline_corners_contact','material_paper_edges','native_and_4x','structure_and_features')
    if set(record.get('faces',{}))!=set(pipeline.FACES):raise ValueError('All six faces require an independent review')
    for face,path in files.items():
        item=record['faces'][face]
        if w.sha(path)!=report['final_sha256'][face] or item.get('sha256')!=w.sha(path):
            raise ValueError('Candidate changed after review: '+face)
        if any(item.get(check) is not True for check in checks) or not item.get('notes'):
            raise ValueError('Standard model review is incomplete: '+face)
        for evidence in item.get('evidence',[]):
            if w.sha(Path(evidence['path']))!=evidence['sha256']:
                raise ValueError('Standard review evidence changed')
        if not item.get('evidence'):raise ValueError('Native/4x review evidence is missing')
    w.check_sources(w.read(work/'work/intake.json'))
    pipeline.fresh(output);output.mkdir(parents=True)
    for path in files.values():shutil.copy2(path,output/path.name)
    w.save(work/'delivery.json',dict(status='reviewed_image_delivery',output=str(output),
        final_sha256=report['final_sha256'],review_sha256=w.sha(Path(review)),model_uv_verified=False))
    return output
