"""Correctness regression: fresh top inference, camera rejection, other faces locked.

No benchmark timing or concurrency claims. Fixtures are test expectations only;
the production calibrator never reads them. Originals and old delivery are read-only.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phone_texture.auto_geometry import frame_geometry
from phone_texture.auto_material import propose_job,lighting_scope
from phone_texture.automatic import measure_masks
from phone_texture import lighting,lighting_review
from phone_texture.geometry_review import digest,write
from phone_texture.top_frame import calibrate_top,verify_top_mask


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','previous','previous_process','records'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=parser.parse_args();records=args.records.resolve();records.mkdir(parents=True,exist_ok=False)
    fixtures=json.loads((ROOT/'tests/fixtures/iphone17-top-camera.json').read_text())
    from phone_normalizer.api import Normalizer,NormalizerConfig
    import torch
    torch.set_num_threads(2)
    results=[]
    with Normalizer(NormalizerConfig(remove_support=False,deskew=False,tight_crop=False,device='mps')) as norm:
        norm.backend.max_inference_side=1024
        for sku,expected in fixtures['cases'].items():
            source=args.source/sku/'top.JPG'
            if digest(source)!=expected['source_sha256']:raise ValueError('Regression source changed: '+sku)
            process=records/'process'/sku;process.mkdir(parents=True)
            raw=ImageOps.exif_transpose(Image.open(source)).convert('RGB')
            matte,_=norm.normalize(raw);matte.save(process/'fresh-matte.png');stages={}
            def capture(n,i):
                stages[n]=i.copy();i.save(process/(n+'.png'))
            candidate,proof=frame_geometry(matte,'top',capture,source_image=raw)
            geom=process/'geometry';geom.mkdir();src=geom/'top.png';candidate.save(src)
            upper,lower,_=calibrate_top(np.array(matte),np.array(raw))
            old_path=args.previous_process/'cases'/sku/'work/geometry-records/geometry-review'/sku/'top/after-support-source-canvas.png'
            old=np.array(Image.open(old_path).convert('RGBA'))
            try:verify_top_mask(np.array(matte),old,upper,lower,proof)
            except ValueError as exc:old_failure=str(exc)
            else:raise AssertionError('Old camera-containing top was accepted')
            new=np.array(stages['after_support'])
            for box in expected['camera_rectangles']:
                x0,y0,x1,y1=box
                if not np.any(old[y0:y1,x0:x1,3]):raise AssertionError('Old fault not reproduced')
                if np.any(new[y0:y1,x0:x1,3]):raise AssertionError('Independent camera witness retained')
            for box in expected['body_rectangles']:
                x0,y0,x1,y1=box
                if np.any(new[y0:y1,x0:x1,3]<128):raise AssertionError('Independent body witness lost')
            overlay=raw.copy();draw=ImageDraw.Draw(overlay)
            for box in expected['camera_rectangles']:draw.rectangle(box,outline='red',width=2)
            for box in expected['body_rectangles']:draw.rectangle(box,outline='lime',width=2)
            x0,x1=proof['x_extent'];draw.line([(x0,float(np.polyval(upper,x0))),(x1,float(np.polyval(upper,x1)))],fill='cyan',width=2)
            overlay.crop((max(0,x0-10),int(np.polyval(upper,x0))-80,min(raw.width,x1+10),int(np.polyval(lower,x0))+20)).save(process/'source-witness-overlay.png')
            assets=process/'material-inputs';job,analysis,allowed,paper=propose_job(src,'top',assets)
            effect=lighting_scope(src,'top',job,assets/'effect-source-witnesses')
            write(assets/'lighting-scope.json',effect)
            job.update(output_dir=str(process/'lighting'),records_dir=str(process/'lighting-records'))
            jp=assets/'job.json';write(jp,job);write(assets/'analysis.json',analysis);lighting.run('aluminum',jp)
            lit=process/'lighting/top.png';checks=measure_masks(src,lit,allowed,paper)
            lighting_review.measure_local(effect,src,lit,allowed,paper)
            dest=records/'six-face-candidates'/sku;shutil.copytree(args.previous/sku,dest)
            locked={p.name:digest(p) for p in dest.glob('*.png') if p.stem!='top'}
            shutil.copy2(lit,dest/'top.png')
            if locked!={p.name:digest(p) for p in dest.glob('*.png') if p.stem!='top'}:raise AssertionError('Unrelated face changed')
            if digest(source)!=expected['source_sha256']:raise AssertionError('Original modified')
            # Full native and 4x top boundary/round-corner evidence on both backgrounds.
            for color,label in [((25,25,25,255),'dark'),((235,235,235,255),'light')]:
                for phase,path in [('geometry',src),('lighting',lit)]:
                    im=Image.open(path).convert('RGBA');bg=Image.new('RGBA',im.size,color);bg.alpha_composite(im);bg=bg.convert('RGB')
                    bg.save(process/(phase+'-'+label+'-1x.png'))
                    bg.resize((bg.width*4,bg.height*4)).save(process/(phase+'-'+label+'-4x.png'))
            row=dict(sku=sku,source_sha256=digest(source),geometry=proof,old_rejected=old_failure,
                independent_witnesses_passed=True,lighting_checks=checks,other_five_sha256=locked,
                top_sha256=digest(dest/'top.png'),runtime_agent_required=False,historical_masks_reused=False)
            write(process/'report.json',row);results.append(row);print(sku,'passed',flush=True)
    font=ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc',21)
    canvas=Image.new('RGB',(1600,110+len(results)*150),(125,125,125));draw=ImageDraw.Draw(canvas)
    draw.text((20,20),'顶面修正：左为旧版，右为程序自动修正；摄像头台阶不参与标定',font=font,fill='white')
    for i,row in enumerate(results):
        sku=row['sku'];y=80+i*150
        draw.text((5,y),sku,font=font,fill='white')
        for col,path in enumerate([args.previous/sku/'top.png',records/'six-face-candidates'/sku/'top.png']):
            im=Image.open(path).convert('RGBA');im.thumbnail((760,120));canvas.paste(im,(20+col*800,y+24),im)
    canvas.save(records/'顶面修正对照.jpg',quality=93)
    write(records/'report.json',dict(kind='correctness_regression_not_benchmark',cases=results,
        count=len(results),all_passed=True,runtime_agent_required=False,performance_test_run=False,
        previous_top_candidates_rejected=True,full_batch_reprocessed=False,other_five_faces='byte-identical prior candidates',
        model_uv_verified=False,final_user_acceptance=False))


if __name__=='__main__':main()
