"""Regression of source-derived rear panel seams, including old output refusal."""
import argparse,contextlib,io,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from phone_texture.auto_material import propose_job,lighting_scope
from phone_texture import lighting,lighting_review
from phone_texture.automatic import measure_masks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--pro-process',type=Path,required=True);parser.add_argument('--promax-work',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(exist_ok=False,parents=True)
    pro=args.pro_process.resolve()
    paths=sorted(pro.glob('cases/*/work/geometry-records/staged/*/back.png'))
    pm=args.promax_work.resolve()
    paths.append(pm/'geometry-records/staged/原图未命名/back.png');rows=[]
    for source in paths:
        name='ProMax' if source.parent.name=='原图未命名' else source.parent.name
        folder=out/name;folder.mkdir();companions={f:source.parent/(f+'.png') for f in ['front','back','left','right','top','bottom']}
        job,proof,allowed,paper=propose_job(source,'back',folder/'materials',companions)
        spec=lighting_scope(source,'back',job,folder/'scope-assets')
        old=(pm/'lighting/原图未命名/back.png') if name=='ProMax' else pro/'complete-candidates'/name/'back.png'
        old_error=None
        try:lighting_review.measure_local(spec,source,old,allowed,paper)
        except ValueError as exc:old_error=str(exc)
        job.update(output_dir=str(folder/'lighting'),records_dir=str(folder/'records'))
        (folder/'job.json').write_text(json.dumps(job));(folder/'scope.json').write_text(json.dumps(spec));(folder/'analysis.json').write_text(json.dumps(proof))
        with contextlib.redirect_stdout(io.StringIO()):lighting.run('aluminum',folder/'job.json')
        output=folder/'lighting/back.png';checks=lighting_review.measure_perimeter(spec,source,output,allowed,paper)
        invariants=measure_masks(source,output,allowed,paper)
        lighting_review.evidence_images(spec,source,output,folder/'visual-review')
        rows.append(dict(sku=name,old_rejected=old_error,checks=checks,invariants=invariants,output=str(output),source=str(source),old=str(old),panel=proof['rear_panel']))
        print(name,'old rejected:',old_error is not None,'new PASS',flush=True)
    (out/'report.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    canvas=Image.new('RGB',(1500,len(rows)*350),'#eceee9');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc',20)
    for i,r in enumerate(rows):
        for j,(label,path) in enumerate([('原图',r['source']),('旧处理',r['old']),('修复后',r['output'])]):
            im=Image.open(path).convert('RGBA');w,h=im.size;im=im.crop((0,int(h*.25),w,int(h*.44)));im.thumbnail((480,295));canvas.paste(im,(j*500+10,i*350+40),im);draw.text((j*500+10,i*350+10),r['sku']+' '+label,font=font,fill='#203331')
    canvas.save(out/'upper-junctions.jpg',quality=94)
    assert len(rows)==11 and all(r['old_rejected'] for r in rows)


if __name__=='__main__':main()
