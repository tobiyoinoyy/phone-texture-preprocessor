"""Explicit frame-only runner with side-view deskew and named orientation."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .api import Normalizer, NormalizerConfig
from .frame_strip import normalize_frame_strip
from PIL import Image


def process_frames(sources, output_dir, *, face='bottom', device='auto', backend_instance=None, deskew=True):
    if face not in ('top','bottom','left','right'):
        raise ValueError('Only horizontal top, bottom, left or right frame photographs are supported')
    paths=[Path(p).expanduser().resolve() for p in sources]
    if not paths:
        raise ValueError('Provide at least one source image')
    names=[p.parent.name for p in paths]
    if len(set(names)) != len(names):
        raise ValueError('Each input must be in a differently named SKU folder')
    out=Path(output_dir).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'Use a new output folder: {out}')
    for p in paths:
        with Image.open(p) as im:
            from scripts.geometry_postprocess import validate_geometry_size
            validate_geometry_size(im.size)
    out.mkdir(parents=True)
    report={'mode':'frame_only','face':face,'success':False,'files':[]}
    try:
        config=NormalizerConfig(device=device,remove_support=False,deskew=False,tight_crop=False)
        with Normalizer(config,backend_instance=backend_instance) as normalizer:
            for p,name in zip(paths,names):
                with Image.open(p) as im:
                    segmented,segmentation=normalizer.normalize(im)
                result,geometry=normalize_frame_strip(segmented,face=face,deskew=deskew)
                folder=out/name;folder.mkdir()
                target=folder/f'{face}.png'
                with target.open('xb') as handle:result.save(handle,format='PNG')
                info={'source':str(p),'output':str(target),'segmentation':asdict(segmentation),'geometry':geometry}
                with (folder/'metadata.json').open('x',encoding='utf-8') as handle:
                    json.dump(info,handle,ensure_ascii=False,indent=2)
                report['files'].append(info)
        report['success']=True
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        with (out/'manifest.json').open('x',encoding='utf-8') as handle:
            json.dump(report,handle,ensure_ascii=False,indent=2)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description='提取边框并排除镜头及支架；侧面摆正后右视图顺时针90度、左视图逆时针90度。')
    parser.add_argument('sources',nargs='+',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--face',choices=('top','bottom','left','right'),default='bottom')
    parser.add_argument('--device',choices=('auto','cpu','mps','cuda'),default='auto')
    parser.add_argument('--no-deskew',action='store_true',help='仅重现历史裁切，不做顶底面或侧面摆正')
    args=parser.parse_args(argv)
    try:
        report=process_frames(args.sources,args.output_dir,face=args.face,device=args.device,deskew=not args.no_deskew)
    except Exception as exc:
        print(f'处理停止，需检查：{exc}')
        return 1
    for item in report['files']:
        print(f"完成：{item['output']}  {item['geometry']['output_size']}")
    print('请检查边框上下沿和圆角；本模式未进行 UV 对齐。')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
