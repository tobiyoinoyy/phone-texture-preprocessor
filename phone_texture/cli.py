"""Public CLI for the unified phone texture workflow."""
import argparse
import json
from pathlib import Path
from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog='phone-texture', description='六面图统一处理：几何 → 材质光照 → 本机背面对齐 → 验收发布')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('auto', help='仅原图：自动生成铝材六面候选、材料范围和本机背面天线标记（需最终视觉验收）')
    command.add_argument('input', type=Path)
    command.add_argument('--model', required=True, choices=('iphone17pro','iphone17promax'), help='导出贴图对应的固定 UV 模型')
    command.add_argument('--work', type=Path, help='过程目录；省略时自动创建独立目录')
    command.add_argument('--output', type=Path, help='六面候选目录；省略时自动创建独立目录')
    command.add_argument('--jobs', type=int, default=2, help='同时执行数 1–4；其他 SKU 自动排队')
    command.add_argument('--device', choices=('auto','cpu','mps','cuda'), default='auto')
    command.add_argument('--cpu-threads', type=int, default=2)
    command.add_argument('--experimental-concurrency', action='store_true', help='压测专用：最多 10 任务、4 份推理；需外部内存监测')
    command.add_argument('--inference-slots', type=int, help='同时推理名额，默认 min(jobs, 2)')
    command.add_argument('--skus', nargs='+')
    command.add_argument('--single-name', default='SKU_01')
    for mode, help_text in (('standardize', '原始六面照片：抠图、去支架、四面摆正、精裁'), ('deskew', '已裁切透明贴图：只摆正选定边框，原样保留其他面')):
        command = sub.add_parser(mode, help=help_text)
        command.add_argument('input', type=Path)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--process', type=Path, required=True)
        if mode == 'deskew':
            command.add_argument('--faces', nargs='+', choices=('left','right','top','bottom'), default=['left','right','top','bottom'])
        else:
            command.add_argument('--device', choices=('auto','cpu','mps','cuda'), default='auto')
            command.add_argument('--geometry-guides', '--frame-guides', dest='frame_guides', type=Path, help='原图坐标轮廓、局部去支架/保护蒙版、角度及哈希 JSON')
    command = sub.add_parser('approve-geometry', help='复核六面几何检查图后发布候选，不执行新图像处理')
    command.add_argument('process', type=Path)
    command.add_argument('--review', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('prepare', help='统一左右侧尺寸并创建结构标注页')
    command.add_argument('input', type=Path)
    command.add_argument('--process', type=Path, required=True)
    command.add_argument('--reference', required=True, help='按钮基准 SKU；不作为天线基准')
    command.add_argument('--width', type=int)
    command.add_argument('--height', type=int)
    from . import workspace
    workspace.add_commands(sub)
    command = sub.add_parser('start', help='新建完整六面处理批次，默认以每台自己的背面为基准')
    command.add_argument('input', type=Path)
    command.add_argument('--work', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--input-kind', choices=('raw','cutout','existing'), default='raw')
    command.add_argument('--material', choices=('aluminum','stainless','mixed'), default='aluminum')
    command.add_argument('--alignment-basis', choices=('back_image','front_back_images'), default='back_image')
    command.add_argument('--model-reference', type=Path)
    command.add_argument('--skus', nargs='+')
    command.add_argument('--single-name', default='SKU_01')
    command.add_argument('--delivery-goal', choices=('model_textures','photos'), default='model_textures')
    command.add_argument('--goal-reason')
    for name, help_text in [('run', '继续完整流程，缺少标定或复核时保存待办'), ('status', '查看本批进度与待办')]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument('work', type=Path)
    command = sub.add_parser('lighting', help='调用同库铝材或不锈钢光照引擎')
    command.add_argument('--material', choices=('aluminum','stainless'), required=True)
    command.add_argument('--job', type=Path, required=True)
    for name in ('model-review-template', 'verify-model'):
        command = sub.add_parser(name, help='固定模型六面坐标复核' + ('模板' if name.endswith('template') else ''))
        command.add_argument('--input', type=Path, required=True)
        command.add_argument('--model-reference', type=Path, required=True)
        command.add_argument('--output' if name.endswith('template') else '--review', type=Path, required=True)
    command = sub.add_parser('gui', help='打开完整六面处理窗口')
    command.add_argument('paths', nargs='*', type=Path)
    command = sub.add_parser('web', help='打开仅原图自动处理的本地浏览器界面')
    command.add_argument('--home', type=Path)
    command.add_argument('--port', type=int, default=18765)
    command.add_argument('--no-open', action='store_true')
    command = sub.add_parser('apply', help='使用逐 SKU 背板与 UV 标定，输出完整六面贴图')
    command.add_argument('work', type=Path)
    command.add_argument('--landmarks', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--process', type=Path, required=True)
    args = parser.parse_args(argv)
    from . import pipeline
    try:
        if args.command == 'web':
            from .web_app import main as browser_main
            options = ['--port', str(args.port)]
            if args.home is not None:
                options += ['--home', str(args.home)]
            if args.no_open:
                options += ['--no-open']
            return browser_main(options)
        elif args.command == 'auto':
            from .automatic import run
            result = run(args.input,args.work,args.output,jobs=args.jobs,device=args.device,
                         cpu_threads=args.cpu_threads,skus=args.skus,single_name=args.single_name,
                         inference_slots=args.inference_slots,experimental_concurrency=args.experimental_concurrency,model=args.model)
            print(json.dumps({k:result.get(k) for k in ('status','processed','total','visual_approved','seconds','output')},ensure_ascii=False,indent=2))
            return 0 if result['status']=='processed_needs_visual_review' else 2
        elif args.command == 'start':
            from .workflow import create
            options = vars(args).copy()
            options.pop('command')
            source = options.pop('input')
            result = create(source, **options)
        elif args.command in ('run', 'status'):
            from . import workflow
            result = getattr(workflow, args.command)(args.work)
        elif args.command == 'gui':
            from .desktop import main as desktop
            return desktop(args.paths)
        elif args.command == 'lighting':
            from .lighting import run
            result = run(args.material, args.job)
        elif args.command in ('model-review-template', 'verify-model'):
            from . import model_reference
            files = pipeline.batch_files(args.input)
            result = (model_reference.template(files, args.model_reference, args.output)
                      if args.command.endswith('template') else model_reference.verify(files, args.model_reference, args.review))
        elif args.command in ('intake','checkpoint','lighting-scope','perimeter-evidence','align-structure','publish'):
            result = workspace.dispatch(args)
        elif args.command in ('standardize','deskew'):
            pipeline.process_batch(args.input, args.output, args.process, mode=args.command,
                                   faces=getattr(args, 'faces', pipeline.FRAMES), device=getattr(args, 'device', 'auto'), frame_guides=getattr(args, 'frame_guides', None))
        elif args.command == 'approve-geometry':
            from .geometry_review import approve
            approve(args.process, args.review, args.output)
        elif args.command == 'prepare':
            pipeline.prepare(args.input, args.process, reference=args.reference, width=args.width, height=args.height)
        elif args.command == 'apply':
            pipeline.apply(args.work, args.landmarks, args.output, args.process)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(2, f'处理停止：{exc}\n')
    if args.command in ('standardize','deskew','approve-geometry','prepare','apply'):
        print(('候选已生成，须完成几何复核后发布。' if args.command in ('standardize','deskew') else '完成。') + f'过程资料：{args.process.resolve()}')
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
