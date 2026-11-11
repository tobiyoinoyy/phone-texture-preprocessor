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
    command.add_argument('--bottom-alignment',choices=('appearance','off'),default='appearance',help='底面按可见结构校正；不能可靠校正时保留原底面并继续')
    command.add_argument('--resume-geometry-from',type=Path,help='从旧批次过程目录校验并复用本SKU完整几何；缺少完整几何则重新处理原图')
    command.add_argument('--require-geometry-resume',action='store_true',help='必须成功复用已绑定几何；缺失或失效时停止，不重新分割原图')
    command.add_argument('--source-body-reference-review',action='store_true',help='依据本机源图机身边界提供天线参考；适用时仍保留位置待核对状态')
    command.add_argument('--side-candidate-review',action='store_true',help='显式尝试来源约束的侧面复核候选；不等于严格材料或结构通过')
    command.add_argument('--saved-review-manifest',type=Path,help='续接清单中与当前六面原图精确匹配的已保存候选；不匹配的SKU仍走普通流程')
    command.add_argument('--saved-review-manifest-sha256',help='固定续接清单SHA-256；清单改变时拒绝运行')
    command.add_argument('--jobs', type=int, default=1, help='同时执行数 1–4；其他 SKU 自动排队')
    command.add_argument('--device', choices=('auto','cpu','mps','cuda'), default='auto')
    command.add_argument('--cpu-threads', type=int, default=1)
    command.add_argument('--experimental-concurrency', action='store_true', help='压测专用：最多 10 任务、4 份推理；需外部内存监测')
    command.add_argument('--inference-slots', type=int, help='同时推理名额，默认 min(jobs, 2)')
    command.add_argument('--skus', nargs='+')
    command.add_argument('--single-name', default='SKU_01')
    command.add_argument('--protection-policy',choices=('sticker_only','none'),default='none',help='none：不使用逐像素保护区；保留材质、机身与结构质量检查')
    command.add_argument('--lighting-profile',choices=('baseline','accepted-shadow-lift-v1','accepted-shadow-lift-v2'),default='accepted-shadow-lift-v2',help='按源图调光；整合版仍需视觉审核')
    command.add_argument('--review-fallback',choices=('fail','preserve-source'),default='preserve-source',help='保留原位置像素并输出待核对候选；fail保留严格对照')
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
    command.add_argument('--protection-policy',choices=('sticker_only','none'),default='sticker_only',help='none：不使用逐像素保护区；保留材质、机身与结构质量检查')
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
    command = sub.add_parser('web', help='打开六面自动处理与显式保存候选续接的本地浏览器界面')
    command.add_argument('--home', type=Path)
    command.add_argument('--port', type=int, default=18765)
    command.add_argument('--no-open', action='store_true')
    command.add_argument('--source-body-reference-review',action='store_true',help='为此服务的新批次显式启用本机源图天线参考复核')
    command.add_argument('--side-candidate-review',action='store_true',help='为此隔离服务的新批次显式启用侧面候选复核')
    command.add_argument('--resume-geometry-from',type=Path,help='新批次使用的几何检查点过程目录')
    command.add_argument('--require-geometry-resume',action='store_true',help='新批次必须成功复用几何，不启动重新分割')
    command.add_argument('--saved-review-manifest',type=Path,help='为此服务的新批次显式配置已保存候选续接清单')
    command.add_argument('--saved-review-manifest-sha256',help='固定续接清单SHA-256')
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
            if args.source_body_reference_review:
                options += ['--source-body-reference-review']
            if args.side_candidate_review:
                options += ['--side-candidate-review']
            if args.resume_geometry_from is not None:
                options += ['--resume-geometry-from', str(args.resume_geometry_from)]
            if args.require_geometry_resume:
                options += ['--require-geometry-resume']
            if args.saved_review_manifest is not None:
                options += ['--saved-review-manifest', str(args.saved_review_manifest)]
            if args.saved_review_manifest_sha256 is not None:
                options += ['--saved-review-manifest-sha256', args.saved_review_manifest_sha256]
            return browser_main(options)
        elif args.command == 'auto':
            if args.saved_review_manifest_sha256 is not None:
                digest = args.saved_review_manifest_sha256
                if (args.saved_review_manifest is None or len(digest) != 64
                        or any(c not in '0123456789abcdef' for c in digest)):
                    raise ValueError('保存候选清单 SHA-256 需与清单路径一同提供，并为64位小写十六进制')
                if workspace.sha(args.saved_review_manifest.expanduser().resolve()) != digest:
                    raise ValueError('保存候选清单已改变，未开始处理')
            from .automatic import run
            result = run(args.input,args.work,args.output,jobs=args.jobs,device=args.device,
                         cpu_threads=args.cpu_threads,skus=args.skus,single_name=args.single_name,
                         inference_slots=args.inference_slots,experimental_concurrency=args.experimental_concurrency,model=args.model,protection_policy=args.protection_policy,review_fallback=args.review_fallback,lighting_profile=args.lighting_profile,resume_geometry_from=args.resume_geometry_from,require_geometry_resume=args.require_geometry_resume,bottom_alignment=args.bottom_alignment,source_body_reference_review=args.source_body_reference_review,side_candidate_review=args.side_candidate_review,saved_review_manifest=args.saved_review_manifest,saved_review_manifest_sha256=args.saved_review_manifest_sha256)
            print(json.dumps({k:result.get(k) for k in ('status','processed','generated','source_preserved_candidates','total','visual_approved','seconds','output')},ensure_ascii=False,indent=2))
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
