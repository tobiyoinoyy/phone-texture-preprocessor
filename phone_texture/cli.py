"""Public CLI for the unified phone texture workflow."""
import argparse
from pathlib import Path
from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog='phone-texture', description='手机贴图预处理：标准化 → 边框摆正 → 结构对齐')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
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
    command = sub.add_parser('apply', help='使用逐 SKU 背板与 UV 标定，输出完整六面贴图')
    command.add_argument('work', type=Path)
    command.add_argument('--landmarks', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--process', type=Path, required=True)
    args = parser.parse_args(argv)
    from . import pipeline
    try:
        if args.command in ('standardize','deskew'):
            pipeline.process_batch(args.input, args.output, args.process, mode=args.command,
                                   faces=getattr(args, 'faces', pipeline.FRAMES), device=getattr(args, 'device', 'auto'), frame_guides=getattr(args, 'frame_guides', None))
        elif args.command == 'approve-geometry':
            from .geometry_review import approve
            approve(args.process, args.review, args.output)
        elif args.command == 'prepare':
            pipeline.prepare(args.input, args.process, reference=args.reference, width=args.width, height=args.height)
        else:
            pipeline.apply(args.work, args.landmarks, args.output, args.process)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(2, f'处理停止：{exc}\n')
    print(('候选已生成，须完成几何复核后发布。' if args.command in ('standardize','deskew') else '完成。') + f'过程资料：{args.process.resolve()}')
    return 0
